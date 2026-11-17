"""Fused FLUX.2 q/k norm + RoPE on MPS: kernel numerics and the Diffusers processor swap."""

import pytest
import torch
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers import transformer_flux2 as flux2

from nunchaku_lite.adapters.flux2 import (
    MPSFusedFlux2AttnProcessor,
    MPSFusedFlux2ParallelSelfAttnProcessor,
    install_mps_fused_attention,
)
from nunchaku_lite.ops.mps_fused import fused_qk_norm_rope_available, qk_norm_rope_mps

needs_mps = pytest.mark.skipif(not fused_qk_norm_rope_available(), reason="requires MPS with torch.mps.compile_shader")
DIM, HEADS = 256, 2


def _tables(n, device):
    ang = torch.rand(n, 64, device=device) * 6.28
    return torch.cos(ang).repeat_interleave(2, -1), torch.sin(ang).repeat_interleave(2, -1)


def _modules(device, dtype):
    torch.manual_seed(0)
    joint = flux2.Flux2Attention(DIM, heads=HEADS, dim_head=128, added_kv_proj_dim=DIM, eps=1e-6)
    single = flux2.Flux2ParallelSelfAttention(DIM, heads=HEADS, dim_head=128, mlp_ratio=3.0, eps=1e-6)
    for m in (joint, single):
        for name, p in m.named_parameters():
            if name.startswith("norm"):
                p.data.uniform_(0.5, 1.5)
    return joint.to(device, dtype), single.to(device, dtype)


def _run(joint, single, device, dtype, lt=5, li=11):
    torch.manual_seed(1)
    x = torch.randn(1, li, DIM, device=device, dtype=dtype)
    ctx = torch.randn(1, lt, DIM, device=device, dtype=dtype)
    rope = _tables(lt + li, device)
    with torch.no_grad():
        img, txt = joint(x, ctx, image_rotary_emb=rope)
        out = single(torch.cat([ctx, x], dim=1), image_rotary_emb=rope)
    return img, txt, out


def test_install_swaps_stock_processors_and_falls_back_off_mps():
    joint, single = _modules("cpu", torch.float32)
    ref = _run(joint, single, "cpu", torch.float32)
    root = torch.nn.ModuleList([joint, single])
    assert install_mps_fused_attention(root) == 2
    assert install_mps_fused_attention(root) == 0  # already swapped
    assert isinstance(joint.processor, MPSFusedFlux2AttnProcessor)
    assert isinstance(single.processor, MPSFusedFlux2ParallelSelfAttnProcessor)
    for got, exp in zip(_run(joint, single, "cpu", torch.float32), ref):
        torch.testing.assert_close(got, exp, rtol=0, atol=0)  # CPU: the stock processor runs unchanged


@needs_mps
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_processors_match_stock_on_mps(dtype):
    joint, single = _modules("mps", dtype)
    ref = _run(joint, single, "mps", dtype)
    install_mps_fused_attention(torch.nn.ModuleList([joint, single]))
    for got, exp in zip(_run(joint, single, "mps", dtype), ref):
        tol = (3e-2 if dtype == torch.bfloat16 else 5e-3) * exp.float().abs().max().item()
        torch.testing.assert_close(got.float(), exp.float(), rtol=0, atol=tol)


@needs_mps
def test_qk_norm_rope_reads_column_slices():
    lt, li, width = 3, 9, HEADS * 128
    proj = torch.randn(1, lt + li, 5 * width, device="mps", dtype=torch.float16) * 2
    q, k, v = (proj[..., i * width : (i + 1) * width] for i in (1, 3, 0))
    wq, wk = torch.rand(128, device="mps") + 0.5, torch.rand(128, device="mps") + 0.5
    cos, sin = _tables(lt + li, "mps")
    fq, fk, fv = qk_norm_rope_mps(q, k, v, wq, wk, cos, sin, HEADS, eps_q=1e-6, eps_k=1e-6)

    def ref(t, w):
        t = t.unflatten(-1, (HEADS, 128)).float()
        t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * w
        return apply_rotary_emb(t, (cos, sin), sequence_dim=1)

    torch.testing.assert_close(fq.float(), ref(q, wq), rtol=0, atol=4e-3 * ref(q, wq).abs().max().item())
    torch.testing.assert_close(fk.float(), ref(k, wk), rtol=0, atol=4e-3 * ref(k, wk).abs().max().item())
    torch.testing.assert_close(fv, v.unflatten(-1, (HEADS, 128)), rtol=0, atol=0)


@needs_mps
def test_qk_norm_rope_rejects_mismatched_tables():
    x = torch.zeros(1, 4, 128, device="mps", dtype=torch.float16)
    cos, sin = _tables(5, "mps")
    with pytest.raises(ValueError):
        qk_norm_rope_mps(x, x, x, None, None, cos, sin, 1, eps_q=1e-6, eps_k=1e-6)
