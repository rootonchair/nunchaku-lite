"""Fused q/k norm + RoPE and Neural Accelerator attention on MPS: kernel numerics and the processor swaps.

MPS tests run twice: with the ``nunchaku_lite_kernels`` ops ("hub") and with the runtime-compiled fallback.
"""

import pytest
import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers import transformer_flux as flux1
from diffusers.models.transformers import transformer_flux2 as flux2
from diffusers.models.transformers import transformer_qwenimage as qwen
from diffusers.models.transformers import transformer_z_image as z_image

from nunchaku_lite.adapters.mps_processors import (
    MPSFusedFlux2AttnProcessor,
    MPSFusedFlux2ParallelSelfAttnProcessor,
    install_mps_fused_processors,
)
from nunchaku_lite.adapters.qwen_image import NunchakuQwenImageAttnProcessor
from nunchaku_lite.ops.mps import (
    attention_mps,
    attention_mps_available,
    attention_mps_supported,
    qk_norm_rope_mps,
    qk_norm_rope_mps_available,
)

needs_mps = pytest.mark.skipif(not qk_norm_rope_mps_available(), reason="requires MPS with torch.mps.compile_shader")
DIM, HEADS = 256, 2


@pytest.fixture(autouse=True, params=["hub", "runtime"])
def kernel_source(request, monkeypatch):
    """Run against the nunchaku_lite_kernels ops, then against the runtime-compiled kernels."""

    from nunchaku_lite.ops import mps as mps_ops

    if request.param == "runtime":
        monkeypatch.setattr(mps_ops, "_lite_op", lambda name: None)
    elif not torch.backends.mps.is_available() or mps_ops._lite_op("attention_mps") is None:
        pytest.skip("the installed nunchaku_lite_kernels build has no attention ops")
    return request.param


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
    assert install_mps_fused_processors(root) == 2
    assert install_mps_fused_processors(root) == 0  # already swapped
    assert isinstance(joint.processor, MPSFusedFlux2AttnProcessor)
    assert isinstance(single.processor, MPSFusedFlux2ParallelSelfAttnProcessor)
    for got, exp in zip(_run(joint, single, "cpu", torch.float32), ref):
        torch.testing.assert_close(got, exp, rtol=0, atol=0)  # CPU: the stock processor runs unchanged


@needs_mps
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_processors_match_stock_on_mps(dtype):
    joint, single = _modules("mps", dtype)
    ref = _run(joint, single, "mps", dtype)
    install_mps_fused_processors(torch.nn.ModuleList([joint, single]))
    for got, exp in zip(_run(joint, single, "mps", dtype), ref):
        tol = (3e-2 if dtype == torch.bfloat16 else 5e-3) * exp.float().abs().max().item()
        torch.testing.assert_close(got.float(), exp.float(), rtol=0, atol=tol)


@pytest.fixture
def kernel_calls(monkeypatch):
    """Counts fused-kernel calls from the processors, so a silent fallback to the stock code fails the test."""
    from nunchaku_lite.adapters import mps_processors

    calls = []
    real = mps_processors.qk_norm_rope_mps
    monkeypatch.setattr(mps_processors, "qk_norm_rope_mps", lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def _randomize_norms(*modules):
    for m in modules:
        for name, p in m.named_parameters():
            if name.startswith("norm"):
                p.data.uniform_(0.5, 1.5)


def _complex_tables(n, device):
    return torch.polar(torch.ones(n, 64), torch.rand(n, 64) * 6.28).to(device)


def _flux1_case(device, dtype):
    torch.manual_seed(0)
    joint = flux1.FluxAttention(
        query_dim=DIM, added_kv_proj_dim=DIM, dim_head=128, heads=HEADS, out_dim=DIM, context_pre_only=False, eps=1e-6
    )
    single = flux1.FluxAttention(query_dim=DIM, dim_head=128, heads=HEADS, out_dim=DIM, eps=1e-6, pre_only=True)
    _randomize_norms(joint, single)
    joint, single = joint.to(device, dtype), single.to(device, dtype)
    x, ctx = (torch.randn(1, n, DIM, device=device, dtype=dtype) for n in (11, 5))
    rope = _tables(16, device)

    def run():
        img, txt = joint(hidden_states=x, encoder_hidden_states=ctx, image_rotary_emb=rope)
        return img, txt, single(hidden_states=torch.cat([ctx, x], dim=1), image_rotary_emb=rope)

    return (joint, single), run, 2


def _qwen_attention(device, dtype):
    attn = Attention(
        query_dim=DIM, added_kv_proj_dim=DIM, dim_head=128, heads=HEADS, out_dim=DIM, context_pre_only=False,
        bias=True, processor=qwen.QwenDoubleStreamAttnProcessor2_0(), qk_norm="rms_norm", eps=1e-6,
    )  # fmt: skip
    _randomize_norms(attn)
    return attn.to(device, dtype)


def _qwen_case(device, dtype, mask=False):
    torch.manual_seed(0)
    attn = _qwen_attention(device, dtype)
    img, txt = torch.randn(1, 11, DIM, device=device, dtype=dtype), torch.randn(1, 5, DIM, device=device, dtype=dtype)
    rope = (_complex_tables(11, device), _complex_tables(5, device))
    text_mask = torch.tensor([[True, True, True, False, False]], device=device) if mask else None

    def run():
        return attn(
            hidden_states=img, encoder_hidden_states=txt, encoder_hidden_states_mask=text_mask, image_rotary_emb=rope
        )

    return (attn,), run, 1


def _z_image_case(device, dtype):
    torch.manual_seed(0)
    attn = Attention(
        query_dim=DIM, dim_head=128, heads=HEADS, qk_norm="rms_norm", eps=1e-5, bias=False, out_bias=False,
        processor=z_image.ZSingleStreamAttnProcessor(),
    )  # fmt: skip
    _randomize_norms(attn)
    attn = attn.to(device, dtype)
    x = torch.randn(1, 13, DIM, device=device, dtype=dtype)
    freqs = _complex_tables(13, device)[None]
    mask = torch.tensor([[True] * 10 + [False] * 3], device=device)

    def run():
        return attn(x, freqs_cis=freqs), attn(x, attention_mask=mask, freqs_cis=freqs)

    return (attn,), run, 2


_CASES = {
    "flux1": _flux1_case,
    "qwen_image": _qwen_case,
    "qwen_image_masked": lambda d, t: _qwen_case(d, t, mask=True),
    "z_image": _z_image_case,
}


def _assert_matches(got, expected, dtype):
    got = got if isinstance(got, tuple) else (got,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    for g, e in zip(got, expected, strict=True):
        tol = (3e-2 if dtype == torch.bfloat16 else 5e-3) * e.float().abs().max().item()
        torch.testing.assert_close(g.float(), e.float(), rtol=0, atol=tol)


@needs_mps
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("family", sorted(_CASES))
def test_fused_stock_processors_match_stock_on_mps(family, dtype, kernel_calls):
    modules, run, calls = _CASES[family]("mps", dtype)
    with torch.no_grad():
        expected = run()
        assert install_mps_fused_processors(torch.nn.ModuleList(modules)) == len(modules)
        got = run()
    assert len(kernel_calls) == calls
    _assert_matches(got, expected, dtype)


@needs_mps
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nunchaku_qwen_processor_fuses_on_mps(dtype, kernel_calls):
    from types import SimpleNamespace

    torch.manual_seed(0)
    stock = _qwen_attention("mps", dtype)

    def fuse(names):
        layers = [getattr(stock, n) for n in names]
        fused = torch.nn.Linear(DIM, 3 * DIM).to("mps", dtype)
        fused.weight.data = torch.cat([layer.weight for layer in layers])
        fused.bias.data = torch.cat([layer.bias for layer in layers])
        return fused

    attn = SimpleNamespace(
        heads=HEADS, to_qkv=fuse(("to_q", "to_k", "to_v")), add_qkv_proj=fuse(("add_q_proj", "add_k_proj", "add_v_proj")),
        norm_q=stock.norm_q, norm_k=stock.norm_k, norm_added_q=stock.norm_added_q, norm_added_k=stock.norm_added_k,
        to_out=stock.to_out, to_add_out=stock.to_add_out,
    )  # fmt: skip
    img, txt = torch.randn(1, 11, DIM, device="mps", dtype=dtype), torch.randn(1, 5, DIM, device="mps", dtype=dtype)
    rope = (_complex_tables(11, "mps"), _complex_tables(5, "mps"))
    with torch.no_grad():
        expected = stock(hidden_states=img, encoder_hidden_states=txt, image_rotary_emb=rope)
        got = NunchakuQwenImageAttnProcessor()(attn, img, txt, image_rotary_emb=rope)
    assert len(kernel_calls) == 1
    _assert_matches(got, expected, dtype)


def test_install_swaps_every_supported_family_off_mps():
    flux_modules, _, _ = _flux1_case("cpu", torch.float32)
    qwen_modules, _, _ = _qwen_case("cpu", torch.float32)
    z_modules, run, _ = _z_image_case("cpu", torch.float32)
    expected = run()
    root = torch.nn.ModuleList([*flux_modules, *qwen_modules, *z_modules])
    assert install_mps_fused_processors(root) == 4
    assert install_mps_fused_processors(root) == 0
    _assert_matches(run(), expected, torch.float32)  # CPU: the stock processor runs (bit-exact)


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


needs_nax = pytest.mark.skipif(not attention_mps_available(), reason="requires an M5-class GPU (Neural Accelerators)")


@needs_nax
@pytest.mark.parametrize(
    "batch,q_len,k_len,heads,kv_heads,dim,dtype",
    [
        (1, 4608, 4608, 24, 24, 128, torch.bfloat16),  # FLUX.2-klein 1024², aligned
        (2, 1000, 777, 8, 2, 64, torch.float16),  # unaligned lengths, grouped-query heads
        (1, 300, 300, 4, 4, 96, torch.bfloat16),
    ],
)
def test_attention_mps_matches_float32_reference(batch, q_len, k_len, heads, kv_heads, dim, dtype):
    torch.manual_seed(0)
    q = torch.randn(batch, q_len, heads, dim, device="mps", dtype=dtype) * 2
    k = torch.randn(batch, k_len, kv_heads, dim, device="mps", dtype=dtype) * 2
    v = torch.randn(batch, k_len, kv_heads, dim, device="mps", dtype=dtype)
    assert attention_mps_supported(q, k, v)
    out = attention_mps(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(
        *(t.transpose(1, 2).float() for t in (q, k, v)), enable_gqa=True
    ).transpose(1, 2)
    assert out.shape == q.shape and out.dtype == dtype and out.is_contiguous()
    tol = (1e-2 if dtype == torch.bfloat16 else 2e-3) * ref.abs().max().item()
    torch.testing.assert_close(out.float(), ref, rtol=0, atol=tol)


@needs_nax
def test_attention_mps_reads_strided_views():
    torch.manual_seed(0)
    qkv = torch.randn(1, 200, 3, 4, 128, device="mps", dtype=torch.float16)  # (B, L, 3, H, D): q/k/v interleaved
    q, k, v = qkv.unbind(2)
    ref = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2).float() for t in (q, k, v)))
    torch.testing.assert_close(attention_mps(q, k, v).float(), ref.transpose(1, 2), rtol=0, atol=2e-3)


def test_attention_supported_rejects_unsupported_inputs():
    q = torch.zeros(1, 8, 2, 128)
    assert not attention_mps_supported(q, q, q)  # CPU


class _Attend(torch.nn.Module):
    """Stand-in transformer: attention through Diffusers' dispatch and through the adapters' lite dispatch."""

    def forward(self, q, k, v, mask=None, fail=False):
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        from nunchaku_lite.adapters.attention_dispatch import dispatch_lite_attention_fn

        if fail:
            raise RuntimeError("boom")
        a = dispatch_attention_fn(q, k, v, attn_mask=mask)
        b = dispatch_lite_attention_fn(
            *(t.transpose(1, 2) for t in (q, k, v)), attn_mask=mask, query_layout="bhsd", output_layout="bhsd"
        ).transpose(1, 2)
        return a, b


def test_mps_attention_scope_resets_and_is_noop_off_mps():
    from nunchaku_lite.adapters import attention_dispatch as ad

    module = _Attend()
    q = torch.randn(1, 6, 2, 64)
    expected = module(q, q, q)
    assert ad.install_mps_attention(module)
    assert not ad.install_mps_attention(module)  # idempotent
    for got, exp in zip(module(q, q, q), expected):
        torch.testing.assert_close(got, exp)  # CPU tensors: PyTorch's attention, unchanged
    with pytest.raises(RuntimeError):
        module(q, q, q, fail=True)
    assert not ad._MPS_ATTENTION.get()  # reset by the always-called forward hook


@needs_nax
def test_mps_attention_only_inside_patched_forward(monkeypatch):
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    from nunchaku_lite.adapters import attention_dispatch as ad

    calls = []
    real = ad.attention_mps
    monkeypatch.setattr(ad, "attention_mps", lambda *a, **k: calls.append(1) or real(*a, **k))
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 130, 4, 128, device="mps", dtype=torch.bfloat16) for _ in range(3))
    mask = torch.ones(1, 130, dtype=torch.bool, device="mps")
    mask[:, 100:] = False
    module = _Attend()
    ad.install_mps_attention(module)

    dispatch_attention_fn(q, k, v)
    assert calls == []  # outside a patched forward: PyTorch's attention

    ref = torch.nn.functional.scaled_dot_product_attention(
        *(t.transpose(1, 2).float() for t in (q, k, v)), attn_mask=mask[:, None, None, :]
    ).transpose(1, 2)
    for got in module(q, k, v, mask):
        torch.testing.assert_close(got.float(), ref, rtol=0, atol=1e-2 * ref.abs().max().item())
    assert len(calls) == 2


@needs_mps
def test_qkv_norm_rotary_in_place_matches_fallback(monkeypatch):
    """The Nunchaku processors' path: shared kernel in place on the dense qkv (batch 2) vs the PyTorch fallback."""
    import torch.nn.functional as F

    from nunchaku_lite.adapters.common import pack_rotemb
    from nunchaku_lite.ops import mps as mps_ops

    torch.manual_seed(0)
    batch, seq, heads = 2, 40, 2
    qkv = (torch.randn(batch, seq, 3 * heads * 128) * 3).to("mps", torch.bfloat16)
    monkeypatch.setattr(mps_ops, "svdq_linear_mps", lambda proj, x: qkv.clone())
    wq, wk = torch.rand(128, device="mps") + 0.5, torch.rand(128, device="mps") + 0.5
    angles = torch.randn(seq, 64)
    rotemb = torch.stack([angles.sin(), angles.cos()], dim=-1).unsqueeze(0).unsqueeze(-2)
    packed = pack_rotemb(F.pad(rotemb, (0, 0, 0, 0, 0, 0, 0, 256 - seq))).to("mps")
    x = torch.zeros(batch, seq, 8, device="mps", dtype=torch.bfloat16)
    fused = mps_ops.qkv_norm_rotary_mps(x, None, wq, wk, packed, eps_q=1e-6, eps_k=1e-5)
    monkeypatch.setattr(mps_ops, "qk_norm_rope_mps_available", lambda: False)
    ref = mps_ops.qkv_norm_rotary_mps(x, None, wq, wk, packed, eps_q=1e-6, eps_k=1e-5)
    torch.testing.assert_close(fused.float(), ref.float(), rtol=0, atol=2e-2 * ref.float().abs().max().item())


def test_hub_shader_header_in_sync():
    """nunchaku-lite-kernels compiles a generated copy of the shared kernels; regenerate it after changing them."""
    import subprocess
    import sys
    from pathlib import Path

    tool = Path(__file__).resolve().parents[1] / "tools/sync_hub_shaders.py"
    result = subprocess.run([sys.executable, str(tool), "--check"], check=False)
    assert result.returncode == 0, "run: python tools/sync_hub_shaders.py"


def test_shared_shader_sources_import_no_framework():
    import subprocess
    import sys

    code = "import sys, nunchaku_lite.shaders; assert 'torch' not in sys.modules and 'mlx' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
