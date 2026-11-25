"""CPU tests for the LTX-2.5 diffusion-decoder fast path (nunchaku_lite.adapters.ltx2_decoder).

The CUDA kernel itself is covered by nunchaku-lite-kernels/tests; here a reference implementation of the op stands in
for it, so these tests check what nunchaku-lite controls: the per-axis tables, the arguments passed to the op, and the
install/fallback behaviour.
"""

import pytest
import torch
import torch.nn.functional as F

ltx2_decoder = pytest.importorskip("diffusers.models.autoencoders.ltx2_diffusion_decoder")

from nunchaku_lite.adapters import ltx2_decoder as adapter
from nunchaku_lite.ops import fused


def _reference_op(q, k, q_weight, k_weight, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w, q_scale, eps):
    """Eager stand-in for nunchaku_lite_kernels.per_head_qk_norm_rope3d_ (same contract, in place)."""
    tables = ((cos_t, sin_t, 1), (cos_h, sin_h, 2), (cos_w, sin_w, 3))
    for x, weight, scale in ((q, q_weight, q_scale), (k, k_weight, 1.0)):
        y = F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)
        if scale != 1.0:
            y = y * scale
        out, start = [], 0
        for cos, sin, axis in tables:
            dim = 2 * cos.shape[1]
            shape = [1, 1, 1, 1, 1, cos.shape[1]]
            shape[axis] = cos.shape[0]
            c, s = cos.reshape(shape), sin.reshape(shape)
            chunk = y[..., start : start + dim]
            pairs = chunk.reshape(*chunk.shape[:-1], dim // 2, 2)
            even, odd = pairs[..., 0].float(), pairs[..., 1].float()
            out.append(torch.stack([even * c - odd * s, even * s + odd * c], -1).reshape(chunk.shape).to(x.dtype))
            start += dim
        x.copy_(torch.cat(out, -1))


class _FakeOps:
    per_head_qk_norm_rope3d_ = staticmethod(_reference_op)


def _attention(head_dim=64, heads=4):
    torch.manual_seed(0)
    attn = ltx2_decoder.LTX2VideoVaeNeighborhoodAttention(head_dim * heads, (3, 3, 3), head_dim=head_dim)
    with torch.no_grad():
        attn.norm_q.weight.uniform_(0.5, 1.5)
        attn.norm_k.weight.uniform_(0.5, 1.5)
    return attn


@pytest.mark.parametrize("head_dim", [64, 128])
def test_axis_tables_match_the_module_rotation(head_dim):
    attn = _attention(head_dim)
    grid = (5, 6, 7)
    tables = fused.rope3d_axis_tables(attn.rope, *grid, torch.device("cpu"))
    assert len(tables) == 6
    for axis, (dim, length) in enumerate(zip(attn.rope.rope_dim_split, grid)):
        cos, sin = tables[2 * axis], tables[2 * axis + 1]
        assert cos.shape == sin.shape == (length, dim // 2)
        assert cos.dtype == torch.float32 and cos.is_contiguous()
        angles = torch.arange(length, dtype=torch.float32)[:, None] * attn.rope._inv_freqs(dim, torch.device("cpu"))
        assert torch.equal(cos, angles.cos()) and torch.equal(sin, angles.sin())
    assert sum(t.shape[1] for t in tables[::2]) == head_dim // 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("grid", [(5, 6, 7), (9, 4, 3)])
def test_wrapper_reproduces_project_qkv(dtype, grid):
    attn = _attention().to(dtype)
    x = torch.randn(2, *grid, 256, dtype=dtype)
    with torch.no_grad():
        ref_q, ref_k, _ = attn.project_qkv(x)
        shape = (2, *grid, attn.heads, attn.head_dim)
        q = attn.to_q(x).view(shape).contiguous()
        k = attn.to_k(x).view(shape).contiguous()
        fused._per_head_qk_norm_rope3d(q, k, attn.norm_q, attn.norm_k, attn.rope, attn.scale, _FakeOps())
    assert torch.equal(q, ref_q)
    assert torch.equal(k, ref_k)


def test_wrapper_mutation_swapped_norms_is_detected():
    attn = _attention()
    x = torch.randn(1, 5, 6, 7, 256)
    with torch.no_grad():
        ref_q, _, _ = attn.project_qkv(x)
        q = attn.to_q(x).view(1, 5, 6, 7, attn.heads, attn.head_dim).contiguous()
        k = attn.to_k(x).view(1, 5, 6, 7, attn.heads, attn.head_dim).contiguous()
        fused._per_head_qk_norm_rope3d(q, k, attn.norm_k, attn.norm_q, attn.rope, attn.scale, _FakeOps())
    assert not torch.allclose(q, ref_q, atol=1e-3)


def test_wrapper_rejects_mismatched_eps():
    attn = _attention()
    attn.norm_k.eps = 1e-5
    q = torch.zeros(1, 3, 3, 3, attn.heads, attn.head_dim)
    with pytest.raises(ValueError, match="eps"):
        fused._per_head_qk_norm_rope3d(q, q.clone(), attn.norm_q, attn.norm_k, attn.rope, attn.scale, _FakeOps())


def test_wrapper_requires_cuda():
    attn = _attention()
    q = torch.zeros(1, 3, 3, 3, attn.heads, attn.head_dim)
    with pytest.raises(ValueError, match="CUDA"):
        fused.per_head_qk_norm_rope3d_(q, q.clone(), attn.norm_q, attn.norm_k, attn.rope, attn.scale)


def test_optimize_leaves_decoder_unchanged_without_the_op(monkeypatch):
    monkeypatch.setattr(adapter, "has_per_head_qk_norm_rope3d", lambda: False)
    attn = _attention()
    before = attn.processor
    assert adapter.optimize_ltx2_diffusion_decoder(torch.nn.ModuleList([attn])) is False
    assert attn.processor is before


def test_optimize_installs_on_every_attention_module(monkeypatch):
    monkeypatch.setattr(adapter, "has_per_head_qk_norm_rope3d", lambda: True)

    class _NoNatten(adapter.LTX2DecoderFusedNattenProcessor):
        def __init__(self, backend=None):
            self.backend = backend

    monkeypatch.setattr(adapter, "LTX2DecoderFusedNattenProcessor", _NoNatten)
    mods = torch.nn.ModuleList([_attention(), torch.nn.Linear(4, 4), _attention()])
    assert adapter.optimize_ltx2_diffusion_decoder(mods) is True
    assert all(isinstance(m.processor, _NoNatten) for m in mods if hasattr(m, "rope"))


def test_optimize_rejects_a_model_without_neighborhood_attention(monkeypatch):
    monkeypatch.setattr(adapter, "has_per_head_qk_norm_rope3d", lambda: True)
    monkeypatch.setattr(adapter, "LTX2DecoderFusedNattenProcessor", lambda backend=None: object())
    with pytest.raises(ValueError, match="no LTX-2.5 neighborhood-attention"):
        adapter.optimize_ltx2_diffusion_decoder(torch.nn.Linear(4, 4))
