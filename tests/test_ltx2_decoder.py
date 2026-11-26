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


def _reference_add_rmsnorm_modulate_(x, residual, weight, one_plus_scale, shift, eps):
    """Eager stand-in for nunchaku_lite_kernels.add_rmsnorm_modulate_ (same contract)."""
    assert one_plus_scale.shape == shift.shape == (x.shape[0], x.shape[-1])
    x.add_(residual)
    view = (x.shape[0],) + (1,) * (x.dim() - 2) + (x.shape[-1],)
    return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps) * one_plus_scale.view(view) + shift.view(view)


def _reference_silu_mul(gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


class _FakeOps:
    per_head_qk_norm_rope3d_ = staticmethod(_reference_op)
    add_rmsnorm_modulate_ = staticmethod(_reference_add_rmsnorm_modulate_)
    silu_mul = staticmethod(_reference_silu_mul)


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
    monkeypatch.setattr(adapter, "has_ltx2_decoder_block_ops", lambda: False)
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


# ---- stage-5 block fusions


class _LinearOnlyProcessor:
    """CPU stand-in for the NATTEN/flex processors: same projections, no neighborhood attention."""

    def __call__(self, attn, hidden_states, block_mask=None):
        return attn.to_out[0](attn.to_v(hidden_states))


def _block(dim=64, context=32, dtype=torch.float32):
    torch.manual_seed(0)
    block = ltx2_decoder.LTX2VideoVaeDiffusionNABlock(dim, (3, 3, 3), context, head_dim=32)
    with torch.no_grad():
        block.scale_shift_table.normal_(0, 0.5)
        block.norm1.weight.uniform_(0.5, 1.5)
        block.norm2.weight.uniform_(0.5, 1.5)
    block.attn.set_processor(_LinearOnlyProcessor())
    return block.to(dtype).eval()


def _block_inputs(block, batch=2, grid=(3, 4, 5), dtype=torch.float32):
    dim, context = block.context_proj.out_features, block.context_proj.in_features
    hidden = torch.randn(batch, *grid, dim, dtype=dtype)
    latent_context = torch.randn(batch, *grid, context, dtype=dtype)
    modulation = tuple(torch.randn(batch, 1, 1, 1, dim, dtype=dtype) * 0.5 for _ in range(7))
    return hidden, latent_context, modulation


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tile", [65536, 7])
def test_fused_block_reproduces_the_diffusers_block(monkeypatch, dtype, tile):
    monkeypatch.setattr(adapter, "SWIGLU_TILE_TOKENS", tile)  # 7 exercises the ragged multi-tile path
    block = _block(dtype=dtype)
    hidden, latent_context, modulation = _block_inputs(block, dtype=dtype)
    before = hidden.clone()
    with torch.no_grad():
        ref = block(hidden, latent_context, modulation)
        got = adapter.fused_diffusion_block_forward(block, hidden, latent_context, modulation, ops=_FakeOps())
    assert torch.equal(hidden, before), "the block input must not be modified in place"
    assert got.shape == ref.shape and got.dtype == ref.dtype
    # merged gate/up matmul may reduce in a different order than two separate ones
    torch.testing.assert_close(got, ref, rtol=0, atol=1e-5 if dtype == torch.float32 else 0)


@pytest.mark.parametrize(
    "mutate",
    ["swap_msa_mlp", "no_context", "scale_not_one_plus", "swap_gate_up"],
)
def test_fused_block_mutations_are_detected(monkeypatch, mutate):
    block = _block()
    hidden, latent_context, modulation = _block_inputs(block)
    with torch.no_grad():
        ref = block(hidden, latent_context, modulation)
        if mutate == "swap_msa_mlp":
            m = list(modulation)
            m[0], m[3] = m[3], m[0]
            got = adapter.fused_diffusion_block_forward(block, hidden, latent_context, tuple(m), ops=_FakeOps())
        elif mutate == "no_context":
            got = adapter.fused_diffusion_block_forward(block, hidden, torch.zeros_like(latent_context), modulation,
                                                        ops=_FakeOps())
        elif mutate == "scale_not_one_plus":
            real = adapter.add_rmsnorm_modulate_
            monkeypatch.setattr(adapter, "add_rmsnorm_modulate_",
                                lambda x, r, n, ops_, sh, ops=None: real(x, r, n, ops_ - 1, sh, ops=ops))
            got = adapter.fused_diffusion_block_forward(block, hidden, latent_context, modulation, ops=_FakeOps())
        else:
            mlp = block.mlp
            setattr(mlp, adapter._GATE_UP_ATTR, torch.cat([mlp.w_up.weight, mlp.w_gate.weight]))
            got = adapter.fused_diffusion_block_forward(block, hidden, latent_context, modulation, ops=_FakeOps())
    assert not torch.allclose(got, ref, atol=1e-3)


def test_add_rmsnorm_modulate_falls_back_to_eager_on_cpu():
    norm = torch.nn.RMSNorm(16, eps=1e-6)
    x, r = torch.randn(2, 3, 16), torch.randn(2, 3, 16)
    scale, shift = torch.randn(2, 1, 16), torch.randn(2, 1, 16)
    ref = norm(x + r) * (1 + scale) + shift
    x2 = x.clone()
    got = fused.add_rmsnorm_modulate_(x2, r, norm, 1 + scale, shift)
    assert torch.equal(x2, x + r)
    assert torch.equal(got, ref)


def test_silu_mul_falls_back_to_eager_on_cpu():
    gate_up = torch.randn(5, 32)
    assert torch.equal(fused.silu_mul(gate_up), F.silu(gate_up[:, :16]) * gate_up[:, 16:])


def test_optimize_installs_fused_blocks_and_stage5_tiles(monkeypatch):
    monkeypatch.setattr(adapter, "has_per_head_qk_norm_rope3d", lambda: True)
    monkeypatch.setattr(adapter, "has_ltx2_decoder_block_ops", lambda: True)

    class _NoNatten(adapter.LTX2DecoderFusedNattenProcessor):
        def __init__(self, backend=None, q_tile_shape=None, kv_tile_shape=None):
            self.backend, self.tiles = backend, (q_tile_shape, kv_tile_shape)

    monkeypatch.setattr(adapter, "LTX2DecoderFusedNattenProcessor", _NoNatten)
    upsampling_attn, block = _attention(), _block()
    model = torch.nn.ModuleList([upsampling_attn, block])
    tiles = ((4, 4, 4), (8, 4, 4))
    assert adapter.optimize_ltx2_diffusion_decoder(model, natten_tile_shapes=tiles) is True
    assert upsampling_attn.processor.tiles == (None, None)
    assert block.attn.processor.tiles == tiles
    assert block.forward.__func__ is adapter.fused_diffusion_block_forward


def test_optimize_keeps_blocks_without_the_block_ops(monkeypatch):
    monkeypatch.setattr(adapter, "has_per_head_qk_norm_rope3d", lambda: False)
    monkeypatch.setattr(adapter, "has_ltx2_decoder_block_ops", lambda: False)
    block = _block()
    assert adapter.optimize_ltx2_diffusion_decoder(torch.nn.ModuleList([block])) is False
    assert "forward" not in vars(block)
