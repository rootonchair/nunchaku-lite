import importlib.util

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from nunchaku_lite.mlx.fused_attention import install_fused_attention
from nunchaku_lite.mlx.kernels import qk_norm_rope

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires a Metal GPU")


def _norm_rope_ref(x, w, cos, sin, eps=1e-5):
    """(L, H, 128) float64 reference: RMSNorm over the head dim, then interleaved-pair rotation."""
    x = x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w
    re, im = x[..., 0::2], x[..., 1::2]
    c, s = cos[:, None, :], sin[:, None, :]
    out = np.empty_like(x)
    out[..., 0::2] = re * c - im * s
    out[..., 1::2] = im * c + re * s
    return out


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("text_rows", [0, 7])
def test_qk_norm_rope_matches_reference(dtype, text_rows):
    rng = np.random.default_rng(0)
    heads, li, width = 3, 21, 3 * 128
    stride = 5 * width  # q/k/v inside a wider projection row (as in the fused single-block projection)
    offs = (width, 3 * width, 0)
    img = rng.standard_normal((li, stride)).astype(np.float32) * 2
    txt = rng.standard_normal((text_rows, stride - width)).astype(np.float32) * 2 if text_rows else None
    w = {n: rng.uniform(0.5, 1.5, 128).astype(np.float32) for n in ("q", "k", "tq", "tk")}
    ang = rng.uniform(0, 2 * np.pi, (text_rows + li, 64))
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    cast = lambda a: mx.array(a).astype(dtype)
    kw = {}
    if txt is not None:
        kw = {"text": cast(txt), "text_q_weight": mx.array(w["tq"]), "text_k_weight": mx.array(w["tk"])}
    q, k, v = qk_norm_rope(
        cast(img), offs, mx.array(w["q"]), mx.array(w["k"]), mx.array(cos), mx.array(sin), heads, **kw
    )

    rows = np.asarray(cast(img).astype(mx.float32)).astype(np.float64)
    trows = None if txt is None else np.asarray(cast(txt).astype(mx.float32)).astype(np.float64)

    def part(src, off):
        return src[:, off : off + width].reshape(len(src), heads, 128)

    def ref(off, wi, wt):
        out = _norm_rope_ref(part(rows, off), wi, cos[text_rows:], sin[text_rows:])
        if trows is not None:
            out = np.concatenate([_norm_rope_ref(part(trows, off), wt, cos[:text_rows], sin[:text_rows]), out])
        return out.transpose(1, 0, 2)[None]

    expect_v = part(rows, offs[2]) if trows is None else np.concatenate([part(trows, offs[2]), part(rows, offs[2])])
    tol = 2e-2 if dtype == mx.bfloat16 else 4e-3
    for got, exp in (
        (q, ref(offs[0], w["q"], w["tq"])),
        (k, ref(offs[1], w["k"], w["tk"])),
        (v, expect_v.transpose(1, 0, 2)[None]),
    ):
        assert got.shape == (1, heads, text_rows + li, 128) and got.dtype == dtype
        np.testing.assert_allclose(np.asarray(got.astype(mx.float32)), exp, atol=tol * np.abs(exp).max(), rtol=0)


def test_qk_norm_rope_rejects_bad_shapes():
    x = mx.zeros((4, 3 * 128), dtype=mx.float16)
    tables = mx.zeros((4, 64))
    with pytest.raises(ValueError):
        qk_norm_rope(x, (0, 128, 384), mx.ones(128), mx.ones(128), tables, tables, 1)  # v past the row
    with pytest.raises(ValueError):
        qk_norm_rope(x, (0, 128, 256), mx.ones(128), mx.ones(128), mx.zeros((5, 64)), tables, 1)


requires_mflux = pytest.mark.skipif(importlib.util.find_spec("mflux") is None, reason="requires mflux")


class _Blocks(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.blocks = list(modules)


def _randomize_norms(*norms):
    for norm in norms:
        norm.weight = mx.random.uniform(0.5, 1.5, (128,))


def _assert_close(got, expected):
    # float32 GPU matmuls on M5 carry ~1e-3 relative error, so compare at 1% of the output range.
    got = got if isinstance(got, tuple) else (got,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert g.shape == e.shape
        np.testing.assert_allclose(np.asarray(g), np.asarray(e), atol=1e-2 * float(mx.abs(e).max()), rtol=0)


def _install(*modules) -> None:
    container = _Blocks(*modules)
    assert install_fused_attention(container) == len(modules)
    assert install_fused_attention(container) == 0  # idempotent
    assert all(getattr(type(m), "_nl_fused", False) for m in modules)


@pytest.fixture
def kernel_calls(monkeypatch):
    """Counts fused-kernel calls, so a silent fallback to mflux's implementation fails the test."""
    from nunchaku_lite.mlx import fused_attention

    calls = []
    real = fused_attention.qk_norm_rope
    monkeypatch.setattr(fused_attention, "qk_norm_rope", lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def _angles(rows):
    ang = mx.random.uniform(0, 6.28, (rows, 64))
    return mx.cos(ang), mx.sin(ang)


@requires_mflux
def test_fused_attention_matches_mflux_flux2(kernel_calls):
    from mflux.models.flux2.model.flux2_transformer.attention import Flux2Attention
    from mflux.models.flux2.model.flux2_transformer.parallel_self_attention import Flux2ParallelSelfAttention

    mx.random.seed(0)
    dim, heads, lt, li = 256, 2, 5, 12
    joint = Flux2Attention(dim, heads, 128, added_kv_proj_dim=dim)
    single = Flux2ParallelSelfAttention(dim, heads, 128, mlp_ratio=3.0)
    _randomize_norms(joint.norm_q, joint.norm_k, joint.norm_added_q, joint.norm_added_k, single.norm_q, single.norm_k)
    x, ctx = mx.random.normal((1, li, dim)), mx.random.normal((1, lt, dim))
    rope = _angles(lt + li)
    joint_x = mx.concatenate([ctx, x], axis=1)
    expected = (*joint(x, ctx, rope), single(joint_x, rope))
    _install(joint, single)
    assert isinstance(joint, Flux2Attention)
    _assert_close((*joint(x, ctx, rope), single(joint_x, rope)), expected)
    assert len(kernel_calls) == 2


@requires_mflux
def test_fused_attention_matches_mflux_flux1(kernel_calls):
    from mflux.models.flux.model.flux_transformer.embed_nd import EmbedND
    from mflux.models.flux.model.flux_transformer.joint_attention import JointAttention
    from mflux.models.flux.model.flux_transformer.single_block_attention import SingleBlockAttention

    mx.random.seed(0)
    dim, heads, lt, li = 256, 2, 5, 12
    joint, single = JointAttention(), SingleBlockAttention()
    for attn, names in (
        (joint, ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out")),
        (single, ("to_q", "to_k", "to_v")),
    ):
        attn.num_heads = heads  # mflux hard-codes FLUX.1's 3072 / 24 heads; shrink for the test
        for name in names:
            setattr(attn, name, nn.Linear(dim, dim))
    joint.to_out = [nn.Linear(dim, dim)]
    _randomize_norms(joint.norm_q, joint.norm_k, joint.norm_added_q, joint.norm_added_k, single.norm_q, single.norm_k)
    x, ctx = mx.random.normal((1, li, dim)), mx.random.normal((1, lt, dim))
    ids = mx.random.randint(0, 64, (1, lt + li, 3)).astype(mx.float32)
    rope = EmbedND()(ids)
    joint_x = mx.concatenate([ctx, x], axis=1)
    expected = (*joint(x, ctx, rope), single(joint_x, rope))
    _install(joint, single)
    _assert_close((*joint(x, ctx, rope), single(joint_x, rope)), expected)
    assert len(kernel_calls) == 2


@requires_mflux
def test_fused_attention_matches_mflux_qwen_image_with_fused_qkv(kernel_calls):
    from mflux.models.qwen.model.qwen_transformer.qwen_attention import QwenAttention

    from nunchaku_lite.mlx.pipeline import ProjectionSlice, SharedProjection

    mx.random.seed(0)
    dim, heads, lt, li = 256, 2, 5, 12
    attn = QwenAttention(dim=dim, num_heads=heads, head_dim=128)
    _randomize_norms(attn.norm_q, attn.norm_k, attn.norm_added_q, attn.norm_added_k)
    for names in (("to_q", "to_k", "to_v"), ("add_q_proj", "add_k_proj", "add_v_proj")):
        # a checkpoint-fused projection serving the three separate ones (as the Nunchaku adapter builds it)
        shared = SharedProjection(nn.Linear(dim, 3 * dim), consumers=3)
        for j, name in enumerate(names):
            setattr(attn, name, ProjectionSlice(shared, j * dim, (j + 1) * dim, owner=j == 0))
    img, txt = mx.random.normal((1, li, dim)), mx.random.normal((1, lt, dim))
    rope = (_angles(li), _angles(lt))
    expected = attn(img, txt, None, rope)
    _install(attn)
    _assert_close(attn(img, txt, None, rope), expected)
    assert len(kernel_calls) == 1


@requires_mflux
def test_fused_attention_matches_mflux_z_image(kernel_calls):
    from mflux.models.z_image.model.z_image_transformer.attention import ZImageAttention

    mx.random.seed(0)
    dim, length = 256, 17
    attn = ZImageAttention(dim=dim, n_heads=2)
    _randomize_norms(attn.norm_q, attn.norm_k)
    x = mx.random.normal((1, length, dim))
    freqs = mx.stack(_angles(length), axis=-1)
    mask = mx.array([[True] * (length - 3) + [False] * 3])
    expected = (attn(x, freqs_cis=freqs), attn(x, attention_mask=mask, freqs_cis=freqs))
    _install(attn)
    _assert_close((attn(x, freqs_cis=freqs), attn(x, attention_mask=mask, freqs_cis=freqs)), expected)
    assert len(kernel_calls) == 2


def test_install_fused_attention_skips_unknown_modules():
    assert install_fused_attention(_Blocks(nn.Linear(4, 4))) == 0
