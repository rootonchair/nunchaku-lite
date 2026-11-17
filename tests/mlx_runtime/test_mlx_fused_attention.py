import importlib.util
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from nunchaku_lite.mlx.flux2_attention import install_fused_attention
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


@pytest.mark.skipif(importlib.util.find_spec("mflux") is None, reason="requires mflux")
def test_install_fused_attention_matches_mflux():
    from mflux.models.flux2.model.flux2_transformer.attention import Flux2Attention
    from mflux.models.flux2.model.flux2_transformer.parallel_self_attention import Flux2ParallelSelfAttention

    mx.random.seed(0)
    dim, heads, lt, li = 256, 2, 5, 12

    t = SimpleNamespace(
        transformer_blocks=[SimpleNamespace(attn=Flux2Attention(dim, heads, 128, added_kv_proj_dim=dim))],
        single_transformer_blocks=[SimpleNamespace(attn=Flux2ParallelSelfAttention(dim, heads, 128, mlp_ratio=3.0))],
    )
    joint, single = t.transformer_blocks[0].attn, t.single_transformer_blocks[0].attn
    for norm in (joint.norm_q, joint.norm_k, joint.norm_added_q, joint.norm_added_k, single.norm_q, single.norm_k):
        norm.weight = mx.random.uniform(0.5, 1.5, (128,))
    x, ctx = mx.random.normal((1, li, dim)), mx.random.normal((1, lt, dim))
    ang = mx.random.uniform(0, 6.28, (lt + li, 64))
    rope = (mx.cos(ang), mx.sin(ang))
    ref_joint = joint(x, ctx, rope)
    ref_single = single(mx.concatenate([ctx, x], axis=1), rope)

    assert install_fused_attention(t) == 2
    assert install_fused_attention(t) == 0  # idempotent
    assert isinstance(joint, Flux2Attention) and getattr(type(joint), "_nl_fused", False)
    # float32 GPU matmuls on M5 carry ~1e-3 relative error, so compare at 1% of the output range.
    for got, exp in zip(
        (*joint(x, ctx, rope), single(mx.concatenate([ctx, x], axis=1), rope)), (*ref_joint, ref_single)
    ):
        np.testing.assert_allclose(np.asarray(got), np.asarray(exp), atol=1e-2 * float(mx.abs(exp).max()), rtol=0)
