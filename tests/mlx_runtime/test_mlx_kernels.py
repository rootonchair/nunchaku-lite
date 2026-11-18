import mlx.core as mx
import numpy as np
import pytest

from nunchaku_lite.mlx.kernels import has_neural_accelerators, svdq_gemm_w4a8, svdq_gemm_w8a8
from nunchaku_lite.mlx.packing import SVDQLogicalWeights, to_kernel_layout

pytestmark = pytest.mark.skipif(not has_neural_accelerators(), reason="requires an M5-class GPU")


def _case(m, n, k, r, seed=0, bias=True):
    rng = np.random.default_rng(seed)
    w = SVDQLogicalWeights(
        q=rng.integers(-8, 8, size=(n, k), dtype=np.int8),
        wscale=rng.uniform(0.001, 0.02, size=(n, k // 64)).astype(np.float32),
        smooth=np.ones(k, np.float32),
        down=np.zeros((r, k), np.float32),
        up=(rng.standard_normal((n, r)) * 0.1).astype(np.float16).astype(np.float32),
        bias=rng.standard_normal(n).astype(np.float32) if bias else None,
    )
    a = rng.integers(-127, 128, size=(m, k), dtype=np.int8)
    ascale = rng.uniform(0.001, 0.01, size=m).astype(np.float32)
    lact = rng.standard_normal((m, r)).astype(np.float16).astype(np.float32)
    packed = to_kernel_layout(w)
    acc = a.astype(np.float64) @ packed["w8"].astype(np.float64).T
    ref = acc * ascale[:, None] * packed["wsc"][None, :] + lact @ w.up.T
    if w.bias is not None:
        ref = ref + w.bias
    args = {
        "a": mx.array(a),
        "ascale": mx.array(ascale),
        "wsc": mx.array(packed["wsc"]),
        "lact": mx.array(lact.astype(np.float16)),
        "lup": mx.array(w.up.astype(np.float16)),
        "bias": None if w.bias is None else mx.array(w.bias),
    }
    return packed, args, ref


@pytest.mark.parametrize("m,n,k,r", [(128, 256, 512, 32), (256, 128, 1024, 64), (64, 384, 192, 32)])
@pytest.mark.parametrize("bias", [True, False])
def test_w8a8_matches_reference(m, n, k, r, bias):
    packed, args, ref = _case(m, n, k, r, bias=bias)
    out = svdq_gemm_w8a8(w8=mx.array(packed["w8"]), out_dtype=mx.float32, **args)
    np.testing.assert_allclose(np.array(out), ref, rtol=1e-4, atol=1e-4 * np.abs(ref).max())


@pytest.mark.parametrize("m,n,k,r", [(128, 256, 512, 32), (256, 128, 1024, 64), (128, 192, 192, 32)])
def test_w4a8_matches_w8a8(m, n, k, r):
    packed, args, ref = _case(m, n, k, r, seed=1)
    out4 = svdq_gemm_w4a8(w4=mx.array(packed["w4"]), wr=mx.array(packed["wr"]), out_dtype=mx.float32, **args)
    out8 = svdq_gemm_w8a8(w8=mx.array(packed["w8"]), out_dtype=mx.float32, **args)
    np.testing.assert_allclose(np.array(out4), ref, rtol=1e-4, atol=1e-4 * np.abs(ref).max())
    np.testing.assert_allclose(np.array(out4), np.array(out8), rtol=1e-5, atol=1e-5 * np.abs(ref).max())


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_low_precision_outputs(dtype):
    packed, args, ref = _case(128, 256, 512, 32, seed=2)
    out = svdq_gemm_w8a8(w8=mx.array(packed["w8"]), out_dtype=dtype, **args)
    assert out.dtype == dtype
    np.testing.assert_allclose(np.array(out.astype(mx.float32)), ref, rtol=2e-2, atol=1e-2 * np.abs(ref).max())


def test_misaligned_shapes_raise():
    packed, args, _ = _case(128, 256, 512, 32)
    args["a"] = args["a"][:100]
    with pytest.raises(ValueError):
        svdq_gemm_w8a8(w8=mx.array(packed["w8"]), **args)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("shift", [0.0, 0.171875])
def test_svdq_quantize_act_int8_matches_reference(dtype, shift):
    from nunchaku_lite.mlx.kernels import svdq_quantize_act_int8

    rng = np.random.default_rng(7)
    x = mx.array(rng.standard_t(5, size=(37, 1536)).astype(np.float32)).astype(dtype)
    inv_smooth = rng.uniform(0.5, 2.0, size=1536).astype(np.float32)
    q, scale = svdq_quantize_act_int8(x, mx.array(inv_smooth), mx.array([shift], dtype=mx.float32))
    xs = (np.array(x.astype(mx.float32)) + shift) * inv_smooth
    ref_scale = np.abs(xs).max(axis=1) / 127.0
    np.testing.assert_allclose(np.array(scale), ref_scale, rtol=1e-6)
    ref_q = np.clip(np.rint(xs / ref_scale[:, None]), -127, 127)
    assert np.abs(np.array(q).astype(np.int32) - ref_q).max() <= 1
    assert np.mean(np.array(q) != ref_q) < 1e-3
