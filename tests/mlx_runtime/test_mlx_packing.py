import numpy as np
import pytest

from nunchaku_lite.mlx.packing import (
    SVDQLogicalWeights,
    pack_lowrank,
    pack_scale,
    pack_weight,
    to_kernel_layout,
    unpack_lowrank,
    unpack_scale,
    unpack_svdq_int4,
    unpack_weight,
)

rng = np.random.default_rng(0)


@pytest.mark.parametrize("shape", [(128, 128), (256, 384), (3072, 256)])
def test_weight_roundtrip(shape):
    q = rng.integers(-8, 8, size=shape, dtype=np.int8)
    packed = pack_weight(q)
    assert packed.shape == (shape[0], shape[1] // 2) and packed.dtype == np.int8
    np.testing.assert_array_equal(unpack_weight(packed), q)


def test_weight_packing_is_not_identity():
    q = rng.integers(-8, 8, size=(128, 128), dtype=np.int8)
    naive = ((q[:, 0::2] & 0xF) | ((q[:, 1::2] & 0xF) << 4)).astype(np.uint8).view(np.int8)
    assert not np.array_equal(pack_weight(q), naive)


@pytest.mark.parametrize("groups", [1, 2, 48])
def test_scale_roundtrip(groups):
    scale = rng.standard_normal((256, groups)).astype(np.float32)
    packed = pack_scale(scale)
    assert packed.shape == ((256,) if groups == 1 else (groups, 256))
    np.testing.assert_array_equal(unpack_scale(packed, 256), scale)


@pytest.mark.parametrize("down", [True, False])
def test_lowrank_roundtrip(down):
    logical = rng.standard_normal((32, 384) if down else (384, 32)).astype(np.float32)
    packed = pack_lowrank(logical, down=down)
    assert packed.shape == (384, 32)
    np.testing.assert_array_equal(unpack_lowrank(packed, down=down), logical)


def test_unpack_svdq_int4_matches_logical():
    n, k, r = 256, 512, 32
    q = rng.integers(-8, 8, size=(n, k), dtype=np.int8)
    wscale = rng.uniform(0.001, 0.02, size=(n, k // 64)).astype(np.float32)
    smooth = rng.uniform(0.5, 2.0, size=(k, 1)).astype(np.float32)
    down = rng.standard_normal((r, k)).astype(np.float32)
    up = rng.standard_normal((n, r)).astype(np.float32)
    bias = rng.standard_normal((n, 1)).astype(np.float32)
    w = unpack_svdq_int4(
        qweight=pack_weight(q),
        wscales=pack_scale(wscale),
        smooth_factor=pack_scale(smooth),
        proj_down=pack_lowrank(down, down=True),
        proj_up=pack_lowrank(up, down=False),
        bias=pack_scale(bias),
    )
    np.testing.assert_array_equal(w.q, q)
    np.testing.assert_array_equal(w.wscale, wscale)
    np.testing.assert_array_equal(w.smooth, smooth[:, 0])
    np.testing.assert_array_equal(w.down, down)
    np.testing.assert_array_equal(w.up, up)
    np.testing.assert_array_equal(w.bias, bias[:, 0])


def test_kernel_layout_bounds_and_reconstruction():
    n, k = 128, 256
    q = rng.integers(-8, 8, size=(n, k), dtype=np.int8)
    wscale = rng.uniform(0.001, 0.02, size=(n, k // 64)).astype(np.float32)
    w = SVDQLogicalWeights(
        q=q,
        wscale=wscale,
        smooth=np.ones(k, np.float32),
        down=np.zeros((32, k), np.float32),
        up=np.zeros((n, 32), np.float32),
        bias=None,
    )
    out = to_kernel_layout(w)
    assert out["w8"].dtype == np.int8 and np.abs(out["w8"].astype(np.int32)).max() <= 127
    assert out["wr"].shape == (k // 64, n) and float(out["wr"].max()) <= 127 / 8 + 1e-3
    # per-row int8 error: half a step from rint, plus |q| <= 8 times the fp16 rounding of wr (<= 2**-7)
    recon = out["w8"].astype(np.float32) * out["wsc"][:, None]
    bound = (0.5 + 8 * 2.0**-7) * out["wsc"][:, None] * 1.001
    assert (np.abs(recon - w.dequantize()) <= bound).all()
    # int4 storage: low nibble holds the even input channel
    lo = ((out["w4"] & 0xF).astype(np.int8) ^ 8) - 8
    hi = ((out["w4"] >> 4).astype(np.int8) ^ 8) - 8
    np.testing.assert_array_equal(lo, q[:, 0::2])
    np.testing.assert_array_equal(hi, q[:, 1::2])
