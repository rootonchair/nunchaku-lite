import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from nunchaku_lite.mlx import (
    GELU_SHIFT,
    AWQW4A16Linear,
    SVDQLogicalWeights,
    SVDQW4A4Linear,
    has_neural_accelerators,
    pack_svdq_int4,
    patch_transformer,
)
from nunchaku_lite.mlx.packing import pack_awq_w4

needs_nax = pytest.mark.skipif(not has_neural_accelerators(), reason="requires an M5-class GPU")
BACKENDS = ["dense", pytest.param("int8", marks=needs_nax), pytest.param("int4", marks=needs_nax)]


def _weights(n=256, k=512, r=32, seed=0) -> SVDQLogicalWeights:
    rng = np.random.default_rng(seed)
    return SVDQLogicalWeights(
        q=rng.integers(-8, 8, size=(n, k), dtype=np.int8),
        wscale=rng.uniform(0.001, 0.02, size=(n, k // 64)).astype(np.float32),
        smooth=rng.uniform(0.5, 2.0, size=k).astype(np.float32),
        down=(rng.standard_normal((r, k)) * 0.05).astype(np.float32),
        up=(rng.standard_normal((n, r)) * 0.05).astype(np.float32),
        bias=(rng.standard_normal(n) * 0.1).astype(np.float32),
    )


def _checkpoint(w: SVDQLogicalWeights, dtype=mx.float32) -> dict[str, mx.array]:
    return {k: (mx.array(v) if k == "qweight" else mx.array(v).astype(dtype)) for k, v in pack_svdq_int4(w).items()}


def _layer(w: SVDQLogicalWeights, act_unsigned=False, dtype=mx.float32, backend="auto") -> SVDQW4A4Linear:
    layer = SVDQW4A4Linear(
        w.in_features, w.out_features, rank=w.down.shape[0], act_unsigned=act_unsigned, dtype=dtype, backend=backend
    )
    layer.load_weights(list(_checkpoint(w, dtype).items()))
    return layer


def _reference(w: SVDQLogicalWeights, x: np.ndarray, shift: float) -> np.ndarray:
    x = x.astype(np.float64)
    out = ((x + shift) / w.smooth) @ w.dequantize().astype(np.float64).T + (x @ w.down.T) @ w.up.T
    return out + w.bias


def _rel_err(out, ref):
    return float(np.linalg.norm(np.asarray(out, np.float64) - ref) / np.linalg.norm(ref))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("act_unsigned", [False, True])
def test_layer_matches_exact_reference(backend, act_unsigned):
    w = _weights()
    layer = _layer(w, act_unsigned=act_unsigned, backend=backend)
    x = np.random.default_rng(1).standard_normal((2, 77, w.in_features)).astype(np.float32)
    out = np.array(layer(mx.array(x)))
    assert out.shape == (2, 77, w.out_features)
    ref = _reference(w, x.reshape(-1, w.in_features), GELU_SHIFT if act_unsigned else 0.0).reshape(out.shape)
    # MLX float32 GPU matmuls run at reduced internal precision (~1e-3 relative on M5).
    assert _rel_err(out, ref) < (2e-3 if backend == "dense" else 2e-2)


def test_parameters_use_nunchaku_names():
    layer = SVDQW4A4Linear(256, 128, rank=16)
    names = {k for k, _ in nn.utils.tree_flatten(layer.parameters())}
    assert names == {"qweight", "wscales", "smooth_factor", "proj_down", "proj_up", "bias"}
    assert layer.qweight.shape == (128, 128) and layer.wscales.shape == (4, 128)
    no_bias = SVDQW4A4Linear(256, 128, bias=False)
    assert {k for k, _ in nn.utils.tree_flatten(no_bias.parameters())} == names - {"bias"}


def test_from_linear_copies_shape_and_bias():
    layer = SVDQW4A4Linear.from_linear(nn.Linear(512, 256, bias=False), rank=32, act_unsigned=True)
    assert (layer.in_features, layer.out_features, layer.bias, layer.act_unsigned) == (512, 256, None, True)


@needs_nax
def test_int8_and_int4_agree():
    w = _weights(seed=3)
    layer = _layer(w, backend="int8")
    x = mx.array(np.random.default_rng(4).standard_normal((130, w.in_features)).astype(np.float32))
    out8 = np.array(layer(x))
    layer.backend = "int4"
    out4 = np.array(layer(x))  # backend switch rebuilds the cache
    # int8 paths are bit-identical; the fp low-rank matmul may sum in a different order per tile shape
    np.testing.assert_allclose(out8, out4, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cache_rebuilds_after_load_weights(backend):
    w = _weights(seed=5)
    layer = _layer(w, backend=backend)
    x = np.random.default_rng(6).standard_normal((8, w.in_features)).astype(np.float32)
    layer(mx.array(x))
    w2 = SVDQLogicalWeights(q=w.q, wscale=w.wscale, smooth=w.smooth, down=w.down, up=w.up * 3, bias=w.bias + 1)
    layer.load_weights([(k, v) for k, v in _checkpoint(w2).items() if k in ("proj_up", "bias")], strict=False)
    tol = 2e-3 if backend == "dense" else 2e-2
    assert _rel_err(np.array(layer(mx.array(x))), _reference(w2, x, 0.0)) < tol


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_low_precision_activations(backend, dtype):
    w = _weights(seed=10)
    layer = _layer(w, act_unsigned=True, dtype=dtype, backend=backend)
    x = np.random.default_rng(12).standard_normal((3, 50, w.in_features)).astype(np.float32)
    out = layer(mx.array(x).astype(dtype))
    assert out.dtype == dtype
    ref = _reference(w, x.reshape(-1, w.in_features), GELU_SHIFT).reshape(out.shape)
    assert _rel_err(np.array(out.astype(mx.float32)), ref) < 3e-2


def test_zero_rank_checkpoint():
    w = _weights(r=16, seed=8)
    w.down, w.up = w.down[:0], w.up[:, :0]
    layer = SVDQW4A4Linear(w.in_features, w.out_features, rank=0, backend="dense")
    layer.load_weights(list(_checkpoint(w).items()))
    x = np.random.default_rng(9).standard_normal((5, w.in_features)).astype(np.float32)
    ref = (x.astype(np.float64) / w.smooth) @ w.dequantize().astype(np.float64).T + w.bias
    assert _rel_err(np.array(layer(mx.array(x))), ref) < 2e-3


def test_nvfp4_rejected():
    with pytest.raises(NotImplementedError):
        SVDQW4A4Linear(256, 256, precision="nvfp4")


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(512)
        self.fc1 = nn.Linear(512, 256)
        self.layers = [nn.Linear(256, 512), nn.Linear(512, 256, bias=False)]


def test_patch_transformer_replaces_layers_and_loads_rest():
    w1, w2 = _weights(n=256, k=512, seed=20), _weights(n=512, k=256, seed=21)
    ckpt = {f"fc1.{k}": v for k, v in _checkpoint(w1).items()}
    # original Nunchaku key names are accepted
    aliases = {"proj_down": "lora_down", "proj_up": "lora_up", "smooth_factor": "smooth"}
    ckpt.update({f"blocks.0.{aliases.get(k, k)}": v for k, v in _checkpoint(w2).items()})
    ckpt["blocks.0.smooth_orig"] = mx.ones((256,))  # ignored, as in nunchaku_lite
    ckpt["norm.weight"] = mx.full((512,), 2.0)

    model = _Block()
    patch_transformer(
        model,
        ckpt,
        act_unsigned=["blocks.0"],
        key_map=lambda k: k.replace("blocks.0", "layers.0"),
        backend="dense",
    )
    assert model.fc1.backend == model.layers[0].backend == "dense"
    assert isinstance(model.fc1, SVDQW4A4Linear) and isinstance(model.layers[0], SVDQW4A4Linear)
    assert isinstance(model.layers[1], nn.Linear)  # not in the checkpoint
    assert model.layers[0].act_unsigned and not model.fc1.act_unsigned
    np.testing.assert_allclose(np.array(model.norm.weight), 2.0)

    x = np.random.default_rng(22).standard_normal((4, 512)).astype(np.float32)
    h = np.array(model.fc1(mx.array(x)))
    assert _rel_err(h, _reference(w1, x, 0.0)) < 2e-3
    y = np.array(model.layers[0](mx.array(h)))
    assert _rel_err(y, _reference(w2, h, GELU_SHIFT)) < 2e-3


def test_patch_transformer_strict_errors():
    ckpt = {f"missing.{k}": v for k, v in _checkpoint(_weights()).items()}
    with pytest.raises(ValueError, match="no nn.Linear"):
        patch_transformer(_Block(), ckpt)
    patch_transformer(_Block(), ckpt, strict=False)  # skipped
    with pytest.raises(ValueError, match="do not match"):
        patch_transformer(_Block(), {"unknown.weight": mx.zeros((3,))})
    awq = {
        "mod.qweight": mx.zeros((4, 32), dtype=mx.int32),
        "mod.wzeros": mx.zeros((1, 16)),
        "mod.wscales": mx.zeros((1, 16)),
    }
    with pytest.raises(ValueError, match="no nn.Linear"):
        patch_transformer(_Block(), awq)


def test_backend_validation_and_auto_resolution(monkeypatch):
    import nunchaku_lite.mlx.linear as linear_module

    with pytest.raises(ValueError, match="backend must be one of"):
        SVDQW4A4Linear(256, 256, backend="w8a8")
    with pytest.raises(ValueError, match="backend must be one of"):
        patch_transformer(_Block(), {}, backend="fast")
    monkeypatch.setattr(linear_module, "has_neural_accelerators", lambda: True)
    monkeypatch.setattr(linear_module.mx, "device_info", lambda: {"max_recommended_working_set_size": int(19e9)})
    assert linear_module.resolve_backend("auto", int(4e9)) == "int8"
    assert linear_module.resolve_backend("auto", int(15e9)) == "int4"  # over half the working set
    assert linear_module.resolve_backend("int8", int(15e9)) == "int8"  # explicit choice wins
    monkeypatch.setattr(linear_module, "has_neural_accelerators", lambda: False)
    assert linear_module.resolve_backend("auto") == "dense"


# ---------------------------------------------------------------------------
# Packed qweight release
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_qweight_released_as_exact_lazy_repack(backend):
    w = _weights(seed=30)
    w.wscale = np.random.default_rng(1).uniform(0.01, 0.02, size=w.wscale.shape).astype(np.float32)  # invertible
    layer = _layer(w, backend=backend)
    original = layer.qweight
    packed = np.array(original)
    x = mx.array(np.random.default_rng(31).standard_normal((8, w.in_features)).astype(np.float32))
    out = np.array(layer(x))
    assert layer.qweight is not original  # replaced by a lazy re-pack of the backend copy
    cache = layer._cache
    np.testing.assert_array_equal(np.array(layer.qweight), packed)  # evaluates to the exact checkpoint bytes
    flat = dict(nn.utils.tree_flatten(layer.parameters()))
    assert flat["qweight"].shape == packed.shape and flat["qweight"].dtype == mx.int8
    np.testing.assert_allclose(np.array(layer(x)), out)
    assert layer._cache is cache  # no rebuild after release

    # weight change after release: the rebuild recovers q from the lazy qweight
    w2 = SVDQLogicalWeights(q=w.q, wscale=w.wscale, smooth=w.smooth, down=w.down, up=w.up * 2, bias=w.bias)
    layer.load_weights([("proj_up", _checkpoint(w2)["proj_up"])], strict=False)
    tol = 2e-3 if backend == "dense" else 2e-2
    assert _rel_err(np.array(layer(x)), _reference(w2, np.array(x), 0.0)) < tol


@needs_nax
def test_backend_switch_after_release_and_non_invertible_kept():
    w = _weights(seed=32)
    w.wscale = np.random.default_rng(2).uniform(0.01, 0.02, size=w.wscale.shape).astype(np.float32)
    layer = _layer(w, backend="int8")
    x = mx.array(np.random.default_rng(33).standard_normal((8, w.in_features)).astype(np.float32))
    layer(x)
    layer.backend = "dense"
    assert _rel_err(np.array(layer(x)), _reference(w, np.array(x), 0.0)) < 2e-3

    w.wscale[:, 0] = 1e-5  # one tiny group per channel: the int8 copy is not invertible
    keep = _layer(w, backend="int8")
    original = keep.qweight
    keep(x)
    assert keep.qweight is original


# ---------------------------------------------------------------------------
# AWQ W4A16
# ---------------------------------------------------------------------------


def _awq_case(n=384, k=256, seed=40):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16, size=(n, k)).astype(np.int32)
    scale = rng.uniform(0.001, 0.02, size=(n, k // 64)).astype(np.float32)
    zero = (-7 * scale).astype(np.float32)
    bias = rng.standard_normal(n).astype(np.float32)
    dense = (codes.reshape(n, k // 64, 64) * scale[..., None] + zero[..., None]).reshape(n, k)
    return codes, scale, zero, bias, dense


def _awq_tensors(codes, scale, zero, bias):
    return {
        "qweight": mx.array(pack_awq_w4(codes)),
        "wscales": mx.array(scale.T.copy()),
        "wzeros": mx.array(zero.T.copy()),
        "bias": mx.array(bias),
    }


def test_awq_layer_matches_dense_reference():
    codes, scale, zero, bias, dense = _awq_case()
    layer = AWQW4A16Linear(256, 384, dtype=mx.float32)
    layer.load_weights(list(_awq_tensors(codes, scale, zero, bias).items()))
    x = np.random.default_rng(41).standard_normal((3, 256)).astype(np.float32)
    assert _rel_err(np.array(layer(mx.array(x))), x.astype(np.float64) @ dense.T + bias) < 2e-3


def test_awq_adanorm_layout_restores_original_order():
    splits = 6
    codes, scale, zero, bias, dense = _awq_case(n=384, seed=42)
    n = codes.shape[0]

    # Nunchaku's AdaNorm transform (deepcompressor convert_to_nunchaku_w4x16_linear_weight)
    def interleave(t):
        return t.reshape(splits, n // splits, *t.shape[1:]).swapaxes(0, 1).reshape(t.shape)

    b = bias.reshape(splits, n // splits).T.copy()
    b[:, 1] += 1
    b[:, -2] += 1
    stored = _awq_tensors(interleave(codes), interleave(scale), interleave(zero), b.reshape(n))
    layer = AWQW4A16Linear(256, n, adanorm_splits=splits, dtype=mx.float32)
    layer.load_weights(list(stored.items()))
    x = np.random.default_rng(43).standard_normal((2, 256)).astype(np.float32)
    assert _rel_err(np.array(layer(mx.array(x))), x.astype(np.float64) @ dense.T + bias) < 2e-3


def test_patch_transformer_loads_awq_with_adanorm_splits():
    codes, scale, zero, bias, dense = _awq_case(n=512, k=256, seed=44)
    ckpt = {f"layers.0.{k}": v for k, v in _awq_tensors(codes, scale, zero, bias).items()}
    model = _Block()
    patch_transformer(model, ckpt, adanorm_splits={"layers.0": 1})
    assert isinstance(model.layers[0], AWQW4A16Linear) and model.layers[0].adanorm_splits == 1
    x = np.random.default_rng(45).standard_normal((2, 256)).astype(np.float32)
    out = np.array(model.layers[0](mx.array(x).astype(mx.bfloat16)).astype(mx.float32))
    assert _rel_err(out, x.astype(np.float64) @ dense.T + bias) < 2e-2
