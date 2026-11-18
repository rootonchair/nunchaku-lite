"""Original Nunchaku Z-Image checkpoints on mflux's Z-Image transformer, built without base weights."""

import mlx.core as mx
import numpy as np
import pytest

from nunchaku_lite.mlx import SVDQLogicalWeights, load_nunchaku_pipeline, pack_svdq_int4
from nunchaku_lite.mlx.pipeline import _Z_IMAGE_RENAMES, _without_base_weights, z_image_key_map

pytest.importorskip("mflux")
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mlx.utils import tree_flatten

D, HEADS, HIDDEN, CAP = 384, 3, 1024, 64  # mflux: hidden = dim / 3 * 8; head dim 128 = sum(axes_dims)


def _transformer():
    return ZImageTransformer(dim=D, n_layers=1, n_refiner_layers=1, n_heads=HEADS, cap_feat_dim=CAP)


class FakeZImagePipeline:
    def __init__(self, model_path: str, **kwargs):
        self.transformer = _transformer()


def _svdq(n, k, seed):
    rng = np.random.default_rng(seed)
    return SVDQLogicalWeights(
        q=rng.integers(-8, 8, size=(n, k), dtype=np.int8),
        wscale=rng.uniform(0.001, 0.02, size=(n, k // 64)).astype(np.float32),
        smooth=rng.uniform(0.5, 2.0, size=k).astype(np.float32),
        down=(rng.standard_normal((32, k)) * 0.05).astype(np.float32),
        up=(rng.standard_normal((n, 32)) * 0.05).astype(np.float32),
        bias=None,
    )


def _ref(w, x):
    x = x.astype(np.float64)
    return (x / w.smooth) @ w.dequantize().astype(np.float64).T + (x @ w.down.T) @ w.up.T


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))


def _run(layer, x):
    return np.array(layer(mx.array(x).astype(mx.bfloat16)).astype(mx.float32))


def _checkpoint():
    """Nunchaku layout: quantized ``layers`` / ``noise_refiner``, dense fused ``context_refiner`` (skip_refiners)."""
    ckpt, ws = {}, {}
    replaced = (".attention.to_q.", ".attention.to_k.", ".attention.to_v.", ".attention.to_out.", ".feed_forward.")
    for name, value in tree_flatten(_transformer().parameters()):
        if any(r in name for r in replaced):
            continue
        for diffusers, mflux in _Z_IMAGE_RENAMES:  # dense weights under their Diffusers names
            if name.startswith(mflux + "."):
                name = diffusers + name[len(mflux) :]
        ckpt[name] = mx.random.normal(value.shape).astype(mx.bfloat16) * 0.02
    for seed, p in enumerate(("layers.0", "noise_refiner.0")):
        for j, (name, (n, k)) in enumerate(
            {
                "attention.to_qkv": (3 * D, D),
                "attention.to_out.0": (D, D),
                "feed_forward.net.0.proj": (2 * HIDDEN, D),
                "feed_forward.net.2": (D, HIDDEN),
            }.items()
        ):
            ws[f"{p}.{name}"] = w = _svdq(n, k, 100 * seed + j)
            for t, v in pack_svdq_int4(w).items():
                ckpt[f"{p}.{name}.{t}"] = mx.array(v)
            ckpt[f"{p}.{name}.smooth_factor_orig"] = mx.ones((k,), dtype=mx.bfloat16)  # extra Nunchaku tensor
    rng = np.random.default_rng(7)
    dense = {
        "attention.to_qkv.weight": (3 * D, D),
        "attention.to_out.0.weight": (D, D),
        "feed_forward.net.0.proj.weight": (2 * HIDDEN, D),
        "feed_forward.net.2.weight": (D, HIDDEN),
    }
    for name, shape in dense.items():
        ws[f"context_refiner.0.{name}"] = w = (rng.standard_normal(shape) * 0.05).astype(np.float32)
        ckpt[f"context_refiner.0.{name}"] = mx.array(w).astype(mx.bfloat16)
    return ckpt, ws


def test_z_image_checkpoint_fills_a_transformer_built_without_weights():
    ckpt, ws = _checkpoint()
    pipe = load_nunchaku_pipeline("p", pipeline_cls=FakeZImagePipeline, checkpoint=ckpt, backend="dense")
    assert pipe._nunchaku_lite_target == "z_image"
    x = np.random.default_rng(3).standard_normal((4, D)).astype(np.float32)

    attn = pipe.transformer.layers[0].attention
    qkv = np.concatenate([_run(getattr(attn, n), x) for n in ("to_q", "to_k", "to_v")], -1)
    assert _rel(qkv, _ref(ws["layers.0.attention.to_qkv"], x)) < 2e-2
    assert _rel(_run(attn.to_out[0], x), _ref(ws["layers.0.attention.to_out.0"], x)) < 2e-2

    # Diffusers' SwiGLU: first half * silu(second half) = mflux's w3(x) * silu(w1(x))
    h = _ref(ws["noise_refiner.0.feed_forward.net.0.proj"], x)
    y = h[:, :HIDDEN] * (h[:, HIDDEN:] / (1 + np.exp(-h[:, HIDDEN:])))
    expected = _ref(ws["noise_refiner.0.feed_forward.net.2"], y)
    assert _rel(_run(pipe.transformer.noise_refiner[0].feed_forward, x), expected) < 3e-2

    ctx = pipe.transformer.context_refiner[0]  # dense fused refiner split into mflux's projections
    w = ws["context_refiner.0.attention.to_qkv.weight"]
    np.testing.assert_allclose(np.array(ctx.attention.to_k.weight.astype(mx.float32)), w[D : 2 * D], atol=1e-2)
    np.testing.assert_allclose(
        np.array(ctx.feed_forward.w1.weight.astype(mx.float32)),
        ws["context_refiner.0.feed_forward.net.0.proj.weight"][HIDDEN:],
        atol=1e-2,
    )


def test_incomplete_checkpoint_raises_without_base_weights():
    ckpt, _ = _checkpoint()
    del ckpt["x_pad_token"]
    with pytest.raises(ValueError, match="x_pad_token"):
        load_nunchaku_pipeline("p", pipeline_cls=FakeZImagePipeline, checkpoint=ckpt, backend="dense")


def test_without_base_weights_skips_one_component(monkeypatch):
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

    seen = []

    def fake_load(weight_definition, model_path=None, download_patterns=None):
        seen.append(download_patterns)
        return {c.name: WeightLoader._load_component(None, c)[0] for c in weight_definition.get_components()}

    def fake_component(root_path, definition, *args, **kwargs):
        return {"loaded": definition.name}, None, None

    monkeypatch.setattr(WeightLoader, "load", staticmethod(fake_load))
    monkeypatch.setattr(WeightLoader, "_load_component", staticmethod(fake_component))
    saved = dict(WeightLoader.__dict__)

    with _without_base_weights("transformer"):
        loaded = WeightLoader.load(ZImageWeightDefinition, model_path="x")
        WeightLoader.load(ZImageWeightDefinition, "x", ["transformer/a.safetensors", "vae/b.safetensors"])  # positional
    assert loaded["transformer"] == {} and loaded["vae"] == {"loaded": "vae"}
    assert "transformer/*.json" in seen[0] and "transformer/*.safetensors" not in seen[0]  # configs still fetched
    assert seen[1] == ["vae/b.safetensors"]
    assert WeightLoader.__dict__["load"] is saved["load"]  # the exact objects are restored

    with pytest.raises(RuntimeError), _without_base_weights("transformer"):
        raise RuntimeError("pipeline construction failed")
    assert WeightLoader.__dict__["_load_component"] is saved["_load_component"]  # restored after an error


def test_adapter_without_loaded_set_raises_clearly():
    from nunchaku_lite.mlx.pipeline import _ADAPTERS, register_adapter

    class OldAdapter:  # pre-load_base_transformer contract: patch() returned None
        target = "old_style"

        def matches(self, pipeline):
            return False

        def patch(self, pipeline, state, **kwargs):
            return None

    register_adapter(OldAdapter())
    try:
        ckpt, _ = _checkpoint()
        with pytest.raises(TypeError, match="load_base_transformer=True"):
            load_nunchaku_pipeline("p", pipeline_cls=FakeZImagePipeline, checkpoint=ckpt, target="old_style")
        load_nunchaku_pipeline(
            "p", pipeline_cls=FakeZImagePipeline, checkpoint=ckpt, target="old_style", load_base_transformer=True
        )
    finally:
        _ADAPTERS.pop("old_style")


def test_z_image_key_map_matches_fallback_renames():
    key_map = z_image_key_map()
    for src, dst in _Z_IMAGE_RENAMES:
        assert key_map(f"{src}.weight") == f"{dst}.weight"
    assert key_map("layers.3.attention.to_out.0.weight") == "layers.3.attention.to_out.0.weight"
