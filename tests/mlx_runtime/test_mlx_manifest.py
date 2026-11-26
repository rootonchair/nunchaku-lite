"""Runtime-manifest checkpoints (Nunchaku Lite / lite-infer layout) and the Diffusers -> MLX name map."""

import dataclasses
import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from nunchaku_lite.manifest import parse_runtime_manifest
from nunchaku_lite.mlx import (
    AWQW4A16Linear,
    PrefixMap,
    SVDQLogicalWeights,
    SVDQW4A4Linear,
    load_checkpoint,
    load_nunchaku_pipeline,
    pack_svdq_int4,
    patch_transformer,
)
from nunchaku_lite.mlx.manifest import SplitLinearInput, SplitLinearOutput
from nunchaku_lite.mlx.packing import pack_awq_w4
from nunchaku_lite.mlx.pipeline import _FLUX1_RENAMES, _FLUX2_RENAMES, flux1_key_map, flux2_key_map

D, HEAD_DIM, MLP = 256, 128, 128


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


def _svdq_ref(w, x):
    x = x.astype(np.float64)
    return (x / w.smooth) @ w.dequantize().astype(np.float64).T + (x @ w.down.T) @ w.up.T


def _awq(n, k, seed):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16, size=(n, k)).astype(np.int32)
    scale = rng.uniform(0.001, 0.02, size=(n, k // 64)).astype(np.float32)
    zero = (-7 * scale).astype(np.float32)
    dense = (codes.reshape(n, k // 64, 64) * scale[..., None] + zero[..., None]).reshape(n, k)
    tensors = {"qweight": pack_awq_w4(codes), "wscales": scale.T.copy(), "wzeros": zero.T.copy()}
    return tensors, dense, (codes, scale, zero)


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))


def _run(layer, x):
    return np.array(layer(mx.array(x).astype(mx.bfloat16)).astype(mx.float32))


def _target(prefix, op="svdq_w4a4", *, has_bias=False, op_options=None, precision="int4"):
    return {
        "checkpoint_prefix": prefix,
        "source_modules": [prefix],
        "roles": [],
        "kind": "linear",
        "nunchaku_op": op,
        "precision": precision,
        "group_size": 16 if precision == "fp4" else 64,
        "rank": 32 if op == "svdq_w4a4" else 0,
        "has_bias": has_bias,
        "op_options": op_options or {},
        "activation": {},
    }


def _manifest(targets, structural=(), precision="int4"):
    return {
        "schema": "nunchaku_lite.runtime_manifest",
        "version": 1,
        "component": "transformer",
        "nunchaku_format_version": 1,
        "producer": {"name": "test", "version": "0"},
        "requirements": {
            "method": "svdquant",
            "precision": precision,
            "rank": 32,
            "weight_dtype": "int4",
            "activation_dtype": "int4",
        },
        "structural_patches": list(structural),
        "targets": targets,
    }


def _save(folder, state, manifest=None):
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {"quantization_config": json.dumps({"runtime_manifest": manifest})} if manifest else {}
    arrays = {k: v if isinstance(v, mx.array) else mx.array(v) for k, v in state.items()}
    mx.save_safetensors(str(folder / "diffusion_pytorch_model.safetensors"), arrays, metadata=metadata)
    return folder


# ---------------------------------------------------------------------------
# Miniature mflux FLUX.2 transformer (mflux module names)
# ---------------------------------------------------------------------------


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_dim = D
        for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_out", "to_add_out"):
            setattr(self, name, nn.Linear(D, D, bias=False))


class _FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_in = nn.Linear(D, 2 * MLP, bias=False)
        self.linear_out = nn.Linear(MLP, D, bias=False)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn, self.ff, self.ff_context = _Attn(), _FF(), _FF()


class _ParallelAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_dim = D
        self.to_qkv_mlp_proj = nn.Linear(D, 3 * D + 2 * MLP, bias=False)
        self.to_out = nn.Linear(D + MLP, D, bias=False)


class _Single(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _ParallelAttn()


class _Modulation(nn.Module):
    def __init__(self, sets):
        super().__init__()
        self.linear = nn.Linear(D, D * 3 * sets, bias=False)


class _TimeGuidance(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(256, D, bias=False)


class Flux2Transformer(nn.Module):  # name matches mflux for adapter auto-selection
    def __init__(self):
        super().__init__()
        self.x_embedder = nn.Linear(128, D, bias=False)
        self.time_guidance_embed = _TimeGuidance()
        self.double_stream_modulation_img = _Modulation(2)
        self.single_stream_modulation = _Modulation(1)
        self.transformer_blocks = [_Block()]
        self.single_transformer_blocks = [_Single()]


class FakeFlux2Pipeline:
    def __init__(self, model_path: str, **kwargs):
        self.transformer = Flux2Transformer()


_LITE_SVDQ = {  # Diffusers names, as in lite-infer FLUX.2 checkpoints
    "transformer_blocks.0.attn.to_q": (D, D),
    "transformer_blocks.0.attn.to_k": (D, D),
    "transformer_blocks.0.attn.to_v": (D, D),
    "transformer_blocks.0.attn.to_out.0": (D, D),
    "transformer_blocks.0.attn.add_q_proj": (D, D),
    "transformer_blocks.0.attn.add_k_proj": (D, D),
    "transformer_blocks.0.attn.add_v_proj": (D, D),
    "transformer_blocks.0.attn.to_add_out": (D, D),
    "transformer_blocks.0.ff.linear_in": (2 * MLP, D),
    "transformer_blocks.0.ff.linear_out": (D, MLP),
    "transformer_blocks.0.ff_context.linear_in": (2 * MLP, D),
    "transformer_blocks.0.ff_context.linear_out": (D, MLP),
    "single_transformer_blocks.0.attn.to_qkv_mlp_proj": (3 * D + 2 * MLP, D),
    "single_transformer_blocks.0.attn.to_out": (D, D + MLP),
}
_LITE_AWQ = {"double_stream_modulation_img.linear": (6 * D, D), "single_stream_modulation.linear": (3 * D, D)}


def _lite_flux2_checkpoint():
    ws, state, targets = {}, {}, []
    for i, (prefix, (n, k)) in enumerate(_LITE_SVDQ.items()):
        ws[prefix] = _svdq(n, k, 200 + i)
        state.update({f"{prefix}.{name}": v for name, v in pack_svdq_int4(ws[prefix]).items()})
        targets.append(_target(prefix))
    for i, (prefix, (n, k)) in enumerate(_LITE_AWQ.items()):
        tensors, ws[prefix], _ = _awq(n, k, 300 + i)
        state.update({f"{prefix}.{name}": v for name, v in tensors.items()})
        targets.append(_target(prefix, "awq_w4a16"))
    state["x_embedder.weight"] = np.full((D, 128), 0.5, np.float32)
    state["time_guidance_embed.timestep_embedder.linear_1.weight"] = np.full((D, 256), 0.25, np.float32)
    return ws, state, _manifest(targets)


def test_lite_flux2_repo_folder_loads_from_manifest(tmp_path):
    ws, state, manifest = _lite_flux2_checkpoint()
    _save(tmp_path / "repo" / "transformer", state, manifest)
    pipe = load_nunchaku_pipeline("base", pipeline_cls=FakeFlux2Pipeline, checkpoint=tmp_path / "repo", backend="dense")
    assert pipe._nunchaku_lite_target == "flux2" and len(pipe._nunchaku_lite_runtime_manifest.targets) == 16
    tr = pipe.transformer
    np.testing.assert_allclose(np.array(tr.x_embedder.weight), 0.5)
    np.testing.assert_allclose(np.array(tr.time_guidance_embed.linear_1.weight), 0.25)  # Diffusers -> mflux name

    x = np.random.default_rng(1).standard_normal((5, D)).astype(np.float32)
    attn = tr.transformer_blocks[0].attn
    for name in ("to_q", "to_k", "to_v", "add_v_proj", "to_add_out"):
        prefix = f"transformer_blocks.0.attn.{name}"
        assert isinstance(getattr(attn, name), SVDQW4A4Linear)
        assert _rel(_run(getattr(attn, name), x), _svdq_ref(ws[prefix], x)) < 2e-2
    assert _rel(_run(attn.to_out, x), _svdq_ref(ws["transformer_blocks.0.attn.to_out.0"], x)) < 2e-2  # to_out.0

    single = tr.single_transformer_blocks[0].attn
    ref = _svdq_ref(ws["single_transformer_blocks.0.attn.to_qkv_mlp_proj"], x)
    assert _rel(_run(single.to_qkv_mlp_proj, x), ref) < 2e-2
    xy = np.random.default_rng(2).standard_normal((5, D + MLP)).astype(np.float32)
    assert _rel(_run(single.to_out, xy), _svdq_ref(ws["single_transformer_blocks.0.attn.to_out"], xy)) < 2e-2

    modulation = tr.double_stream_modulation_img.linear
    assert isinstance(modulation, AWQW4A16Linear) and modulation.adanorm_splits == 1
    assert _rel(_run(modulation, x), x.astype(np.float64) @ ws["double_stream_modulation_img.linear"].T) < 2e-2
    layers = [m for _, m in tr.named_modules() if isinstance(m, SVDQW4A4Linear)]
    assert len(layers) == 14 and all(m.backend == "dense" for m in layers)


def test_manifest_strict_rejects_unknown_dense_tensors(tmp_path):
    _, state, manifest = _lite_flux2_checkpoint()
    state["unknown.weight"] = np.zeros((2,), np.float32)
    folder = _save(tmp_path / "ckpt", state, manifest)
    with pytest.raises(ValueError, match="do not match model parameters"):
        load_nunchaku_pipeline("p", pipeline_cls=FakeFlux2Pipeline, checkpoint=folder, backend="dense")
    load_nunchaku_pipeline("p", pipeline_cls=FakeFlux2Pipeline, checkpoint=folder, backend="dense", strict=False)


def test_manifest_target_without_model_module_raises(tmp_path):
    _, state, manifest = _lite_flux2_checkpoint()
    manifest["targets"].append(_target("transformer_blocks.0.attn.missing"))
    folder = _save(tmp_path / "ckpt", state, manifest)
    with pytest.raises(ValueError, match="has no module at model path"):
        load_nunchaku_pipeline("p", pipeline_cls=FakeFlux2Pipeline, checkpoint=folder, backend="dense")


def test_original_nunchaku_flux2_keys_are_normalized_like_nunchaku_lite():
    """Original Nunchaku FLUX.2 names (``qkv_proj`` under the block, ``lora_*``) go through the shared converter."""

    original = {
        "transformer_blocks.0.qkv_proj": (3 * D, D),
        "transformer_blocks.0.qkv_proj_context": (3 * D, D),
        "transformer_blocks.0.out_proj": (D, D),
        "transformer_blocks.0.out_proj_context": (D, D),
        "transformer_blocks.0.mlp_fc1": (2 * MLP, D),
        "transformer_blocks.0.mlp_fc2": (D, MLP),
        "transformer_blocks.0.mlp_context_fc1": (2 * MLP, D),
        "transformer_blocks.0.mlp_context_fc2": (D, MLP),
        "single_transformer_blocks.0.qkv_proj": (3 * D, D),
        "single_transformer_blocks.0.mlp_fc1": (2 * MLP, D),
        "single_transformer_blocks.0.out_proj": (D, D),
        "single_transformer_blocks.0.mlp_fc2": (D, MLP),
    }
    ws, ckpt = {}, {}
    for i, (prefix, (n, k)) in enumerate(original.items()):
        ws[prefix] = _svdq(n, k, 400 + i)
        for name, v in pack_svdq_int4(ws[prefix]).items():
            name = {"proj_down": "lora_down", "proj_up": "lora_up", "smooth_factor": "smooth"}.get(name, name)
            ckpt[f"{prefix}.{name}"] = mx.array(v)
    for name in ("double_stream_modulation_img", "single_stream_modulation"):
        ckpt[f"{name}.linear.weight"] = mx.zeros(Flux2Transformer()[name].linear.weight.shape)
    pipe = load_nunchaku_pipeline(  # partial checkpoint: the rest comes from the base transformer
        "p", pipeline_cls=FakeFlux2Pipeline, checkpoint=ckpt, backend="dense", load_base_transformer=True
    )
    attn = pipe.transformer.transformer_blocks[0].attn
    x = np.random.default_rng(3).standard_normal((4, D)).astype(np.float32)
    qkv = np.concatenate([_run(getattr(attn, n), x) for n in ("add_q_proj", "add_k_proj", "add_v_proj")], -1)
    assert _rel(qkv, _svdq_ref(ws["transformer_blocks.0.qkv_proj_context"], x)) < 2e-2
    assert _rel(_run(attn.to_out, x), _svdq_ref(ws["transformer_blocks.0.out_proj"], x)) < 2e-2


class QuantizedBaseFlux2Pipeline:
    """Like an mflux ``-q4`` repo: the base transformer is made of ``QuantizedLinear`` layers."""

    model_config = SimpleNamespace(transformer_overrides={})

    def __init__(self, model_path: str, **kwargs):
        self.transformer = Flux2Transformer()
        nn.quantize(self.transformer, group_size=64, bits=4)


def test_quantized_base_transformer_is_rebuilt_from_the_checkpoint(tmp_path):
    ws, state, manifest = _lite_flux2_checkpoint()
    folder = _save(tmp_path / "ckpt", state, manifest)
    pipe = load_nunchaku_pipeline("p", pipeline_cls=QuantizedBaseFlux2Pipeline, checkpoint=folder, backend="dense")
    tr = pipe.transformer
    assert not any(isinstance(m, nn.QuantizedLinear) for _, m in tr.named_modules())
    assert tr.x_embedder.weight.dtype == mx.float32 and float(tr.x_embedder.weight[0, 0]) == 0.5
    assert tr.transformer_blocks[0].attn.to_q.wscales.dtype == mx.float32  # checkpoint dtype, not the skeleton's
    x = np.random.default_rng(5).standard_normal((3, D)).astype(np.float32)
    prefix = "transformer_blocks.0.attn.to_q"
    assert _rel(_run(tr.transformer_blocks[0].attn.to_q, x), _svdq_ref(ws[prefix], x)) < 2e-2

    del state["x_embedder.weight"]
    folder = _save(tmp_path / "incomplete", state, manifest)
    with pytest.raises(ValueError, match="missing from it, e.g. 'x_embedder.weight'"):
        load_nunchaku_pipeline("p", pipeline_cls=QuantizedBaseFlux2Pipeline, checkpoint=folder, backend="dense")


# ---------------------------------------------------------------------------
# FLUX.1 (lite-infer layout without a runtime manifest: config.json target lists)
# ---------------------------------------------------------------------------


class _Flux1Attn(nn.Module):
    def __init__(self, joint: bool):
        super().__init__()
        names = ("to_q", "to_k", "to_v") + (("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out") if joint else ())
        for name in names:
            setattr(self, name, nn.Linear(D, D))
        if joint:
            self.to_out = [nn.Linear(D, D)]


class _Flux1FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1, self.linear2 = nn.Linear(D, MLP), nn.Linear(MLP, D)


class _Flux1Joint(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm1_context = _Norm(), _Norm()
        self.attn, self.ff, self.ff_context = _Flux1Attn(True), _Flux1FF(), _Flux1FF()


class _Flux1Single(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm, self.attn = _Norm(), _Flux1Attn(False)
        self.proj_mlp, self.proj_out = nn.Linear(D, MLP), nn.Linear(D + MLP, D)


class Transformer(nn.Module):
    """Miniature mflux FLUX.1 transformer (mflux module names and constructor)."""

    def __init__(self, model_config=None, num_transformer_blocks=1, num_single_transformer_blocks=1):
        super().__init__()
        self.x_embedder = nn.Linear(128, D)
        self.transformer_blocks = [_Flux1Joint() for _ in range(num_transformer_blocks)]
        self.single_transformer_blocks = [_Flux1Single() for _ in range(num_single_transformer_blocks)]


Transformer.__module__ = "mflux.models.flux.model.flux_transformer.transformer"


class FakeFlux1Pipeline:
    model_config = SimpleNamespace(transformer_overrides={})

    def __init__(self, model_path: str, quantize: int | None = None, **kwargs):
        self.transformer = Transformer(self.model_config)
        if quantize:
            nn.quantize(self.transformer, group_size=64, bits=quantize)


class FakeFlux1PipelineWithoutWeights(FakeFlux1Pipeline):
    """mflux sizes FLUX.1 from the loaded weights: built without base weights, the transformer has no blocks."""

    def __init__(self, model_path: str, **kwargs):
        self.transformer = Transformer(self.model_config, num_transformer_blocks=0, num_single_transformer_blocks=0)


_FLUX1_SVDQ = {
    **{f"transformer_blocks.0.attn.{n}": (D, D) for n in ("to_q", "to_k", "to_v", "add_q_proj", "to_out.0")},
    "transformer_blocks.0.ff.net.0.proj": (MLP, D),
    "transformer_blocks.0.ff_context.net.2": (D, MLP),
    "single_transformer_blocks.0.attn.to_q": (D, D),
    "single_transformer_blocks.0.proj_mlp": (MLP, D),
    "single_transformer_blocks.0.proj_out": (D, D + MLP),
}
_FLUX1_AWQ = {"transformer_blocks.0.norm1.linear": (6 * 64, D), "single_transformer_blocks.0.norm.linear": (6 * 64, D)}


def _lite_flux1_repo(root):
    """A lite-infer FLUX.1 repo: biases on every target, targets listed in config.json, no safetensors metadata."""

    rng = np.random.default_rng(7)
    ws, state, biases = {}, {}, {}
    for prefix, (n, _) in {**_FLUX1_SVDQ, **_FLUX1_AWQ}.items():
        if prefix != "transformer_blocks.0.attn.to_k":  # one target without bias
            biases[prefix] = rng.standard_normal(n).astype(np.float32)
    for i, (prefix, (n, k)) in enumerate(_FLUX1_SVDQ.items()):
        ws[prefix] = dataclasses.replace(_svdq(n, k, 500 + i), bias=biases.get(prefix))
        state.update({f"{prefix}.{name}": v for name, v in pack_svdq_int4(ws[prefix]).items()})
    for i, (prefix, (n, k)) in enumerate(_FLUX1_AWQ.items()):
        tensors, ws[prefix], _ = _awq(n, k, 600 + i)
        state.update({f"{prefix}.{name}": v for name, v in tensors.items()})
        state[f"{prefix}.bias"] = biases[prefix]  # AWQ bias: plain nn.Linear order
    tr = Transformer()
    for path, value in nn.utils.tree_flatten(tr.parameters()):
        diffusers = path.replace(".linear1.", ".net.0.proj.").replace(".linear2.", ".net.2.")
        if not any(diffusers.startswith(p + ".") for p in (*_FLUX1_SVDQ, *_FLUX1_AWQ)):
            state[diffusers] = np.full(value.shape, 0.5, np.float32)
    folder = _save(root / "transformer", state)
    config = {
        "quant_method": "nunchaku_lite",
        "svdq_w4a4": {"precision": "int4", "group_size": 64, "rank": 32, "targets": sorted(_FLUX1_SVDQ)},
        "awq_w4a16": {"precision": "int4", "group_size": 64, "targets": sorted(_FLUX1_AWQ)},
    }
    (folder / "config.json").write_text(
        json.dumps({"_class_name": "FluxTransformer2DModel", "quantization_config": config})
    )
    return ws, state, config, biases


def test_manifest_from_config_target_lists(tmp_path):
    _, state, config, _ = _lite_flux1_repo(tmp_path)
    assert parse_runtime_manifest(config) is None  # without checkpoint keys: no manifest
    assert parse_runtime_manifest({**config, "quant_method": "other"}, checkpoint_keys=state) is None
    manifest = parse_runtime_manifest(config, checkpoint_keys=state)
    assert len(manifest.targets) == len(_FLUX1_SVDQ) + len(_FLUX1_AWQ) and manifest.structural_patches == ()
    assert manifest.precision == "int4" and manifest.rank == 32
    by_name = {t.checkpoint_prefix: t for t in manifest.targets}
    assert by_name["transformer_blocks.0.attn.to_q"].has_bias and by_name["transformer_blocks.0.attn.to_q"].rank == 32
    assert not by_name["transformer_blocks.0.attn.to_k"].has_bias
    assert by_name["transformer_blocks.0.norm1.linear"].nunchaku_op == "awq_w4a16"


@pytest.mark.parametrize("quantize", [None, 4, "no-base-weights"])
def test_lite_flux1_repo_loads_from_config_target_lists(tmp_path, quantize):
    ws, _, _, biases = _lite_flux1_repo(tmp_path / "repo")
    if quantize == "no-base-weights":
        cls, kwargs = FakeFlux1PipelineWithoutWeights, {}
    else:
        cls, kwargs = FakeFlux1Pipeline, {"quantize": quantize}
    pipe = load_nunchaku_pipeline("base", pipeline_cls=cls, checkpoint=tmp_path / "repo", backend="dense", **kwargs)
    assert pipe._nunchaku_lite_target == "flux1"
    tr = pipe.transformer
    assert not any(isinstance(m, nn.QuantizedLinear) for _, m in tr.named_modules())
    np.testing.assert_allclose(np.array(tr.x_embedder.weight), 0.5)
    np.testing.assert_allclose(np.array(tr.transformer_blocks[0].attn.add_v_proj.weight), 0.5)  # unquantized
    np.testing.assert_allclose(np.array(tr.transformer_blocks[0].ff.linear2.weight), 0.5)  # Diffusers ff.net.2

    def check(module, prefix, k):
        x = np.random.default_rng(len(prefix)).standard_normal((4, k)).astype(np.float32)
        ref = x.astype(np.float64) @ ws[prefix].T if prefix in _FLUX1_AWQ else _svdq_ref(ws[prefix], x)
        if prefix in biases:
            ref = ref + biases[prefix]
        assert _rel(_run(module, x), ref) < 2e-2, prefix

    block, single = tr.transformer_blocks[0], tr.single_transformer_blocks[0]
    assert isinstance(block.attn.to_k, SVDQW4A4Linear) and "bias" not in block.attn.to_k
    check(block.attn.to_q, "transformer_blocks.0.attn.to_q", D)
    check(block.attn.to_k, "transformer_blocks.0.attn.to_k", D)
    check(block.attn.to_out[0], "transformer_blocks.0.attn.to_out.0", D)
    check(block.ff.linear1, "transformer_blocks.0.ff.net.0.proj", D)
    check(block.ff_context.linear2, "transformer_blocks.0.ff_context.net.2", MLP)
    check(block.norm1.linear, "transformer_blocks.0.norm1.linear", D)
    check(single.proj_out, "single_transformer_blocks.0.proj_out", D + MLP)
    check(single.norm.linear, "single_transformer_blocks.0.norm.linear", D)


def test_flux1_adapter_rejects_checkpoints_without_targets():
    with pytest.raises(ValueError, match="original Nunchaku FLUX.1 checkpoints"):
        load_nunchaku_pipeline(
            "p", pipeline_cls=FakeFlux1Pipeline, checkpoint={"x_embedder.weight": mx.zeros((D, 128))}
        )


def test_flux1_key_map_matches_mflux_mapping():
    pytest.importorskip("mflux")
    key_map, fallback = flux1_key_map(), PrefixMap(_FLUX1_RENAMES)
    for name in (
        "transformer_blocks.3.ff.net.0.proj.qweight",
        "transformer_blocks.3.ff.net.2.bias",
        "transformer_blocks.0.ff_context.net.0.proj.wscales",
        "transformer_blocks.18.ff_context.net.2.qweight",
        "transformer_blocks.0.attn.to_out.0.qweight",
        "transformer_blocks.0.norm1_context.linear.wzeros",
        "single_transformer_blocks.37.proj_out.qweight",
        "single_transformer_blocks.0.norm.linear.qweight",
        "time_text_embed.guidance_embedder.linear_1.weight",
    ):
        assert key_map(name) == fallback(name), name
    assert key_map("transformer_blocks.3.ff.net.2.bias") == "transformer_blocks.3.ff.linear2.bias"


# ---------------------------------------------------------------------------
# Generic manifest adapter, structural patches, AdaNorm targets
# ---------------------------------------------------------------------------


class _Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(D, 6 * 64, bias=True)


class Generic(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D, 2 * D, bias=False)
        self.out = nn.Linear(2 * D, D, bias=False)
        self.norm = _Norm()


class GenericPipeline:
    def __init__(self, model_path: str, **kwargs):
        self.transformer = Generic()


def _generic_checkpoint():
    ws, state, targets = {}, {}, []
    for i, prefix in enumerate(("proj.linears.0", "proj.linears.1", "out.linears.0", "out.linears.1")):
        ws[prefix] = _svdq(D, D, 500 + i)
        state.update({f"{prefix}.{name}": v for name, v in pack_svdq_int4(ws[prefix]).items()})
        targets.append(_target(prefix))
    splits, n = 6, 6 * 64
    _, dense, (codes, scale, zero) = _awq(n, D, 600)
    bias = np.random.default_rng(601).standard_normal(n).astype(np.float32)

    def interleave(t):  # Nunchaku AdaNorm layout (see test_linear)
        return t.reshape(splits, n // splits, *t.shape[1:]).swapaxes(0, 1).reshape(t.shape)

    stored_bias = bias.reshape(splits, n // splits).T.copy()
    stored_bias[:, 1] += 1
    stored_bias[:, -2] += 1
    state["norm.linear.qweight"] = pack_awq_w4(interleave(codes))
    state["norm.linear.wscales"] = interleave(scale).T.copy()
    state["norm.linear.wzeros"] = interleave(zero).T.copy()
    state["norm.linear.bias"] = stored_bias.reshape(n)
    targets.append(_target("norm.linear", "adanorm_awq_w4a16", has_bias=True, op_options={"adanorm_splits": 6}))
    structural = [
        {"type": "split_linear_output", "module": "proj", "args": {"splits": [D]}},
        {"type": "split_linear_input", "module": "out", "args": {"splits": [D]}},
    ]
    return ws, (dense, bias), state, _manifest(targets, structural)


def test_generic_manifest_adapter_with_structural_patches_and_adanorm(tmp_path):
    ws, (norm_dense, norm_bias), state, manifest = _generic_checkpoint()
    folder = _save(tmp_path / "generic", state, manifest)
    pipe = load_nunchaku_pipeline("p", pipeline_cls=GenericPipeline, checkpoint=folder, backend="dense")
    assert pipe._nunchaku_lite_target == "manifest"
    tr = pipe.transformer
    assert isinstance(tr.proj, SplitLinearOutput) and isinstance(tr.out, SplitLinearInput)

    x = np.random.default_rng(7).standard_normal((3, D)).astype(np.float32)
    ref = np.concatenate([_svdq_ref(ws["proj.linears.0"], x), _svdq_ref(ws["proj.linears.1"], x)], -1)
    assert _rel(_run(tr.proj, x), ref) < 2e-2
    xy = np.random.default_rng(8).standard_normal((3, 2 * D)).astype(np.float32)
    ref = _svdq_ref(ws["out.linears.0"], xy[:, :D]) + _svdq_ref(ws["out.linears.1"], xy[:, D:])
    assert _rel(_run(tr.out, xy), ref) < 2e-2

    assert isinstance(tr.norm.linear, AWQW4A16Linear) and tr.norm.linear.adanorm_splits == 6
    assert _rel(_run(tr.norm.linear, x), x.astype(np.float64) @ norm_dense.T + norm_bias) < 2e-2


def test_patch_transformer_uses_manifest_and_key_map(tmp_path):
    ws, _, state, manifest = _generic_checkpoint()

    def rename(name):  # the checkpoint calls the model's ``proj`` module ``renamed``
        return "renamed" + name[len("proj") :] if name.split(".")[0] == "proj" else name

    renamed = {rename(k): v for k, v in state.items()}
    for target in manifest["targets"]:
        target["checkpoint_prefix"] = rename(target["checkpoint_prefix"])
        target["source_modules"] = [rename(m) for m in target["source_modules"]]
    manifest["structural_patches"][0]["module"] = "renamed"
    folder = _save(tmp_path / "ckpt", renamed, manifest)
    model = patch_transformer(Generic(), folder, key_map=PrefixMap([("renamed", "proj")]), backend="dense")
    assert isinstance(model.proj, SplitLinearOutput) and isinstance(model.proj.linears[1], SVDQW4A4Linear)
    x = np.random.default_rng(9).standard_normal((2, D)).astype(np.float32)
    assert _rel(_run(model.proj.linears[1], x), _svdq_ref(ws["proj.linears.1"], x)) < 2e-2


def test_manifest_errors(tmp_path):
    with pytest.raises(ValueError, match="requires a checkpoint with quantization_config.runtime_manifest"):
        load_nunchaku_pipeline("p", pipeline_cls=FakeFlux2Pipeline, checkpoint={}, target="manifest")
    fp4 = _manifest([_target("transformer_blocks.0.attn.to_q", precision="fp4")], precision="fp4")
    folder = _save(tmp_path / "fp4", {"x_embedder.weight": np.zeros((D, 128), np.float32)}, fp4)
    with pytest.raises(NotImplementedError, match="fp4"):
        load_nunchaku_pipeline("p", pipeline_cls=FakeFlux2Pipeline, checkpoint=folder)
    with pytest.raises(ValueError, match="Could not select an adapter"):
        load_nunchaku_pipeline("p", pipeline_cls=GenericPipeline, checkpoint={"proj.weight": mx.zeros((2 * D, D))})


def _rotated(manifest):
    manifest = {**manifest, "nunchaku_format_version": 2}
    manifest["targets"] = [dict(t) for t in manifest["targets"]]
    manifest["targets"][0]["op_options"] = {"rotation": {"type": "hadamard"}}
    return manifest


@pytest.mark.parametrize("strict", [True, False])
def test_rotated_manifest_is_refused(tmp_path, strict):
    # MLX applies no activation rotation; loading rotated weights unrotated would be silently wrong.
    _, _, state, manifest = _generic_checkpoint()
    folder = _save(tmp_path / "ckpt", state, _rotated(manifest))
    with pytest.raises(NotImplementedError, match="not supported on MLX"):
        patch_transformer(Generic(), folder, backend="dense", strict=strict)
    _, flux2_state, flux2_manifest = _lite_flux2_checkpoint()
    folder = _save(tmp_path / "repo" / "transformer", flux2_state, _rotated(flux2_manifest))
    with pytest.raises(NotImplementedError, match="not supported on MLX"):
        load_nunchaku_pipeline("base", pipeline_cls=FakeFlux2Pipeline, checkpoint=tmp_path / "repo", backend="dense")


def test_apply_runtime_manifest_refuses_rotation(tmp_path):
    from nunchaku_lite.mlx import apply_runtime_manifest

    _, _, state, manifest = _generic_checkpoint()
    with pytest.raises(NotImplementedError, match="not supported on MLX"):
        apply_runtime_manifest(Generic(), state, parse_runtime_manifest({"runtime_manifest": _rotated(manifest)}))


def test_apply_runtime_manifest_refuses_marker_in_state():
    from nunchaku_lite.mlx import apply_runtime_manifest

    _, _, state, manifest = _generic_checkpoint()
    state = {**state, "nunchaku_lite.input_rotation": mx.zeros((0,))}
    with pytest.raises(NotImplementedError, match="not supported on MLX"):
        apply_runtime_manifest(Generic(), state, parse_runtime_manifest({"runtime_manifest": manifest}))


def test_marker_checkpoint_is_refused_end_to_end(tmp_path):
    _, _, state, manifest = _generic_checkpoint()
    folder = _save(tmp_path / "ckpt", {**state, "nunchaku_lite.input_rotation": mx.zeros((0,))}, manifest)
    with pytest.raises(NotImplementedError, match="not supported on MLX"):
        patch_transformer(Generic(), folder, backend="dense", strict=False)


def test_rotation_declaration_and_legacy_transforms_are_refused():
    from nunchaku_lite.mlx.manifest import refuse_rotated_checkpoint

    refuse_rotated_checkpoint(None, {}, ["proj.qweight"])
    refuse_rotated_checkpoint(None, {"input_rotation": None}, [])  # null means absent, as in torch
    with pytest.raises(NotImplementedError, match="not supported on MLX"):
        refuse_rotated_checkpoint(None, {"input_rotation": {"format_version": 2}}, [])
    with pytest.raises(NotImplementedError, match="not supported on MLX"):
        refuse_rotated_checkpoint(None, {}, ["proj.input_transform"])
    with pytest.raises(NotImplementedError, match="not supported on MLX"):
        refuse_rotated_checkpoint(None, {}, ["proj.qweight", "nunchaku_lite.input_rotation"])


def test_quantization_config_falls_back_to_config_json(tmp_path):
    _, state, manifest = _lite_flux2_checkpoint()
    folder = _save(tmp_path / "repo" / "transformer", state)  # no safetensors metadata
    (folder / "config.json").write_text(json.dumps({"quantization_config": {"runtime_manifest": manifest}}))
    _, config = load_checkpoint(tmp_path / "repo", return_quantization_config=True)
    assert config["runtime_manifest"]["targets"][0]["checkpoint_prefix"] == "transformer_blocks.0.attn.to_q"
    assert load_checkpoint({}, return_quantization_config=True) == ({}, {})


def test_sharded_checkpoint_folder(tmp_path):
    _, state, manifest = _lite_flux2_checkpoint()
    keys = sorted(state)
    folder = tmp_path / "sharded"
    folder.mkdir()
    for i, part in enumerate((keys[: len(keys) // 2], keys[len(keys) // 2 :])):
        metadata = {"quantization_config": json.dumps({"runtime_manifest": manifest})} if i == 1 else {}
        arrays = {k: mx.array(state[k]) for k in part}
        mx.save_safetensors(str(folder / f"model-0000{i + 1}-of-00002.safetensors"), arrays, metadata=metadata)
    loaded, config = load_checkpoint(folder, return_quantization_config=True)
    assert set(loaded) == set(state) and len(config["runtime_manifest"]["targets"]) == 16


# ---------------------------------------------------------------------------
# PrefixMap
# ---------------------------------------------------------------------------


def test_prefix_map_longest_ancestor_and_placeholders():
    key_map = PrefixMap(
        [("a.{block}.to_out.0", "a.{block}.to_out"), ("time.embedder.linear_1", "time.linear_1"), ("same", "same")]
    )
    assert key_map("a.3.to_out.0") == "a.3.to_out"
    assert key_map("a.12.to_out.0.qweight") == "a.12.to_out.qweight"
    assert key_map("a.x.to_out.0") == "a.x.to_out.0"  # placeholders match integers only
    assert key_map("time.embedder.linear_1.weight") == "time.linear_1.weight"
    assert key_map("unrelated.key") == "unrelated.key"
    assert not PrefixMap() and key_map


def test_prefix_map_from_mflux_weight_targets():
    targets = [
        SimpleNamespace(to_pattern="t.{block}.out.weight", from_pattern=["t.{block}.out.0.weight"]),
        SimpleNamespace(to_pattern="t.{block}.out.bias", from_pattern=["t.{block}.out.0.bias"]),
        SimpleNamespace(to_pattern="x.weight", from_pattern=["x.weight"]),  # identity: no rule
        SimpleNamespace(to_pattern="n.scale", from_pattern=["n.weight"]),  # leaf rename: not a module rule
    ]
    key_map = PrefixMap.from_mflux(targets)
    assert key_map("t.4.out.0.wscales") == "t.4.out.wscales"
    assert key_map("n.weight") == "n.weight"


def test_flux2_key_map_matches_mflux_mapping():
    pytest.importorskip("mflux")
    key_map, fallback = flux2_key_map(), PrefixMap(_FLUX2_RENAMES)
    for name in (
        "time_guidance_embed.timestep_embedder.linear_1.weight",
        "time_guidance_embed.timestep_embedder.linear_2.weight",
        "transformer_blocks.7.attn.to_out.0.qweight",
        "transformer_blocks.0.attn.to_q.qweight",
        "single_transformer_blocks.3.attn.to_qkv_mlp_proj.wscales",
        "double_stream_modulation_img.linear.qweight",
    ):
        assert key_map(name) == fallback(name)
