import copy
import json
from types import SimpleNamespace

import pytest
import torch
from diffusers import Flux2Transformer2DModel
from safetensors.torch import save_file
from torch import nn

from nunchaku_lite import linear as linear_mod
from nunchaku_lite import patch_transformer
from nunchaku_lite.adapters.flux2 import Flux2Adapter
from nunchaku_lite.adapters.manifest import ManifestAdapter
from nunchaku_lite.linear import SVDQW4A4Linear
from nunchaku_lite.manifest import parse_runtime_manifest, with_target_list_manifest
from nunchaku_lite.ops.fused import fused_gelu_mlp
from nunchaku_lite.rotation import (
    FUSED_MLP_FC2_ATTR,
    HADAMARD,
    bind_input_rotations,
    check_rotation_runtime,
    mark_rotated_targets,
    parse_rotation_declaration,
    resolve_input_rotation,
)

ROTATION = {"type": "hadamard"}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 128)
        self.other = nn.Linear(128, 128)


def _target(prefix, *, op="svdq_w4a4", precision="int4", rotation=None):
    op_options = {"rotation": rotation} if rotation is not None else {}
    return {
        "name": prefix,
        "checkpoint_prefix": prefix,
        "source_modules": [prefix],
        "roles": [],
        "kind": "linear",
        "nunchaku_op": op,
        "precision": precision,
        "group_size": 64 if op == "svdq_w4a4" else 128,
        "rank": 16,
        "has_bias": True,
        "op_options": op_options,
        "activation": {},
    }


def _manifest(*, format_version=2, precision="int4", rotation=ROTATION, other_rotation=None):
    return {
        "schema": "nunchaku_lite.runtime_manifest",
        "version": 1,
        "component": "transformer",
        "nunchaku_format_version": format_version,
        "producer": {"name": "test", "version": "0"},
        "requirements": {
            "method": "svdquant",
            "precision": precision,
            "rank": 16,
            "weight_dtype": "int4",
            "activation_dtype": "int4",
        },
        "structural_patches": [],
        "targets": [
            _target("proj", precision=precision, rotation=rotation),
            _target("other", precision=precision, rotation=other_rotation),
        ],
    }


def _parse(manifest):
    return parse_runtime_manifest({"runtime_manifest": manifest})


def _options(adapter_options=None):
    return SimpleNamespace(
        precision="int4", torch_dtype=torch.bfloat16, device=None, strict=True, adapter_options=adapter_options or {}
    )


def _tiny_state():
    model = TinyModel()
    ManifestAdapter().patch(model, {}, {"runtime_manifest": _manifest(rotation=None)}, _options())
    return {k: v.contiguous() for k, v in model.state_dict().items()}


def _write(tmp_path, quantization_config, extra=None, name="ckpt.safetensors"):
    state = _tiny_state()
    state.update(extra or {})
    path = tmp_path / name
    save_file(state, path, metadata={"quantization_config": json.dumps(quantization_config)})
    return path


def _declaration(targets):
    return {"format_version": 2, "type": "hadamard", "targets": list(targets)}


# --- manifest parsing -------------------------------------------------------------------


def test_format_v2_with_rotation_parses():
    parsed = _parse(_manifest())
    assert parsed.nunchaku_format_version == 2
    assert [t.checkpoint_prefix for t in parsed.rotated_targets] == ["proj"]
    assert parsed.targets[0].rotation == ROTATION
    assert parsed.targets[1].rotation is None


def test_format_v1_without_rotation_still_parses():
    assert _parse(_manifest(format_version=1, rotation=None)).rotated_targets == ()


def test_rotation_requires_format_v2():
    with pytest.raises(ValueError, match="nunchaku_format_version >= 2"):
        _parse(_manifest(format_version=1))


def test_unknown_format_version_is_refused():
    with pytest.raises(ValueError, match="nunchaku_format_version 3"):
        _parse(_manifest(format_version=3))


def test_pre_release_block_matrix_format_is_refused_with_conversion_hint():
    rotation = {"type": "block_matrix", "block_size": 64, "shared": True, "generator": "hadamard"}
    with pytest.raises(ValueError, match="drop the .input_transform tensors"):
        _parse(_manifest(rotation=rotation))


@pytest.mark.parametrize(
    "rotation,error,match",
    [
        ({"type": "learned"}, ValueError, "unsupported rotation type"),
        ({"type": "hadamard", "block_size": 64}, ValueError, "unknown rotation fields"),
        ("hadamard", TypeError, "JSON object"),
    ],
)
def test_malformed_rotation_metadata_is_refused(rotation, error, match):
    with pytest.raises(error, match=match):
        _parse(_manifest(rotation=rotation))


def test_rotation_on_non_svdq_target_is_refused():
    manifest = _manifest()
    manifest["targets"][0] = _target("proj", op="awq_w4a16", rotation=ROTATION)
    with pytest.raises(ValueError, match="only supported for svdq_w4a4"):
        _parse(manifest)


# --- model-level declaration ------------------------------------------------------------


def test_rotation_declaration_parses():
    assert parse_rotation_declaration({}) is None
    assert parse_rotation_declaration({"input_rotation": _declaration(["a", "b"])}) == ("a", "b")


@pytest.mark.parametrize(
    "declaration,error,match",
    [
        ({**_declaration(["a"]), "format_version": 1}, ValueError, "format_version"),
        ({"format_version": 2, "type": "hadamard"}, TypeError, "targets"),
        (_declaration([]), TypeError, "targets"),
        (_declaration(["a", 3]), TypeError, "targets"),
        (_declaration(["a", "a"]), ValueError, "more than once"),
        ({**_declaration(["a"]), "type": "block_matrix"}, ValueError, "pre-release"),
        ({**_declaration(["a"]), "block_size": 64}, ValueError, "unknown rotation fields"),
        ("hadamard", TypeError, "JSON object"),
    ],
)
def test_malformed_rotation_declaration_is_refused(declaration, error, match):
    with pytest.raises(error, match=match):
        parse_rotation_declaration({"input_rotation": declaration})


def _target_list_config(declaration=None):
    config = {
        "quant_method": "nunchaku_lite",
        "svdq_w4a4": {"targets": ["proj", "other"], "precision": "int4", "group_size": 64, "rank": 16},
    }
    if declaration is not None:
        config["input_rotation"] = declaration
    return config


def test_target_list_manifest_carries_the_declaration():
    keys = _tiny_state().keys()
    synthesized = with_target_list_manifest(_target_list_config(_declaration(["proj"])), keys)
    parsed = parse_runtime_manifest(synthesized)
    assert parsed.nunchaku_format_version == 2
    assert [t.checkpoint_prefix for t in parsed.rotated_targets] == ["proj"]
    unrotated = parse_runtime_manifest(with_target_list_manifest(_target_list_config(), keys))
    assert unrotated.nunchaku_format_version == 1 and unrotated.rotated_targets == ()
    with pytest.raises(ValueError, match="not svdq_w4a4 targets"):
        with_target_list_manifest(_target_list_config(_declaration(["missing"])), keys)


# --- resolve / runtime checks -----------------------------------------------------------


def test_resolve_input_rotation_and_fp4_opt_in():
    assert resolve_input_rotation(_parse(_manifest(rotation=None)).targets[0]) is None
    assert resolve_input_rotation(_parse(_manifest()).targets[0]) == HADAMARD
    fp4 = _parse(_manifest(precision="fp4")).targets[0]
    with pytest.raises(ValueError, match="allow_fp4_rotation"):
        resolve_input_rotation(fp4)
    assert resolve_input_rotation(fp4, allow_fp4=True) == HADAMARD


def test_check_rotation_runtime(monkeypatch):
    from nunchaku_lite.ops import backend

    check_rotation_runtime(None)
    check_rotation_runtime("cpu")
    with pytest.raises(NotImplementedError, match="MPS"):
        check_rotation_runtime("mps")
    monkeypatch.setattr(backend, "get_ops", lambda: SimpleNamespace())
    with pytest.raises(NotImplementedError, match="quantize_w4a4_act_hadamard_fuse_lora"):
        check_rotation_runtime("cuda")
    monkeypatch.setattr(backend, "get_ops", lambda: SimpleNamespace(quantize_w4a4_act_hadamard_fuse_lora=object()))
    check_rotation_runtime("cuda")


# --- manifest-path loading --------------------------------------------------------------


def test_manifest_load_binds_hadamard_op_per_layer(tmp_path):
    checkpoint = _write(tmp_path, {"runtime_manifest": _manifest()})
    transformer = TinyModel()
    patch_transformer(transformer, checkpoint, target="auto", torch_dtype=torch.bfloat16, device="cpu")

    assert transformer._nunchaku_lite_target == "manifest"
    assert isinstance(transformer.proj, SVDQW4A4Linear) and transformer.proj.input_rotation == HADAMARD
    assert transformer.other.input_rotation is None
    assert not transformer._nunchaku_lite_incompatible_keys.unexpected_keys


def test_manifest_with_matching_declaration_loads(tmp_path):
    checkpoint = _write(tmp_path, {"runtime_manifest": _manifest(), "input_rotation": _declaration(["proj"])})
    transformer = TinyModel()
    patch_transformer(transformer, checkpoint, target="auto", torch_dtype=torch.bfloat16, device="cpu")
    assert transformer.proj.input_rotation == HADAMARD


@pytest.mark.parametrize("manifest_rotation", [ROTATION, None])
def test_manifest_with_inconsistent_declaration_is_refused(tmp_path, manifest_rotation):
    config = {"runtime_manifest": _manifest(rotation=manifest_rotation), "input_rotation": _declaration(["other"])}
    checkpoint = _write(tmp_path, config)
    model = TinyModel()
    with pytest.raises(ValueError, match="inconsistent"):
        patch_transformer(model, checkpoint, target="auto", torch_dtype=torch.bfloat16, device="cpu")
    assert isinstance(model.proj, nn.Linear)


def test_target_list_checkpoint_with_declaration_loads(tmp_path):
    checkpoint = _write(tmp_path, _target_list_config(_declaration(["proj"])))
    transformer = TinyModel()
    patch_transformer(transformer, checkpoint, target="auto", torch_dtype=torch.bfloat16, device="cpu")
    assert transformer._nunchaku_lite_target == "manifest"
    assert transformer.proj.input_rotation == HADAMARD and transformer.other.input_rotation is None


def test_pre_release_input_transform_tensors_are_refused(tmp_path):
    checkpoint = _write(
        tmp_path, {"runtime_manifest": _manifest(rotation=None)}, {"proj.input_transform": torch.eye(64)}
    )
    with pytest.raises(ValueError, match="block_matrix"):
        patch_transformer(TinyModel(), checkpoint, target="auto", torch_dtype=torch.bfloat16, device="cpu")


def test_rotated_checkpoint_on_mps_is_refused_at_load(tmp_path):
    checkpoint = _write(tmp_path, {"runtime_manifest": _manifest()})
    model = TinyModel()
    with pytest.raises(NotImplementedError, match="MPS"):
        patch_transformer(model, checkpoint, target="auto", torch_dtype=torch.bfloat16, device="mps")
    assert isinstance(model.proj, nn.Linear)


def test_model_specific_adapter_refuses_manifest_rotation(tmp_path, monkeypatch):
    from nunchaku_lite import core

    class FakeAdapter:
        target = "rotation_fake"

        def matches(self, transformer):
            return False

        def patch(self, transformer, checkpoint_state, quantization_config, options):
            raise AssertionError("must refuse before patching")

    checkpoint = _write(tmp_path, {"runtime_manifest": _manifest()})
    monkeypatch.setitem(core._ADAPTERS, FakeAdapter.target, FakeAdapter())
    with pytest.raises(ValueError, match="only target='manifest'"):
        patch_transformer(TinyModel(), checkpoint, target=FakeAdapter.target, torch_dtype=torch.bfloat16, device="cpu")


# --- runtime binding --------------------------------------------------------------------


def _record_quantize(monkeypatch):
    calls = []
    monkeypatch.setattr(linear_mod, "svdq_quantize_w4a4_act_fuse_lora_cuda", lambda *a, **k: calls.append("plain"))
    monkeypatch.setattr(
        linear_mod, "svdq_quantize_w4a4_act_hadamard_fuse_lora_cuda", lambda *a, **k: calls.append("hadamard")
    )
    return calls


def test_linear_quantize_dispatches_on_input_rotation(monkeypatch):
    calls = _record_quantize(monkeypatch)
    SVDQW4A4Linear(128, 128, rank=16).quantize(torch.zeros(4, 128))
    SVDQW4A4Linear(128, 128, rank=16, input_rotation=HADAMARD).quantize(torch.zeros(4, 128))
    assert calls == ["plain", "hadamard"]


def test_linear_rejects_unknown_rotation_and_unsigned_combination(monkeypatch):
    with pytest.raises(ValueError, match="Invalid input_rotation"):
        SVDQW4A4Linear(128, 128, input_rotation="random")
    with pytest.raises(ValueError, match="signed activations"):
        SVDQW4A4Linear(128, 128, act_unsigned=True, input_rotation=HADAMARD)
    # Adapters set act_unsigned after construction; quantize must re-check.
    calls = _record_quantize(monkeypatch)
    layer = SVDQW4A4Linear(128, 128, rank=16, input_rotation=HADAMARD)
    layer.act_unsigned = True
    with pytest.raises(ValueError, match="signed activations"):
        layer.quantize(torch.zeros(4, 128))
    assert calls == []


def test_fused_gelu_mlp_refuses_rotated_fc2():
    fc1 = SVDQW4A4Linear(128, 256, rank=16)
    fc2 = SVDQW4A4Linear(256, 128, rank=16, input_rotation=HADAMARD)
    with pytest.raises(ValueError, match="input-rotated fc2"):
        fused_gelu_mlp(torch.zeros(1, 4, 128), fc1, fc2)


def test_mps_path_refuses_rotated_layer():
    # Backstop for a model moved to MPS after loading; checked before any device work, so this runs on CPU.
    from nunchaku_lite.ops.mps import svdq_linear_mps

    layer = SVDQW4A4Linear(128, 128, rank=16, input_rotation=HADAMARD)
    with pytest.raises(NotImplementedError, match="not supported on MPS"):
        svdq_linear_mps(layer, torch.zeros(4, 128))


# --- model-specific adapters (quantization_config.input_rotation) -----------------------


def _tiny_flux2():
    return Flux2Transformer2DModel(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=32,
        num_attention_heads=2,
        joint_attention_dim=64,
        guidance_embeds=False,
        axes_dims_rope=(8, 8, 8, 8),
        timestep_guidance_channels=32,
    )


def _flux2_checkpoint(tmp_path, quantization_config, *, rename=None):
    source = _tiny_flux2()
    Flux2Adapter().patch(source, {}, {"rank": 16}, _options())
    state = {k: v.contiguous() for k, v in source.state_dict().items()}
    if rename is not None:
        old, new = rename
        state = {(new + k[len(old) :] if k.startswith(old) else k): v for k, v in state.items()}
    path = tmp_path / "flux2-rotated.safetensors"
    save_file(state, path, metadata={"quantization_config": json.dumps({"rank": 16, **quantization_config})})
    return path


def _flux2_is_unpatched(transformer):
    return not any(isinstance(m, SVDQW4A4Linear) for m in transformer.modules()) and not hasattr(
        transformer, "_nunchaku_lite_flux2_original_forward"
    )


@pytest.mark.parametrize(
    "qkv_target,rename",
    [
        ("transformer_blocks.0.attn.to_qkv", None),  # lite module name
        # original nunchaku export name, renamed by the adapter's checkpoint conversion
        ("transformer_blocks.0.qkv_proj", ("transformer_blocks.0.attn.to_qkv.", "transformer_blocks.0.qkv_proj.")),
    ],
)
def test_flux2_adapter_binds_declared_targets(tmp_path, qkv_target, rename):
    targets = [qkv_target, "single_transformer_blocks.0.attn.mlp_fc2"]
    path = _flux2_checkpoint(tmp_path, {"input_rotation": _declaration(targets)}, rename=rename)
    transformer = _tiny_flux2()
    patch_transformer(transformer, path, target="flux2", precision="int4", torch_dtype=torch.bfloat16)

    assert transformer.transformer_blocks[0].attn.to_qkv.input_rotation == HADAMARD
    assert transformer.single_transformer_blocks[0].attn.mlp_fc2.input_rotation == HADAMARD
    assert transformer.single_transformer_blocks[0].attn.qkv_proj.input_rotation is None
    assert transformer.transformer_blocks[0].ff.linear_in.input_rotation is None
    assert not transformer._nunchaku_lite_incompatible_keys.unexpected_keys


def test_flux2_unknown_declared_target_fails_before_patching(tmp_path):
    path = _flux2_checkpoint(tmp_path, {"input_rotation": _declaration(["transformer_blocks.0.missing"])})
    transformer = _tiny_flux2()
    with pytest.raises(ValueError, match="no packed SVDQ weight"):
        patch_transformer(transformer, path, target="flux2", precision="int4")
    assert _flux2_is_unpatched(transformer)


def test_flux2_nvfp4_rotation_fails_before_patching(tmp_path):
    path = _flux2_checkpoint(tmp_path, {"input_rotation": _declaration(["transformer_blocks.0.attn.to_qkv"])})
    transformer = _tiny_flux2()
    with pytest.raises(ValueError, match="allow_fp4_rotation"):
        patch_transformer(transformer, path, target="flux2", precision="fp4", device="cpu")
    assert _flux2_is_unpatched(transformer)


def test_adapter_without_rotation_support_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(Flux2Adapter, "supports_input_rotation", False)
    path = _flux2_checkpoint(tmp_path, {"input_rotation": _declaration(["transformer_blocks.0.attn.to_qkv"])})
    transformer = _tiny_flux2()
    with pytest.raises(ValueError, match="does not support"):
        patch_transformer(transformer, path, target="flux2", precision="int4")
    assert _flux2_is_unpatched(transformer)


class _BindModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = SVDQW4A4Linear(128, 128, rank=16)
        self.dense = nn.Linear(128, 128)


def _bind(model, targets):
    state = {"a.qweight": torch.zeros(1)}
    mark_rotated_targets(state, targets)
    bound = bind_input_rotations(model, state)
    assert list(state) == ["a.qweight"]  # markers consumed
    return bound


def test_bind_input_rotations_checks_the_patched_structure():
    model = _BindModel()
    assert _bind(model, ["a"]) == ["a"] and model.a.input_rotation == HADAMARD

    with pytest.raises(ValueError, match="not a module"):
        _bind(_BindModel(), ["missing"])
    with pytest.raises(TypeError, match="not an SVDQ"):
        _bind(_BindModel(), ["dense"])

    unsigned = _BindModel()
    unsigned.a.act_unsigned = True
    with pytest.raises(ValueError, match="unsigned or fused-MLP"):
        _bind(unsigned, ["a"])

    fused = _BindModel()
    setattr(fused.a, FUSED_MLP_FC2_ATTR, True)
    with pytest.raises(ValueError, match="unsigned or fused-MLP"):
        _bind(fused, ["a"])


def test_bad_manifest_rotation_fails_before_the_model_is_modified():
    model = TinyModel()
    original = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="allow_fp4_rotation"):
        ManifestAdapter().patch(model, {}, {"runtime_manifest": _manifest(precision="fp4")}, _options())
    assert isinstance(model.proj, nn.Linear)
    assert all(torch.equal(original[k], v) for k, v in model.state_dict().items())


def test_fused_adapters_mark_their_fused_fc2():
    from diffusers.models.attention import FeedForward

    from nunchaku_lite.adapters.common import SVDQPatchContext
    from nunchaku_lite.adapters.ltx2 import _patch_ltx2_feed_forward

    context = SVDQPatchContext(precision="int4", rank=16, torch_dtype=torch.bfloat16)
    ff = _patch_ltx2_feed_forward(FeedForward(64, activation_fn="gelu-approximate"), context)
    assert getattr(ff.net[2], FUSED_MLP_FC2_ATTR, False)
