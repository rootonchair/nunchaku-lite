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
from nunchaku_lite.manifest import parse_runtime_manifest
from nunchaku_lite.ops.fused import fused_gelu_mlp
from nunchaku_lite.ops.quantize import hadamard_matrix
from nunchaku_lite.rotation import (
    FUSED_MLP_FC2_ATTR,
    HADAMARD64,
    bind_input_rotations,
    parse_rotation_declaration,
    resolve_input_rotation,
)

ROTATION = {"type": "block_matrix", "block_size": 64, "shared": True, "generator": "hadamard"}


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
    return type(
        "Options",
        (),
        {
            "precision": "int4",
            "torch_dtype": torch.bfloat16,
            "device": None,
            "strict": True,
            "adapter_options": adapter_options or {},
        },
    )()


def _write_checkpoint(tmp_path, manifest, transforms):
    quantization_config = {"runtime_manifest": manifest}
    model = TinyModel()
    ManifestAdapter().patch(model, dict(transforms), quantization_config, _options())
    state = dict(model.state_dict())
    state.update(transforms)
    path = tmp_path / "rotated.safetensors"
    metadata = {"quantization_config": json.dumps(quantization_config)}
    save_file({k: v.contiguous() for k, v in state.items()}, path, metadata=metadata)
    return path


# --- manifest parsing -------------------------------------------------------------------


def test_format_v2_with_rotation_parses():
    parsed = _parse(_manifest())
    assert parsed.nunchaku_format_version == 2
    assert [t.checkpoint_prefix for t in parsed.rotated_targets] == ["proj"]
    assert parsed.targets[0].rotation == ROTATION
    assert parsed.targets[1].rotation is None


def test_format_v1_without_rotation_still_parses():
    parsed = _parse(_manifest(format_version=1, rotation=None))
    assert parsed.rotated_targets == ()


def test_rotation_requires_format_v2():
    with pytest.raises(ValueError, match="nunchaku_format_version >= 2"):
        _parse(_manifest(format_version=1))


def test_unknown_format_version_is_refused():
    with pytest.raises(ValueError, match="nunchaku_format_version 3"):
        _parse(_manifest(format_version=3))


def test_legacy_hadamard_rotation_format_is_refused_with_reexport_hint():
    with pytest.raises(ValueError, match="re-export"):
        _parse(_manifest(rotation={"type": "hadamard", "block_size": 64}))


@pytest.mark.parametrize(
    "rotation,error,match",
    [
        ({"type": "learned", "block_size": 64, "shared": True}, ValueError, "unsupported rotation type"),
        ({"type": "block_matrix", "block_size": 0, "shared": True}, ValueError, "block_size"),
        ({"type": "block_matrix", "block_size": 64}, TypeError, "shared"),
        ({"type": "block_matrix", "block_size": 64, "shared": True, "generator": 1}, TypeError, "generator"),
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


# --- resolve_input_rotation -------------------------------------------------------------


def _rotated_target(**kwargs):
    return _parse(_manifest(**kwargs)).targets[0]


def test_shared_h64_resolves_to_hadamard64_and_pops_tensor():
    state = {"proj.input_transform": hadamard_matrix(64), "proj.qweight": torch.zeros(1)}
    assert resolve_input_rotation(_rotated_target(), state) == HADAMARD64
    assert "proj.input_transform" not in state and "proj.qweight" in state


def test_h64_without_generator_hint_still_resolves_from_the_data():
    rotation = {k: v for k, v in ROTATION.items() if k != "generator"}
    state = {"proj.input_transform": hadamard_matrix(64)}
    assert resolve_input_rotation(_rotated_target(rotation=rotation), state) == HADAMARD64


def test_missing_transform_tensor_is_refused():
    with pytest.raises(ValueError, match="missing"):
        resolve_input_rotation(_rotated_target(), {})


def test_stray_transform_on_unrotated_target_is_refused():
    target = _parse(_manifest(rotation=None)).targets[0]
    with pytest.raises(ValueError, match="declares no rotation"):
        resolve_input_rotation(target, {"proj.input_transform": hadamard_matrix(64)})


def test_generator_hint_never_overrides_the_data():
    q, _ = torch.linalg.qr(torch.randn(64, 64, dtype=torch.float64))
    with pytest.raises(ValueError, match="generator='hadamard'"):
        resolve_input_rotation(_rotated_target(), {"proj.input_transform": q.float()})


def test_general_block_matrix_is_not_implemented():
    rotation = {"type": "block_matrix", "block_size": 64, "shared": True}
    q, _ = torch.linalg.qr(torch.randn(64, 64, dtype=torch.float64))
    with pytest.raises(NotImplementedError, match="general block_matrix"):
        resolve_input_rotation(_rotated_target(rotation=rotation), {"proj.input_transform": q.float()})


def test_per_block_h64_is_not_implemented_yet():
    rotation = {"type": "block_matrix", "block_size": 64, "shared": False}
    blocks = hadamard_matrix(64).expand(2, 64, 64).clone()
    with pytest.raises(NotImplementedError):
        resolve_input_rotation(_rotated_target(rotation=rotation), {"proj.input_transform": blocks})


@pytest.mark.parametrize(
    "shape,shared", [((32, 32), True), ((64, 64, 64), True), ((2, 64, 64), True), ((64, 64), False)]
)
def test_transform_shape_must_match_metadata(shape, shared):
    rotation = {**ROTATION, "shared": shared}
    with pytest.raises(ValueError, match="input_transform"):
        resolve_input_rotation(_rotated_target(rotation=rotation), {"proj.input_transform": torch.zeros(shape)})


def test_fp4_rotation_is_refused_by_default_and_allowed_with_opt_in():
    target = _rotated_target(precision="fp4")
    with pytest.raises(ValueError, match="allow_fp4_rotation"):
        resolve_input_rotation(target, {"proj.input_transform": hadamard_matrix(64)})
    assert resolve_input_rotation(target, {"proj.input_transform": hadamard_matrix(64)}, allow_fp4=True) == HADAMARD64


# --- end-to-end loading -----------------------------------------------------------------


def test_manifest_load_binds_hadamard_op_per_layer(tmp_path):
    checkpoint = _write_checkpoint(tmp_path, _manifest(), {"proj.input_transform": hadamard_matrix(64)})
    transformer = TinyModel()
    patch_transformer(transformer, checkpoint, target="auto", torch_dtype=torch.bfloat16, device="cpu")

    assert transformer._nunchaku_lite_target == "manifest"
    assert isinstance(transformer.proj, SVDQW4A4Linear) and transformer.proj.input_rotation == HADAMARD64
    assert transformer.other.input_rotation is None
    assert not transformer._nunchaku_lite_incompatible_keys.unexpected_keys


def test_bad_transform_fails_before_the_model_is_modified():
    model = TinyModel()
    original = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="missing"):
        ManifestAdapter().patch(model, {}, {"runtime_manifest": _manifest()}, _options())
    assert isinstance(model.proj, nn.Linear)
    assert all(torch.equal(original[k], v) for k, v in model.state_dict().items())


def test_model_specific_adapter_refuses_rotated_checkpoint(tmp_path, monkeypatch):
    from nunchaku_lite import core

    class FakeAdapter:
        target = "rotation_fake"

        def matches(self, transformer):
            return False

        def patch(self, transformer, checkpoint_state, quantization_config, options):
            raise AssertionError("must refuse before patching")

    checkpoint = _write_checkpoint(tmp_path, _manifest(), {"proj.input_transform": hadamard_matrix(64)})
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
    SVDQW4A4Linear(128, 128, rank=16, input_rotation=HADAMARD64).quantize(torch.zeros(4, 128))
    assert calls == ["plain", "hadamard"]


def test_linear_rejects_unknown_rotation_and_unsigned_combination(monkeypatch):
    with pytest.raises(ValueError, match="Invalid input_rotation"):
        SVDQW4A4Linear(128, 128, input_rotation="random")
    with pytest.raises(ValueError, match="signed activations"):
        SVDQW4A4Linear(128, 128, act_unsigned=True, input_rotation=HADAMARD64)
    # Adapters set act_unsigned after construction; quantize must re-check.
    calls = _record_quantize(monkeypatch)
    layer = SVDQW4A4Linear(128, 128, rank=16, input_rotation=HADAMARD64)
    layer.act_unsigned = True
    with pytest.raises(ValueError, match="signed activations"):
        layer.quantize(torch.zeros(4, 128))
    assert calls == []


def test_fused_gelu_mlp_refuses_rotated_fc2():
    fc1 = SVDQW4A4Linear(128, 256, rank=16)
    fc2 = SVDQW4A4Linear(256, 128, rank=16, input_rotation=HADAMARD64)
    with pytest.raises(ValueError, match="input-rotated fc2"):
        fused_gelu_mlp(torch.zeros(1, 4, 128), fc1, fc2)


def test_mps_path_refuses_rotated_layer():
    # The MPS kernels apply no activation rotation; checked before any device work, so this runs on CPU.
    from nunchaku_lite.ops.mps import svdq_linear_mps

    layer = SVDQW4A4Linear(128, 128, rank=16, input_rotation=HADAMARD64)
    with pytest.raises(NotImplementedError, match="not supported on MPS"):
        svdq_linear_mps(layer, torch.zeros(4, 128))


@pytest.mark.parametrize("with_manifest", [False, True])
def test_rotation_tensor_without_rotation_metadata_is_refused(tmp_path, monkeypatch, with_manifest):
    from nunchaku_lite import core

    class FakeAdapter:
        target = "rotation_fake_nomanifest"

        def matches(self, transformer):
            return False

        def patch(self, transformer, checkpoint_state, quantization_config, options):
            raise AssertionError("must refuse before patching")

    state = {k: v.contiguous() for k, v in TinyModel().state_dict().items()}
    state["proj.input_transform"] = hadamard_matrix(64)
    quantization_config = {"runtime_manifest": _manifest(format_version=1, rotation=None)} if with_manifest else {}
    path = tmp_path / "stray.safetensors"
    save_file(state, path, metadata={"quantization_config": json.dumps(quantization_config)})
    monkeypatch.setitem(core._ADAPTERS, FakeAdapter.target, FakeAdapter())
    target = "manifest" if with_manifest else FakeAdapter.target
    with pytest.raises(ValueError, match="activation-rotation tensors"):
        patch_transformer(TinyModel(), path, target=target, torch_dtype=torch.bfloat16, device="cpu", strict=False)


# --- model-specific adapters (quantization_config.input_rotation) -----------------------

DECLARATION = {"format_version": 2, **ROTATION}


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


def _flux2_checkpoint(tmp_path, transforms, quantization_config):
    source = _tiny_flux2()
    options = SimpleNamespace(
        precision="int4", torch_dtype=torch.bfloat16, device=None, strict=True, adapter_options={}
    )
    Flux2Adapter().patch(source, {}, {"rank": 16}, options)
    state = {k: v.contiguous() for k, v in source.state_dict().items()}
    state.update(transforms)
    path = tmp_path / "flux2-rotated.safetensors"
    save_file(state, path, metadata={"quantization_config": json.dumps({"rank": 16, **quantization_config})})
    return path


def test_rotation_declaration_parses_and_validates():
    assert parse_rotation_declaration({}) is None
    assert parse_rotation_declaration({"input_rotation": DECLARATION}) == ROTATION
    with pytest.raises(ValueError, match="format_version"):
        parse_rotation_declaration({"input_rotation": {**DECLARATION, "format_version": 1}})
    with pytest.raises(ValueError, match="re-export"):
        parse_rotation_declaration({"input_rotation": {"format_version": 2, "type": "hadamard", "block_size": 64}})


@pytest.mark.parametrize(
    "transform_key",
    [
        "transformer_blocks.0.attn.to_qkv.input_transform",  # lite module name
        "transformer_blocks.0.qkv_proj.input_transform",  # original nunchaku export name
    ],
)
def test_flux2_adapter_binds_declared_rotation_per_layer(tmp_path, transform_key):
    path = _flux2_checkpoint(
        tmp_path,
        {
            transform_key: hadamard_matrix(64),
            "single_transformer_blocks.0.attn.mlp_fc2.input_transform": hadamard_matrix(64),
        },
        {"input_rotation": DECLARATION},
    )
    transformer = _tiny_flux2()
    patch_transformer(transformer, path, target="flux2", precision="int4", torch_dtype=torch.bfloat16)

    assert transformer.transformer_blocks[0].attn.to_qkv.input_rotation == HADAMARD64
    assert transformer.single_transformer_blocks[0].attn.mlp_fc2.input_rotation == HADAMARD64
    assert transformer.single_transformer_blocks[0].attn.qkv_proj.input_rotation is None
    assert transformer.transformer_blocks[0].ff.linear_in.input_rotation is None
    assert not transformer._nunchaku_lite_incompatible_keys.unexpected_keys


def test_flux2_rotation_tensors_without_declaration_are_refused(tmp_path):
    path = _flux2_checkpoint(tmp_path, {"transformer_blocks.0.attn.to_qkv.input_transform": hadamard_matrix(64)}, {})
    with pytest.raises(ValueError, match="no rotation metadata"):
        patch_transformer(_tiny_flux2(), path, target="flux2", precision="int4", strict=False)


def test_declaration_without_tensors_is_refused(tmp_path):
    path = _flux2_checkpoint(tmp_path, {}, {"input_rotation": DECLARATION})
    with pytest.raises(ValueError, match="no input_transform tensors"):
        patch_transformer(_tiny_flux2(), path, target="flux2", precision="int4")


def test_adapter_without_rotation_support_is_refused(tmp_path, monkeypatch):
    from nunchaku_lite import core

    monkeypatch.setattr(Flux2Adapter, "supports_input_rotation", False)
    path = _flux2_checkpoint(
        tmp_path,
        {"transformer_blocks.0.attn.to_qkv.input_transform": hadamard_matrix(64)},
        {"input_rotation": DECLARATION},
    )
    assert core._ADAPTERS["flux2"].supports_input_rotation is False
    with pytest.raises(ValueError, match="does not support"):
        patch_transformer(_tiny_flux2(), path, target="flux2", precision="int4")


class _BindModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = SVDQW4A4Linear(128, 128, rank=16)
        self.dense = nn.Linear(128, 128)


def test_bind_input_rotations_validates_each_layer():
    model = _BindModel()
    state = {"a.input_transform": hadamard_matrix(64), "a.qweight": torch.zeros(1)}
    assert bind_input_rotations(model, state, ROTATION) == ["a"]
    assert model.a.input_rotation == HADAMARD64 and list(state) == ["a.qweight"]

    with pytest.raises(ValueError, match="does not belong"):
        bind_input_rotations(_BindModel(), {"missing.input_transform": hadamard_matrix(64)}, ROTATION)
    with pytest.raises(TypeError, match="not an SVDQ"):
        bind_input_rotations(_BindModel(), {"dense.input_transform": hadamard_matrix(64)}, ROTATION)

    unsigned = _BindModel()
    unsigned.a.act_unsigned = True
    with pytest.raises(ValueError, match="unsigned or fused-MLP"):
        bind_input_rotations(unsigned, {"a.input_transform": hadamard_matrix(64)}, ROTATION)

    fused = _BindModel()
    setattr(fused.a, FUSED_MLP_FC2_ATTR, True)
    with pytest.raises(ValueError, match="unsigned or fused-MLP"):
        bind_input_rotations(fused, {"a.input_transform": hadamard_matrix(64)}, ROTATION)

    fp4 = _BindModel()
    fp4.a = SVDQW4A4Linear(128, 128, rank=16, precision="nvfp4")
    with pytest.raises(ValueError, match="allow_fp4_rotation"):
        bind_input_rotations(fp4, {"a.input_transform": hadamard_matrix(64)}, ROTATION)
    assert bind_input_rotations(fp4, {"a.input_transform": hadamard_matrix(64)}, ROTATION, allow_fp4=True) == ["a"]


def test_fused_adapters_mark_their_fused_fc2():
    from diffusers.models.attention import FeedForward

    from nunchaku_lite.adapters.common import SVDQPatchContext
    from nunchaku_lite.adapters.ltx2 import _patch_ltx2_feed_forward

    context = SVDQPatchContext(precision="int4", rank=16, torch_dtype=torch.bfloat16)
    ff = _patch_ltx2_feed_forward(FeedForward(64, activation_fn="gelu-approximate"), context)
    assert getattr(ff.net[2], FUSED_MLP_FC2_ATTR, False)
