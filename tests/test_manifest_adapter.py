import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from nunchaku_lite import patch_transformer
from nunchaku_lite.adapters.manifest import (
    ManifestAdapter,
    ManifestAdaNormAWQW4A16,
    SplitLinearInput,
    SplitLinearOutput,
)
from nunchaku_lite.adapters.common import PATCHED_MODULE_ATTR
from nunchaku_lite.linear import AWQW4A16Linear, SVDQW4A4Linear
from nunchaku_lite.lora.core.layout import unpack_lowrank_weight
from nunchaku_lite.lora.manifest import COMFYUI_FORMAT, KOHYA_FORMAT, PEFT_FORMAT, detect_manifest_lora_format
from nunchaku_lite.manifest import parse_runtime_manifest


class TinyManifestModel(nn.Module):
    def __init__(self, out_features: int = 128):
        super().__init__()
        self.proj = nn.Linear(128, out_features)

    def forward(self, x):
        return self.proj(x)


class TinyFusedManifestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(128, 128)
        self.k = nn.Linear(128, 128)
        self.v = nn.Linear(128, 128)
        self.qkv = nn.Linear(128, 384)


class TinyAdaNormParent(nn.Module):
    def __init__(self, *, in_features: int = 128, out_features: int = 768, with_emb: bool = False):
        super().__init__()
        self.emb = nn.Linear(in_features, in_features) if with_emb else None
        self.silu = nn.Identity()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.Identity()


class TinyAdaNormModel(nn.Module):
    def __init__(self, *, splits: int = 6):
        super().__init__()
        self.norm = TinyAdaNormParent(out_features=128 * splits)


class FixedLinear(nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output

    def forward(self, x):
        return self.output.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1)


class MatchingFakeAdapter:
    target = "manifest_fake"

    def matches(self, transformer):
        return isinstance(transformer, TinyManifestModel)

    def patch(self, transformer, checkpoint_state, quantization_config, options):
        transformer.fake_adapter_used = True
        return checkpoint_state


def _manifest(*, op="svdq_w4a4", precision="int4", group_size=64, rank=4, patches=None, has_bias=True):
    return {
        "schema": "nunchaku_lite.runtime_manifest",
        "version": 1,
        "component": "transformer",
        "nunchaku_format_version": 1,
        "producer": {"name": "test", "version": "0"},
        "requirements": {
            "method": "svdquant",
            "precision": precision,
            "rank": rank,
            "weight_dtype": "fp4_e2m1_all" if precision == "fp4" else "int4",
            "activation_dtype": "int4",
            "torch_dtype": None,
        },
        "structural_patches": patches or [],
        "targets": [
            {
                "name": "proj",
                "checkpoint_prefix": "proj",
                "source_modules": ["proj"],
                "roles": [],
                "kind": "linear",
                "nunchaku_op": op,
                "precision": precision,
                "group_size": group_size,
                "rank": rank,
                "has_bias": has_bias,
                "op_options": {"adanorm_splits": 6} if op == "adanorm_awq_w4a16" else {},
                "activation": {},
            }
        ],
    }


def _quantization_config(manifest):
    return {"runtime_manifest": manifest}


def _write_manifest_checkpoint(tmp_path, model, manifest):
    quantization_config = _quantization_config(manifest)
    ManifestAdapter().patch(
        model,
        {},
        quantization_config,
        type(
            "Options",
            (),
            {
                "precision": parse_runtime_manifest(quantization_config).runtime_precision or "int4",
                "torch_dtype": torch.bfloat16,
                "device": None,
                "strict": True,
                "adapter_options": {},
            },
        )(),
    )
    checkpoint = tmp_path / "manifest.safetensors"
    save_file(
        model.state_dict(),
        checkpoint,
        metadata={"quantization_config": json.dumps(quantization_config)},
    )
    return checkpoint


def test_parse_runtime_manifest_rejects_unsupported_schema():
    quantization_config = _quantization_config({**_manifest(), "schema": "other"})

    with pytest.raises(ValueError, match="Unsupported runtime_manifest schema"):
        parse_runtime_manifest(quantization_config)


def test_parse_runtime_manifest_allows_missing_torch_dtype_requirement():
    manifest = _manifest(precision="fp4")
    manifest["requirements"].pop("torch_dtype")

    parsed = parse_runtime_manifest(_quantization_config(manifest))

    assert parsed is not None
    assert parsed.runtime_precision == "nvfp4"


def test_patch_transformer_manifest_target_replaces_svdq_linear(tmp_path):
    manifest = _manifest()
    checkpoint = _write_manifest_checkpoint(tmp_path, TinyManifestModel(), manifest)

    transformer = TinyManifestModel()
    patch_transformer(
        transformer,
        checkpoint,
        target="manifest",
        precision="auto",
        torch_dtype=torch.bfloat16,
        device="cpu",
    )

    assert transformer._nunchaku_lite_target == "manifest"
    assert isinstance(transformer.proj, SVDQW4A4Linear)
    assert transformer.proj.rank == 4
    assert transformer.proj.precision == "int4"
    assert getattr(transformer.proj, PATCHED_MODULE_ATTR)


def test_patch_transformer_manifest_accepts_legacy_smooth_factor_orig_checkpoint(tmp_path):
    manifest = _manifest()
    checkpoint = _write_manifest_checkpoint(tmp_path, TinyManifestModel(), manifest)

    checkpoint_state = load_file(checkpoint)
    checkpoint_state["proj.smooth_factor_orig"] = checkpoint_state["proj.smooth_factor"].clone()
    legacy_checkpoint = tmp_path / "manifest-legacy-smooth-orig.safetensors"
    save_file(
        checkpoint_state,
        legacy_checkpoint,
        metadata={"quantization_config": json.dumps(_quantization_config(manifest))},
    )

    transformer = TinyManifestModel()
    patch_transformer(
        transformer,
        legacy_checkpoint,
        target="manifest",
        precision="int4",
        torch_dtype=torch.bfloat16,
        device="cpu",
    )

    assert isinstance(transformer.proj, SVDQW4A4Linear)
    assert all("smooth_factor_orig" not in key for key in transformer.state_dict())


def test_patch_transformer_auto_uses_manifest_before_matching_adapter(tmp_path, monkeypatch):
    import nunchaku_lite.core as core

    manifest = _manifest()
    checkpoint = _write_manifest_checkpoint(tmp_path, TinyManifestModel(), manifest)
    monkeypatch.setitem(core._ADAPTERS, MatchingFakeAdapter.target, MatchingFakeAdapter())

    transformer = TinyManifestModel()
    patch_transformer(transformer, checkpoint, target="auto", torch_dtype=torch.bfloat16, device="cpu")

    assert transformer._nunchaku_lite_target == "manifest"
    assert not hasattr(transformer, "fake_adapter_used")


def test_patch_transformer_auto_falls_back_when_manifest_absent(tmp_path, monkeypatch):
    import nunchaku_lite.core as core

    checkpoint = tmp_path / "dense.safetensors"
    save_file(
        TinyManifestModel().state_dict(),
        checkpoint,
        metadata={"quantization_config": json.dumps({})},
    )
    monkeypatch.setitem(core._ADAPTERS, MatchingFakeAdapter.target, MatchingFakeAdapter())

    transformer = TinyManifestModel()
    patch_transformer(transformer, checkpoint, target="auto", torch_dtype=torch.bfloat16, device="cpu")

    assert transformer._nunchaku_lite_target == MatchingFakeAdapter.target
    assert transformer.fake_adapter_used


def test_patch_transformer_manifest_target_requires_manifest(tmp_path):
    checkpoint = tmp_path / "dense.safetensors"
    save_file(
        TinyManifestModel().state_dict(),
        checkpoint,
        metadata={"quantization_config": json.dumps({})},
    )

    with pytest.raises(ValueError, match="requires quantization_config.runtime_manifest"):
        patch_transformer(TinyManifestModel(), checkpoint, target="manifest", torch_dtype=torch.bfloat16, device="cpu")


def test_manifest_adapter_applies_split_linear_output_before_replacement(tmp_path):
    manifest = _manifest(
        patches=[{"type": "split_linear_output", "module": "proj", "args": {"splits": [64]}}],
    )
    source = TinyManifestModel(out_features=128)
    checkpoint = _write_manifest_checkpoint(tmp_path, source, manifest)
    assert isinstance(source.proj, SVDQW4A4Linear)

    transformer = TinyManifestModel(out_features=128)
    ManifestAdapter().patch(
        transformer,
        {},
        _quantization_config(manifest),
        type(
            "Options",
            (),
            {
                "precision": "int4",
                "torch_dtype": torch.bfloat16,
                "device": None,
                "strict": True,
                "adapter_options": {},
            },
        )(),
    )

    assert isinstance(transformer.proj, SVDQW4A4Linear)
    assert getattr(transformer.proj, PATCHED_MODULE_ATTR)
    patch_transformer(
        TinyManifestModel(out_features=128),
        checkpoint,
        target="manifest",
        torch_dtype=torch.bfloat16,
        device="cpu",
    )


def test_manifest_structural_patch_classes_preserve_linear_metadata():
    linear = nn.Linear(128, 128)

    split_input = SplitLinearInput.from_linear(linear, [64])
    split_output = SplitLinearOutput.from_linear(linear, [64])

    assert split_input.in_features == 128
    assert split_input.out_features == 128
    assert split_output.in_features == 128
    assert split_output.out_features == 128


def test_manifest_adapter_replaces_awq_target(tmp_path):
    manifest = _manifest(op="awq_w4a16", precision="int4", group_size=64, rank=0)
    checkpoint = _write_manifest_checkpoint(tmp_path, TinyManifestModel(), manifest)

    transformer = TinyManifestModel()
    patch_transformer(
        transformer,
        checkpoint,
        target="manifest",
        precision="int4",
        torch_dtype=torch.bfloat16,
        device="cpu",
    )

    assert isinstance(transformer.proj, AWQW4A16Linear)
    assert getattr(transformer.proj, PATCHED_MODULE_ATTR)


@pytest.mark.parametrize(
    ("input_shape", "chunk_sizes"),
    [
        ((8, 128), [8]),
        ((9, 128), [8, 1]),
        ((2, 9, 128), [8, 8, 2]),
        ((17, 1, 128), [8, 8, 1]),
    ],
)
def test_awq_w4a16_linear_chunks_manifest_gemv_inputs(monkeypatch, input_shape, chunk_sizes):
    import nunchaku_lite.linear as linear_module

    calls = []

    def fake_gemv(in_feats, kernel, scaling_factors, zeros, m, n, k, group_size):
        calls.append(
            {
                "input_shape": tuple(in_feats.shape),
                "m": m,
                "n": n,
                "k": k,
                "group_size": group_size,
            }
        )
        return torch.zeros(in_feats.shape[0], n, dtype=in_feats.dtype, device=in_feats.device)

    def fail_gemm(*args, **kwargs):
        raise AssertionError("Small manifest AWQ inputs should stay on GEMV")

    monkeypatch.setattr(linear_module, "awq_gemv_w4a16_cuda", fake_gemv)
    monkeypatch.setattr(linear_module, "awq_gemm_w4a16_g64_int32", fail_gemm)
    layer = AWQW4A16Linear(128, 16, bias=False, group_size=64, torch_dtype=torch.bfloat16)

    output = layer(torch.zeros(input_shape, dtype=torch.bfloat16))

    assert tuple(output.shape) == (*input_shape[:-1], 16)
    assert [call["m"] for call in calls] == chunk_sizes
    assert [call["input_shape"] for call in calls] == [(chunk_size, 128) for chunk_size in chunk_sizes]
    assert all(call["n"] == 16 and call["k"] == 128 and call["group_size"] == 64 for call in calls)


def test_awq_w4a16_linear_empty_input_skips_native_gemv(monkeypatch):
    import nunchaku_lite.linear as linear_module

    def fail_gemv(*args, **kwargs):
        raise AssertionError("Empty AWQ input should not call native GEMV")

    monkeypatch.setattr(linear_module, "awq_gemv_w4a16_cuda", fail_gemv)
    layer = AWQW4A16Linear(128, 16, bias=False, group_size=64, torch_dtype=torch.bfloat16)

    output = layer(torch.zeros(0, 2, 128, dtype=torch.bfloat16))

    assert tuple(output.shape) == (0, 2, 16)
    assert output.numel() == 0


def test_awq_w4a16_linear_dispatches_large_compatible_inputs_to_gemm(monkeypatch):
    import nunchaku_lite.linear as linear_module

    gemm_calls = []

    def fake_gemm(in_feats, kernel, scaling_factors, zeros):
        gemm_calls.append(
            {
                "input_shape": tuple(in_feats.shape),
                "kernel_dtype": kernel.dtype,
                "scales_shape": tuple(scaling_factors.shape),
                "zeros_shape": tuple(zeros.shape),
            }
        )
        return torch.zeros(in_feats.shape[0], 128, dtype=in_feats.dtype, device=in_feats.device)

    def fail_gemv(*args, **kwargs):
        raise AssertionError("Large compatible manifest AWQ inputs should use GEMM")

    monkeypatch.setattr(linear_module, "awq_gemm_w4a16_g64_int32", fake_gemm)
    monkeypatch.setattr(linear_module, "awq_gemv_w4a16_cuda", fail_gemv)
    layer = AWQW4A16Linear(128, 128, bias=False, group_size=64, torch_dtype=torch.bfloat16)

    output = layer(torch.zeros(2, 9, 128, dtype=torch.bfloat16))

    assert tuple(output.shape) == (2, 9, 128)
    assert gemm_calls == [
        {
            "input_shape": (18, 128),
            "kernel_dtype": torch.int32,
            "scales_shape": (2, 128),
            "zeros_shape": (2, 128),
        }
    ]


def test_awq_w4a16_linear_keeps_incompatible_large_inputs_on_gemv(monkeypatch):
    import nunchaku_lite.linear as linear_module

    calls = []

    def fake_gemv(in_feats, kernel, scaling_factors, zeros, m, n, k, group_size):
        calls.append(m)
        return torch.zeros(in_feats.shape[0], n, dtype=in_feats.dtype, device=in_feats.device)

    def fail_gemm(*args, **kwargs):
        raise AssertionError("Incompatible output features should not use GEMM")

    monkeypatch.setattr(linear_module, "awq_gemv_w4a16_cuda", fake_gemv)
    monkeypatch.setattr(linear_module, "awq_gemm_w4a16_g64_int32", fail_gemm)
    layer = AWQW4A16Linear(128, 16, bias=False, group_size=64, torch_dtype=torch.bfloat16)

    output = layer(torch.zeros(18, 128, dtype=torch.bfloat16))

    assert tuple(output.shape) == (18, 16)
    assert calls == [8, 8, 2]


def test_awq_w4a16_linear_bias_broadcasts_after_chunking(monkeypatch):
    import nunchaku_lite.linear as linear_module

    def fake_gemv(in_feats, kernel, scaling_factors, zeros, m, n, k, group_size):
        return torch.zeros(in_feats.shape[0], n, dtype=in_feats.dtype, device=in_feats.device)

    monkeypatch.setattr(linear_module, "awq_gemv_w4a16_cuda", fake_gemv)
    layer = AWQW4A16Linear(128, 16, bias=True, group_size=64, torch_dtype=torch.bfloat16)
    with torch.no_grad():
        layer.bias.copy_(torch.arange(16, dtype=torch.bfloat16))

    output = layer(torch.zeros(2, 9, 128, dtype=torch.bfloat16))

    expected = layer.bias.view(1, 1, -1).expand_as(output)
    assert torch.equal(output, expected)


def test_awq_w4a16_linear_rejects_wrong_input_features():
    layer = AWQW4A16Linear(128, 16, bias=False, group_size=64, torch_dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="expected input last dimension 128"):
        layer(torch.zeros(2, 127, dtype=torch.bfloat16))


@pytest.mark.parametrize("splits", [3, 6])
def test_manifest_adapter_wraps_adanorm_awq_target(splits):
    manifest = _manifest(op="adanorm_awq_w4a16", precision="int4", group_size=64, rank=0)
    manifest["targets"][0].update(
        {
            "name": "norm.linear",
            "checkpoint_prefix": "norm.linear",
            "source_modules": ["norm.linear"],
            "op_options": {"adanorm_splits": splits},
        }
    )
    transformer = TinyAdaNormModel(splits=splits)

    ManifestAdapter().patch(
        transformer,
        {},
        _quantization_config(manifest),
        type(
            "Options",
            (),
            {
                "precision": "int4",
                "torch_dtype": torch.bfloat16,
                "device": None,
                "strict": True,
                "adapter_options": {},
            },
        )(),
    )

    assert isinstance(transformer.norm, ManifestAdaNormAWQW4A16)
    assert transformer.norm.splits == splits
    assert getattr(transformer.norm, PATCHED_MODULE_ATTR)
    assert isinstance(transformer.norm.linear, AWQW4A16Linear)
    assert getattr(transformer.norm.linear, PATCHED_MODULE_ATTR)


def test_manifest_adanorm_wrapper_decodes_six_way_interleaved_output():
    parent = TinyAdaNormParent(in_features=4, out_features=24)
    output = torch.arange(24, dtype=torch.float32)
    wrapper = ManifestAdaNormAWQW4A16(parent, FixedLinear(output), splits=6)
    x = torch.ones(1, 2, 4)

    norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = wrapper(x, emb=torch.zeros(1, 4))
    decoded = output.view(1, -1, 6).permute(2, 0, 1)

    assert torch.equal(norm_x, x * decoded[1][:, None] + decoded[0][:, None])
    assert torch.equal(gate_msa, decoded[2])
    assert torch.equal(shift_mlp, decoded[3])
    assert torch.equal(scale_mlp, decoded[4] - 1)
    assert torch.equal(gate_mlp, decoded[5])


def test_manifest_adanorm_wrapper_decodes_three_way_interleaved_output():
    parent = TinyAdaNormParent(in_features=4, out_features=12)
    output = torch.arange(12, dtype=torch.float32)
    wrapper = ManifestAdaNormAWQW4A16(parent, FixedLinear(output), splits=3)
    x = torch.ones(1, 2, 4)

    norm_x, gate_msa = wrapper(x, emb=torch.zeros(1, 4))
    decoded = output.view(1, -1, 3).permute(2, 0, 1)

    assert torch.equal(norm_x, x * decoded[1][:, None] + decoded[0][:, None])
    assert torch.equal(gate_msa, decoded[2])


def test_parse_runtime_manifest_rejects_invalid_adanorm_splits():
    manifest = _manifest(op="adanorm_awq_w4a16", precision="int4", group_size=64, rank=0)
    manifest["targets"][0]["op_options"] = {"adanorm_splits": 4}

    with pytest.raises(ValueError, match="adanorm_splits to be 3 or 6"):
        parse_runtime_manifest(_quantization_config(manifest))


def test_manifest_adapter_rejects_adanorm_target_without_linear_child():
    manifest = _manifest(op="adanorm_awq_w4a16", precision="int4", group_size=64, rank=0)
    manifest["targets"][0].update(
        {
            "name": "norm.proj",
            "checkpoint_prefix": "norm.proj",
            "source_modules": ["norm.proj"],
            "op_options": {"adanorm_splits": 6},
        }
    )
    transformer = nn.Module()
    transformer.norm = nn.Module()
    transformer.norm.proj = nn.Linear(128, 768)

    with pytest.raises(ValueError, match="must reference a '.linear' child"):
        ManifestAdapter().patch(
            transformer,
            {},
            _quantization_config(manifest),
            type(
                "Options",
                (),
                {
                    "precision": "int4",
                    "torch_dtype": torch.bfloat16,
                    "device": None,
                    "strict": True,
                    "adapter_options": {},
                },
            )(),
        )


def test_manifest_lora_format_detection_routes_external_formats():
    assert (
        detect_manifest_lora_format(
            {
                "transformer.proj.lora_A.weight": torch.ones(1, 128),
                "transformer.proj.lora_B.weight": torch.ones(128, 1),
            }
        )
        == PEFT_FORMAT
    )
    assert (
        detect_manifest_lora_format(
            {
                "lora_transformer_proj.lora_down.weight": torch.ones(1, 128),
                "lora_transformer_proj.lora_up.weight": torch.ones(128, 1),
            }
        )
        == KOHYA_FORMAT
    )
    assert (
        detect_manifest_lora_format(
            {
                "diffusion_model.proj.lora_A.weight": torch.ones(1, 128),
                "diffusion_model.proj.lora_B.weight": torch.ones(128, 1),
            }
        )
        == COMFYUI_FORMAT
    )


def test_manifest_adapter_binds_generic_lora_runtime(tmp_path):
    manifest = _manifest()
    checkpoint = _write_manifest_checkpoint(tmp_path, TinyManifestModel(), manifest)

    transformer = TinyManifestModel()
    patch_transformer(transformer, checkpoint, target="manifest", torch_dtype=torch.bfloat16, device="cpu")

    assert transformer._nunchaku_lite_target == "manifest"
    assert callable(transformer.load_lora)
    assert callable(transformer.set_adapters)
    assert transformer._nunchaku_lite_loras == {}

    lora = {
        "transformer.proj.lora_A.weight": torch.ones(2, 128, dtype=torch.bfloat16),
        "transformer.proj.lora_B.weight": torch.ones(128, 2, dtype=torch.bfloat16),
    }
    transformer.load_lora(lora, name="style")

    assert transformer.get_list_adapters() == ["style"]
    assert transformer.get_active_adapters() == ["style"]


def test_manifest_generic_lora_accepts_kohya_suffix_alias(tmp_path):
    manifest = _manifest()
    checkpoint = _write_manifest_checkpoint(tmp_path, TinyManifestModel(), manifest)
    transformer = patch_transformer(
        TinyManifestModel(),
        checkpoint,
        target="manifest",
        torch_dtype=torch.bfloat16,
        device="cpu",
    )

    converted = transformer._convert_lora_to_nunchaku(
        {
            "lora_transformer_proj.lora_down.weight": torch.ones(2, 128, dtype=torch.bfloat16),
            "lora_transformer_proj.lora_up.weight": torch.ones(128, 2, dtype=torch.bfloat16),
        }
    )

    assert set(converted) == {"proj.proj_down", "proj.proj_up"}


def test_manifest_generic_lora_accepts_comfyui_component_prefix(tmp_path):
    manifest = _manifest()
    checkpoint = _write_manifest_checkpoint(tmp_path, TinyManifestModel(), manifest)
    transformer = patch_transformer(
        TinyManifestModel(),
        checkpoint,
        target="manifest",
        torch_dtype=torch.bfloat16,
        device="cpu",
    )

    converted = transformer._convert_lora_to_nunchaku(
        {
            "diffusion_model.proj.lora_A.weight": torch.ones(2, 128, dtype=torch.bfloat16),
            "diffusion_model.proj.lora_B.weight": torch.ones(128, 2, dtype=torch.bfloat16),
        }
    )

    assert set(converted) == {"proj.proj_down", "proj.proj_up"}


def test_manifest_generic_lora_fuses_source_module_branches(tmp_path):
    manifest = _manifest()
    manifest["targets"][0].update(
        {
            "name": "qkv",
            "checkpoint_prefix": "qkv",
            "source_modules": ["q", "k", "v"],
            "has_bias": True,
        }
    )
    checkpoint = _write_manifest_checkpoint(tmp_path, TinyFusedManifestModel(), manifest)
    transformer = patch_transformer(
        TinyFusedManifestModel(),
        checkpoint,
        target="manifest",
        torch_dtype=torch.bfloat16,
        device="cpu",
    )

    lora = {}
    for value, branch in enumerate(("q", "k", "v"), start=1):
        lora[f"transformer.{branch}.lora_A.weight"] = torch.ones(2, 128, dtype=torch.bfloat16)
        lora[f"transformer.{branch}.lora_B.weight"] = torch.full((128, 2), value, dtype=torch.bfloat16)

    converted = transformer._convert_lora_to_nunchaku(lora)
    logical_up = unpack_lowrank_weight(converted["qkv.proj_up"], down=False)[:384, :2]

    assert set(converted) == {"qkv.proj_down", "qkv.proj_up"}
    assert torch.equal(logical_up[:128], torch.ones_like(logical_up[:128]))
    assert torch.equal(logical_up[128:256], torch.full_like(logical_up[128:256], 2))
    assert torch.equal(logical_up[256:384], torch.full_like(logical_up[256:384], 3))


def test_manifest_pipeline_lora_methods_bind_when_component_runtime_is_bound(tmp_path):
    manifest = _manifest()
    checkpoint = _write_manifest_checkpoint(tmp_path, TinyManifestModel(), manifest)
    transformer = patch_transformer(
        TinyManifestModel(),
        checkpoint,
        target="manifest",
        torch_dtype=torch.bfloat16,
        device="cpu",
    )
    pipeline = SimpleNamespace(transformer=transformer)

    ManifestAdapter().patch_pipeline(pipeline, component_name="transformer", component=transformer)

    assert callable(pipeline.load_lora_weights)
    assert pipeline._nunchaku_lite_lora_component_name == "transformer"
