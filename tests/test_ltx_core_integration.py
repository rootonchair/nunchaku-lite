"""Nunchaku Lite checkpoints in Lightricks' ltx-core, end to end on a tiny real ``LTXModel`` (CPU, no kernels).

Covers the native splice (byte-identity), ltx-core's own builder with our ``QuantizationPolicy`` (every target bound
bit-exact, wtscale bound, nothing left on meta, non-targets equal to the official file), the feed-forward and
attention replacements, and their fallbacks.
"""

from __future__ import annotations

import json
import re
import warnings

import pytest
import torch

ltx_core = pytest.importorskip("ltx_core")

from ltx_core.loader.simple_builder import SimpleBuilder  # noqa: E402
from ltx_core.model.transformer.attention import Attention  # noqa: E402
from ltx_core.model.transformer.model_configurator import (  # noqa: E402
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
    transformer_checkpoint_selection_ops,
)
from ltx_core.quantization.policy import compose_quantization  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from nunchaku_lite.integrations import ltx_core as integration  # noqa: E402
from nunchaku_lite.integrations.ltx_core import build_nunchaku_policy  # noqa: E402
from nunchaku_lite.linear import SVDQW4A4Linear  # noqa: E402
from nunchaku_lite.loader import NATIVE_PREFIX, read_checkpoint_spec  # noqa: E402
from nunchaku_lite.native_export import _select_targets, splice_native, verify_native_splice  # noqa: E402
from nunchaku_lite.ops.fused import FUSED_MLP_FC2_ATTR  # noqa: E402
from svdq_files import WTSCALE, quant_config, svdq_tensors, write_checkpoint  # noqa: E402

TINY = {
    "transformer": {
        "dropout": 0.0, "attention_bias": True, "num_vector_embeds": None, "activation_fn": "gelu-approximate",
        "num_embeds_ada_norm": 1000, "use_linear_projection": False, "only_cross_attention": False,
        "cross_attention_norm": True, "double_self_attention": False, "upcast_attention": False,
        "standardization_norm": "rms_norm", "norm_elementwise_affine": False, "qk_norm": "rms_norm",
        "positional_embedding_type": "rope", "use_audio_video_cross_attention": True, "share_ff": False,
        "av_cross_ada_norm": True, "use_middle_indices_grid": True,
        "num_attention_heads": 2, "attention_head_dim": 64, "audio_num_attention_heads": 2,
        "audio_attention_head_dim": 64, "num_layers": 2, "cross_attention_dim": 128, "audio_cross_attention_dim": 128,
        "in_channels": 128, "out_channels": 128, "audio_in_channels": 128, "audio_out_channels": 128,
        "apply_gated_attention": True, "cross_attention_adaln": True, "ff_bias": False, "audio_ff_bias": True,
        "use_keyframes_abs_pos_embedding": True, "caption_proj_before_connector": True,
        "use_embeddings_connector": True,
    }
}
CONNECTOR_KEY = f"{NATIVE_PREFIX}video_embeddings_connector.learnable_registers"


def _official(tmp_path):
    torch.manual_seed(0)
    model = LTXModelConfigurator.from_metadata({"config": TINY})
    for p in model.parameters():  # some LTX parameters initialize to zeros; make every tensor distinctive
        p.data.normal_()
    tensors = {f"{NATIVE_PREFIX}{k}": v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    tensors[CONNECTOR_KEY] = torch.randn(4, 128).to(torch.bfloat16)
    path = tmp_path / "official.safetensors"
    save_file(tensors, str(path), metadata={"config": json.dumps(TINY), "model_version": "tiny"})
    return path, model


def _targets(model):
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith("transformer_blocks.")
        and module.in_features % 128 == 0
        and module.out_features % 128 == 0
    }


def _quantized(tmp_path, model, precision, rotate=()):
    targets = _targets(model)
    layers = {
        name: svdq_tensors(m.in_features, m.out_features, precision=precision, bias=m.bias is not None, seed=i)
        for i, (name, m) in enumerate(sorted(targets.items()))
    }
    rotated = [n for n in targets if any(n.endswith(s) for s in rotate)]
    path = tmp_path / "quantized.safetensors"
    return write_checkpoint(path, layers, config=quant_config(list(layers), precision=precision, rotated=rotated),
                            marker=bool(rotated)), layers


def _spliced(tmp_path, precision, rotate=()):
    official, model = _official(tmp_path)
    quantized, layers = _quantized(tmp_path, model, precision, rotate)
    out = tmp_path / "native.safetensors"
    splice_native(official, quantized, out)
    return official, quantized, out, layers


def _build(path, policy):
    """As ltx-pipelines' DiffusionStage: compose the policy, then add the connector selection stage."""
    builder = compose_quantization(SimpleBuilder(LTXModelConfigurator, (path,), (LTXV_MODEL_COMFY_RENAMING_MAP,)), policy)
    builder = builder.with_sd_ops((*builder.sd_ops, transformer_checkpoint_selection_ops(TINY)))
    return builder.build(device=torch.device("cpu"), dtype=torch.bfloat16)


# --- splice ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("precision,rotate", [("nvfp4", ()), ("int4", ()), ("int4", ("",))])
def test_splice_is_byte_exact(tmp_path, precision, rotate):
    official, quantized, out, layers = _spliced(tmp_path, precision, rotate)
    verify_native_splice(official, out, quantized)
    spec = read_checkpoint_spec(out)
    assert spec.layout == "native" and set(spec.layers) == set(layers)
    assert len(spec.layers) == 2 * 28  # per block: 6 attentions x (q, k, v, out) + 2 FFs x 2; gate logits excluded
    with safe_open(str(out), framework="pt") as f:
        assert json.loads(f.metadata()["config"]) == TINY
        assert CONNECTOR_KEY in f.keys()
        assert not any(k.endswith(".weight") and k.removeprefix(NATIVE_PREFIX).rpartition(".")[0] in layers for k in f.keys())


def test_splice_refuses_wrong_model(tmp_path):
    official, model = _official(tmp_path)
    quantized, _ = _quantized(tmp_path, model, "int4")
    tensors = load_file(str(official))
    key = f"{NATIVE_PREFIX}transformer_blocks.0.attn1.to_q.weight"
    tensors[key] = torch.zeros(256, 128, dtype=torch.bfloat16)
    save_file(tensors, str(official), metadata={"config": json.dumps(TINY)})
    with pytest.raises(ValueError, match="not the same model"):
        splice_native(official, quantized, tmp_path / "out.safetensors")


def test_splice_refuses_quantized_official(tmp_path):
    official, quantized, out, _ = _spliced(tmp_path, "int4")
    with pytest.raises(ValueError, match="already quantized"):
        splice_native(out, quantized, tmp_path / "again.safetensors")


# Lightricks' nvfp4 layer set (ltx_core.quantization.nvfp4.convert): video-side linears only.
LIGHTRICKS_SET = re.compile(
    r"transformer_blocks\.\d+\.(attn1\.to_(q|k|v|out\.0)|attn2\.to_(q|out\.0)|audio_to_video_attn\.to_(q|out\.0)"
    r"|video_to_audio_attn\.to_(k|v)|ff\.net\.(0\.proj|2))"
)


@pytest.mark.parametrize("rotate", [(), ("",), ("audio_ff.net.2",)])
def test_subset_splice(tmp_path, rotate):
    official, model = _official(tmp_path)
    quantized, layers = _quantized(tmp_path, model, "int4", rotate)
    subset = sorted(n for n in layers if LIGHTRICKS_SET.fullmatch(n))
    assert len(subset) == 2 * 12
    out = tmp_path / "subset.safetensors"
    report = splice_native(official, quantized, out, targets=subset)
    assert report["layers"] == len(subset)
    verify_native_splice(official, out, quantized, targets=subset)
    with pytest.raises(AssertionError, match="targets"):
        verify_native_splice(official, out, quantized)  # not every layer: the default check must fail
    with pytest.raises(AssertionError, match="targets"):
        verify_native_splice(official, out, quantized, targets=subset[:-1])  # one more spliced than requested
    spec = read_checkpoint_spec(out)
    assert sorted(spec.layers) == subset
    rotated_in_subset = sorted(n for n in subset if any(n.endswith(r) for r in rotate))
    assert sorted(n for n, la in spec.layers.items() if la.input_rotation is not None) == rotated_in_subset
    off, got = load_file(str(official)), load_file(str(out))
    with safe_open(str(out), framework="pt") as f:
        config = json.loads(f.metadata()["quantization_config"])
    assert config["svdq_w4a4"]["targets"] == subset
    assert config.get("input_rotation", {}).get("targets", []) == rotated_in_subset
    assert ("nunchaku_lite.input_rotation" in got) == bool(rotated_in_subset)
    for name in set(layers) - set(subset):  # left out: official bf16 tensors, byte for byte
        for p in ("weight", "bias"):
            key = f"{NATIVE_PREFIX}{name}.{p}"
            assert (key in got) == (key in off) and (key not in off or torch.equal(got[key], off[key]))
        assert f"{NATIVE_PREFIX}{name}.qweight" not in got
    built = _build(out, build_nunchaku_policy(out, fused_attention=False))
    for name, module in built.named_modules():
        if name in layers:
            assert isinstance(module, SVDQW4A4Linear) == (name in subset), name


def test_subset_splice_refusals(tmp_path):
    official, model = _official(tmp_path)
    quantized, layers = _quantized(tmp_path, model, "int4")
    out = tmp_path / "subset.safetensors"
    with pytest.raises(ValueError, match="empty"):
        splice_native(official, quantized, out, targets=[])
    with pytest.raises(ValueError, match="not quantized layers"):
        splice_native(official, quantized, out, targets=[*sorted(layers)[:2], "transformer_blocks.0.nope"])
    with safe_open(str(quantized), framework="pt") as f:
        config = json.loads(f.metadata()["quantization_config"])
    with pytest.raises(ValueError, match="target-list"):  # runtime manifests are not rewritten
        _select_targets({**config, "runtime_manifest": {}}, read_checkpoint_spec(quantized), sorted(layers)[:1])
    assert not out.exists()


def test_subset_cli(tmp_path, monkeypatch, capsys):
    import importlib.util
    import pathlib

    official, model = _official(tmp_path)
    quantized, layers = _quantized(tmp_path, model, "nvfp4")
    out = tmp_path / "cli.safetensors"
    tool = pathlib.Path(__file__).resolve().parents[1] / "tools" / "ltx2_native_splice.py"
    mod_spec = importlib.util.spec_from_file_location("splice_tool", tool)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    monkeypatch.setattr("sys.argv", ["x", "--official", str(official), "--quantized", str(quantized), "--out", str(out),
                                     "--targets-regex", LIGHTRICKS_SET.pattern])
    mod.main()
    report = json.loads(capsys.readouterr().out)
    assert report == {**report, "layers": 24, "verified": True}
    assert sorted(read_checkpoint_spec(out).layers) == sorted(n for n in layers if LIGHTRICKS_SET.fullmatch(n))


def test_verify_catches_tampering(tmp_path):
    official, quantized, out, _ = _spliced(tmp_path, "int4")
    with safe_open(str(out), framework="pt") as f:
        meta = f.metadata()
    tensors = load_file(str(out))
    tensors[CONNECTOR_KEY] = tensors[CONNECTOR_KEY] + 1
    save_file(tensors, str(out), metadata=meta)
    with pytest.raises(AssertionError, match="differs from the official tensor"):
        verify_native_splice(official, out, quantized)


# --- ltx-core build --------------------------------------------------------------------------------


@pytest.mark.parametrize("precision,rotate", [("nvfp4", ()), ("int4", ()), ("int4", ("",)), ("int4", ("ff.net.2",))])
def test_ltx_core_builder_binds_everything(tmp_path, precision, rotate):
    official, _, out, layers = _spliced(tmp_path, precision, rotate)
    policy = build_nunchaku_policy(out, device="cpu")
    model = _build(out, policy)
    spec = read_checkpoint_spec(out)

    reference = {k.removeprefix(NATIVE_PREFIX): v for k, v in load_file(str(official)).items()}
    state = model.state_dict()
    assert not [k for k, v in state.items() if v.is_meta]
    for name, tensors in layers.items():
        module = model.get_submodule(name)
        assert isinstance(module, SVDQW4A4Linear) and module.precision == precision
        assert module.input_rotation == spec.layers[name].input_rotation
        assert module.act_unsigned is False
        for suffix, value in tensors.items():
            got = getattr(module, suffix)
            if suffix == "wtscale":
                assert got.item() == pytest.approx(WTSCALE, rel=1e-2) and got.item() == value.float().item()
            elif value.dtype in (torch.int8, torch.float8_e4m3fn):
                assert torch.equal(got.view(torch.uint8), value.view(torch.uint8))
            else:
                assert torch.equal(got, value)
    quantized_keys = {f"{n}.{s}" for n, t in layers.items() for s in t}
    for key, value in state.items():
        if key in quantized_keys:
            continue
        assert key in reference, key
        assert torch.equal(value, reference[key].to(value.dtype)), key


def test_feed_forward_plan_applied(tmp_path):
    for precision in ("nvfp4", "int4"):
        sub = tmp_path / precision
        sub.mkdir()
        _, _, out, _ = _spliced(sub, precision)
        model = _build(out, build_nunchaku_policy(out, device="cpu"))
        ff = model.transformer_blocks[0].ff
        fused = precision == "nvfp4"
        assert getattr(ff.net[2], FUSED_MLP_FC2_ATTR, False) is fused
        assert ("forward" in ff.__dict__) is fused


def test_rotated_nvfp4_fc2_runs_unfused(tmp_path):
    official, model = _official(tmp_path)
    quantized, _ = _quantized(tmp_path, model, "nvfp4", ("ff.net.2",))
    out = tmp_path / "native.safetensors"
    with pytest.raises(ValueError, match="NVFP4 is refused"):
        splice_native(official, quantized, out)
    splice_native(official, quantized, out, allow_fp4_rotation=True)
    with pytest.raises(ValueError, match="NVFP4 is refused"):
        build_nunchaku_policy(out, device="cpu")
    built = _build(out, build_nunchaku_policy(out, device="cpu", allow_fp4_rotation=True))
    ff = built.transformer_blocks[0].ff
    assert ff.net[2].input_rotation == "hadamard" and ff.net[0].proj.input_rotation is None
    assert not getattr(ff.net[2], FUSED_MLP_FC2_ATTR, False) and "forward" not in ff.__dict__


def test_attention_forward_patched_only_on_quantized_attention(tmp_path):
    _, _, out, _ = _spliced(tmp_path, "int4")
    model = _build(out, build_nunchaku_policy(out, device="cpu"))
    patched = [n for n, m in model.named_modules() if isinstance(m, Attention) and "forward" in m.__dict__]
    assert len(patched) == 2 * 6
    model = _build(out, build_nunchaku_policy(out, device="cpu", fused_attention=False))
    assert not [n for n, m in model.named_modules() if isinstance(m, Attention) and "forward" in m.__dict__]


def test_attention_left_stock_when_unquantized(tmp_path):
    official, model = _official(tmp_path)
    ff_only = {n: m for n, m in _targets(model).items() if ".ff." in n or ".audio_ff." in n}
    layers = {n: svdq_tensors(m.in_features, m.out_features, bias=m.bias is not None, seed=i) for i, (n, m) in enumerate(ff_only.items())}
    quantized = write_checkpoint(tmp_path / "q.safetensors", layers, config=quant_config(list(layers)))
    out = tmp_path / "native.safetensors"
    splice_native(official, quantized, out)
    built = _build(out, build_nunchaku_policy(out, device="cpu"))
    assert not [m for m in built.modules() if isinstance(m, Attention) and "forward" in m.__dict__]
    assert isinstance(built.transformer_blocks[0].attn1.to_q, torch.nn.Linear)


def test_policy_refuses_diffusers_layout(tmp_path):
    _, model = _official(tmp_path)
    quantized, _ = _quantized(tmp_path, model, "int4")
    with pytest.raises(ValueError, match="Diffusers-layout"):
        build_nunchaku_policy(quantized, device="cpu")


def test_policy_names_differ_by_checkpoint_and_options(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(), b.mkdir()
    _, _, out_a, _ = _spliced(a, "int4")
    _, _, out_b, _ = _spliced(b, "int4", ("",))
    names = {
        build_nunchaku_policy(out_a, device="cpu").module_ops[0].name,
        build_nunchaku_policy(out_b, device="cpu").module_ops[0].name,
        build_nunchaku_policy(out_a, device="cpu", fused_attention=False).module_ops[0].name,
    }
    assert len(names) == 3


# --- attention replacement is the stock forward when the kernel does not apply -----------------------


def _attention(gated=True):
    torch.manual_seed(1)
    attn = Attention(query_dim=128, context_dim=128, heads=2, dim_head=64, apply_gated_attention=gated)
    for p in attn.parameters():
        p.data.normal_(0, 0.2)
    return attn


def _rope(batch, seq):
    cos = torch.randn(batch, 2, seq, 32)
    return cos, torch.randn(batch, 2, seq, 32)


@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"pe": "self"},
        {"pe": "self", "k_pe": "ctx", "context": True},
        {"context": True, "mask": True},
        {"pe": "self", "perturbation_mask": True},
        {"pe": "self", "all_perturbed": True},
    ],
)
def test_attention_replacement_matches_stock_on_cpu(gated, kwargs):
    import types

    attn = _attention(gated)
    x = torch.randn(1, 6, 128)
    ctx = torch.randn(1, 5, 128)
    args = {}
    if kwargs.get("context"):
        args["context"] = ctx
    if kwargs.get("pe"):
        args["pe"] = _rope(1, 6)
    if kwargs.get("k_pe"):
        args["k_pe"] = _rope(1, 5)
    if kwargs.get("mask"):
        args["mask"] = torch.zeros(1, 1, 6, 5)
    if kwargs.get("perturbation_mask"):
        args["perturbation_mask"] = torch.rand(1, 6, 1)
    if kwargs.get("all_perturbed"):
        args["all_perturbed"] = True
    expected = attn(x, **args)
    attn.forward = types.MethodType(integration._attention_forward, attn)
    assert torch.equal(attn(x, **args), expected)


def test_fused_qk_declines_unsupported_inputs():
    attn = _attention()
    q = torch.randn(1, 6, 128, dtype=torch.bfloat16)
    assert integration._fused_qk(attn, q, q, _rope(1, 6), None) is None  # CPU
    assert integration._fused_qk(attn, q, q, None, None) is None  # no RoPE


def test_upstream_drift_falls_back_to_stock(tmp_path, monkeypatch):
    _, _, out, _ = _spliced(tmp_path, "nvfp4")
    monkeypatch.setattr(integration, "_ATTENTION_FORWARD_SHA", "0" * 16)
    monkeypatch.setattr(integration, "_FEED_FORWARD_FORWARD_SHA", "0" * 16)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _build(out, build_nunchaku_policy(out, device="cpu"))
    assert any("changed upstream" in str(w.message) for w in caught)
    assert not [m for m in model.modules() if "forward" in m.__dict__]


def test_lora_rule_refuses_quantized_layers():
    rule = integration._refuse_lora_rule()
    with pytest.raises(NotImplementedError, match="W4A4"):
        rule.patch("blocks.0.attn1.to_q.qweight", torch.zeros(2, dtype=torch.int8), torch.zeros(2), {})
    with pytest.raises(NotImplementedError, match="W4A4"):  # NVFP4 scales are floating point (fp8)
        rule.patch("blocks.0.attn1.to_q.wscales", torch.zeros(2, dtype=torch.float8_e4m3fn), torch.zeros(2), {})
    out = rule.patch("x.weight", torch.zeros(2, dtype=torch.bfloat16), torch.ones(2), {})
    assert torch.equal(out["x.weight"], torch.ones(2, dtype=torch.bfloat16))


# --- CLI -------------------------------------------------------------------------------------------


def test_cli_injects_policy_into_pipeline_args(tmp_path, monkeypatch):
    import argparse
    import sys
    import types

    _, _, out, _ = _spliced(tmp_path, "int4")
    seen = {}
    args_module = types.ModuleType("ltx_pipelines.utils.args")
    args_module._resolve_quantization = lambda ns: None
    pipeline = types.ModuleType("ltx_pipelines.fake")

    def fake_main():
        ns = argparse.Namespace(transformer_path=str(out), quantization=None)
        args_module._resolve_quantization(ns)  # what the pipeline's parser does after parse_args
        seen["argv"], seen["policy"] = list(sys.argv[1:]), ns.quantization

    pipeline.main = fake_main
    for name, module in {"ltx_pipelines": types.ModuleType("ltx_pipelines"), "ltx_pipelines.utils": types.ModuleType("ltx_pipelines.utils"),
                         "ltx_pipelines.utils.args": args_module, "ltx_pipelines.fake": pipeline}.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    integration.main(["fake", "--prompt", "x", "--nunchaku-stock-attention"])
    assert seen["argv"] == ["--prompt", "x"]
    assert seen["policy"].module_ops[0].name.endswith("attn=stock:swap")
    assert args_module._resolve_quantization.__name__ == "<lambda>"  # restored
    with pytest.raises(SystemExit, match="--quantization"):
        integration.main(["fake", "--quantization", "fp8-cast"])
