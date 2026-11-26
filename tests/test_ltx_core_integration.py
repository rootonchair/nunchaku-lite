"""Nunchaku Lite checkpoints in Lightricks' ltx-core, end to end on a tiny real ``LTXModel`` (CPU, no kernels).

Covers reading a native-layout file (Lightricks' transformer file with our SVDQ layers in place of the quantized
linears), ltx-core's own builder with our ``QuantizationPolicy`` (every target bound bit-exact, wtscale bound, nothing
left on meta, non-targets equal to the official file), partial quantization, the feed-forward and attention
replacements, and their fallbacks.
"""

from __future__ import annotations

import json
import re
import warnings

import pytest
import torch

ltx_core = pytest.importorskip("ltx_core")

from ltx_core.loader.simple_builder import SimpleBuilder
from ltx_core.model.transformer.attention import Attention
from ltx_core.model.transformer.model_configurator import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
    transformer_checkpoint_selection_ops,
)
from ltx_core.quantization.policy import compose_quantization
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from svdq_files import WTSCALE, quant_config, svdq_tensors, write_checkpoint

from nunchaku_lite.integrations import ltx_core as integration
from nunchaku_lite.integrations.ltx_core import build_nunchaku_policy
from nunchaku_lite.linear import SVDQW4A4Linear
from nunchaku_lite.loader import NATIVE_PREFIX, read_checkpoint_spec
from nunchaku_lite.manifest import ROTATION_MARKER_KEY
from nunchaku_lite.ops.fused import FUSED_MLP_FC2_ATTR

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


def _native(tmp_path, precision="int4", rotate=(), keep=None):
    """A native-layout file: the official file with each quantized layer's weight/bias replaced by its SVDQ tensors,
    the official metadata plus ``quantization_config``, and the rotation marker when any layer is rotated."""
    official, model = _official(tmp_path)
    targets = {n: m for n, m in sorted(_targets(model).items()) if keep is None or keep(n)}
    layers = {
        name: svdq_tensors(m.in_features, m.out_features, precision=precision, bias=m.bias is not None, seed=i)
        for i, (name, m) in enumerate(targets.items())
    }
    rotated = [n for n in layers if any(n.endswith(r) for r in rotate)]
    tensors = load_file(str(official))
    for name in layers:
        for p in ("weight", "bias"):
            tensors.pop(f"{NATIVE_PREFIX}{name}.{p}", None)
    tensors.update({f"{NATIVE_PREFIX}{n}.{k}": v for n, ts in layers.items() for k, v in ts.items()})
    if rotated:
        tensors[ROTATION_MARKER_KEY] = torch.empty(0)
    with safe_open(str(official), framework="pt") as f:
        metadata = dict(f.metadata())
    metadata["quantization_config"] = json.dumps(quant_config(list(layers), precision=precision, rotated=rotated))
    out = tmp_path / "native.safetensors"
    save_file(tensors, str(out), metadata=metadata)
    return official, out, layers


def _build(path, policy):
    """As ltx-pipelines' DiffusionStage: compose the policy, then add the connector selection stage."""
    builder = compose_quantization(SimpleBuilder(LTXModelConfigurator, (path,), (LTXV_MODEL_COMFY_RENAMING_MAP,)), policy)
    builder = builder.with_sd_ops((*builder.sd_ops, transformer_checkpoint_selection_ops(TINY)))
    return builder.build(device=torch.device("cpu"), dtype=torch.bfloat16)


# --- native layout --------------------------------------------------------------------------------


@pytest.mark.parametrize("precision,rotate", [("nvfp4", ()), ("int4", ()), ("int4", ("",))])
def test_native_file_is_read_as_native(tmp_path, precision, rotate):
    _, out, layers = _native(tmp_path, precision, rotate)
    spec = read_checkpoint_spec(out)
    assert spec.layout == "native" and set(spec.layers) == set(layers)
    assert len(spec.layers) == 2 * 28  # per block: 6 attentions x (q, k, v, out) + 2 FFs x 2; gate logits excluded
    assert all((la.input_rotation is not None) == bool(rotate) for la in spec.layers.values())


# Lightricks' nvfp4 layer set (ltx_core.quantization.nvfp4.convert): video-side linears only.
LIGHTRICKS_SET = re.compile(
    r"transformer_blocks\.\d+\.(attn1\.to_(q|k|v|out\.0)|attn2\.to_(q|out\.0)|audio_to_video_attn\.to_(q|out\.0)"
    r"|video_to_audio_attn\.to_(k|v)|ff\.net\.(0\.proj|2))"
)


@pytest.mark.parametrize("rotate", [(), ("",), ("audio_ff.net.2",)])
def test_partially_quantized_file(tmp_path, rotate):
    """Only the listed layers are quantized; every other linear keeps its official bf16 weights."""
    official, out, layers = _native(tmp_path, "int4", rotate, keep=LIGHTRICKS_SET.fullmatch)
    assert len(layers) == 2 * 12
    reference = {k.removeprefix(NATIVE_PREFIX): v for k, v in load_file(str(official)).items()}
    candidates = set(_targets(LTXModelConfigurator.from_metadata({"config": TINY})))
    built = _build(out, build_nunchaku_policy(out, device="cpu", fused_attention=False))
    for name, module in built.named_modules():
        if name in candidates:
            assert isinstance(module, SVDQW4A4Linear) == (name in layers), name
            if name not in layers:
                assert torch.equal(module.weight, reference[f"{name}.weight"].to(module.weight.dtype)), name


# --- ltx-core build --------------------------------------------------------------------------------


@pytest.mark.parametrize("precision,rotate", [("nvfp4", ()), ("int4", ()), ("int4", ("",)), ("int4", ("ff.net.2",))])
def test_ltx_core_builder_binds_everything(tmp_path, precision, rotate):
    official, out, layers = _native(tmp_path, precision, rotate)
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
        _, out, _ = _native(sub, precision)
        model = _build(out, build_nunchaku_policy(out, device="cpu"))
        ff = model.transformer_blocks[0].ff
        fused = precision == "nvfp4"
        assert getattr(ff.net[2], FUSED_MLP_FC2_ATTR, False) is fused
        assert ("forward" in ff.__dict__) is fused


def test_rotated_nvfp4_fc2_runs_unfused(tmp_path):
    _, out, _ = _native(tmp_path, "nvfp4", ("ff.net.2",))
    with pytest.raises(ValueError, match="NVFP4 is refused"):
        build_nunchaku_policy(out, device="cpu")
    built = _build(out, build_nunchaku_policy(out, device="cpu", allow_fp4_rotation=True))
    ff = built.transformer_blocks[0].ff
    assert ff.net[2].input_rotation == "hadamard" and ff.net[0].proj.input_rotation is None
    assert not getattr(ff.net[2], FUSED_MLP_FC2_ATTR, False) and "forward" not in ff.__dict__


def test_attention_forward_patched_only_on_quantized_attention(tmp_path):
    _, out, _ = _native(tmp_path, "int4")
    model = _build(out, build_nunchaku_policy(out, device="cpu"))
    patched = [n for n, m in model.named_modules() if isinstance(m, Attention) and "forward" in m.__dict__]
    assert len(patched) == 2 * 6
    model = _build(out, build_nunchaku_policy(out, device="cpu", fused_attention=False))
    assert not [n for n, m in model.named_modules() if isinstance(m, Attention) and "forward" in m.__dict__]


def test_attention_left_stock_when_unquantized(tmp_path):
    _, out, _ = _native(tmp_path, keep=lambda n: ".ff." in n or ".audio_ff." in n)
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
    _, out_a, _ = _native(a, "int4")
    _, out_b, _ = _native(b, "int4", ("",))
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
    _, out, _ = _native(tmp_path, "nvfp4")
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

    _, out, _ = _native(tmp_path, "int4")
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
