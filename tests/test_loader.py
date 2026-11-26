"""Decision table and invariants of the framework-agnostic loader (``nunchaku_lite.loader``)."""

from __future__ import annotations

import itertools
import json

import pytest
import torch
from svdq_files import WTSCALE, quant_config, svdq_tensors, write_checkpoint
from torch import nn

from nunchaku_lite.linear import SVDQW4A4Linear, wtscale_alpha
from nunchaku_lite.loader import (
    make_linear,
    read_checkpoint_spec,
    replace_linears,
)
from nunchaku_lite.ltx2_conventions import ltx2_gelu_mlp_plan
from nunchaku_lite.manifest import ROTATION_MARKER_KEY

A, B = "blocks.0.attn.to_q", "blocks.0.ff.net.2"
PREFIX = "unet."  # any caller-chosen prefix; the loader knows none
SHAPES = {A: (128, 256), B: (256, 128)}  # name -> (in, out)


def _layers(precision="int4", **kw):
    return {name: svdq_tensors(i, o, precision=precision, seed=n, **kw) for n, (name, (i, o)) in enumerate(SHAPES.items())}


def _write(tmp_path, precision="int4", *, prefix="", rotated=None, marker=None, layers=None, config=None, **kw):
    layers = layers if layers is not None else _layers(precision)
    config = config if config is not None else quant_config(list(layers), precision=precision, rotated=rotated)
    marker = bool(rotated) if marker is None else marker
    return write_checkpoint(tmp_path / "ckpt.safetensors", layers, config=config, prefix=prefix, marker=marker, **kw)


class _Model(nn.Module):
    def __init__(self, device="cpu", linear=nn.Linear):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Module()])
        self.blocks[0].attn = nn.Module()
        self.blocks[0].attn.to_q = linear(128, 256, device=device, dtype=torch.bfloat16)
        self.blocks[0].ff = nn.Module()
        self.blocks[0].ff.net = nn.ModuleList([nn.Identity(), nn.Identity(), linear(256, 128, device=device, dtype=torch.bfloat16)])
        self.norm = nn.LayerNorm(128, device=device)


# --- layout / metadata -----------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", ["", PREFIX])
@pytest.mark.parametrize("precision", ["int4", "nvfp4"])
def test_spec_layers(tmp_path, prefix, precision):
    spec = read_checkpoint_spec(_write(tmp_path, precision, prefix=prefix), key_prefix=prefix)
    assert spec.key_prefix == prefix
    assert spec.precision == precision and spec.rank == 16
    assert set(spec.layers) == set(SHAPES)
    for name, (i, o) in SHAPES.items():
        layer = spec.layers[name]
        assert (layer.in_features, layer.out_features, layer.rank) == (i, o, 16)
        assert layer.group_size == SVDQW4A4Linear(i, o, rank=16, precision=precision, device="meta").group_size
        assert layer.has_bias and layer.input_rotation is None
        assert layer.has_wtscale == (precision == "nvfp4")


def test_prefix_without_tensors_refused(tmp_path):
    with pytest.raises(ValueError, match="no tensors under the key prefix"):
        read_checkpoint_spec(_write(tmp_path), key_prefix=PREFIX)


def test_prefix_must_end_with_a_dot(tmp_path):
    with pytest.raises(ValueError, match="must end with"):
        read_checkpoint_spec(_write(tmp_path, prefix=PREFIX), key_prefix=PREFIX.rstrip("."))


def test_unsupported_dtypes_outside_prefix_ignored(tmp_path):
    from safetensors.torch import load_file, save_file

    path = _write(tmp_path, prefix=PREFIX)
    tensors = load_file(str(path))
    tensors["vae.index"] = torch.zeros(4, dtype=torch.uint32)  # a dtype the loader has no use for
    save_file(tensors, str(path), metadata={"quantization_config": json.dumps(quant_config(list(SHAPES)))})
    assert set(read_checkpoint_spec(path, key_prefix=PREFIX).layers) == set(SHAPES)


def test_prefixed_file_reads_sidecar_config(tmp_path):
    path = _write(tmp_path, prefix=PREFIX, sidecar=True)
    assert set(read_checkpoint_spec(path, key_prefix=PREFIX).layers) == set(SHAPES)


def test_diffusers_layout_reads_sidecar_config(tmp_path):
    assert set(read_checkpoint_spec(_write(tmp_path, sidecar=True)).layers) == set(SHAPES)


def test_no_config_refused(tmp_path):
    path = write_checkpoint(tmp_path / "c.safetensors", _layers(), config=None)
    with pytest.raises(ValueError, match="no 'quantization_config'"):
        read_checkpoint_spec(path)


def test_tensors_outside_prefix_ignored(tmp_path):
    from safetensors.torch import load_file, save_file

    path = _write(tmp_path, prefix=PREFIX)
    tensors = load_file(str(path))
    tensors["vae.decoder.weight"] = torch.zeros(2)  # a single-file checkpoint may carry other components
    save_file(tensors, str(path), metadata={"quantization_config": json.dumps(quant_config(list(SHAPES)))})
    assert set(read_checkpoint_spec(path, key_prefix=PREFIX).layers) == set(SHAPES)


# --- per-layer tensors -----------------------------------------------------------------------------


@pytest.mark.parametrize("precision", ["int4", "nvfp4"])
@pytest.mark.parametrize("suffix", ["qweight", "wscales", "smooth_factor", "proj_down", "proj_up", "bias"])  # wcscales: defaulted
def test_missing_tensor_refused(tmp_path, precision, suffix):
    layers = _layers(precision)
    if suffix not in layers[A]:
        pytest.skip(f"{precision} has no {suffix}")
    config = quant_config(list(layers), precision=precision)
    del layers[A][suffix]
    if suffix == "bias":  # target lists derive has_bias from the keys: declare it through a manifest instead
        config = _manifest_config(layers, precision, has_bias=True)
    with pytest.raises(ValueError, match="missing tensors"):
        read_checkpoint_spec(_write(tmp_path, precision, layers=layers, config=config))


def test_missing_nvfp4_wcscales_defaults_to_ones(tmp_path):
    """As on the Diffusers path: an NVFP4 file without channel scales means ones."""
    layers = _layers("nvfp4")
    del layers[A]["wcscales"]
    path = _write(tmp_path, "nvfp4", layers=layers)
    spec = read_checkpoint_spec(path)
    assert spec.layers[A].defaulted == ("wcscales",) and spec.layers[B].defaulted == ()
    model = _Model(device="meta")
    replace_linears(model, spec)
    module = model.get_submodule(A)
    assert module.wcscales.device.type == "cpu" and torch.equal(module.wcscales, torch.ones(256, dtype=torch.bfloat16))
    assert "wcscales" not in module.state_dict()  # a strict framework loader does not expect it
    assert "wcscales" in model.get_submodule(B).state_dict()


def test_wtscale_is_optional_for_nvfp4(tmp_path):
    layers = _layers("nvfp4", wtscale=False)
    spec = read_checkpoint_spec(_write(tmp_path, "nvfp4", layers=layers))
    assert not any(layer.has_wtscale for layer in spec.layers.values())


@pytest.mark.parametrize("suffix", ["wcscales", "wtscale", "lora_down", "weight"])
def test_unexpected_int4_tensor_refused(tmp_path, suffix):
    layers = _layers("int4")
    layers[A][suffix] = torch.ones(256, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="unexpected tensors"):
        read_checkpoint_spec(_write(tmp_path, layers=layers))


def test_bias_present_but_declared_absent_refused(tmp_path):
    layers = _layers()
    with pytest.raises(ValueError, match="unexpected tensors"):
        read_checkpoint_spec(_write(tmp_path, layers=layers, config=_manifest_config(layers, "int4", has_bias=False)))


def test_nested_tensor_under_target_refused(tmp_path):
    with pytest.raises(ValueError, match="Unexpected tensor"):
        read_checkpoint_spec(_write(tmp_path, extra={f"{A}.lora.down": torch.zeros(1)}))


@pytest.mark.parametrize(
    "suffix,value,match",
    [
        ("qweight", torch.zeros(256, 64, dtype=torch.uint8), "qweight"),
        ("qweight", torch.zeros(256 * 64, dtype=torch.int8), "qweight"),
        ("wscales", torch.zeros(2, 256, dtype=torch.float8_e4m3fn), "wscales"),  # fp8 scales on INT4
        ("wscales", torch.zeros(4, 256, dtype=torch.bfloat16), "wscales"),  # wrong group count
        ("smooth_factor", torch.zeros(64, dtype=torch.bfloat16), "smooth_factor"),
        ("smooth_factor", torch.zeros(128, dtype=torch.int8), "smooth_factor"),
        ("proj_up", torch.zeros(256, 32, dtype=torch.bfloat16), "proj_up"),  # rank differs from proj_down
        ("bias", torch.zeros(128, dtype=torch.bfloat16), "bias"),
    ],
)
def test_bad_tensor_shape_or_dtype_refused(tmp_path, suffix, value, match):
    layers = _layers()
    layers[A][suffix] = value
    with pytest.raises(ValueError, match=match):
        read_checkpoint_spec(_write(tmp_path, layers=layers))


def test_nvfp4_scales_must_be_fp8(tmp_path):
    layers = _layers("nvfp4")
    layers[A]["wscales"] = layers[A]["wscales"].to(torch.bfloat16)
    with pytest.raises(ValueError, match="wscales"):
        read_checkpoint_spec(_write(tmp_path, "nvfp4", layers=layers))


def test_wtscale_must_be_one_element(tmp_path):
    layers = _layers("nvfp4")
    layers[A]["wtscale"] = torch.ones(2, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="wtscale must be one floating element"):
        read_checkpoint_spec(_write(tmp_path, "nvfp4", layers=layers))


def test_group_size_mismatch_refused(tmp_path):
    config = quant_config(list(SHAPES))
    config["svdq_w4a4"]["group_size"] = 16
    with pytest.raises(ValueError, match="group_size"):
        read_checkpoint_spec(_write(tmp_path, config=config))


def test_declared_rank_mismatch_refused(tmp_path):
    config = quant_config(list(SHAPES), rank=32)
    with pytest.raises(ValueError, match="rank"):
        read_checkpoint_spec(_write(tmp_path, config=config))


def test_per_target_rank_mismatch_refused(tmp_path):
    layers = _layers()
    config = _manifest_config(layers, "int4", has_bias=True)
    config["runtime_manifest"]["targets"][0]["rank"] = 32  # requirements say 16, one target says 32
    with pytest.raises(ValueError, match="declares rank 32"):
        read_checkpoint_spec(_write(tmp_path, layers=layers, config=config))


def test_missing_target_module_tensors_refused(tmp_path):
    config = quant_config([*SHAPES, "blocks.0.attn.to_k"])
    with pytest.raises(ValueError, match="missing tensors"):
        read_checkpoint_spec(_write(tmp_path, config=config))


def test_legacy_smooth_factor_orig_dropped_but_consumed(tmp_path):
    layers = _layers()
    layers[A]["smooth_factor_orig"] = layers[A]["smooth_factor"].clone()
    spec = read_checkpoint_spec(_write(tmp_path, layers=layers))
    assert "smooth_factor_orig" not in spec.layers[A].tensors


def test_non_svdq_ops_refused(tmp_path):
    config = quant_config(list(SHAPES))
    config["awq_w4a16"] = {"targets": ["blocks.0.norm"], "group_size": 64}
    with pytest.raises(NotImplementedError, match="Only svdq_w4a4"):
        read_checkpoint_spec(_write(tmp_path, config=config, extra={"blocks.0.norm.qweight": torch.zeros(1)}))


# --- rotation --------------------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", ["", PREFIX])
def test_rotation_declared_with_marker(tmp_path, prefix):
    spec = read_checkpoint_spec(_write(tmp_path, rotated=[A, B], prefix=prefix), key_prefix=prefix)
    assert set(spec.rotated) == {A, B} and spec.has_rotation_marker
    assert all(spec.layers[n].input_rotation == "hadamard" for n in (A, B))


def test_partial_rotation(tmp_path):
    spec = read_checkpoint_spec(_write(tmp_path, rotated=[A]))
    assert spec.layers[A].input_rotation == "hadamard" and spec.layers[B].input_rotation is None


def test_marker_without_declaration_refused(tmp_path):
    with pytest.raises(ValueError, match="marker but no"):
        read_checkpoint_spec(_write(tmp_path, marker=True))


def test_declaration_without_marker_refused(tmp_path):
    with pytest.raises(ValueError, match="marker tensor"):
        read_checkpoint_spec(_write(tmp_path, rotated=[A], marker=False))


def test_marker_under_prefix_refused(tmp_path):
    path = _write(tmp_path, prefix=PREFIX, rotated=[A], marker=False, extra={ROTATION_MARKER_KEY: torch.empty(0)})
    with pytest.raises(ValueError, match="root key"):
        read_checkpoint_spec(path, key_prefix=PREFIX)


def test_rotated_non_target_refused(tmp_path):
    config = quant_config(list(SHAPES), rotated=["blocks.0.attn.to_k"])
    with pytest.raises(ValueError, match="no packed SVDQ weight"):
        read_checkpoint_spec(_write(tmp_path, config=config, marker=True))


def test_nvfp4_rotation_refused_unless_allowed(tmp_path):
    path = _write(tmp_path, "nvfp4", rotated=[A])
    with pytest.raises(ValueError, match="NVFP4 is refused"):
        read_checkpoint_spec(path)
    assert read_checkpoint_spec(path, allow_fp4_rotation=True).layers[A].input_rotation == "hadamard"


def test_legacy_input_transform_refused(tmp_path):
    with pytest.raises(ValueError, match="block_matrix"):
        read_checkpoint_spec(_write(tmp_path, extra={f"{A}.input_transform": torch.zeros(1)}))


# --- LTX GELU MLP rule ----------------------------------------------------------------------------------


@pytest.mark.parametrize("p1,p2,r1,r2", list(itertools.product(["int4", "nvfp4"], ["int4", "nvfp4"], [None, "hadamard"], [None, "hadamard"])))
def test_gelu_mlp_plan_full_table(p1, p2, r1, r2):
    class L:
        def __init__(self, precision, rotation):
            self.precision, self.input_rotation = precision, rotation

    plan = ltx2_gelu_mlp_plan(L(p1, r1), L(p2, r2))
    expected = p1 == p2 == "nvfp4" and r1 is None and r2 is None
    assert plan.fused == expected and plan.fc2_fused_flag == expected
    assert plan.fc2_act_unsigned is False


# --- make / replace -----------------------------------------------------------------------------


@pytest.mark.parametrize("precision", ["int4", "nvfp4"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_make_linear(tmp_path, precision, dtype):
    spec = read_checkpoint_spec(_write(tmp_path, precision, rotated=[A] if precision == "int4" else None))
    module = make_linear(spec.layers[A], torch_dtype=dtype)
    assert module.qweight.is_meta and module.smooth_factor.dtype == dtype
    assert module.input_rotation == spec.layers[A].input_rotation and module.act_unsigned is False
    if precision == "nvfp4":
        assert "wtscale" in module._buffers and module.wtscale.dtype == torch.float32
        assert "wtscale" in module.state_dict()
    else:
        assert module.wtscale is None and "wtscale" not in module.state_dict()


def test_make_linear_refuses_fp32(tmp_path):
    spec = read_checkpoint_spec(_write(tmp_path))
    with pytest.raises(ValueError, match="bfloat16 or float16"):
        make_linear(spec.layers[A], torch_dtype=torch.float32)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_replace_linears_keeps_device_and_dtype(tmp_path, device):
    spec = read_checkpoint_spec(_write(tmp_path))
    model = _Model(device=device)
    replaced = replace_linears(model, spec)
    assert set(replaced) == set(SHAPES)
    for name, module in replaced.items():
        assert model.get_submodule(name) is module and isinstance(module, SVDQW4A4Linear)
        assert module.qweight.device.type == device and module.smooth_factor.dtype == torch.bfloat16


def test_replace_linears_accepts_non_linear_modules(tmp_path):
    class OpsLinear(nn.Module):  # e.g. a framework's own Linear that is not an nn.Linear subclass
        def __init__(self, in_features, out_features, device=None, dtype=None):
            super().__init__()
            self.in_features, self.out_features = in_features, out_features

    spec = read_checkpoint_spec(_write(tmp_path))
    model = _Model(linear=OpsLinear)
    replace_linears(model, spec, torch_dtype=torch.float16, device="meta")
    assert model.get_submodule(A).smooth_factor.dtype == torch.float16


def test_replace_linears_is_atomic(tmp_path):
    spec = read_checkpoint_spec(_write(tmp_path))
    model = _Model()
    model.blocks[0].ff.net[2] = nn.Linear(256, 64)  # wrong out_features on the second target
    with pytest.raises(ValueError, match="out_features=64"):
        replace_linears(model, spec)
    assert isinstance(model.get_submodule(A), nn.Linear)  # the first target was not swapped


def test_replace_linears_missing_module(tmp_path):
    spec = read_checkpoint_spec(_write(tmp_path))
    model = _Model()
    del model.blocks[0].attn.to_q
    with pytest.raises(ValueError, match="has no module"):
        replace_linears(model, spec)


# --- moves and casts keep the quantized tensors ------------------------------------------------------


@pytest.mark.parametrize("precision", ["int4", "nvfp4"])
def test_to_dtype_and_device_keep_quantized_tensors(tmp_path, precision):
    path = _write(tmp_path, precision, rotated=[A] if precision == "int4" else None)
    spec = read_checkpoint_spec(path)
    model = _Model()
    replace_linears(model, spec)
    _fill(model, path, spec)
    module = model.get_submodule(A)
    before = {k: v.clone() for k, v in module.state_dict().items()}
    for dtype in (torch.float16, torch.bfloat16, torch.float16):
        model.to(dtype)
        model.to("cpu")
    after = module.state_dict()
    assert module.qweight.dtype == torch.int8
    assert module.wscales.dtype == (torch.float8_e4m3fn if precision == "nvfp4" else torch.float16)
    assert torch.equal(after["qweight"], before["qweight"])
    if precision == "nvfp4":
        assert after["wscales"].view(torch.uint8).equal(before["wscales"].view(torch.uint8))
        assert module.wtscale.dtype == torch.float32 and module.wtscale.item() == pytest.approx(WTSCALE, rel=1e-2)
        assert after["wtscale"].equal(before["wtscale"])
    assert module.input_rotation == spec.layers[A].input_rotation


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a non-CPU device")
def test_wtscale_alpha_reads_device_buffer_once_per_load(tmp_path):
    path = _write(tmp_path, "nvfp4")
    spec = read_checkpoint_spec(path)
    model = _Model()
    replace_linears(model, spec)
    _fill(model, path, spec)
    module = model.get_submodule(A)
    assert wtscale_alpha(module) is module.wtscale  # CPU tensor: passed through, as for the Diffusers loader
    model.to("mps")
    alpha = wtscale_alpha(module)
    assert alpha.device.type == "cpu" and alpha.dtype == torch.float32
    assert alpha.item() == module.wtscale.cpu().item()
    assert wtscale_alpha(module) is alpha  # cached: no further device reads
    module.wtscale.fill_(0.5)  # a reload / in-place update invalidates the cache
    assert wtscale_alpha(module).item() == 0.5


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a non-CPU device")
def test_wtscale_alpha_handles_inference_tensors(tmp_path):
    """ltx-core builds and binds under torch.inference_mode: the wtscale buffer is then an inference tensor."""
    path = _write(tmp_path, "nvfp4")
    spec = read_checkpoint_spec(path)
    with torch.inference_mode():
        model = _Model()
        replace_linears(model, spec)
        _fill(model, path, spec)
        model.to("mps")
    module = model.get_submodule(A)
    assert module.wtscale.is_inference() and module.wtscale.device.type == "mps"
    alpha = wtscale_alpha(module)  # outside inference mode, as the first forward may be
    assert alpha.item() == module.wtscale.cpu().item() and wtscale_alpha(module) is alpha
    with torch.inference_mode():
        assert wtscale_alpha(module) is alpha


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple silicon")
def test_mps_forward_on_inference_mode_model(tmp_path):
    """The MPS backend (INT4 only) caches derived weights keyed by tensor version; inference tensors have none."""
    path = _write(tmp_path, "int4")
    spec = read_checkpoint_spec(path)
    models = []
    for inference in (False, True):
        with torch.inference_mode(inference):
            model = _Model()
            replace_linears(model, spec)
            _fill(model, path, spec)
            model.to("mps")
        models.append(model)
    x = torch.randn(3, spec.layers[A].in_features, generator=torch.Generator().manual_seed(0)).to("mps", torch.bfloat16)
    with torch.inference_mode():
        ref = models[0].get_submodule(A)(x)
        out = models[1].get_submodule(A)(x)
        again = models[1].get_submodule(A)(x)  # second call hits the derived-weight cache
    assert models[1].get_submodule(A).qweight.is_inference()
    assert torch.equal(out, ref) and torch.equal(again, ref)


def _fill(model, path, spec):
    """What a framework's own weight loader does after replace_linears: load the file's tensors by name."""
    from safetensors.torch import load_file

    state = {k.removeprefix(spec.key_prefix): v for k, v in load_file(str(path)).items() if k.startswith(spec.key_prefix)}
    state = {k: v for k, v in state.items() if k.rpartition(".")[0] in spec.layers and k.rpartition(".")[2] in spec.layers[k.rpartition(".")[0]].tensors}
    model.load_state_dict(state, strict=False)


def _manifest_config(layers, precision, *, has_bias):
    targets = [
        {
            "name": name,
            "checkpoint_prefix": name,
            "source_modules": [name],
            "roles": [],
            "kind": "linear",
            "nunchaku_op": "svdq_w4a4",
            "precision": "fp4" if precision == "nvfp4" else precision,
            "group_size": 64 if precision == "int4" else 16,
            "rank": 16,
            "has_bias": has_bias,
            "op_options": {},
            "activation": {},
        }
        for name in layers
    ]
    return {
        "quant_method": "nunchaku_lite",
        "runtime_manifest": {
            "schema": "nunchaku_lite.runtime_manifest",
            "version": 1,
            "component": "transformer",
            "nunchaku_format_version": 1,
            "producer": {"name": "test", "version": "1"},
            "requirements": {"method": "svdquant", "precision": "fp4" if precision == "nvfp4" else precision, "rank": 16,
                             "weight_dtype": precision, "activation_dtype": precision},
            "structural_patches": [],
            "targets": targets,
        },
    }
