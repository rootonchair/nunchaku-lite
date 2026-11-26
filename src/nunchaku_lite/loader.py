"""Framework-agnostic loading of Nunchaku Lite SVDQ W4A4 checkpoints.

Used by integrations that build the model themselves (Lightricks' ltx-core, ComfyUI): they read a
:class:`CheckpointSpec` from the safetensors header, swap their linears for SVDQ layers by exact
module name and bind the quantized tensors. Torch only: no Diffusers, ltx-core or ComfyUI imports.

Frozen spec (see ``docs/models/ltx2-reference-code.md`` for the decision table):

* **Layout.** ``native`` when any key starts with ``model.diffusion_model.`` (Lightricks' official
  single-file transformer layout, also read by ComfyUI); every model tensor then carries that
  prefix, and layer names in the spec are relative to it. Otherwise ``diffusers``: no prefix.
  The ``nunchaku_lite.input_rotation`` marker always sits at the file root.
* **Metadata.** ``quantization_config`` (JSON) in the safetensors metadata; a Diffusers-layout
  file may instead keep it in a sibling ``config.json``. A native file must carry it in metadata.
* **Targets.** The ``svdq_w4a4`` targets of the config (target lists or a runtime manifest).
  Other ops, and manifests with structural patches, are refused here: they need a model adapter.
* **Per-layer tensors.** ``qweight`` int8 ``(out, in/2)``; ``wscales`` ``(in/g, out)``, bf16/fp16
  for INT4 (``g`` = 64) and float8_e4m3fn for NVFP4 (``g`` = 16); ``smooth_factor`` ``(in,)``;
  ``proj_down`` ``(in, r)``; ``proj_up`` ``(out, r)``; ``bias`` ``(out,)`` iff declared; NVFP4
  also ``wcscales`` ``(out,)`` and optionally ``wtscale`` (one element). ``smooth_factor_orig`` is
  a legacy leftover and is dropped. Anything else under a target, or a missing tensor, is refused.
* **Rotation.** As :func:`nunchaku_lite.core.patch_transformer`: a model-level ``input_rotation``
  declaration requires the marker tensor and vice versa (or a stored version 2 manifest declaring
  rotation per target, which must agree); rotated layers must be SVDQ targets; NVFP4 rotation is
  refused unless ``allow_fp4_rotation``.
* **wtscale.** When the file has it, the layer keeps it as a persistent float32 buffer so that a
  generic state-dict loader binds it; the GEMM reads it once per load (``wtscale_alpha``). An
  NVFP4 layer without ``wtscale`` uses 1.0.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from .linear import SVDQW4A4Linear
from .manifest import ROTATION_MARKER_KEY, RuntimeManifest, parse_runtime_manifest, with_target_list_manifest
from .rotation_loader import (
    HADAMARD,
    check_rotation_runtime,
    parse_rotation_declaration,
    refuse_legacy_transform_tensors,
)

__all__ = [
    "NATIVE_PREFIX",
    "CheckpointSpec",
    "GeluMlpPlan",
    "LayerSpec",
    "bind_quantized_state",
    "check_runtime_support",
    "gelu_mlp_plan",
    "make_linear",
    "read_checkpoint_spec",
    "replace_linears",
]

NATIVE_PREFIX = "model.diffusion_model."
LEGACY_DROPPED_SUFFIXES = ("smooth_factor_orig",)
_INT_SUFFIXES = {"qweight"}
_REQUIRED = ("qweight", "wscales", "smooth_factor", "proj_down", "proj_up")
_GROUP_SIZE = {"int4": 64, "nvfp4": 16}
_FLOATS = (torch.bfloat16, torch.float16, torch.float32)

TensorInfo = tuple[tuple[int, ...], torch.dtype]
NameMap = Callable[[str], str]


@dataclass(frozen=True)
class LayerSpec:
    """One SVDQ W4A4 layer of a checkpoint, as declared by its header."""

    name: str  # module path relative to the model root (no file prefix)
    precision: str  # "int4" | "nvfp4"
    in_features: int
    out_features: int
    rank: int
    group_size: int
    has_bias: bool
    has_wtscale: bool
    input_rotation: str | None
    tensors: Mapping[str, TensorInfo] = field(repr=False)  # suffix -> (shape, dtype)
    dropped: tuple[str, ...] = field(default=(), repr=False)  # legacy suffixes present in the file


@dataclass(frozen=True)
class CheckpointSpec:
    """Header-level description of a Nunchaku Lite checkpoint."""

    path: Path
    layout: str  # "native" | "diffusers"
    key_prefix: str
    precision: str
    rank: int
    layers: Mapping[str, LayerSpec]
    quantization_config: Mapping[str, Any] = field(repr=False)
    has_rotation_marker: bool = False
    model_keys: frozenset[str] = field(default=frozenset(), repr=False)  # every key under key_prefix, unprefixed

    @property
    def rotated(self) -> tuple[str, ...]:
        return tuple(name for name, layer in self.layers.items() if layer.input_rotation is not None)

    def file_key(self, name: str, suffix: str) -> str:
        return f"{self.key_prefix}{name}.{suffix}"

    def layer_file_keys(self, name: str) -> set[str]:
        layer = self.layers[name]
        return {self.file_key(name, s) for s in (*layer.tensors, *layer.dropped)}

    def quantized_file_keys(self) -> set[str]:
        """Every file key that :func:`bind_quantized_state` consumes."""

        keys = set().union(*(self.layer_file_keys(name) for name in self.layers)) if self.layers else set()
        if self.has_rotation_marker:
            keys.add(ROTATION_MARKER_KEY)
        return keys


@dataclass(frozen=True)
class GeluMlpPlan:
    """How a ``fc1 -> GELU -> fc2`` pair of SVDQ layers runs."""

    fused: bool  # through ops.fused.fused_gelu_mlp (fc2's input quantized in fc1's epilogue)
    fc2_act_unsigned: bool
    fc2_fused_flag: bool  # set ops.fused.FUSED_MLP_FC2_ATTR on fc2


class _QuantizedLayer(Protocol):
    precision: str
    input_rotation: str | None


def gelu_mlp_plan(fc1: _QuantizedLayer, fc2: _QuantizedLayer) -> GeluMlpPlan:
    """The one rule for SVDQ GELU MLPs, shared by the Diffusers, ltx-core and ComfyUI integrations.

    Takes :class:`LayerSpec` objects or bound SVDQ layers (anything with ``precision`` and
    ``input_rotation``).

    Fused only for NVFP4 with neither layer rotated: for INT4, fc1's fused epilogue quantizes fc2's
    input unsigned with the shifted-GELU convention, which LTX exports do not use, and a rotated
    layer must quantize its own input on the signed standalone path. Unfused, fc2 is signed.
    """

    fused = (
        fc1.precision == "nvfp4"
        and fc2.precision == "nvfp4"
        and fc1.input_rotation is None
        and fc2.input_rotation is None
    )
    return GeluMlpPlan(fused=fused, fc2_act_unsigned=False, fc2_fused_flag=fused)


def read_checkpoint_spec(path: str | os.PathLike, *, allow_fp4_rotation: bool = False) -> CheckpointSpec:
    """Read and validate a checkpoint's header without loading any tensor data."""

    from safetensors import safe_open

    path = Path(path)
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {}
        header: dict[str, TensorInfo] = {}
        for key in f.keys():  # noqa: SIM118 - safe_open handle, not a dict
            sl = f.get_slice(key)
            header[key] = (tuple(int(d) for d in sl.get_shape()), _dtype(sl.get_dtype(), key))

    native = any(key.startswith(NATIVE_PREFIX) for key in header)
    prefix = NATIVE_PREFIX if native else ""
    model = {key[len(prefix) :]: info for key, info in header.items() if key.startswith(prefix)}
    if native:
        model.pop(ROTATION_MARKER_KEY, None)  # the marker belongs at the root, never under the prefix
    has_marker = ROTATION_MARKER_KEY in header
    if f"{NATIVE_PREFIX}{ROTATION_MARKER_KEY}" in header:
        raise ValueError(f"The {ROTATION_MARKER_KEY!r} marker must be a root key, not under {NATIVE_PREFIX!r}.")

    config = _quantization_config(metadata, path, native=native)
    refuse_legacy_transform_tensors(model)
    config = with_target_list_manifest(config, model.keys())
    manifest = parse_runtime_manifest(config)
    if manifest is None:
        raise ValueError(f"{path} declares no nunchaku_lite quantization targets.")
    if manifest.structural_patches:
        raise NotImplementedError("Checkpoints with runtime_manifest structural patches need a model adapter.")
    other_ops = sorted({t.nunchaku_op for t in manifest.targets} - {"svdq_w4a4"})
    if other_ops:
        raise NotImplementedError(f"Only svdq_w4a4 targets are supported here; checkpoint also has {other_ops}.")

    rotated = _rotated_targets(config, manifest, has_marker)
    targets = {t.checkpoint_prefix: t for t in manifest.targets}
    if len(targets) != len(manifest.targets):
        raise ValueError("Checkpoint declares a quantization target more than once.")
    unknown = sorted(set(rotated) - set(targets))
    if unknown:
        raise ValueError(f"Rotated targets {unknown[:3]} are not svdq_w4a4 targets of the checkpoint.")

    by_layer: dict[str, dict[str, TensorInfo]] = {name: {} for name in targets}
    for key, info in model.items():
        owner, _, suffix = key.rpartition(".")
        if owner in by_layer:
            by_layer[owner][suffix] = info
    # A target's tensors must sit directly under it: nothing may hide one level deeper.
    for key in model:
        for name in targets:
            if key.startswith(f"{name}.") and key.rpartition(".")[0] != name:
                raise ValueError(f"Unexpected tensor {key!r} under quantization target {name!r}.")

    layers = {}
    for name, target in targets.items():
        precision = target.runtime_precision
        if precision not in _GROUP_SIZE:
            raise ValueError(f"Target {name!r} has unsupported precision {target.precision!r}.")
        rotation = HADAMARD if name in rotated else None
        if rotation is not None and precision == "nvfp4" and not allow_fp4_rotation:
            raise ValueError(
                "Rotated checkpoint: rotation on NVFP4 is refused by default (it measured slightly worse than no "
                "rotation); pass allow_fp4_rotation=True to load it anyway."
            )
        layers[name] = _layer_spec(name, precision, target.group_size, target.has_bias, rotation, by_layer[name])
        if target.rank and layers[name].rank != target.rank:
            raise ValueError(f"Target {name!r} declares rank {target.rank}, but its tensors have rank {layers[name].rank}.")

    ranks = {layer.rank for layer in layers.values()}
    precisions = {layer.precision for layer in layers.values()}
    declared_rank = manifest.requirements.get("rank")
    if declared_rank and ranks != {declared_rank}:
        raise ValueError(f"Checkpoint declares rank {declared_rank} but its layers have ranks {sorted(ranks)}.")
    return CheckpointSpec(
        path=path,
        layout="native" if native else "diffusers",
        key_prefix=prefix,
        precision=precisions.pop() if len(precisions) == 1 else "mixed",
        rank=max(ranks),
        layers=layers,
        quantization_config=config,
        has_rotation_marker=has_marker,
        model_keys=frozenset(model),
    )


def check_runtime_support(spec: CheckpointSpec, device: torch.device | str) -> None:
    """Refuse a checkpoint the target device cannot run, before anything is built."""

    device = torch.device(device)
    if spec.rotated:
        check_rotation_runtime(device)
    if device.type != "cuda":
        return
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    sm = major * 10 + minor
    if any(layer.precision == "nvfp4" for layer in spec.layers.values()) and sm < 100:
        raise ValueError(f"NVFP4 checkpoints need a Blackwell GPU (sm100+); this GPU is sm{sm}. Use an INT4 checkpoint.")
    if sm < 75 or (80 < sm < 100 and sm not in (86, 89)):
        raise ValueError(f"Unsupported GPU architecture sm{sm}; Nunchaku Lite needs Turing, Ampere, Ada or Blackwell.")


def make_linear(
    layer: LayerSpec,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = "meta",
) -> SVDQW4A4Linear:
    """An empty SVDQ layer shaped by ``layer``; its tensors are bound later.

    ``torch_dtype`` (bf16 or fp16) is the dtype of the floating tensors (bias, smooth_factor,
    low-rank factors, INT4 wscales, NVFP4 wcscales). ``wtscale``, when the file has one, is a
    persistent float32 buffer.
    """

    if torch_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"SVDQ layers compute in bfloat16 or float16, not {torch_dtype}.")
    module = SVDQW4A4Linear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        rank=layer.rank,
        bias=layer.has_bias,
        precision=layer.precision,
        act_unsigned=False,
        torch_dtype=torch_dtype,
        device=device,
        input_rotation=layer.input_rotation,
    )
    if layer.has_wtscale:
        del module.wtscale  # the plain float attribute set by __init__
        shape = layer.tensors["wtscale"][0]
        module.register_buffer("wtscale", torch.ones(shape, dtype=torch.float32, device=device), persistent=True)
    return module


def replace_linears(
    root: nn.Module,
    spec: CheckpointSpec,
    *,
    torch_dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    name_map: NameMap | None = None,
) -> dict[str, SVDQW4A4Linear]:
    """Swap every target of ``spec`` in ``root`` for an empty SVDQ layer, by exact module name.

    The old module need not be an ``nn.Linear``; its ``in_features`` / ``out_features`` are checked
    when it has them. ``device`` defaults to the old module's (``meta`` included) and
    ``torch_dtype`` to its floating dtype (bfloat16 when it has none). All targets are resolved and
    checked before ``root`` is modified. Returns ``{spec name: new module}``.
    """

    plan = []
    for name, layer in spec.layers.items():
        path = name_map(name) if name_map is not None else name
        try:
            old = root.get_submodule(path)
        except AttributeError as exc:
            raise ValueError(f"Quantization target {name!r} has no module {path!r} in the model.") from exc
        for attr, expected in (("in_features", layer.in_features), ("out_features", layer.out_features)):
            actual = getattr(old, attr, None)
            if isinstance(actual, int) and actual != expected:
                raise ValueError(f"Module {path!r} has {attr}={actual}, but the checkpoint layer has {expected}.")
        dtype = torch_dtype or _floating_dtype(old) or torch.bfloat16
        target_device = device if device is not None else _module_device(old)
        plan.append((name, path, layer, dtype, target_device))

    replaced = {}
    for name, path, layer, dtype, target_device in plan:
        module = make_linear(layer, torch_dtype=dtype, device=target_device)
        parent_path, _, child = path.rpartition(".")
        parent = root.get_submodule(parent_path) if parent_path else root
        setattr(parent, child, module)
        replaced[name] = module
    return replaced


@torch.no_grad()
def bind_quantized_state(
    root: nn.Module,
    spec: CheckpointSpec,
    *,
    device: torch.device | str | None = None,
    name_map: NameMap | None = None,
) -> set[str]:
    """Load the quantized tensors of every target into the SVDQ layers of ``root``.

    The layers must come from :func:`replace_linears` / :func:`make_linear` for the same spec.
    Meta layers are materialized on ``device`` (default: CPU); others stay where they are unless
    ``device`` is given. Floating tensors are cast to the layer's dtype, fp8 and int8 tensors are
    loaded bit-exact, ``wtscale`` as float32. Returns every file key consumed, so a caller that
    loads the rest itself can check full coverage.
    """

    from safetensors import safe_open

    modules = {}
    for name, layer in spec.layers.items():
        path = name_map(name) if name_map is not None else name
        module = root.get_submodule(path)
        _check_bound_module(path, module, layer)
        modules[name] = module

    consumed: set[str] = set()
    with safe_open(str(spec.path), framework="pt") as f:
        for name, layer in spec.layers.items():
            module = modules[name]
            current = _module_device(module)
            target = torch.device(device) if device is not None else (torch.device("cpu") if current.type == "meta" else current)
            for suffix in layer.tensors:
                value = f.get_tensor(spec.file_key(name, suffix))
                if suffix == "wtscale":
                    value = value.to(torch.float32).reshape(module.wtscale.shape)
                else:
                    expected = getattr(module, suffix)
                    if value.dtype in _FLOATS and expected.dtype in _FLOATS:
                        value = value.to(expected.dtype)
                    if value.dtype != expected.dtype or tuple(value.shape) != tuple(expected.shape):
                        raise ValueError(f"{spec.file_key(name, suffix)}: {tuple(value.shape)} {value.dtype} does not "
                                         f"fit the layer's {tuple(expected.shape)} {expected.dtype}.")
                _assign(module, suffix, value.to(target))
            consumed |= spec.layer_file_keys(name)
            meta = [n for n, t in (*module.named_parameters(), *module.named_buffers()) if t.is_meta]
            if meta:
                raise RuntimeError(f"Layer {name!r} still has meta tensors after binding: {meta}.")
            module.__dict__.pop("_nunchaku_lite_alpha_cache", None)
    if spec.has_rotation_marker:
        consumed.add(ROTATION_MARKER_KEY)
    return consumed


# --- helpers -------------------------------------------------------------------------------------

_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def _dtype(name: str, key: str) -> torch.dtype:
    try:
        return _DTYPES[name]
    except KeyError:
        raise ValueError(f"Tensor {key!r} has unsupported dtype {name!r}.") from None


def _quantization_config(metadata: Mapping[str, str], path: Path, *, native: bool) -> dict[str, Any]:
    raw = metadata.get("quantization_config")
    if raw is not None:
        try:
            config = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} metadata 'quantization_config': {exc}") from exc
        if not isinstance(config, dict):
            raise TypeError(f"{path} metadata 'quantization_config' must be a JSON object.")
        return config
    if native:
        raise ValueError(f"{path} is a native-layout checkpoint without 'quantization_config' in its metadata.")
    sibling = path.parent / "config.json"
    if sibling.is_file():
        config = json.loads(sibling.read_text()).get("quantization_config")
        if isinstance(config, dict):
            return config
    raise ValueError(f"{path} has no 'quantization_config' (metadata or sibling config.json).")


def _rotated_targets(config: Mapping[str, Any], manifest: RuntimeManifest, has_marker: bool) -> tuple[str, ...]:
    declared = parse_rotation_declaration(dict(config))
    stored = "runtime_manifest" in config and manifest.nunchaku_format_version >= 2
    manifest_rotated = tuple(t.checkpoint_prefix for t in manifest.rotated_targets) if stored else ()
    if has_marker and declared is None:
        raise ValueError(
            f"Checkpoint has the {ROTATION_MARKER_KEY!r} rotation marker but no quantization_config.input_rotation "
            "declaration listing the rotated targets."
        )
    if declared is None:
        return manifest_rotated
    if manifest_rotated and set(declared) != set(manifest_rotated):
        raise ValueError(
            "quantization_config.input_rotation.targets do not match the runtime_manifest targets that declare "
            "op_options.rotation; the checkpoint metadata is inconsistent."
        )
    if not manifest_rotated and not has_marker:
        raise ValueError(
            "Checkpoint declares quantization_config.input_rotation without the "
            f"{ROTATION_MARKER_KEY!r} marker tensor (or a version 2 runtime_manifest declaring it)."
        )
    return declared


def _layer_spec(
    name: str,
    precision: str,
    group_size: int,
    has_bias: bool,
    rotation: str | None,
    tensors: dict[str, TensorInfo],
) -> LayerSpec:
    expected_group = _GROUP_SIZE[precision]
    if group_size != expected_group:
        raise ValueError(f"Target {name!r}: {precision} needs group_size {expected_group}, got {group_size}.")
    dropped = tuple(s for s in LEGACY_DROPPED_SUFFIXES if s in tensors)
    tensors = {s: info for s, info in tensors.items() if s not in dropped}
    nvfp4_extra = {"wcscales", "wtscale"} if precision == "nvfp4" else set()
    allowed = set(_REQUIRED) | ({"bias"} if has_bias else set()) | nvfp4_extra
    required = set(_REQUIRED) | ({"bias"} if has_bias else set()) | ({"wcscales"} if precision == "nvfp4" else set())
    missing = sorted(required - set(tensors))
    if missing:
        raise ValueError(f"Target {name!r} is missing tensors {missing}.")
    extra = sorted(set(tensors) - allowed)
    if extra:
        raise ValueError(f"Target {name!r} has unexpected tensors {extra}.")

    qshape, qdtype = tensors["qweight"]
    if len(qshape) != 2 or qdtype != torch.int8:
        raise ValueError(f"Target {name!r}: qweight must be 2-D int8, got {qshape} {qdtype}.")
    out_features, half_in = qshape
    in_features = 2 * half_in
    down_shape = tensors["proj_down"][0]
    rank = down_shape[1] if len(down_shape) == 2 else -1
    scale_dtypes = (torch.float8_e4m3fn,) if precision == "nvfp4" else (torch.bfloat16, torch.float16)
    shapes = {
        "wscales": ((in_features // group_size, out_features), scale_dtypes),
        "smooth_factor": ((in_features,), _FLOATS),
        "proj_down": ((in_features, rank), _FLOATS),
        "proj_up": ((out_features, rank), _FLOATS),
        "bias": ((out_features,), _FLOATS),
        "wcscales": ((out_features,), _FLOATS),
    }
    for suffix, (shape, dtypes) in shapes.items():
        if suffix not in tensors:
            continue
        actual_shape, actual_dtype = tensors[suffix]
        if tuple(actual_shape) != shape or actual_dtype not in dtypes:
            raise ValueError(f"Target {name!r}: {suffix} is {tuple(actual_shape)} {actual_dtype}, expected {shape} "
                             f"in {[str(d) for d in dtypes]}.")
    if in_features % group_size or in_features % 128 or out_features % 128:
        raise ValueError(f"Target {name!r}: ({out_features}, {in_features}) does not fit SVDQ packing.")
    if rank < 0 or rank % 16:
        raise ValueError(f"Target {name!r}: low-rank rank {rank} must be a non-negative multiple of 16.")
    if "wtscale" in tensors:
        shape, dtype = tensors["wtscale"]
        if int(torch.Size(shape).numel()) != 1 or dtype not in _FLOATS:
            raise ValueError(f"Target {name!r}: wtscale must be one floating element, got {shape} {dtype}.")
    return LayerSpec(
        name=name,
        precision=precision,
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        group_size=group_size,
        has_bias=has_bias,
        has_wtscale="wtscale" in tensors,
        input_rotation=rotation,
        tensors=tensors,
        dropped=dropped,
    )


def _check_bound_module(path: str, module: nn.Module, layer: LayerSpec) -> None:
    if not isinstance(module, SVDQW4A4Linear):
        raise TypeError(f"{path!r} is a {type(module).__name__}; call replace_linears first.")
    got = (module.in_features, module.out_features, module.rank, module.precision, module.input_rotation,
           module.bias is not None, isinstance(getattr(module, "wtscale", None), torch.Tensor))
    want = (layer.in_features, layer.out_features, layer.rank, layer.precision, layer.input_rotation,
            layer.has_bias, layer.has_wtscale)
    if got != want:
        raise ValueError(f"{path!r} does not match its checkpoint layer: module {got}, checkpoint {want}.")
    if module.act_unsigned:
        raise ValueError(f"{path!r} is set to unsigned activations; checkpoint layers quantize signed inputs.")


def _assign(module: nn.Module, name: str, value: torch.Tensor) -> None:
    if name in module._parameters:
        old = module._parameters[name]
        module._parameters[name] = nn.Parameter(value, requires_grad=bool(old is not None and old.requires_grad and value.is_floating_point()))
    elif name in module._buffers:
        module._buffers[name] = value
    else:
        raise AttributeError(f"SVDQ layer has no tensor {name!r}.")


def _module_device(module: nn.Module) -> torch.device:
    for tensor in (*module.parameters(recurse=False), *module.buffers(recurse=False)):
        return tensor.device
    for tensor in (*module.parameters(), *module.buffers()):
        return tensor.device
    return torch.device("cpu")


def _floating_dtype(module: nn.Module) -> torch.dtype | None:
    for tensor in (*module.parameters(), *module.buffers()):
        if tensor.dtype in (torch.bfloat16, torch.float16):
            return tensor.dtype
    return None
