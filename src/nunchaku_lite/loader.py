"""Framework-agnostic loading of Nunchaku Lite SVDQ W4A4 checkpoints.

For integrations that build the model themselves (Lightricks' ltx-core, later ComfyUI): read a :class:`CheckpointSpec`
from the safetensors header, then swap the model's linears for SVDQ layers by exact module name; the framework's own
weight loader fills them. Torch only: no Diffusers or framework imports.

This module defines no checkpoint format of its own:

* the per-layer tensors (names, shapes, dtypes, group size) are checked against an empty
  :class:`~nunchaku_lite.linear.SVDQW4A4Linear` built for that layer, so the layer class is the only definition;
* ``quantization_config`` lookup and GPU support rules come from :mod:`nunchaku_lite.checkpoint_config`, and the
  rotation rules from :mod:`nunchaku_lite.rotation_loader`, shared with :func:`nunchaku_lite.core.patch_transformer`;
* the key layout is the caller's: ``key_prefix`` names the prefix the model's tensors sit under in the file (``""``
  for a file that holds only the model). The rotation marker is always a root key.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from .checkpoint_config import check_device_precision, resolve_quantization_config
from .linear import SVDQ_CHECKPOINT_SCALARS, SVDQ_LEGACY_CHECKPOINT_SUFFIXES, SVDQW4A4Linear
from .manifest import ROTATION_MARKER_KEY, parse_runtime_manifest, with_target_list_manifest
from .rotation_loader import (
    HADAMARD,
    check_rotation_runtime,
    refuse_legacy_transform_tensors,
    resolve_rotated_targets,
    validate_declared_targets,
)

__all__ = [
    "CheckpointSpec",
    "GeluMlpPlan",
    "LayerSpec",
    "check_runtime_support",
    "gelu_mlp_plan",
    "make_linear",
    "read_checkpoint_spec",
    "replace_linears",
]

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
    tensors: Mapping[str, TensorInfo] = field(repr=False)  # suffix -> (shape, dtype), legacy suffixes excluded


@dataclass(frozen=True)
class CheckpointSpec:
    """Header-level description of a Nunchaku Lite checkpoint."""

    path: Path
    key_prefix: str
    precision: str  # the layers' precision, or "mixed"
    rank: int
    layers: Mapping[str, LayerSpec]
    quantization_config: Mapping[str, Any] = field(repr=False)
    has_rotation_marker: bool = False

    @property
    def rotated(self) -> tuple[str, ...]:
        return tuple(name for name, layer in self.layers.items() if layer.input_rotation is not None)


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
    """The one rule for SVDQ GELU MLPs, shared by the Diffusers and framework integrations.

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


def read_checkpoint_spec(
    path: str | os.PathLike, *, key_prefix: str = "", allow_fp4_rotation: bool = False
) -> CheckpointSpec:
    """Read and validate a checkpoint's header without loading any tensor data.

    Args:
        path: Local safetensors file.
        key_prefix: Prefix of the model's tensor keys in the file; keys outside it are ignored (a single-file
            checkpoint may also hold other components). Layer names in the spec are relative to it.
        allow_fp4_rotation: Accept rotated NVFP4 layers.
    """

    from safetensors import safe_open

    path = Path(path)
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {}
        header: dict[str, TensorInfo] = {}
        for key in f.keys():  # noqa: SIM118 - safe_open handle, not a dict
            sl = f.get_slice(key)
            header[key] = (tuple(int(d) for d in sl.get_shape()), _dtype(sl.get_dtype(), key))

    if key_prefix and f"{key_prefix}{ROTATION_MARKER_KEY}" in header:
        raise ValueError(f"The {ROTATION_MARKER_KEY!r} marker must be a root key, not under {key_prefix!r}.")
    has_marker = ROTATION_MARKER_KEY in header
    model = {key[len(key_prefix) :]: info for key, info in header.items() if key.startswith(key_prefix)}
    model.pop(ROTATION_MARKER_KEY, None)
    if not model:
        raise ValueError(f"{path} has no tensors under the key prefix {key_prefix!r}.")

    config = resolve_quantization_config(metadata, path)
    if not config:
        raise ValueError(f"{path} has no 'quantization_config' (metadata or sibling config.json).")
    refuse_legacy_transform_tensors(model)
    has_stored_manifest = "runtime_manifest" in config
    config = with_target_list_manifest(config, model.keys())
    manifest = parse_runtime_manifest(config)
    if manifest is None:
        raise ValueError(f"{path} declares no nunchaku_lite quantization targets.")
    if manifest.structural_patches:
        raise NotImplementedError("Checkpoints with runtime_manifest structural patches need a model adapter.")
    other_ops = sorted({t.nunchaku_op for t in manifest.targets} - {"svdq_w4a4"})
    if other_ops:
        raise NotImplementedError(f"Only svdq_w4a4 targets are supported here; checkpoint also has {other_ops}.")
    targets = {t.checkpoint_prefix: t for t in manifest.targets}
    if len(targets) != len(manifest.targets):
        raise ValueError("Checkpoint declares a quantization target more than once.")

    rotated = resolve_rotated_targets(
        config, has_marker=has_marker, stored_manifest=manifest if has_stored_manifest else None
    ) or ()
    if rotated:
        validate_declared_targets(
            model,
            rotated,
            is_fp4=any(targets[n].runtime_precision == "nvfp4" for n in rotated if n in targets),
            allow_fp4=allow_fp4_rotation,
        )
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
        layer = _layer_spec(
            name,
            target.runtime_precision,
            target.group_size,
            target.has_bias,
            HADAMARD if name in rotated else None,
            by_layer[name],
        )
        if target.rank and layer.rank != target.rank:
            raise ValueError(f"Target {name!r} declares rank {target.rank}, but its tensors have rank {layer.rank}.")
        layers[name] = layer

    ranks = {layer.rank for layer in layers.values()}
    precisions = {layer.precision for layer in layers.values()}
    declared_rank = manifest.requirements.get("rank")
    if declared_rank and ranks != {declared_rank}:
        raise ValueError(f"Checkpoint declares rank {declared_rank} but its layers have ranks {sorted(ranks)}.")
    return CheckpointSpec(
        path=path,
        key_prefix=key_prefix,
        precision=precisions.pop() if len(precisions) == 1 else "mixed",
        rank=max(ranks),
        layers=layers,
        quantization_config=config,
        has_rotation_marker=has_marker,
    )


def check_runtime_support(spec: CheckpointSpec, device: torch.device | str) -> None:
    """Refuse a checkpoint the target device cannot run, before anything is built (same rules as the Diffusers path)."""

    if spec.rotated:
        check_rotation_runtime(torch.device(device))
    for precision in sorted({layer.precision for layer in spec.layers.values()}):
        check_device_precision(precision, device)


def make_linear(
    layer: LayerSpec,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = "meta",
) -> SVDQW4A4Linear:
    """An empty SVDQ layer shaped by ``layer``; its tensors are filled by the framework's weight loader.

    ``torch_dtype`` (bf16 or fp16) is the dtype of the floating tensors. A one-element scale the file carries
    (``wtscale``) becomes a persistent float32 buffer, so that a generic state-dict loader binds it too.
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
    for scalar in SVDQ_CHECKPOINT_SCALARS:
        if scalar in layer.tensors:
            delattr(module, scalar)  # the plain attribute set by __init__
            shape = layer.tensors[scalar][0]
            module.register_buffer(scalar, torch.ones(shape, dtype=torch.float32, device=device), persistent=True)
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


# --- helpers -------------------------------------------------------------------------------------

_DTYPES = {  # safetensors dtype names
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
_LOADABLE_FLOATS = (torch.bfloat16, torch.float16, torch.float32)  # cast to the layer's dtype on load


def _dtype(name: str, key: str) -> torch.dtype:
    try:
        return _DTYPES[name]
    except KeyError:
        raise ValueError(f"Tensor {key!r} has unsupported dtype {name!r}.") from None


def _layer_spec(
    name: str,
    precision: str,
    group_size: int,
    has_bias: bool,
    rotation: str | None,
    tensors: dict[str, TensorInfo],
) -> LayerSpec:
    """Check one target's tensors against the SVDQ layer they will load into."""

    tensors = {s: info for s, info in tensors.items() if s not in SVDQ_LEGACY_CHECKPOINT_SUFFIXES}
    try:
        in_features, out_features, rank = SVDQW4A4Linear.checkpoint_dims({s: shape for s, (shape, _) in tensors.items()})
    except KeyError as exc:
        raise ValueError(f"Target {name!r} is missing tensors [{exc.args[0]!r}].") from None
    except ValueError as exc:
        raise ValueError(f"Target {name!r}: {exc}.") from None
    try:
        reference = SVDQW4A4Linear(
            in_features, out_features, rank=rank, bias=has_bias, precision=precision, device="meta",
            input_rotation=rotation,
        )
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"Target {name!r}: {exc}") from None
    if group_size != reference.group_size:
        raise ValueError(f"Target {name!r}: {precision} needs group_size {reference.group_size}, got {group_size}.")

    expected = {
        suffix: (tuple(t.shape), t.dtype)
        for suffix, t in itertools.chain(reference.named_parameters(recurse=False), reference.named_buffers(recurse=False))
    }
    scalars = {s for s in SVDQ_CHECKPOINT_SCALARS if getattr(reference, s, None) is not None}
    missing = sorted(set(expected) - set(tensors))
    if missing:
        raise ValueError(f"Target {name!r} is missing tensors {missing}.")
    extra = sorted(set(tensors) - set(expected) - scalars)
    if extra:
        raise ValueError(f"Target {name!r} has unexpected tensors {extra}.")
    for suffix, (shape, dtype) in expected.items():
        actual_shape, actual_dtype = tensors[suffix]
        loadable = actual_dtype in _LOADABLE_FLOATS if dtype in _LOADABLE_FLOATS else actual_dtype == dtype
        if tuple(actual_shape) != shape or not loadable:
            raise ValueError(f"Target {name!r}: {suffix} is {tuple(actual_shape)} {actual_dtype}, but the layer's "
                             f"{suffix} is {shape} {dtype}.")
    for suffix in scalars & set(tensors):
        shape, dtype = tensors[suffix]
        if int(torch.Size(shape).numel()) != 1 or dtype not in _LOADABLE_FLOATS:
            raise ValueError(f"Target {name!r}: {suffix} must be one floating element, got {shape} {dtype}.")
    return LayerSpec(
        name=name,
        precision=precision,
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        group_size=group_size,
        has_bias=has_bias,
        has_wtscale=any(s in tensors for s in SVDQ_CHECKPOINT_SCALARS),
        input_rotation=rotation,
        tensors=tensors,
    )


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
