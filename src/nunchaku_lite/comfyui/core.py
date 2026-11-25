"""Load a Nunchaku Lite checkpoint into ComfyUI's native model class for its family.

Flow (:func:`load_comfy_diffusion_model`):

1. Read the checkpoint and its ``quantization_config``; resolve precision and rank.
2. Select the :class:`ComfyFamilyAdapter` that recognizes the checkpoint.
3. The family splits the checkpoint into ComfyUI-named dense tensors, tensors for the quantized modules it will
   install, and a detection view (dense tensors plus meta stand-ins with the dense shapes of quantized layers).
4. ComfyUI detects the model config from the detection view and builds its model on the meta device.
5. The family installs Nunchaku Lite modules into ``model.diffusion_model`` by module name.
6. Every tensor is loaded with ``assign=True``. The load is exact: an unexpected key, a missing key or a
   parameter/buffer left on the meta device is an error, never a silently random weight.
7. The model is wrapped in a ComfyUI ``ModelPatcher``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from ..adapters.common import SVDQPatchContext, drop_obsolete_svdq_tensors
from ..core import (
    _check_rotation_support,
    _infer_svdq_rank,
    _precision_from_quantization_config,
    _validate_quantization_compatibility,
)
from ..manifest import parse_runtime_manifest
from ..rotation_loader import bind_input_rotations, refuse_legacy_transform_tensors
from ..utils import convert_fp16, load_state_dict_in_safetensors, patch_scale_key

logger = logging.getLogger(__name__)


@dataclass
class ComfyLoadPlan:
    """ComfyUI-side view of one checkpoint, produced by a family adapter.

    Attributes:
        dense: Tensors for ComfyUI's own modules, keyed by ``diffusion_model`` state-dict names.
        quant: Tensors for the modules the family installs, keyed by their state-dict names under
            ``diffusion_model`` after :meth:`ComfyFamilyAdapter.install`.
        detect: Keys ComfyUI's model detection reads: ``dense`` plus meta stand-ins carrying the dense
            ``(out, in)`` weight shapes of the quantized layers.
        detect_metadata: Optional safetensors-style metadata passed to ComfyUI's detection.
        modules: Family-specific description of the modules to install, keyed by module path.
    """

    dense: dict[str, torch.Tensor]
    quant: dict[str, torch.Tensor]
    detect: dict[str, torch.Tensor]
    detect_metadata: dict[str, str] | None = None
    modules: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _RotationAdapterView:
    """The two adapter attributes the shared rotation checks read."""

    target: str
    supports_input_rotation: bool


class ComfyFamilyAdapter(Protocol):
    """Maps one Nunchaku Lite checkpoint family onto a ComfyUI model class.

    ``plan`` must not touch ComfyUI's model; ``install`` mutates ``diffusion_model`` (built on the meta device)
    so that its state-dict keys are exactly ``plan.dense | plan.quant``.

    A family with ``supports_input_rotation`` must map every tensor stored under a quantized layer's checkpoint
    prefix into ``plan.quant`` under the installed module's path (as :func:`.common.module_tensors` does), so the
    rotation placeholders follow, and must run those layers on the signed standalone quantize path.
    """

    name: str
    supports_input_rotation: bool

    def matches(self, state_dict: dict[str, torch.Tensor], metadata: dict[str, str]) -> bool: ...

    def plan(
        self, state_dict: dict[str, torch.Tensor], metadata: dict[str, str], context: SVDQPatchContext
    ) -> ComfyLoadPlan: ...

    def install(self, diffusion_model: nn.Module, plan: ComfyLoadPlan, context: SVDQPatchContext) -> None: ...


_FAMILIES: dict[str, ComfyFamilyAdapter] = {}
_BUILTINS_LOADED = False


def register_comfy_family(family: ComfyFamilyAdapter) -> None:
    """Register a ComfyUI family adapter under its ``name``."""

    if not getattr(family, "name", None):
        raise ValueError("ComfyUI family adapter must define a non-empty 'name'.")
    _FAMILIES[family.name] = family


def list_comfy_families() -> list[str]:
    """Return the names of the registered ComfyUI family adapters."""

    _ensure_builtin_families()
    return sorted(_FAMILIES)


def _ensure_builtin_families() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from . import flux, qwen_image, sdxl, z_image  # noqa: F401  (register on import)

    _BUILTINS_LOADED = True


def select_comfy_family(state_dict: dict[str, torch.Tensor], metadata: dict[str, str]) -> ComfyFamilyAdapter:
    """Return the single registered family adapter that recognizes the checkpoint."""

    _ensure_builtin_families()
    matches = [family for family in _FAMILIES.values() if family.matches(state_dict, metadata)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            "No Nunchaku Lite ComfyUI family recognizes this checkpoint. "
            f"Supported families: {', '.join(list_comfy_families())}."
        )
    names = ", ".join(sorted(family.name for family in matches))
    raise ValueError(f"Several Nunchaku Lite ComfyUI families match this checkpoint: {names}.")


def resolve_checkpoint_precision(state_dict: dict[str, torch.Tensor], quantization_config: dict[str, Any]) -> str:
    """Return the native precision (``"int4"`` or ``"nvfp4"``) the checkpoint was exported for.

    The declared precision (``quantization_config``) and the tensors must agree: NVFP4 weight scales are
    ``float8_e4m3fn``, INT4 weight scales are floating point.
    """

    scale_dtypes = {
        value.dtype
        for key, value in state_dict.items()
        if key.endswith(".wscales") and f"{key[: -len('.wscales')]}.wzeros" not in state_dict
    }
    if not scale_dtypes:
        raise ValueError("Checkpoint has no SVDQ W4A4 layers (no '.wscales' tensors without '.wzeros').")
    tensor_precisions = {"nvfp4" if dtype == torch.float8_e4m3fn else "int4" for dtype in scale_dtypes}
    if len(tensor_precisions) != 1:
        raise ValueError(f"Checkpoint mixes INT4 and NVFP4 SVDQ layers: {sorted(tensor_precisions)}.")
    tensor_precision = tensor_precisions.pop()
    declared = _precision_from_quantization_config(quantization_config)
    if declared is not None and declared != tensor_precision:
        raise ValueError(f"quantization_config declares {declared}, but the checkpoint tensors are {tensor_precision}.")
    return tensor_precision


def _parse_quantization_config(metadata: dict[str, str] | None) -> dict[str, Any]:
    raw = (metadata or {}).get("quantization_config")
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise TypeError("quantization_config metadata must be a JSON object.")
    return parsed


def build_comfy_context(
    state_dict: dict[str, torch.Tensor],
    quantization_config: dict[str, Any],
    torch_dtype: torch.dtype,
) -> SVDQPatchContext:
    """Resolve precision, rank and runtime dtype for a ComfyUI load."""

    precision = resolve_checkpoint_precision(state_dict, quantization_config)
    # Nunchaku v1 checkpoints name the low-rank down projection ``lora_down`` and carry no rank metadata.
    lowrank = {
        (key[: -len(".lora_down")] + ".proj_down" if key.endswith(".lora_down") else key): value
        for key, value in state_dict.items()
        if key.endswith((".proj_down", ".lora_down"))
    }
    rank = quantization_config.get("rank") or _infer_svdq_rank(lowrank)
    if not rank:
        raise ValueError("Could not determine the SVDQ low-rank rank from the checkpoint.")
    if torch_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"Nunchaku Lite layers run in bfloat16 or float16, not {torch_dtype}.")
    return SVDQPatchContext(
        precision=precision, rank=int(rank), torch_dtype=torch_dtype, requested_torch_dtype=torch_dtype
    )


def load_state_exactly(diffusion_model: nn.Module, tensors: dict[str, torch.Tensor]) -> None:
    """Assign ``tensors`` to ``diffusion_model`` and verify every parameter and buffer was provided.

    Floating-point tensors are cast to the dtype of the (meta) parameter they replace, so ComfyUI's dtype
    choice for dense modules and the context dtype for quantized modules both hold after loading.
    """

    expected = diffusion_model.state_dict(keep_vars=True)
    unexpected = sorted(set(tensors) - set(expected))
    missing = sorted(set(expected) - set(tensors))
    if unexpected or missing:
        raise ValueError(
            "Checkpoint does not match the ComfyUI model built for it: "
            f"{len(missing)} missing key(s) {missing[:8]}, {len(unexpected)} unexpected key(s) {unexpected[:8]}."
        )
    cast = {}
    for key, value in tensors.items():
        target = expected[key]
        if value.dtype != target.dtype:
            if not (value.is_floating_point() and target.is_floating_point()):
                raise ValueError(f"{key}: checkpoint dtype {value.dtype} cannot load into {target.dtype}.")
            value = value.to(target.dtype)
        if value.shape != target.shape:
            raise ValueError(f"{key}: checkpoint shape {tuple(value.shape)} != model shape {tuple(target.shape)}.")
        cast[key] = value
    diffusion_model.load_state_dict(cast, strict=True, assign=True)
    left_on_meta = [name for name, tensor in _named_tensors(diffusion_model) if tensor.is_meta]
    if left_on_meta:
        raise ValueError(f"Parameters or buffers left uninitialized on the meta device: {left_on_meta[:8]}.")


def _named_tensors(module: nn.Module):
    yield from module.named_parameters()
    yield from module.named_buffers()


def load_comfy_diffusion_model(
    checkpoint: str | Path,
    *,
    model_options: dict[str, Any] | None = None,
    load_device: torch.device | None = None,
    offload_device: torch.device | None = None,
    disable_dynamic: bool = True,
):
    """Load a Nunchaku Lite checkpoint as a ComfyUI ``ModelPatcher``.

    Args:
        checkpoint: Local safetensors path (or Hugging Face ``org/repo/file.safetensors`` path).
        model_options: ComfyUI model options; ``dtype`` overrides the compute dtype, and ``allow_fp4_rotation``
            loads Hadamard-rotated NVFP4 checkpoints, which are refused by default.
        load_device: Device to run on; defaults to ComfyUI's torch device.
        offload_device: Device to offload to; defaults to ComfyUI's unet offload device.
        disable_dynamic: Accepted for ComfyUI's ``cached_patcher_init`` factory signature. Nunchaku Lite layers
            always load through the classic (non-dynamic) ``ModelPatcher``.

    Returns:
        ``comfy.model_patcher.ModelPatcher`` wrapping ComfyUI's native model for the checkpoint's family.
    """

    import comfy.model_detection
    import comfy.model_management
    import comfy.model_patcher

    model_options = model_options or {}
    state_dict, metadata = load_state_dict_in_safetensors(checkpoint, return_metadata=True)
    metadata = metadata or {}
    quantization_config = _parse_quantization_config(metadata)
    refuse_legacy_transform_tensors(state_dict.keys())
    family = select_comfy_family(state_dict, metadata)

    load_device = torch.device(load_device or comfy.model_management.get_torch_device())
    if load_device.type == "cuda" and load_device.index is None:
        # ComfyUI matches loaded models by device equality, and torch.device("cuda") != torch.device("cuda:0"):
        # without an index, unload_all_models()/free_memory() would never unload this model.
        load_device = torch.device("cuda", torch.cuda.current_device())
    offload_device = offload_device or comfy.model_management.unet_offload_device()
    precision = resolve_checkpoint_precision(state_dict, quantization_config)
    _validate_quantization_compatibility(
        {"weight": {"dtype": "int4" if precision == "int4" else "fp4_e2m1_all"}}, precision, load_device, checkpoint
    )
    # The Diffusers loader's rotation decision table, unchanged: it refuses inconsistent declarations, families
    # without rotation support and devices without the Hadamard op, and marks the rotated targets with placeholder
    # keys. Families carry the placeholders through their key mapping into ``plan.quant`` like any other tensor
    # of a quantized layer, so they arrive at ComfyUI module paths and are bound after ``install``.
    rotation_targets = _check_rotation_support(
        _RotationAdapterView(f"comfyui:{family.name}", family.supports_input_rotation),
        None,
        state_dict,
        quantization_config,
        parse_runtime_manifest(quantization_config) if "runtime_manifest" in quantization_config else None,
        precision=precision,
        device=load_device,
        allow_fp4=bool(model_options.get("allow_fp4_rotation", False)),
    )

    # The dtype decision needs the model config, which needs the detection view, which needs the context only
    # for module shapes, not dtypes; plan with bf16 and re-derive the dtype-dependent context afterwards.
    plan = family.plan(state_dict, metadata, build_comfy_context(state_dict, quantization_config, torch.bfloat16))
    model_config = comfy.model_detection.model_config_from_unet(plan.detect, "", metadata=plan.detect_metadata)
    if model_config is None:
        raise ValueError(f"ComfyUI could not detect a model config for this {family.name} checkpoint.")
    supported = [dtype for dtype in model_config.supported_inference_dtypes if dtype in (torch.bfloat16, torch.float16)]
    unet_dtype = model_options.get("dtype") or comfy.model_management.unet_dtype(
        device=load_device, supported_dtypes=supported or [torch.bfloat16, torch.float16]
    )
    if unet_dtype not in (torch.bfloat16, torch.float16):
        unet_dtype = torch.bfloat16
    manual_cast_dtype = comfy.model_management.unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype, device=load_device)
    context = build_comfy_context(state_dict, quantization_config, unet_dtype)

    model = model_config.get_model(plan.detect, "", device=torch.device("meta"))
    diffusion_model = model.diffusion_model
    family.install(diffusion_model, plan, context)
    if rotation_targets is not None:
        bind_input_rotations(diffusion_model, plan.quant, rotation_targets)

    tensors = dict(model_config.process_unet_state_dict(dict(plan.dense)))
    tensors.update(plan.quant)
    drop_obsolete_svdq_tensors(tensors)
    patch_scale_key(diffusion_model, tensors)
    if context.torch_dtype == torch.float16:
        convert_fp16(diffusion_model, tensors)
    load_state_exactly(diffusion_model, tensors)
    from ..ops.mps import configure_backend

    configure_backend(diffusion_model, "auto")

    diffusion_model._nunchaku_lite_modules = plan.modules  # installed module path -> spec (incl. checkpoint source)
    model._nunchaku_lite_family = family.name
    model._nunchaku_lite_precision = context.precision
    model._nunchaku_lite_quantization_config = quantization_config
    del disable_dynamic
    patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
    patcher.cached_patcher_init = (load_comfy_diffusion_model, (checkpoint, model_options))
    if not comfy.model_management.is_device_cpu(offload_device):
        model.to(offload_device)
    return patcher
