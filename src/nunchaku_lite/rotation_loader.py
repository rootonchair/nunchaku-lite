"""Load-time binding of online activation rotations declared by checkpoints.

A rotated ``svdq_w4a4`` layer was exported with its residual weight packed as
``Q(R @ blockdiag(H64 / 8))`` and must run the Hadamard quantize op. Checkpoints declare it by
name, ``{"type": "hadamard"}`` (defined in :mod:`nunchaku_lite.manifest`):

* per target in a ``nunchaku_format_version`` 2 runtime manifest stored in the checkpoint; or
* for checkpoints without a manifest, in ``quantization_config["input_rotation"]`` listing the
  rotated targets, together with the empty ``nunchaku_lite.input_rotation`` marker tensor.

Both make loaders that predate rotation refuse the checkpoint. The manifest adapter binds
manifest rotations itself; every other case is bound by :func:`bind_input_rotations`.

The kernel never decides anything at run time: the loader binds the op per layer, once, and
refuses anything it cannot honour before the model is modified wherever possible.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn

from .manifest import (
    LEGACY_INPUT_TRANSFORM_SUFFIX,
    ROTATION_DECLARATION_KEY,
    ROTATION_MARKER_KEY,
    RuntimeManifest,
    RuntimeManifestTarget,
    parse_rotation_declaration,
)

HADAMARD = "hadamard"
HADAMARD_OP_NAME = "quantize_w4a4_act_hadamard_fuse_lora"
# Placeholder keys that carry declared targets through an adapter's checkpoint-key renaming.
_MARKER_SUFFIX = "._nunchaku_lite_input_rotation"

__all__ = [
    "HADAMARD",
    "ROTATION_DECLARATION_KEY",
    "bind_input_rotations",
    "check_rotation_runtime",
    "mark_rotated_targets",
    "parse_rotation_declaration",
    "refuse_legacy_transform_tensors",
    "resolve_input_rotation",
    "resolve_rotated_targets",
    "validate_declared_targets",
]


def resolve_rotated_targets(
    quantization_config: dict,
    *,
    has_marker: bool,
    stored_manifest: RuntimeManifest | None,
) -> tuple[str, ...] | None:
    """The checkpoint prefixes declared rotated, or ``None`` when the checkpoint declares no rotation.

    Rotation is declared per target in a ``nunchaku_format_version`` 2 runtime manifest stored in the checkpoint
    (``stored_manifest``; never one synthesized from target lists), or by the model-level ``input_rotation``
    declaration together with the ``ROTATION_MARKER_KEY`` tensor. Both make loaders that predate rotation refuse the
    checkpoint; when a manifest and a declaration are both present they must agree.
    """

    declared = parse_rotation_declaration(quantization_config)
    manifest_rotated = (
        tuple(target.checkpoint_prefix for target in stored_manifest.rotated_targets) if stored_manifest else ()
    )
    if has_marker and declared is None:
        raise ValueError(
            f"Checkpoint has the {ROTATION_MARKER_KEY!r} rotation marker but no quantization_config.input_rotation "
            "declaration listing the rotated targets."
        )
    if declared is None and not manifest_rotated:
        return None
    if stored_manifest is not None and declared is not None and set(declared) != set(manifest_rotated):
        if not manifest_rotated:
            raise ValueError(
                "Checkpoint declares quantization_config.input_rotation but its runtime_manifest declares no "
                "rotation; checkpoints with a manifest must declare rotation per target in op_options.rotation."
            )
        raise ValueError(
            "quantization_config.input_rotation.targets do not match the runtime_manifest targets that declare "
            "op_options.rotation; the checkpoint metadata is inconsistent."
        )
    if not manifest_rotated and not has_marker:
        raise ValueError(
            "Checkpoint declares quantization_config.input_rotation without a runtime_manifest "
            f"(nunchaku_format_version 2) declaring the rotation or the {ROTATION_MARKER_KEY!r} marker tensor. "
            "One of them is required, so that older Nunchaku Lite versions refuse the checkpoint instead of "
            "loading it without rotation."
        )
    return manifest_rotated or declared


def resolve_input_rotation(target: RuntimeManifestTarget, *, allow_fp4: bool = False) -> str | None:
    """Return the runtime rotation for a manifest target (``None`` or ``"hadamard"``).

    Raises:
        ValueError: If the target is NVFP4 and ``allow_fp4`` is not set.
    """

    if target.rotation is None:
        return None
    _check_fp4(target.checkpoint_prefix, is_fp4=target.precision == "fp4", allow_fp4=allow_fp4)
    return HADAMARD


def refuse_legacy_transform_tensors(checkpoint_keys: Iterable[str]) -> None:
    """Refuse ``.input_transform`` tensors of the abandoned pre-release ``block_matrix`` format."""

    legacy = [key for key in checkpoint_keys if key.endswith(LEGACY_INPUT_TRANSFORM_SUFFIX)]
    if legacy:
        raise ValueError(
            f"Checkpoint contains {len(legacy)} .input_transform tensors (e.g. {legacy[0]!r}) from the pre-release "
            "'block_matrix' rotation format. Convert it: drop those tensors and declare the rotation by name, "
            "{'type': 'hadamard'} (the packed weights are unchanged)."
        )


def check_rotation_runtime(device: torch.device | str | None) -> None:
    """Refuse rotated checkpoints the target device cannot run, at load time.

    Raises:
        NotImplementedError: On MPS (its kernels apply no rotation), or on CUDA when the
            installed kernels predate the Hadamard quantize op.
    """

    device_type = torch.device(device).type if device is not None else None
    if device_type == "mps":
        raise NotImplementedError(
            "Rotated (hadamard) SVDQ checkpoints are not supported on MPS yet; use an unrotated checkpoint."
        )
    if device_type == "cuda":
        from .ops.backend import get_ops

        if getattr(get_ops(), HADAMARD_OP_NAME, None) is None:
            raise NotImplementedError(
                f"The installed Nunchaku Lite kernels have no {HADAMARD_OP_NAME} op, which rotated checkpoints "
                'need; update the kernels or run "pip install ./nunchaku-lite-kernels".'
            )


def validate_declared_targets(
    checkpoint_state: dict[str, torch.Tensor],
    targets: Iterable[str],
    *,
    is_fp4: bool,
    allow_fp4: bool = False,
) -> None:
    """Check rotated targets against the checkpoint, before any model is modified.

    Raises:
        ValueError: If a rotated target has no packed SVDQ weight in the checkpoint (or is an
            AWQ layer), or the checkpoint is NVFP4 without ``allow_fp4``.
    """

    awq = [target for target in targets if f"{target}.wzeros" in checkpoint_state]
    if awq:
        raise ValueError(f"Rotated targets {awq[:3]} are AWQ W4A16 layers; only SVDQ W4A4 layers can be rotated.")
    missing = [target for target in targets if f"{target}.qweight" not in checkpoint_state]
    if missing:
        raise ValueError(
            f"Rotated targets {missing[:3]} have no packed SVDQ weight (<target>.qweight) in the checkpoint."
        )
    _check_fp4("Rotated checkpoint", is_fp4=is_fp4, allow_fp4=allow_fp4)


def mark_rotated_targets(checkpoint_state: dict[str, torch.Tensor], targets: Iterable[str]) -> None:
    """Add placeholder keys so the rotated targets follow the adapter's key renaming.

    :func:`bind_input_rotations` consumes them after patching.

    Raises:
        ValueError: If the checkpoint already contains placeholder keys, which would be bound
            as if they had been declared.
    """

    stray = [key for key in checkpoint_state if key.endswith(_MARKER_SUFFIX)]
    if stray:
        raise ValueError(f"Checkpoint contains reserved rotation placeholder keys (e.g. {stray[0]!r}).")
    for target in targets:
        checkpoint_state[target + _MARKER_SUFFIX] = torch.empty(0)


def bind_input_rotations(
    transformer: nn.Module, checkpoint_state: dict[str, torch.Tensor], expected_targets: Sequence[str]
) -> list[str]:
    """Bind ``"hadamard"`` to every layer marked by :func:`mark_rotated_targets`, all or nothing.

    Used after an adapter has patched the transformer and renamed the checkpoint keys (markers
    included) to module paths. The markers are removed from ``checkpoint_state``. Every marked
    layer is resolved and checked first, and the count is compared with ``expected_targets``;
    only if all of that passes is any layer bound, so a failure binds nothing.

    Returns:
        The module paths that were bound.

    Raises:
        ValueError: If the number of markers found differs from ``expected_targets`` (the
            adapter dropped, renamed away or duplicated some), a marked target maps to no
            module, or to a layer whose input is unsigned or quantized inside a fused MLP
            epilogue.
        TypeError: If a marked target maps to a module that is not an SVDQ W4A4 layer.
    """

    from .linear import SVDQW4A4Linear
    from .ops.fused import FUSED_MLP_FC2_ATTR

    paths = [key[: -len(_MARKER_SUFFIX)] for key in checkpoint_state if key.endswith(_MARKER_SUFFIX)]
    for path in paths:
        checkpoint_state.pop(path + _MARKER_SUFFIX)
    if len(paths) != len(expected_targets):
        what = "lost" if len(paths) < len(expected_targets) else "duplicated"
        raise ValueError(
            f"Found {len(paths)} of {len(expected_targets)} rotated targets after patching; the adapter {what} some, "
            "and loading them with the wrong rotation would give wrong outputs."
        )

    modules = []
    for path in paths:
        try:
            module = transformer.get_submodule(path)
        except AttributeError as exc:
            raise ValueError(f"Rotated target {path!r} is not a module of the patched model.") from exc
        if not isinstance(module, SVDQW4A4Linear):
            raise TypeError(f"Rotated target {path!r} is a {type(module).__name__}, not an SVDQ W4A4 layer.")
        if module.act_unsigned or getattr(module, FUSED_MLP_FC2_ATTR, False):
            raise ValueError(
                f"Layer {path!r} quantizes unsigned or fused-MLP activations, which cannot be rotated "
                "(rotation needs the signed standalone quantize path)."
            )
        modules.append(module)
    for module in modules:
        module.input_rotation = HADAMARD
    return paths


def _check_fp4(where: str, *, is_fp4: bool, allow_fp4: bool) -> None:
    if is_fp4 and not allow_fp4:
        raise ValueError(
            f"{where}: rotation on NVFP4 is refused by default (it measured slightly worse than no rotation); "
            "pass adapter_options={'allow_fp4_rotation': True} to load it anyway."
        )
