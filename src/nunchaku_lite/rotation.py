"""Load-time resolution of online activation rotations declared by runtime manifests.

A rotated ``svdq_w4a4`` layer carries an authoritative ``<prefix>.input_transform``
tensor (the orthogonal blocks ``T``; the packed weight is ``Q(R @ blockdiag(T))``) plus
rotation metadata, declared either per target in a runtime manifest
(``op_options.rotation``) or, for checkpoints loaded through model-specific adapters,
once in ``quantization_config["input_rotation"]``. The kernel never inspects the tensor:
the loader decides once, here, which native transform matches it and binds the
corresponding quantize op to the layer. Today the only native transform is the
blockwise Sylvester Hadamard ``H64 / 8``.
"""

from __future__ import annotations

from typing import Any

import torch

from torch import nn

from .manifest import ROTATION_FORMAT_VERSION, RuntimeManifestTarget, _validate_rotation
from .ops.quantize import HADAMARD_BLOCK_SIZE, hadamard_matrix

INPUT_TRANSFORM_SUFFIX = ".input_transform"
ROTATION_DECLARATION_KEY = "input_rotation"
HADAMARD64 = "hadamard64"
_HADAMARD_ATOL = 1e-6
# Set by adapters on the fc2 of an MLP run through ops.fused.fused_gelu_mlp, whose input is
# quantized inside fc1's GEMM epilogue (no rotation there).
FUSED_MLP_FC2_ATTR = "_nunchaku_lite_fused_mlp_fc2"


def resolve_input_rotation(
    target: RuntimeManifestTarget,
    checkpoint_state: dict[str, torch.Tensor],
    *,
    allow_fp4: bool = False,
) -> str | None:
    """Return the native input rotation for ``target`` and drop its transform tensor.

    Args:
        target: Parsed manifest target.
        checkpoint_state: Mutable checkpoint state dict. The target's
            ``input_transform`` tensor is removed from it, since it is metadata for
            the loader and not a module parameter.
        allow_fp4: Opt in to rotation on NVFP4 targets. Off by default because the
            64-wide rotation measured slightly worse than no rotation for NVFP4.

    Returns:
        ``None`` for an unrotated target, or ``"hadamard64"``.

    Raises:
        ValueError: If the transform tensor is missing or malformed, or the target
            is NVFP4 without ``allow_fp4``.
        NotImplementedError: If the transform is valid but no native kernel path
            implements it (anything other than shared ``H64 / 8``).
    """

    rotation = target.rotation
    key = target.checkpoint_prefix + INPUT_TRANSFORM_SUFFIX
    if rotation is None:
        if key in checkpoint_state:
            raise ValueError(f"Checkpoint has {key!r} but target {target.checkpoint_prefix!r} declares no rotation.")
        return None

    transform = checkpoint_state.pop(key, None)
    if transform is None:
        raise ValueError(f"Rotated target {target.checkpoint_prefix!r} is missing its {key!r} tensor.")
    return _resolve_transform(
        target.checkpoint_prefix, rotation, transform, is_fp4=target.precision == "fp4", allow_fp4=allow_fp4
    )


def parse_rotation_declaration(quantization_config: dict[str, Any]) -> dict[str, Any] | None:
    """Parse ``quantization_config["input_rotation"]`` for non-manifest checkpoints.

    The declaration applies to every layer that has an ``.input_transform`` tensor:
    ``{"format_version": 2, "type": "block_matrix", "block_size": 64, "shared": True,
    "generator": "hadamard"}`` (``generator`` optional).

    Returns:
        The rotation metadata without ``format_version``, or ``None`` if absent.

    Raises:
        ValueError: If the declaration is malformed or has an unsupported version.
    """

    raw = quantization_config.get(ROTATION_DECLARATION_KEY)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError(f"quantization_config.{ROTATION_DECLARATION_KEY} must be a JSON object.")
    version = raw.get("format_version")
    if version != ROTATION_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported quantization_config.{ROTATION_DECLARATION_KEY}.format_version {version!r}; "
            f"expected {ROTATION_FORMAT_VERSION}."
        )
    rotation = {key: value for key, value in raw.items() if key != "format_version"}
    _validate_rotation(f"quantization_config.{ROTATION_DECLARATION_KEY}", "svdq_w4a4", rotation)
    return rotation


def bind_input_rotations(
    transformer: nn.Module,
    checkpoint_state: dict[str, torch.Tensor],
    rotation: dict[str, Any],
    *,
    allow_fp4: bool = False,
) -> list[str]:
    """Bind the declared rotation to every layer that has an ``.input_transform`` tensor.

    Used after a model-specific adapter has patched the transformer and renamed the
    checkpoint keys to module paths. Each transform tensor is validated, removed from
    ``checkpoint_state``, and turned into ``input_rotation`` on its layer.

    Returns:
        The module paths that were bound.

    Raises:
        ValueError: If a transform belongs to no SVDQ layer, to a layer whose input is
            unsigned or quantized inside a fused MLP epilogue, or is malformed.
        NotImplementedError: For valid transforms without a native kernel path.
    """

    from .linear import SVDQW4A4Linear

    bound = []
    for key in [key for key in checkpoint_state if key.endswith(INPUT_TRANSFORM_SUFFIX)]:
        prefix = key[: -len(INPUT_TRANSFORM_SUFFIX)]
        try:
            module = transformer.get_submodule(prefix)
        except AttributeError as exc:
            raise ValueError(f"Rotation tensor {key!r} does not belong to any module of the patched model.") from exc
        if not isinstance(module, SVDQW4A4Linear):
            raise ValueError(f"Rotation tensor {key!r} belongs to {type(module).__name__}, not an SVDQ W4A4 layer.")
        if module.act_unsigned or getattr(module, FUSED_MLP_FC2_ATTR, False):
            raise ValueError(
                f"Layer {prefix!r} quantizes unsigned or fused-MLP activations, which cannot be rotated "
                "(rotation needs the signed standalone quantize path)."
            )
        kind = _resolve_transform(
            prefix, rotation, checkpoint_state.pop(key), is_fp4=module.precision == "nvfp4", allow_fp4=allow_fp4
        )
        module.input_rotation = kind
        bound.append(prefix)
    return bound


def _resolve_transform(
    name: str, rotation: dict[str, Any], transform: torch.Tensor, *, is_fp4: bool, allow_fp4: bool
) -> str:
    if is_fp4 and not allow_fp4:
        raise ValueError(
            f"Rotated layer {name!r} is NVFP4. Rotation on NVFP4 is refused by default "
            "(it measured slightly worse than no rotation); pass adapter_options={'allow_fp4_rotation': True} "
            "to load it anyway."
        )
    block_size = rotation["block_size"]
    _validate_transform_shape(name, rotation, transform, block_size)
    if _is_shared_hadamard64(rotation, transform):
        return HADAMARD64
    if rotation.get("generator") == "hadamard":
        raise ValueError(
            f"Rotated layer {name!r} declares generator='hadamard' but its input_transform "
            f"is not the shared Sylvester H{block_size}/sqrt({block_size}) this runtime implements."
        )
    raise NotImplementedError(
        f"Rotated layer {name!r} uses a general block_matrix transform "
        f"(block_size={block_size}, shared={rotation['shared']}); only the shared Sylvester H64/8 "
        "has a native kernel path."
    )


def _validate_transform_shape(name: str, rotation: dict[str, Any], transform: torch.Tensor, block_size: int) -> None:
    shape = tuple(transform.shape)
    if len(shape) == 2:
        expected = (block_size, block_size)
    elif len(shape) == 3:
        expected = (shape[0], block_size, block_size)
    else:
        expected = None
    if expected is None or shape != expected:
        raise ValueError(
            f"Rotated layer {name!r} input_transform has shape {shape}; expected "
            f"[{block_size}, {block_size}] (shared) or [K/{block_size}, {block_size}, {block_size}]."
        )
    if (len(shape) == 2) != bool(rotation["shared"]):
        raise ValueError(
            f"Rotated layer {name!r} input_transform shape {shape} does not match rotation.shared={rotation['shared']}."
        )


def _is_shared_hadamard64(rotation: dict[str, Any], transform: torch.Tensor) -> bool:
    if not rotation["shared"] or rotation["block_size"] != HADAMARD_BLOCK_SIZE or transform.dim() != 2:
        return False
    reference = hadamard_matrix(HADAMARD_BLOCK_SIZE, dtype=torch.float64)
    return torch.allclose(transform.detach().to("cpu", torch.float64), reference, rtol=0.0, atol=_HADAMARD_ATOL)
