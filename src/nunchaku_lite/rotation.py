"""Load-time resolution of online activation rotations declared by runtime manifests.

A rotated ``svdq_w4a4`` target carries ``op_options.rotation`` metadata and an
authoritative ``<checkpoint_prefix>.input_transform`` tensor (the orthogonal blocks
``T``; the packed weight is ``Q(R @ blockdiag(T))``). The kernel never inspects that
tensor: the loader decides once, here, which native transform matches it and binds
the corresponding quantize op to the layer. Today the only native transform is the
blockwise Sylvester Hadamard ``H64 / 8``.
"""

from __future__ import annotations

from typing import Any

import torch

from .manifest import RuntimeManifestTarget
from .ops.quantize import HADAMARD_BLOCK_SIZE, hadamard_matrix

INPUT_TRANSFORM_SUFFIX = ".input_transform"
HADAMARD64 = "hadamard64"
_HADAMARD_ATOL = 1e-6


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
    if target.precision == "fp4" and not allow_fp4:
        raise ValueError(
            f"Rotated target {target.checkpoint_prefix!r} is NVFP4. Rotation on NVFP4 is refused by default "
            "(it measured slightly worse than no rotation); pass adapter_options={'allow_fp4_rotation': True} "
            "to load it anyway."
        )

    block_size = rotation["block_size"]
    _validate_transform_shape(target, transform, block_size)
    if _is_shared_hadamard64(rotation, transform):
        return HADAMARD64
    if rotation.get("generator") == "hadamard":
        raise ValueError(
            f"Rotated target {target.checkpoint_prefix!r} declares generator='hadamard' but its input_transform "
            f"is not the shared Sylvester H{block_size}/sqrt({block_size}) this runtime implements."
        )
    raise NotImplementedError(
        f"Rotated target {target.checkpoint_prefix!r} uses a general block_matrix transform "
        f"(block_size={block_size}, shared={rotation['shared']}); only the shared Sylvester H64/8 "
        "has a native kernel path."
    )


def _validate_transform_shape(target: RuntimeManifestTarget, transform: torch.Tensor, block_size: int) -> None:
    shape = tuple(transform.shape)
    if len(shape) == 2:
        expected = (block_size, block_size)
    elif len(shape) == 3:
        expected = (shape[0], block_size, block_size)
    else:
        expected = None
    if expected is None or shape != expected:
        raise ValueError(
            f"Rotated target {target.checkpoint_prefix!r} input_transform has shape {shape}; expected "
            f"[{block_size}, {block_size}] (shared) or [K/{block_size}, {block_size}, {block_size}]."
        )
    if (len(shape) == 2) != bool(target.rotation["shared"]):
        raise ValueError(
            f"Rotated target {target.checkpoint_prefix!r} input_transform shape {shape} does not match "
            f"rotation.shared={target.rotation['shared']}."
        )


def _is_shared_hadamard64(rotation: dict[str, Any], transform: torch.Tensor) -> bool:
    if not rotation["shared"] or rotation["block_size"] != HADAMARD_BLOCK_SIZE or transform.dim() != 2:
        return False
    reference = hadamard_matrix(HADAMARD_BLOCK_SIZE, dtype=torch.float64)
    return torch.allclose(transform.detach().to("cpu", torch.float64), reference, rtol=0.0, atol=_HADAMARD_ATOL)
