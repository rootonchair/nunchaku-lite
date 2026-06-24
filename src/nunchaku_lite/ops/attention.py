"""Attention kernel wrappers exposed by the active backend."""

import torch

from .backend import get_ops


def attention_fp16_cuda(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, scale: float) -> None:
    """Run native fp16 attention into a preallocated output tensor.

    Args:
        q: Query tensor in the native packed attention layout.
        k: Key tensor in the native packed attention layout.
        v: Value tensor in the native packed attention layout.
        o: Output tensor to fill.
        scale: Attention scale factor, typically ``head_dim ** -0.5``.

    Returns:
        None.
    """

    get_ops().attention_fp16(q, k, v, o, scale)
