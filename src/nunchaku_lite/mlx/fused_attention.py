"""Fused attention prologue for mflux attention modules: shared helpers and the registry.

mflux computes the q/k RMSNorm and RoPE of attention layers with reshapes, float32 casts and stacks that
``mx.compile`` cannot fuse, and passes strided q/k/v views to attention. On FLUX.2-klein-4B at 1024² that costs
about as much as the attention itself. The fused path does the prologue in one kernel
(:func:`~nunchaku_lite.mlx.kernels.qk_norm_rope`, head dim 128, interleaved-pair RoPE) reading q/k/v straight from
the projection output.

Nothing here is specific to a model family. Each adapter registers a call function per mflux attention class with
:func:`register_fused_attention` (see ``pipeline.py``); :func:`install_fused_attention` swaps the registered modules
of a loaded transformer for subclasses that route ``__call__`` through it. A call function receives
``(module, base_class, *args, **kwargs)`` and falls back to ``base_class.__call__`` for anything the kernel does not
cover.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import mlx.core as mx

from .kernels import qk_norm_rope

__all__ = [
    "HEAD_DIM",
    "IMAGE_QKV",
    "TEXT_QKV",
    "attention",
    "install_fused_attention",
    "joint_qkv",
    "qk_norm_rope",
    "qkv_rows",
    "register_fused_attention",
    "tables_fit",
]

HEAD_DIM = 128
IMAGE_QKV = ("to_q", "to_k", "to_v")
TEXT_QKV = ("add_q_proj", "add_k_proj", "add_v_proj")
_CALLS: dict[tuple[str, str], Callable] = {}


def register_fused_attention(family: str, class_name: str, call: Callable) -> None:
    """Route mflux attention class ``mflux.models.<family>...<class_name>`` (exact class, not subclasses) through
    ``call(module, base_class, *args, **kwargs)``."""

    _CALLS[(family, class_name)] = call


def install_fused_attention(transformer: Any) -> int:
    """Route the transformer's registered attention modules (head dim 128) through their fused call. Returns how
    many were patched (0 when nothing is registered for the model); calling it again patches nothing."""

    patched = 0
    for _, module in transformer.named_modules():
        call = _CALLS.get(_class_key(type(module)))
        if call is None or getattr(type(module), "_nl_fused", False) or _head_dim(module) != HEAD_DIM:
            continue
        module.__class__ = _fused_class(type(module), call)
        patched += 1
    return patched


def _class_key(cls: type) -> tuple[str, str] | None:
    # Exact mflux classes only: subclasses (e.g. FLUX.1 concept attention) have other signatures.
    parts = cls.__module__.split(".")
    return (parts[2], cls.__name__) if len(parts) > 2 and parts[:2] == ["mflux", "models"] else None


def _head_dim(attn) -> int | None:
    for name in ("dim_head", "head_dimension", "head_dim"):
        value = getattr(attn, name, None)
        if isinstance(value, int):
            return value
    return None


@functools.cache
def _fused_class(base: type, call: Callable) -> type:
    def __call__(self, *args, **kwargs):
        return call(self, base, *args, **kwargs)

    return type(base.__name__, (base,), {"__call__": __call__, "_nl_fused": True, "__module__": base.__module__})


def attention(q: mx.array, k: mx.array, v: mx.array, mask: mx.array | None = None) -> mx.array:
    """``(1, H, L, 128)`` q/k/v -> ``(1, L, H * 128)``."""

    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5, mask=mask)
    heads, length = out.shape[1], out.shape[2]
    return out.transpose(0, 2, 1, 3).reshape(1, length, heads * HEAD_DIM)


def qkv_rows(attn, x: mx.array, names: tuple[str, str, str], width: int) -> tuple[mx.array, tuple[int, int, int]]:
    """Projection rows ``(L, C)`` holding q, k, v and their column offsets, without slicing copies when the three
    projections are slices of one shared (checkpoint-fused) layer."""

    from .pipeline import ProjectionSlice

    mods = [getattr(attn, n) for n in names]
    if all(isinstance(m, ProjectionSlice) for m in mods):
        shared = [m.get("shared") if "shared" in m else m._shared for m in mods]
        if all(s is shared[0] for s in shared):
            out = shared[0].layer(x)
            return out.reshape(-1, out.shape[-1]), tuple(m._start for m in mods)
    out = mx.concatenate([m(x) for m in mods], axis=-1)
    return out.reshape(-1, out.shape[-1]), (0, width, 2 * width)


def joint_qkv(attn, x, context, width, image_names=IMAGE_QKV, text_names=TEXT_QKV):
    """Image and text projection rows for joint attention, ``(image, text, offsets)``, or ``None`` when their
    layouts differ."""

    img, offs = qkv_rows(attn, x, image_names, width)
    txt, txt_offs = qkv_rows(attn, context, text_names, width)
    return (img, txt, offs) if offs == txt_offs and img.dtype == txt.dtype else None


def tables_fit(cos: mx.array, sin: mx.array, rows: int) -> bool:
    """Whether ``cos`` / ``sin`` are per-pair rotary tables ``(rows, 64)``."""

    return cos.shape == (rows, HEAD_DIM // 2) and sin.shape == (rows, HEAD_DIM // 2)
