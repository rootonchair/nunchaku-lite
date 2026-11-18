"""Fused MPS q/k norm + RoPE for attention processors (Apple silicon): shared helpers and the processor registry.

Before attention, many diffusion transformers RMS-normalize each q/k head and apply an interleaved-pair rotary
embedding. On MPS this module runs both, plus the head layout, in one kernel
(:func:`~nunchaku_lite.ops.mps.qk_norm_rope_mps`, head dim 128) reading q/k/v straight from the projections.

Nothing here is specific to a model family. Each adapter module defines fused subclasses of its family's Diffusers
processors and registers them with :func:`register_mps_processor`; :func:`install_mps_fused_processors` swaps the
registered stock processors of a loaded transformer (manifest checkpoints keep Diffusers' processors). A fused
processor checks :func:`fusable` before projecting and otherwise runs the stock code unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn

from ..ops.fused import _rms_norm_eps, _rms_norm_weight
from ..ops.mps import _derived, qk_norm_rope_mps, qk_norm_rope_mps_available

__all__ = [
    "complex_rope_tables",
    "fusable",
    "fused_attention",
    "fused_qk_norm_rope",
    "install_mps_fused_processors",
    "real_rope_tables",
    "register_mps_processor",
]

_QKV = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_HEAD_DIM = 128
_REPLACEMENTS: dict[type, type] = {}


def register_mps_processor(stock: type, fused: type) -> None:
    """Swap Diffusers processor class ``stock`` for ``fused`` (a subclass with the fused prologue) on MPS."""

    _REPLACEMENTS[stock] = fused


def install_mps_fused_processors(transformer: torch.nn.Module) -> int:
    """Swap the transformer's registered stock attention processors for their fused-MPS subclasses. Returns how many
    modules were updated (0 when already installed or when nothing is registered for the model)."""

    patched = 0
    for module in transformer.modules():
        processor = getattr(module, "processor", None)
        replacement = _REPLACEMENTS.get(type(processor))
        if replacement is None or not hasattr(module, "set_processor"):
            continue
        new = replacement()
        new._attention_backend = processor._attention_backend
        new._parallel_config = processor._parallel_config
        module.set_processor(new)
        patched += 1
    return patched


def fusable(hidden_states: torch.Tensor, head_dim: int, norms: Sequence) -> bool:
    """Whether the fused kernel applies: MPS, batch 1, fp16/bf16, head dim 128, q/k norms present (the kernel always
    normalizes) with ``(128,)`` or no affine weights. Check before projecting, so a fallback computes nothing twice."""

    if hidden_states.device.type != "mps" or hidden_states.shape[0] != 1 or head_dim != _HEAD_DIM:
        return False
    if hidden_states.dtype not in (torch.float16, torch.bfloat16) or any(norm is None for norm in norms):
        return False
    if any(w is not None and w.shape != (_HEAD_DIM,) for w in map(_rms_norm_weight, norms)):
        return False
    return qk_norm_rope_mps_available()


def fused_qk_norm_rope(
    image: _QKV,
    norms: Sequence,
    tables: tuple[torch.Tensor, torch.Tensor],
    heads: int,
    text: _QKV | None = None,
) -> _QKV:
    """q/k RMSNorm + RoPE + ``(1, L, heads, 128)`` layout in one MPS kernel (call after :func:`fusable`).

    Args:
        image: Image-token ``(q, k, v)`` projections ``(1, Li, heads * 128)`` (column slices of one fused
            projection are fine).
        norms: ``(norm_q, norm_k)`` for the image rows, followed by ``(norm_added_q, norm_added_k)`` with ``text``.
        tables: Rotary ``(cos, sin)`` for all rows, text first (see :func:`~nunchaku_lite.ops.mps.qk_norm_rope_mps`).
        heads: Number of attention heads.
        text: Optional text-token ``(q, k, v)``, placed before the image rows.
    """

    weights = [_rms_norm_weight(n) for n in norms]
    return qk_norm_rope_mps(
        *image, weights[0], weights[1], *tables, heads,
        eps_q=_rms_norm_eps(norms[0]), eps_k=_rms_norm_eps(norms[1]),
        text=text,
        text_q_weight=weights[2] if text is not None else None,
        text_k_weight=weights[3] if text is not None else None,
    )  # fmt: skip


def fused_attention(
    processor,
    image: _QKV,
    norms: Sequence,
    tables: tuple[torch.Tensor, torch.Tensor],
    heads: int,
    text: _QKV | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """:func:`fused_qk_norm_rope`, then the processor's attention backend; returns ``(1, Lt + Li, heads * 128)``."""

    query, key, value = fused_qk_norm_rope(image, norms, tables, heads, text=text)
    out = dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=mask,
        backend=getattr(processor, "_attention_backend", None),
        parallel_config=getattr(processor, "_parallel_config", None),
    )
    return out.flatten(2, 3).to(query.dtype)


def real_rope_tables(rotary, rows: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Diffusers' real ``(cos, sin)`` tables ``(rows, 128)`` (each pair's value repeated), or ``None``."""

    if not (isinstance(rotary, tuple) and len(rotary) == 2):
        return None
    if any(not torch.is_tensor(t) or tuple(t.shape) != (rows, _HEAD_DIM) for t in rotary):
        return None
    return rotary


def complex_rope_tables(freqs: Sequence[torch.Tensor], rows: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Per-pair ``(cos, sin)`` float32 tables ``(rows, 64)`` from complex rotary tables ``(L_i, 64)``, concatenated
    in order and cached while the tables live; ``None`` for other layouts."""

    if any(not torch.is_tensor(f) or not f.is_complex() or f.ndim != 2 or f.shape[-1] != _HEAD_DIM // 2 for f in freqs):
        return None
    if sum(f.shape[0] for f in freqs) != rows:
        return None
    head, rest = freqs[0], freqs[1:]
    key = "complex_rope:" + ",".join(f"{id(f)}.{f._version}.{f.shape[0]}" for f in rest)

    def build(_):
        table = torch.cat(list(freqs)) if rest else head
        return table.real.float().contiguous(), table.imag.float().contiguous()

    return _derived(head, key, build)
