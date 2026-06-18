"""Attention dispatch helpers for Nunchaku Lite adapter attention calls."""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn

_ACTIVE_LITE_BACKEND: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "nunchaku_lite_attention_backend",
    default=None,
)

QUERY_LAYOUTS = frozenset({"bshd", "bhsd"})
OUTPUT_LAYOUTS = frozenset({"bshd", "bhsd", "bs_flat"})


def _normalize_backend(backend: str | None) -> str | None:
    if backend is None:
        return None
    return str(backend).lower()


def _effective_backend(backend: str | None) -> str | None:
    return _normalize_backend(backend) if backend is not None else _ACTIVE_LITE_BACKEND.get()


@contextlib.contextmanager
def lite_attention_backend(backend: str | None = None):
    """Set the active Nunchaku Lite attention backend for adapter dispatch calls."""

    normalized = _normalize_backend(backend)
    token = _ACTIVE_LITE_BACKEND.set(normalized)
    try:
        yield
    finally:
        _ACTIVE_LITE_BACKEND.reset(token)


def dispatch_lite_attention_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    attention_kwargs: dict[str, Any] | None = None,
    *,
    backend: str | None = None,
    parallel_config: Any | None = None,
    query_layout: str = "bshd",
    output_layout: str = "bshd",
) -> torch.Tensor:
    """Dispatch attention through Diffusers with local adapter backend scoping."""

    _validate_layout_args(query_layout, output_layout)
    effective_backend = _effective_backend(backend)
    if _is_sageattention_backend(effective_backend) and _requires_native_fallback_for_masked_attention(
        attn_mask, parallel_config
    ):
        effective_backend = "native"

    return _dispatch_diffusers_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
        attention_kwargs=attention_kwargs,
        backend=effective_backend,
        parallel_config=parallel_config,
        query_layout=query_layout,
        output_layout=output_layout,
    )


def _dispatch_diffusers_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    enable_gqa: bool,
    attention_kwargs: dict[str, Any] | None,
    backend: str | None,
    parallel_config: Any | None,
    query_layout: str,
    output_layout: str,
) -> torch.Tensor:
    if query_layout != "bshd":
        query = _convert_attention_layout(query, query_layout, "bshd")
        key = _convert_attention_layout(key, query_layout, "bshd")
        value = _convert_attention_layout(value, query_layout, "bshd")
    output = dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
        attention_kwargs=attention_kwargs,
        backend=backend,
        parallel_config=parallel_config,
    )
    return _convert_attention_layout(output, "bshd", output_layout)


def _is_sageattention_backend(backend: str | None) -> bool:
    normalized = _normalize_backend(backend)
    return normalized is not None and "sage" in normalized


def _requires_native_fallback_for_masked_attention(
    attn_mask: torch.Tensor | None,
    parallel_config: Any | None,
) -> bool:
    return attn_mask is not None or parallel_config is not None


def _validate_layout_args(query_layout: str, output_layout: str) -> None:
    if query_layout not in QUERY_LAYOUTS:
        raise ValueError(f"query_layout must be one of {sorted(QUERY_LAYOUTS)}; got {query_layout!r}.")
    if output_layout not in OUTPUT_LAYOUTS:
        raise ValueError(f"output_layout must be one of {sorted(OUTPUT_LAYOUTS)}; got {output_layout!r}.")


def _convert_attention_layout(tensor: torch.Tensor, source_layout: str, target_layout: str) -> torch.Tensor:
    if target_layout == source_layout:
        return tensor
    if target_layout == "bs_flat":
        if source_layout == "bshd":
            return tensor.flatten(2, 3)
        if source_layout == "bhsd":
            return tensor.transpose(1, 2).reshape(tensor.shape[0], tensor.shape[2], tensor.shape[1] * tensor.shape[3])
    if source_layout == "bs_flat":
        raise ValueError("Cannot convert from flattened attention layout without head metadata.")
    if source_layout == "bshd" and target_layout == "bhsd":
        return tensor.transpose(1, 2)
    if source_layout == "bhsd" and target_layout == "bshd":
        return tensor.transpose(1, 2).contiguous()
    raise ValueError(f"Unsupported attention layout conversion: {source_layout!r} to {target_layout!r}.")
