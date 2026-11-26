"""Data-free weight-only INT4 (W4A16) for large Hugging Face text encoders, e.g. LTX-2.5's Gemma 4 12B.

Each selected ``nn.Linear`` is replaced by :class:`~nunchaku_lite.linear.AWQW4A16Linear` (Nunchaku AWQ group-64 int32
layout, served by the native AWQ GEMM/GEMV kernels). Weights are quantized round-to-nearest per group of 64 input
channels with an asymmetric (scale, zero) pair; for each group the clipping range is chosen from a few shrink ratios by
the reconstruction error of the bf16-stored scale/zero. No calibration data, so quantization runs at load time in
seconds on a GPU and needs no separate checkpoint.

Usage::

    from transformers import AutoModelForImageTextToText
    from nunchaku_lite.text_encoders.w4a16 import quantize_text_encoder_w4a16_

    te = AutoModelForImageTextToText.from_pretrained(path, torch_dtype=torch.bfloat16)  # on the CPU
    quantize_text_encoder_w4a16_(te, device="cuda")  # decoder-layer linears -> W4A16 on the GPU
    te.to("cuda")  # the remaining (bf16) modules
"""

from __future__ import annotations

import re
from collections.abc import Callable

import torch
from torch import nn

from ..linear import AWQW4A16Linear

GROUP_SIZE = 64
CLIP_RATIOS = (1.0, 0.95, 0.9, 0.85, 0.8)
# decoder-layer projections of the text model; vision/audio towers, embeddings and lm_head stay dense
DEFAULT_INCLUDE = re.compile(r"(^|\.)language_model\.layers\.\d+\.")


def _channel_maps(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    channel = torch.arange(64, device=device)
    return (channel // 32) * 4 + (channel % 8) // 2, ((channel % 32) // 8) + 4 * (channel % 2)


@torch.no_grad()
def pack_awq_g64(codes: torch.Tensor) -> torch.Tensor:
    """Unsigned 4-bit codes ``(N, K)`` -> Nunchaku AWQ group-64 int32 ``qweight`` ``(N / 4, K / 2)``, on ``codes``'s
    device (same layout as ``nunchaku_lite.ops.mps._awq_g64_pack``)."""
    n, k = codes.shape
    groups = k // 64
    c = codes.to(torch.int64).view(n // 4, 4, groups, 64).permute(0, 2, 1, 3)
    packed_index, nibble = _channel_maps(codes.device)
    words = torch.zeros(n // 4, groups, 4, 8, dtype=torch.int64, device=codes.device)
    words.index_add_(-1, packed_index, c << (4 * nibble))  # disjoint nibbles: add == or
    words = torch.where(words >= 2**31, words - 2**32, words)
    return words.to(torch.int32).view(n // 4, groups * 32)


@torch.no_grad()
def quantize_weight_w4a16(
    weight: torch.Tensor, group_size: int = GROUP_SIZE, dtype: torch.dtype = torch.bfloat16, clip_ratios=CLIP_RATIOS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``weight (N, K)`` -> ``(codes (N, K) uint8, scales (K/g, N), zeros (K/g, N))`` with ``w ~= code * scale + zero``.

    Scales and zeros are rounded to ``dtype`` before choosing codes, so the error that picks the clip ratio is the
    error the kernel will actually see.
    """
    n, k = weight.shape
    if k % group_size:
        raise ValueError(f"in_features {k} is not a multiple of group_size {group_size}")
    w = weight.float().reshape(n, k // group_size, group_size)
    wmin, wmax = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
    best = None
    for ratio in clip_ratios:
        lo = (wmin * ratio).to(dtype).float()
        scale = ((wmax * ratio - lo) / 15).clamp_min(1e-8).to(dtype).float()
        codes = ((w - lo) / scale).round().clamp_(0, 15)
        err = (codes * scale + lo - w).pow_(2).sum(-1, keepdim=True)
        if best is None:
            best = [err, codes, scale, lo]
        else:
            better = err < best[0]
            best[0] = torch.where(better, err, best[0])
            best[1] = torch.where(better, codes, best[1])
            best[2] = torch.where(better, scale, best[2])
            best[3] = torch.where(better, lo, best[3])
    _, codes, scale, lo = best
    return (
        codes.to(torch.uint8).reshape(n, k),
        scale.squeeze(-1).t().to(dtype).contiguous(),
        lo.squeeze(-1).t().to(dtype).contiguous(),
    )


def _eligible(name: str, module: nn.Module, include: Callable[[str, nn.Module], bool]) -> bool:
    return (
        type(module) is nn.Linear
        and module.in_features % GROUP_SIZE == 0
        and module.out_features % 4 == 0
        and include(name, module)
    )


@torch.no_grad()
def quantize_text_encoder_w4a16_(
    model: nn.Module,
    device: str | torch.device = "cuda",
    include: Callable[[str, nn.Module], bool] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Replace eligible linears of ``model`` in place with W4A16 modules on ``device``; returns a summary.

    Each layer is quantized on ``device`` and its dense weight dropped right after, so peak extra memory is one dense
    layer. ``include(name, module)`` selects layers (default: decoder-layer linears of a ``language_model``). Other
    modules are left where they are; move the model to ``device`` afterwards.
    """
    include = include or (lambda name, _m: bool(DEFAULT_INCLUDE.search(name)))
    device = torch.device(device)
    targets = [(n, m) for n, m in model.named_modules() if _eligible(n, m, include)]
    dense_bytes = packed_bytes = 0
    for name, linear in targets:
        weight = linear.weight.detach().to(device)
        codes, scales, zeros = quantize_weight_w4a16(weight, dtype=dtype)
        q = AWQW4A16Linear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            group_size=GROUP_SIZE,
            torch_dtype=dtype,
            device=device,
        )
        q.qweight.data = pack_awq_g64(codes)
        q.wscales.data, q.wzeros.data = scales, zeros
        if linear.bias is not None:
            q.bias.data = linear.bias.detach().to(device, dtype)
        parent_name, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_name) if parent_name else model, child, q)
        dense_bytes += linear.weight.numel() * linear.weight.element_size()
        packed_bytes += sum(t.numel() * t.element_size() for t in (q.qweight, q.wscales, q.wzeros))
        del weight, codes, linear
    return {"layers": len(targets), "dense_bytes": dense_bytes, "packed_bytes": packed_bytes}
