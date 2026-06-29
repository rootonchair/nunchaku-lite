import math

import torch

from ._ops import ops

gemm_w4a4 = ops.gemm_w4a4
quantize_w4a4_act_fuse_lora = ops.quantize_w4a4_act_fuse_lora
fused_rms_norm_modulate = ops.fused_rms_norm_modulate
fused_affine_modulate = ops.fused_affine_modulate
fused_cross_head_qk_norm_rope = ops.fused_cross_head_qk_norm_rope
gemv_awq = ops.gemv_awq
attention_fp16 = ops.attention_fp16


def _ceil_divide(a: int, b: int) -> int:
    return -(-a // b)


def svdq_gemm_w4a4_cuda(
    act: torch.Tensor,
    wgt: torch.Tensor,
    out: torch.Tensor | None = None,
    qout: torch.Tensor | None = None,
    ascales: torch.Tensor | None = None,
    wscales: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    poolout: torch.Tensor | None = None,
    lora_act_in: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    norm_q: torch.Tensor | None = None,
    norm_k: torch.Tensor | None = None,
    rotary_emb: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    smooth_factor: torch.Tensor | None = None,
    out_vk: torch.Tensor | None = None,
    out_linearattn: torch.Tensor | None = None,
    act_unsigned: bool = False,
    lora_scales: list[float] | None = None,
    fuse_silu: bool = False,
    fp4: bool = False,
    alpha: float | None = 1.0,
    wcscales: torch.Tensor | None = None,
    out_q: torch.Tensor | None = None,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
    attn_tokens: int = 0,
) -> None:
    if lora_scales is None:
        if lora_up is None:
            lora_scales = []
        else:
            rank = lora_up.shape[1]
            lora_scales = [1.0] * math.ceil(rank / 16)
    if isinstance(alpha, torch.Tensor):
        alpha_arg = alpha
    elif alpha is None or float(alpha) == 1.0:
        alpha_arg = None
    else:
        alpha_arg = torch.tensor(float(alpha), device=act.device)
    ops.gemm_w4a4(
        act,
        wgt,
        out,
        qout,
        ascales,
        wscales,
        oscales,
        poolout,
        lora_act_in,
        lora_up,
        lora_down,
        lora_act_out,
        norm_q,
        norm_k,
        rotary_emb,
        bias,
        smooth_factor,
        out_vk,
        out_linearattn,
        act_unsigned,
        lora_scales,
        fuse_silu,
        fp4,
        alpha_arg,
        wcscales,
        out_q,
        out_k,
        out_v,
        attn_tokens,
    )


def svdq_quantize_w4a4_act_fuse_lora_cuda(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    fuse_glu: bool = False,
    fp4: bool = False,
    pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if lora_down is None:
        raise ValueError("lora_down is required")

    batch_size, channels = input.shape
    rank = lora_down.shape[1]
    batch_size_pad = _ceil_divide(batch_size, pad_size) * pad_size
    if output is None:
        output = torch.empty(batch_size_pad, channels // 2, dtype=torch.uint8, device=input.device)
    if oscales is None:
        if fp4:
            if channels % 16 != 0:
                raise ValueError("NVFP4 activation channels must be divisible by 16")
            oscales = torch.empty(channels // 16, batch_size_pad, dtype=torch.float8_e4m3fn, device=input.device)
        else:
            if channels % 64 != 0:
                raise ValueError("INT4 activation channels must be divisible by 64")
            oscales = torch.empty(channels // 64, batch_size_pad, dtype=input.dtype, device=input.device)
    if lora_act_out is None:
        lora_act_out = torch.empty(batch_size_pad, rank, dtype=torch.float32, device=input.device)

    ops.quantize_w4a4_act_fuse_lora(input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4)
    return output, oscales, lora_act_out


def awq_gemm_w4a16_g128_int16(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
) -> torch.Tensor:
    return ops.awq_gemm_w4a16_g128_int16(in_feats, kernel, scaling_factors, zeros)


def awq_gemm_w4a16_g64_int32(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
) -> torch.Tensor:
    return ops.awq_gemm_w4a16_g64_int32(in_feats, kernel, scaling_factors, zeros)


def awq_gemv_w4a16_cuda(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    scaling_factors: torch.Tensor,
    zeros: torch.Tensor,
    m: int,
    n: int,
    k: int,
    group_size: int = 64,
) -> torch.Tensor:
    return ops.gemv_awq(in_feats, kernel, scaling_factors, zeros, m, n, k, group_size)


def attention_fp16_cuda(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, scale: float) -> None:
    ops.attention_fp16(q, k, v, o, scale)


__all__ = [
    "attention_fp16_cuda",
    "attention_fp16",
    "awq_gemm_w4a16_g128_int16",
    "awq_gemm_w4a16_g64_int32",
    "awq_gemv_w4a16_cuda",
    "fused_affine_modulate",
    "fused_cross_head_qk_norm_rope",
    "fused_rms_norm_modulate",
    "gemm_w4a4",
    "gemv_awq",
    "quantize_w4a4_act_fuse_lora",
    "svdq_gemm_w4a4_cuda",
    "svdq_quantize_w4a4_act_fuse_lora_cuda",
]
