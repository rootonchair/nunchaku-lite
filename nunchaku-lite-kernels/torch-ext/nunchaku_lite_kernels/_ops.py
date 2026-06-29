from __future__ import annotations

import importlib
import pkgutil
import sys

import torch


_DISPATCH_NAMESPACE = "nunchaku_lite_kernels"


def add_op_namespace_prefix(op_name: str) -> str:
    return f"{_DISPATCH_NAMESPACE}::{op_name}"


def _load_extension_module():
    try:
        from . import _C

        return _C
    except ImportError:
        package = sys.modules[__package__]
        for module_info in pkgutil.iter_modules(package.__path__):
            if module_info.name.startswith("_nunchaku_lite_kernels_cuda"):
                return importlib.import_module(f"{__package__}.{module_info.name}")
        raise


_extension_module = _load_extension_module()
ops = getattr(torch.ops, _DISPATCH_NAMESPACE)


@torch.library.register_fake(add_op_namespace_prefix("gemm_w4a4"))
def _fake_gemm_w4a4(
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
    act_unsigned: bool,
    lora_scales: list[float],
    fuse_silu: bool,
    fp4: bool,
    alpha,
    wcscales,
    out_q,
    out_k,
    out_v,
    attn_tokens: int,
) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("quantize_w4a4_act_fuse_lora"))
def _fake_quantize_w4a4_act_fuse_lora(
    input,
    output,
    oscales,
    lora_down,
    lora_act_out,
    smooth,
    fuse_glu: bool,
    fp4: bool,
) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("fused_rms_norm_modulate"))
def _fake_fused_rms_norm_modulate(x, norm_weight, scale, shift, eps: float):
    return torch.empty_like(x)


@torch.library.register_fake(add_op_namespace_prefix("fused_affine_modulate"))
def _fake_fused_affine_modulate(x, scale, shift):
    return torch.empty_like(x)


@torch.library.register_fake(add_op_namespace_prefix("awq_gemm_w4a16_g128_int16"))
def _fake_awq_gemm_w4a16_g128_int16(in_feats, qweight, scaling_factors, zeros):
    return in_feats.new_empty((*in_feats.shape[:-1], qweight.shape[0] * 4))


@torch.library.register_fake(add_op_namespace_prefix("awq_gemm_w4a16_g64_int32"))
def _fake_awq_gemm_w4a16_g64_int32(in_feats, qweight, scaling_factors, zeros):
    return in_feats.new_empty((*in_feats.shape[:-1], qweight.shape[0] * 4))


@torch.library.register_fake(add_op_namespace_prefix("gemv_awq"))
def _fake_gemv_awq(in_feats, qweight, scaling_factors, zeros, m: int, n: int, k: int, group_size: int):
    return in_feats.new_empty((m, n))


@torch.library.register_fake(add_op_namespace_prefix("attention_fp16"))
def _fake_attention_fp16(q, k, v, o, scale: float) -> None:
    return None


@torch.library.register_fake(add_op_namespace_prefix("fused_cross_head_qk_norm_rope"))
def _fake_fused_cross_head_qk_norm_rope(
    q,
    k,
    q_weight,
    k_weight,
    q_cos,
    q_sin,
    k_cos,
    k_sin,
    q_heads: int,
    k_heads: int,
    head_dim: int,
    eps: float,
    interleaved: bool,
):
    return [torch.empty_like(q), torch.empty_like(k)]
