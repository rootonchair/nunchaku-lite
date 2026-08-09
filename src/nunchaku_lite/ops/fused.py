"""Higher-level fused operations assembled from native quantization and GEMM kernels."""

import torch
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from torch.nn import RMSNorm

from ..linear import SVDQW4A4Linear
from ..utils import ceil_divide
from .backend import get_ops
from .gemm import svdq_gemm_w4a4_cuda


def fused_gelu_mlp(x: torch.Tensor, fc1: SVDQW4A4Linear, fc2: SVDQW4A4Linear, pad_size: int = 256) -> torch.Tensor:
    """Run a two-layer GELU MLP while keeping the intermediate activation quantized.

    Args:
        x: Input tensor with shape ``(batch, sequence, channels)``.
        fc1: First quantized linear projection, wrapped by Diffusers GELU.
        fc2: Second quantized linear projection.
        pad_size: Token padding multiple for the intermediate quantized buffer.

    Returns:
        MLP output with shape ``(batch, sequence, fc2.out_features)``.
    """

    batch_size, seq_len, channels = x.shape
    x = x.view(batch_size * seq_len, channels)
    quantized_x, ascales, lora_act = fc1.quantize(x)

    batch_size_pad = ceil_divide(batch_size * seq_len, pad_size) * pad_size
    qout_act = torch.empty(batch_size_pad, fc1.out_features // 2, dtype=torch.uint8, device=x.device)
    if fc2.precision == "nvfp4":
        qout_ascales = torch.empty(fc1.out_features // 16, batch_size_pad, dtype=torch.float8_e4m3fn, device=x.device)
    else:
        qout_ascales = torch.empty(fc1.out_features // 64, batch_size_pad, dtype=x.dtype, device=x.device)
    qout_lora_act = torch.empty(batch_size_pad, fc2.proj_down.shape[1], dtype=torch.float32, device=x.device)

    svdq_gemm_w4a4_cuda(
        act=quantized_x,
        wgt=fc1.qweight,
        qout=qout_act,
        ascales=ascales,
        wscales=fc1.wscales,
        oscales=qout_ascales,
        lora_act_in=lora_act,
        lora_up=fc1.proj_up,
        lora_down=fc2.proj_down,
        lora_act_out=qout_lora_act,
        bias=fc1.bias,
        smooth_factor=fc2.smooth_factor,
        fp4=fc1.precision == "nvfp4",
        alpha=fc1.wtscale,
        wcscales=fc1.wcscales,
    )
    output = torch.empty(batch_size * seq_len, fc2.out_features, dtype=x.dtype, device=x.device)
    output = fc2.forward_quant(qout_act, qout_ascales, qout_lora_act, output=output)
    return output.view(batch_size, seq_len, -1)


def fused_qkv_norm_rotary(
    x: torch.Tensor,
    proj: SVDQW4A4Linear,
    norm_q: RMSNorm | DiffusersRMSNorm | None = None,
    norm_k: RMSNorm | DiffusersRMSNorm | None = None,
    rotary_emb: torch.Tensor | None = None,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    attn_tokens: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run QKV projection with fused Q/K normalization and packed RoPE.

    Args:
        x: Input tensor with shape ``(batch, sequence, channels)``.
        proj: Fused SVDQ QKV projection module.
        norm_q: Optional query RMSNorm module.
        norm_k: Optional key RMSNorm module.
        rotary_emb: Optional packed rotary embedding consumed by the native
            GEMM kernel.
        output: Optional dense output buffer, or a tuple of preallocated
            ``(query, key, value)`` tensors for packed attention.
        attn_tokens: Number of unpadded tokens when ``output`` is a Q/K/V
            tuple.

    Returns:
        Dense fused QKV tensor when ``output`` is not a tuple, otherwise the
        populated ``(query, key, value)`` tuple.
    """

    batch_size, seq_len, channels = x.shape
    x = x.view(batch_size * seq_len, channels)
    quantized_x, ascales, lora_act = proj.quantize(x)

    if output is None:
        output = torch.empty(batch_size * seq_len, proj.out_features, dtype=x.dtype, device=x.device)

    norm_q_weight = norm_q.weight if norm_q is not None else None
    norm_k_weight = norm_k.weight if norm_k is not None else None

    if isinstance(output, tuple):
        out_q, out_k, out_v = output
        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=proj.qweight,
            ascales=ascales,
            wscales=proj.wscales,
            lora_act_in=lora_act,
            lora_up=proj.proj_up,
            bias=proj.bias,
            fp4=proj.precision == "nvfp4",
            alpha=proj.wtscale,
            wcscales=proj.wcscales,
            norm_q=norm_q_weight,
            norm_k=norm_k_weight,
            rotary_emb=rotary_emb,
            out_q=out_q,
            out_k=out_k,
            out_v=out_v,
            attn_tokens=attn_tokens,
        )
        return out_q, out_k, out_v

    svdq_gemm_w4a4_cuda(
        act=quantized_x,
        wgt=proj.qweight,
        out=output,
        ascales=ascales,
        wscales=proj.wscales,
        lora_act_in=lora_act,
        lora_up=proj.proj_up,
        bias=proj.bias,
        fp4=proj.precision == "nvfp4",
        alpha=proj.wtscale,
        wcscales=proj.wcscales,
        norm_q=norm_q_weight,
        norm_k=norm_k_weight,
        rotary_emb=rotary_emb,
    )
    return output.view(batch_size, seq_len, -1)


def _rms_norm_eps(norm: RMSNorm | DiffusersRMSNorm | None) -> float:
    if norm is None:
        return 1e-6
    eps = getattr(norm, "eps", None)
    if eps is None:
        eps = getattr(norm, "variance_epsilon", None)
    return float(eps) if eps is not None else 1e-6


def _rms_norm_weight(norm: RMSNorm | DiffusersRMSNorm | None) -> torch.Tensor | None:
    if norm is None:
        return None
    return getattr(norm, "weight", None)


def _broadcast_shape_supported_for_modulation(x: torch.Tensor, value: torch.Tensor) -> bool:
    if value.ndim == 3 and value.shape[1] == 1:
        value = value.squeeze(1)
    return value.numel() in (x.shape[-1], x.shape[0] * x.shape[-1], x.numel())


def _torch_broadcast_modulation_param(x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3 and value.ndim == 2 and value.shape == (x.shape[0], x.shape[-1]):
        return value.unsqueeze(1)
    return value


def fused_rms_norm_modulate(
    x: torch.Tensor,
    norm: RMSNorm | DiffusersRMSNorm | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """Apply ``norm(x) * (1 + scale) + shift`` with a guarded CUDA fast path."""

    weight = _rms_norm_weight(norm)
    eps = _rms_norm_eps(norm)
    native_scale = scale.squeeze(1) if scale.ndim == 3 and scale.shape[1] == 1 else scale
    native_shift = shift.squeeze(1) if shift.ndim == 3 and shift.shape[1] == 1 else shift
    if (
        x.device.type == "cuda"
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.ndim == 3
        and native_scale.dtype == x.dtype
        and native_shift.dtype == x.dtype
        and native_scale.device == x.device
        and native_shift.device == x.device
        and native_scale.numel() in (x.shape[-1], x.shape[0] * x.shape[-1])
        and native_shift.numel() in (x.shape[-1], x.shape[0] * x.shape[-1])
        and not torch.is_grad_enabled()
    ):
        try:
            return get_ops().fused_rms_norm_modulate(
                x.contiguous(),
                weight,
                native_scale.contiguous(),
                native_shift.contiguous(),
                eps,
            )
        except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError):
            pass
    if norm is None:
        normalized = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=None, eps=eps)
    else:
        normalized = norm(x)
    scale = _torch_broadcast_modulation_param(x, scale)
    shift = _torch_broadcast_modulation_param(x, shift)
    return normalized * (1 + scale) + shift


def fused_affine_modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Apply ``x * (1 + scale) + shift`` with a guarded CUDA fast path."""

    native_scale = scale.squeeze(1) if scale.ndim == 3 and scale.shape[1] == 1 else scale
    native_shift = shift.squeeze(1) if shift.ndim == 3 and shift.shape[1] == 1 else shift
    if (
        x.device.type == "cuda"
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.ndim == 3
        and native_scale.dtype == x.dtype
        and native_shift.dtype == x.dtype
        and native_scale.device == x.device
        and native_shift.device == x.device
        and _broadcast_shape_supported_for_modulation(x, native_scale)
        and _broadcast_shape_supported_for_modulation(x, native_shift)
        and not torch.is_grad_enabled()
    ):
        try:
            return get_ops().fused_affine_modulate(
                x.contiguous(), native_scale.contiguous(), native_shift.contiguous()
            )
        except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError):
            pass
    scale = _torch_broadcast_modulation_param(x, scale)
    shift = _torch_broadcast_modulation_param(x, shift)
    return x * (1 + scale) + shift


def _cross_head_weight(
    norm: RMSNorm | DiffusersRMSNorm | None,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    weight = _rms_norm_weight(norm)
    if weight is None:
        return None
    if int(weight.numel()) != int(channels):
        raise ValueError("cross-head norm weight must match the full Q/K channel dimension")
    if weight.dtype not in (dtype, torch.float32):
        weight = weight.to(dtype=dtype)
    return weight.to(device=device).contiguous()


def _cached_cross_head_rope_contiguous(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    target_device = torch.device(device)
    if x.device == target_device and x.dtype == torch.float32 and x.is_contiguous():
        return x

    cache_key = (str(target_device), torch.float32)
    cache = getattr(x, "_nunchaku_lite_cross_head_rope_contiguous_cache", None)
    if cache is not None:
        cached = cache.get(cache_key)
        if (
            cached is not None
            and cached.shape == x.shape
            and cached.device == target_device
            and cached.dtype == torch.float32
            and cached.is_contiguous()
        ):
            return cached

    cached = x.to(device=target_device, dtype=torch.float32).contiguous()
    try:
        if cache is None:
            cache = {}
            x._nunchaku_lite_cross_head_rope_contiguous_cache = cache
        cache[cache_key] = cached
    except Exception:
        pass
    return cached


def _prepare_cross_head_rope(
    rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    batch_size: int,
    seq_len: int,
    heads: int,
    head_dim: int,
    channels: int,
    rope_type: str,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if rotary_emb is None:
        return None, None
    if len(rotary_emb) != 2:
        raise ValueError("rotary_emb must be a (cos, sin) tuple")
    cos, sin = rotary_emb
    if cos.shape != sin.shape:
        raise ValueError("rotary cos/sin shapes must match")
    if rope_type == "split":
        expected = (batch_size, heads, seq_len, head_dim // 2)
        if tuple(cos.shape) != expected:
            raise ValueError(f"split RoPE must have shape {expected}")
        return (
            _cached_cross_head_rope_contiguous(cos, device),
            _cached_cross_head_rope_contiguous(sin, device),
        )
    if rope_type == "interleaved":
        if cos.ndim == 2:
            if tuple(cos.shape) != (seq_len, channels):
                raise ValueError("interleaved RoPE must have shape [sequence, channels]")
            cos = cos.unsqueeze(0).expand(batch_size, -1, -1)
            sin = sin.unsqueeze(0).expand(batch_size, -1, -1)
        elif cos.ndim == 3:
            if cos.shape[0] == 1 and batch_size != 1:
                cos = cos.expand(batch_size, -1, -1)
                sin = sin.expand(batch_size, -1, -1)
            if tuple(cos.shape) != (batch_size, seq_len, channels):
                raise ValueError("interleaved RoPE must have shape [batch, sequence, channels]")
        else:
            raise ValueError("interleaved RoPE must be 2D or 3D")
        return (
            _cached_cross_head_rope_contiguous(cos, device),
            _cached_cross_head_rope_contiguous(sin, device),
        )
    raise ValueError("cross-head fused RoPE supports only split and interleaved rope types")


def fused_cross_head_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    norm_q: RMSNorm | DiffusersRMSNorm | None,
    norm_k: RMSNorm | DiffusersRMSNorm | None,
    query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    q_heads: int,
    k_heads: int,
    head_dim: int,
    rope_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply full-channel Q/K RMSNorm and RoPE in one native launch."""

    if q.device.type != "cuda" or k.device.type != "cuda":
        raise ValueError("fused_cross_head_qk_norm_rope requires CUDA q/k tensors")
    if q.device != k.device:
        raise ValueError("q and k must be on the same CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype:
        raise ValueError("q and k must both be float16 or bfloat16 with matching dtype")
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("q and k must have shape [batch, sequence, channels]")
    if q.shape[0] != k.shape[0]:
        raise ValueError("q and k batch sizes must match")
    q_heads = int(q_heads)
    k_heads = int(k_heads)
    head_dim = int(head_dim)
    if q_heads <= 0 or k_heads <= 0 or head_dim <= 0:
        raise ValueError("q_heads, k_heads, and head_dim must be positive")
    if head_dim % 2 != 0:
        raise ValueError("fused RoPE requires an even head_dim")
    if q.shape[-1] != q_heads * head_dim or k.shape[-1] != k_heads * head_dim:
        raise ValueError("q/k channels must match heads * head_dim")

    q_weight = _cross_head_weight(norm_q, q.shape[-1], q.device, q.dtype)
    k_weight = _cross_head_weight(norm_k, k.shape[-1], k.device, k.dtype)
    eps_q = _rms_norm_eps(norm_q)
    eps_k = _rms_norm_eps(norm_k)
    if eps_q != eps_k:
        raise ValueError("fused cross-head Q/K norm requires matching q/k eps")

    q_cos, q_sin = _prepare_cross_head_rope(
        query_rotary_emb,
        batch_size=q.shape[0],
        seq_len=q.shape[1],
        heads=q_heads,
        head_dim=head_dim,
        channels=q.shape[-1],
        rope_type=rope_type,
        device=q.device,
    )
    k_cos, k_sin = _prepare_cross_head_rope(
        key_rotary_emb if key_rotary_emb is not None else query_rotary_emb,
        batch_size=k.shape[0],
        seq_len=k.shape[1],
        heads=k_heads,
        head_dim=head_dim,
        channels=k.shape[-1],
        rope_type=rope_type,
        device=k.device,
    )

    q_out, k_out = get_ops().fused_cross_head_qk_norm_rope(
        q.contiguous(),
        k.contiguous(),
        q_weight,
        k_weight,
        q_cos,
        q_sin,
        k_cos,
        k_sin,
        q_heads,
        k_heads,
        head_dim,
        float(eps_q),
        rope_type == "interleaved",
    )
    return q_out, k_out
