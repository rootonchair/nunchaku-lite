import pytest
import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_ltx2 import apply_interleaved_rotary_emb, apply_split_rotary_emb

from nunchaku_lite.ops.fused import (
    fused_affine_modulate,
    fused_cross_head_qk_norm_rope,
    fused_rms_norm_modulate,
)


def _has_native_cross_head_qk_op() -> bool:
    try:
        from nunchaku_lite._C import ops

        return hasattr(ops, "fused_cross_head_qk_norm_rope")
    except Exception:  # noqa: BLE001 - feature probe: any failure means "unavailable", so skip
        return False


def _has_native_modulation_ops() -> bool:
    try:
        from nunchaku_lite.ops.backend import get_ops

        ops = get_ops()
        return hasattr(ops, "fused_rms_norm_modulate") and hasattr(ops, "fused_affine_modulate")
    except Exception:  # noqa: BLE001 - feature probe: any failure means "unavailable", so skip
        return False


def _cross_head_ref(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    *,
    channels: int,
    rope_type: str,
) -> torch.Tensor:
    normed = F.rms_norm(x, (channels,), weight=weight, eps=1e-6)
    if rope_type == "split":
        return apply_split_rotary_emb(normed, rotary_emb)
    return apply_interleaved_rotary_emb(normed, rotary_emb)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_native_cross_head_qk_op(),
    reason="cross-head Q/K norm+RoPE fusion requires the CUDA extension",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_seq,k_seq", [(7, 7), (7, 5)])
@pytest.mark.parametrize("weighted", [False, True])
@torch.inference_mode()
def test_native_cross_head_qk_norm_split_rope_matches_reference(dtype, q_seq, k_seq, weighted):
    torch.manual_seed(37 + q_seq + k_seq + int(weighted))
    device = torch.device("cuda")
    batch_size = 2
    heads = 4
    head_dim = 64
    channels = heads * head_dim
    q = torch.randn(batch_size, q_seq, channels, dtype=dtype, device=device)
    k = torch.randn(batch_size, k_seq, channels, dtype=dtype, device=device)
    q_phase = torch.randn(batch_size, heads, q_seq, head_dim // 2, dtype=torch.float32, device=device)
    k_phase = torch.randn(batch_size, heads, k_seq, head_dim // 2, dtype=torch.float32, device=device)
    q_rope = (q_phase.cos(), q_phase.sin())
    k_rope = (k_phase.cos(), k_phase.sin())
    norm_q = torch.nn.RMSNorm(channels, eps=1e-6).to(device=device, dtype=dtype) if weighted else None
    norm_k = torch.nn.RMSNorm(channels, eps=1e-6).to(device=device, dtype=dtype) if weighted else None

    q_out, k_out = fused_cross_head_qk_norm_rope(
        q,
        k,
        norm_q,
        norm_k,
        q_rope,
        k_rope,
        q_heads=heads,
        k_heads=heads,
        head_dim=head_dim,
        rope_type="split",
    )

    q_weight = norm_q.weight if norm_q is not None else None
    k_weight = norm_k.weight if norm_k is not None else None
    q_ref = _cross_head_ref(q, q_weight, q_rope, channels=channels, rope_type="split")
    k_ref = _cross_head_ref(k, k_weight, k_rope, channels=channels, rope_type="split")
    torch.testing.assert_close(q_out, q_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(k_out, k_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_native_cross_head_qk_op(),
    reason="cross-head Q/K norm+RoPE fusion requires the CUDA extension",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weighted", [False, True])
@torch.inference_mode()
def test_native_cross_head_qk_norm_interleaved_rope_matches_reference(dtype, weighted):
    torch.manual_seed(41 + int(weighted))
    device = torch.device("cuda")
    batch_size = 1
    heads = 4
    head_dim = 64
    channels = heads * head_dim
    q = torch.randn(batch_size, 11, channels, dtype=dtype, device=device)
    k = torch.randn(batch_size, 9, channels, dtype=dtype, device=device)
    q_phase = torch.randn(batch_size, 11, channels // 2, dtype=torch.float32, device=device)
    k_phase = torch.randn(batch_size, 9, channels // 2, dtype=torch.float32, device=device)
    q_rope = (q_phase.cos().repeat_interleave(2, dim=-1), q_phase.sin().repeat_interleave(2, dim=-1))
    k_rope = (k_phase.cos().repeat_interleave(2, dim=-1), k_phase.sin().repeat_interleave(2, dim=-1))
    norm_q = torch.nn.RMSNorm(channels, eps=1e-6).to(device=device, dtype=dtype) if weighted else None
    norm_k = torch.nn.RMSNorm(channels, eps=1e-6).to(device=device, dtype=dtype) if weighted else None

    q_out, k_out = fused_cross_head_qk_norm_rope(
        q,
        k,
        norm_q,
        norm_k,
        q_rope,
        k_rope,
        q_heads=heads,
        k_heads=heads,
        head_dim=head_dim,
        rope_type="interleaved",
    )

    q_weight = norm_q.weight if norm_q is not None else None
    k_weight = norm_k.weight if norm_k is not None else None
    q_ref = _cross_head_ref(q, q_weight, q_rope, channels=channels, rope_type="interleaved")
    k_ref = _cross_head_ref(k, k_weight, k_rope, channels=channels, rope_type="interleaved")
    torch.testing.assert_close(q_out, q_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(k_out, k_ref, atol=3e-2, rtol=3e-2)


def test_cross_head_qk_norm_rope_rejects_non_cuda_direct_call():
    q = torch.randn(1, 3, 8)
    k = torch.randn(1, 3, 8)
    rope = (torch.randn(1, 2, 3, 2), torch.randn(1, 2, 3, 2))

    with pytest.raises(ValueError, match="requires CUDA"):
        fused_cross_head_qk_norm_rope(
            q,
            k,
            None,
            None,
            rope,
            rope,
            q_heads=2,
            k_heads=2,
            head_dim=4,
            rope_type="split",
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_native_modulation_ops(),
    reason="modulation fusion requires the CUDA extension",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("shape", ["channels", "batch_channels"])
@torch.inference_mode()
def test_native_rms_norm_modulate_matches_reference(dtype, weighted, shape):
    torch.manual_seed(51 + int(weighted))
    device = torch.device("cuda")
    batch_size, seq_len, channels = 2, 5, 64
    x = torch.randn(batch_size, seq_len, channels, dtype=dtype, device=device)
    norm = torch.nn.RMSNorm(channels, eps=1e-6, elementwise_affine=weighted).to(device=device, dtype=dtype)
    if shape == "channels":
        scale = torch.randn(channels, dtype=dtype, device=device)
        shift = torch.randn(channels, dtype=dtype, device=device)
    else:
        scale = torch.randn(batch_size, channels, dtype=dtype, device=device)
        shift = torch.randn(batch_size, channels, dtype=dtype, device=device)

    output = fused_rms_norm_modulate(x, norm, scale, shift)
    ref_scale = scale.unsqueeze(1) if scale.ndim == 2 else scale
    ref_shift = shift.unsqueeze(1) if shift.ndim == 2 else shift
    reference = norm(x) * (1 + ref_scale) + ref_shift

    torch.testing.assert_close(output, reference, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_native_modulation_ops(),
    reason="modulation fusion requires the CUDA extension",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", ["channels", "batch_channels", "full"])
@torch.inference_mode()
def test_native_affine_modulate_matches_reference(dtype, shape):
    torch.manual_seed(61)
    device = torch.device("cuda")
    batch_size, seq_len, channels = 2, 5, 64
    x = torch.randn(batch_size, seq_len, channels, dtype=dtype, device=device)
    if shape == "channels":
        scale = torch.randn(channels, dtype=dtype, device=device)
        shift = torch.randn(channels, dtype=dtype, device=device)
        ref_scale = scale
        ref_shift = shift
    elif shape == "batch_channels":
        scale = torch.randn(batch_size, channels, dtype=dtype, device=device)
        shift = torch.randn(batch_size, channels, dtype=dtype, device=device)
        ref_scale = scale.unsqueeze(1)
        ref_shift = shift.unsqueeze(1)
    else:
        scale = torch.randn(batch_size, seq_len, channels, dtype=dtype, device=device)
        shift = torch.randn(batch_size, seq_len, channels, dtype=dtype, device=device)
        ref_scale = scale
        ref_shift = shift

    output = fused_affine_modulate(x, scale, shift)
    reference = x * (1 + ref_scale) + ref_shift

    torch.testing.assert_close(output, reference, atol=3e-2, rtol=3e-2)
