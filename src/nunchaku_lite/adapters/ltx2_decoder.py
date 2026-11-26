"""Faster diffusers LTX-2.5 diffusion decoder (LTX2VideoDiffusionDecoderModel).

Stage 5 (the ``LTX2VideoVaeDiffusionNABlock`` stack, ~93% of decode time at 768x512x121) is mostly eager glue
around two big ops (NATTEN attention and the SwiGLU matmuls). This module replaces the glue with nunchaku-lite
kernels and keeps the big ops:

- attention: per-head q/k RMSNorm + q scale + 3D RoPE in one in-place kernel (``per_head_qk_norm_rope3d_``),
  then NATTEN ``na3d`` (optionally with explicit tile shapes);
- stage-5 block: ``residual add + RMSNorm + (1 + scale) * x + shift`` in one kernel (``add_rmsnorm_modulate_``),
  used for both the context injection + norm1 and the attention residual + norm2; the SwiGLU runs gate and up as one
  matmul and ``silu(gate) * up`` as one kernel (``silu_mul``).

Every kernel reproduces the eager rounding; the remaining differences are reduction order (norm sums, and the merged
gate/up matmul if the GEMM library picks a different algorithm for the wider output).

Usage::

    from nunchaku_lite.adapters.ltx2_decoder import optimize_ltx2_diffusion_decoder

    optimize_ltx2_diffusion_decoder(pipe.diffusion_decoder)

Parts whose kernels the installed nunchaku-lite kernels lack are left unchanged.
"""

from __future__ import annotations

import logging
import types

import torch
import torch.nn.functional as F

from ..ops.fused import (
    add_rmsnorm_modulate_,
    has_ltx2_decoder_block_ops,
    has_per_head_qk_norm_rope3d,
    per_head_qk_norm_rope3d_,
    silu_mul,
)

logger = logging.getLogger(__name__)

# Tokens per SwiGLU tile: bounds the merged (tokens, 2 * hidden) gate/up activation (256 MB at hidden 1024, bf16).
SWIGLU_TILE_TOKENS = 65536
_GATE_UP_ATTR = "_nunchaku_lite_gate_up_weight"


class LTX2DecoderFusedNattenProcessor:
    """Drop-in for diffusers ``LTX2VideoVaeNeighborhoodNattenProcessor`` with fused q/k norm + scale + 3D RoPE."""

    def __init__(self, backend: str | None = None, q_tile_shape=None, kv_tile_shape=None):
        from kernels import get_kernel

        if (q_tile_shape is None) != (kv_tile_shape is None):
            raise ValueError("pass both q_tile_shape and kv_tile_shape, or neither")
        self._na3d = get_kernel("shi-labs/natten", version=1).na3d
        self.backend = backend
        self._tiles = {} if q_tile_shape is None else {"q_tile_shape": tuple(q_tile_shape),
                                                       "kv_tile_shape": tuple(kv_tile_shape)}

    def __call__(self, attn, hidden_states: torch.Tensor, block_mask=None) -> torch.Tensor:
        batch_size, num_frames, height, width, channels = hidden_states.shape
        shape = (batch_size, num_frames, height, width, attn.heads, attn.head_dim)
        query = attn.to_q(hidden_states).view(shape).contiguous()
        key = attn.to_k(hidden_states).view(shape).contiguous()
        value = attn.to_v(hidden_states).view(shape).contiguous()
        per_head_qk_norm_rope3d_(query, key, attn.norm_q, attn.norm_k, attn.rope, attn.scale)
        # `scale=1.0`: the query is already scaled, as in the diffusers processor.
        hidden_states = self._na3d(query, key, value, kernel_size=attn.kernel_size, scale=1.0, backend=self.backend,
                                   **self._tiles)
        hidden_states = hidden_states.reshape(batch_size, num_frames, height, width, channels)
        return attn.to_out[0](hidden_states)


def _is_diffusion_block(module: torch.nn.Module) -> bool:
    mlp = getattr(module, "mlp", None)
    return (
        all(hasattr(module, n) for n in ("scale_shift_table", "context_proj", "norm1", "norm2", "attn"))
        and mlp is not None
        and all(hasattr(mlp, n) for n in ("w_gate", "w_up", "w_down"))
        and all(getattr(mlp, n).bias is None for n in ("w_gate", "w_up", "w_down"))
    )


def fused_swiglu(mlp: torch.nn.Module, x: torch.Tensor, ops=None) -> torch.Tensor:
    """``w_down(silu(w_gate(x)) * w_up(x))`` with one merged gate/up matmul per tile and a fused ``silu * up``."""
    weight = getattr(mlp, _GATE_UP_ATTR, None)
    if weight is None or weight.device != x.device or weight.dtype != x.dtype:
        weight = torch.cat([mlp.w_gate.weight, mlp.w_up.weight], dim=0).detach().to(x.device, x.dtype).contiguous()
        setattr(mlp, _GATE_UP_ATTR, weight)
    flat = x.reshape(-1, x.shape[-1])
    out = torch.empty((flat.shape[0], mlp.w_down.weight.shape[0]), device=x.device, dtype=x.dtype)
    for start in range(0, flat.shape[0], SWIGLU_TILE_TOKENS):
        stop = min(start + SWIGLU_TILE_TOKENS, flat.shape[0])
        hidden = silu_mul(F.linear(flat[start:stop], weight), ops=ops)
        torch.mm(hidden, mlp.w_down.weight.t(), out=out[start:stop])  # = F.linear(hidden, w_down), no copy
    return out.view(*x.shape[:-1], out.shape[-1])


def fused_diffusion_block_forward(block, hidden_states, latent_context, modulation, block_mask=None, ops=None):
    """Same result as ``LTX2VideoVaeDiffusionNABlock.forward`` (eager rounding), with fused glue kernels."""
    table = block.scale_shift_table
    scale_msa, shift_msa, scale_mlp, shift_mlp = (
        modulation[i] + table[i].view(1, 1, 1, 1, -1) for i in (0, 1, 3, 4)
    )
    # residual = hidden_states + context_proj(latent_context), accumulated into the fresh projection output
    residual = block.context_proj(latent_context)
    attn_in = add_rmsnorm_modulate_(residual, hidden_states, block.norm1, 1 + scale_msa, shift_msa, ops=ops)
    # residual = residual + attn(...), accumulated into the fresh attention output
    attn_out = block.attn(attn_in, block_mask)
    mlp_in = add_rmsnorm_modulate_(attn_out, residual, block.norm2, 1 + scale_mlp, shift_mlp, ops=ops)
    out = fused_swiglu(block.mlp, mlp_in, ops=ops)
    return out.add_(attn_out)


def optimize_ltx2_diffusion_decoder(
    decoder: torch.nn.Module,
    backend: str | None = None,
    natten_tile_shapes: tuple | None = None,
    fuse_blocks: bool = True,
) -> bool:
    """Install the fused attention processor and (``fuse_blocks``) the fused stage-5 block forward.

    ``natten_tile_shapes``: optional ``(q_tile_shape, kv_tile_shape)`` for NATTEN, applied to the stage-5 blocks'
    attention only (where it was tuned). Returns True if anything was installed.
    """
    modules = list(decoder.modules())
    attention = [m for m in modules if hasattr(m, "project_qkv") and hasattr(m, "rope") and hasattr(m, "set_processor")]
    blocks = [m for m in modules if _is_diffusion_block(m)]
    if not attention:
        raise ValueError("no LTX-2.5 neighborhood-attention modules found in the given decoder")
    applied = False
    if has_per_head_qk_norm_rope3d():
        processor = LTX2DecoderFusedNattenProcessor(backend=backend)
        for module in attention:
            module.set_processor(processor)
        if natten_tile_shapes is not None:
            tuned = LTX2DecoderFusedNattenProcessor(backend, *natten_tile_shapes)
            for block in blocks:
                block.attn.set_processor(tuned)
        applied = True
    else:
        logger.warning("nunchaku-lite kernels lack per_head_qk_norm_rope3d_; decoder attention left unchanged")
    if fuse_blocks:
        if has_ltx2_decoder_block_ops():
            for block in blocks:
                block.forward = types.MethodType(fused_diffusion_block_forward, block)
            applied = applied or bool(blocks)
        else:
            logger.warning("nunchaku-lite kernels lack the LTX-2.5 decoder block ops; stage-5 blocks left unchanged")
    return applied
