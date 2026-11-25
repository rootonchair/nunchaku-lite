"""Faster attention for the diffusers LTX-2.5 diffusion decoder (LTX2VideoDiffusionDecoderModel).

The decoder's neighborhood attention normalizes q and k per head, scales q and applies an axis-factorized 3D RoPE in
fp32 as a chain of eager ops (~99 ms of a ~318 ms stage-5 block on an RTX PRO 6000 at 768x512x121). This processor
replaces that chain with one in-place nunchaku-lite kernel (``per_head_qk_norm_rope3d_``) and keeps everything else
identical: the q/k/v and output projections, and NATTEN ``na3d`` for the attention itself.

Usage::

    from nunchaku_lite.adapters.ltx2_decoder import optimize_ltx2_diffusion_decoder

    optimize_ltx2_diffusion_decoder(pipe.diffusion_decoder)

If the installed kernels do not provide the op, the decoder is left unchanged and ``False`` is returned.
"""

from __future__ import annotations

import logging

import torch

from ..ops.fused import has_per_head_qk_norm_rope3d, per_head_qk_norm_rope3d_

logger = logging.getLogger(__name__)


class LTX2DecoderFusedNattenProcessor:
    """Drop-in for diffusers ``LTX2VideoVaeNeighborhoodNattenProcessor`` with fused q/k norm + scale + 3D RoPE."""

    def __init__(self, backend: str | None = None):
        from kernels import get_kernel

        self._na3d = get_kernel("shi-labs/natten", version=1).na3d
        self.backend = backend

    def __call__(self, attn, hidden_states: torch.Tensor, block_mask=None) -> torch.Tensor:
        batch_size, num_frames, height, width, channels = hidden_states.shape
        shape = (batch_size, num_frames, height, width, attn.heads, attn.head_dim)
        query = attn.to_q(hidden_states).view(shape).contiguous()
        key = attn.to_k(hidden_states).view(shape).contiguous()
        value = attn.to_v(hidden_states).view(shape).contiguous()
        per_head_qk_norm_rope3d_(query, key, attn.norm_q, attn.norm_k, attn.rope, attn.scale)
        # `scale=1.0`: the query is already scaled, as in the diffusers processor.
        hidden_states = self._na3d(query, key, value, kernel_size=attn.kernel_size, scale=1.0, backend=self.backend)
        hidden_states = hidden_states.reshape(batch_size, num_frames, height, width, channels)
        return attn.to_out[0](hidden_states)


def optimize_ltx2_diffusion_decoder(decoder: torch.nn.Module, backend: str | None = None) -> bool:
    """Install the fused processor on every neighborhood-attention module of ``decoder``. Returns True if applied."""
    if not has_per_head_qk_norm_rope3d():
        logger.warning("nunchaku-lite kernels lack per_head_qk_norm_rope3d_; LTX-2.5 decoder left unchanged")
        return False
    processor = LTX2DecoderFusedNattenProcessor(backend=backend)
    applied = 0
    for module in decoder.modules():
        if hasattr(module, "project_qkv") and hasattr(module, "rope") and hasattr(module, "set_processor"):
            module.set_processor(processor)
            applied += 1
    if applied == 0:
        raise ValueError("no LTX-2.5 neighborhood-attention modules found in the given decoder")
    return True
