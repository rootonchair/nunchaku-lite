"""Fused attention prologue for mflux's FLUX.2 blocks.

mflux computes the q/k RMSNorm and RoPE of every FLUX.2 attention layer in float32 with reshapes and stacks that
``mx.compile`` cannot fuse, and passes strided q/k/v views to attention. On FLUX.2-klein-4B at 1024² that costs
about as much as the attention itself. :func:`install_fused_attention` swaps each block's attention module for a
subclass that does the prologue in one kernel (:func:`~nunchaku_lite.mlx.kernels.qk_norm_rope`) reading q/k/v
straight from the projection output. Anything the kernel does not cover (KV cache, batch > 1, head dim != 128,
no rotary embedding) falls back to mflux's own implementation.
"""

from __future__ import annotations

import functools
from typing import Any

import mlx.core as mx

from .kernels import qk_norm_rope

__all__ = ["install_fused_attention"]


def install_fused_attention(transformer: Any) -> int:
    """Route the transformer's FLUX.2 attention modules through the fused prologue. Returns how many were patched."""

    patched = 0
    for blocks, call in (
        (getattr(transformer, "transformer_blocks", ()), _joint_call),
        (getattr(transformer, "single_transformer_blocks", ()), _parallel_call),
    ):
        for block in blocks:
            attn = getattr(block, "attn", None)
            if attn is None or getattr(attn, "dim_head", None) != 128 or getattr(type(attn), "_nl_fused", False):
                continue
            attn.__class__ = _fused_class(type(attn), call)
            patched += 1
    return patched


@functools.cache
def _fused_class(base: type, call) -> type:
    def __call__(self, *args, **kwargs):
        return call(self, base, *args, **kwargs)

    return type(base.__name__, (base,), {"__call__": __call__, "_nl_fused": True, "__module__": base.__module__})


def _attention(attn, q, k, v):
    from mflux.models.flux.model.flux_transformer.common.attention_utils import AttentionUtils

    return AttentionUtils.compute_attention(
        query=q, key=k, value=v, batch_size=1, num_heads=attn.heads, head_dim=attn.dim_head
    )


def _qkv_rows(attn, x: mx.array, names: tuple[str, str, str]) -> tuple[mx.array, tuple[int, int, int]]:
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
    inner = attn.inner_dim
    return out.reshape(-1, out.shape[-1]), (0, inner, 2 * inner)


def _joint_call(
    self, base, hidden_states, encoder_hidden_states, image_rotary_emb, kv_cache=None, kv_cache_layer_idx=None
):
    if (
        kv_cache is not None
        or image_rotary_emb is None
        or encoder_hidden_states is None
        or self.added_kv_proj_dim is None
        or hidden_states.shape[0] != 1
    ):
        return base.__call__(
            self,
            hidden_states,
            encoder_hidden_states,
            image_rotary_emb,
            kv_cache=kv_cache,
            kv_cache_layer_idx=kv_cache_layer_idx,
        )
    dtype, context_dtype = hidden_states.dtype, encoder_hidden_states.dtype
    img, offs = _qkv_rows(self, self.compute_precision.to_compute(hidden_states), ("to_q", "to_k", "to_v"))
    txt, txt_offs = _qkv_rows(
        self, self.compute_precision.to_compute(encoder_hidden_states), ("add_q_proj", "add_k_proj", "add_v_proj")
    )
    if offs != txt_offs or img.dtype != txt.dtype:
        return base.__call__(self, hidden_states, encoder_hidden_states, image_rotary_emb)
    cos, sin = image_rotary_emb
    q, k, v = qk_norm_rope(
        img,
        offs,
        self.norm_q.weight,
        self.norm_k.weight,
        cos,
        sin,
        self.heads,
        eps=self.norm_q.eps,
        text=txt,
        text_q_weight=self.norm_added_q.weight,
        text_k_weight=self.norm_added_k.weight,
    )
    out = _attention(self, q, k, v)
    lt = encoder_hidden_states.shape[1]
    encoder_out = self.compute_precision.to_stream(self.to_add_out(out[:, :lt]), context_dtype)
    return self.compute_precision.to_stream(self.to_out(out[:, lt:]), dtype), encoder_out


def _parallel_call(self, base, hidden_states, image_rotary_emb, kv_cache=None, kv_cache_layer_idx=None):
    if kv_cache is not None or image_rotary_emb is None or hidden_states.shape[0] != 1:
        return base.__call__(
            self, hidden_states, image_rotary_emb, kv_cache=kv_cache, kv_cache_layer_idx=kv_cache_layer_idx
        )
    dtype = hidden_states.dtype
    proj = self.to_qkv_mlp_proj(self.compute_precision.to_compute(hidden_states))
    inner = self.inner_dim
    cos, sin = image_rotary_emb
    q, k, v = qk_norm_rope(
        proj.reshape(-1, proj.shape[-1]),
        (0, inner, 2 * inner),
        self.norm_q.weight,
        self.norm_k.weight,
        cos,
        sin,
        self.heads,
        eps=self.norm_q.eps,
    )
    out = _attention(self, q, k, v)
    mlp_hidden = self.mlp_act(proj[..., 3 * inner :])
    out = self.to_out(mx.concatenate([out, mlp_hidden], axis=-1))
    return self.compute_precision.to_stream(out, dtype)
