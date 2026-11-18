"""Fused attention prologue for mflux's transformers (FLUX.1, FLUX.2, Qwen-Image, Z-Image).

mflux computes the q/k RMSNorm and RoPE of every attention layer with reshapes, float32 casts and stacks that
``mx.compile`` cannot fuse, and passes strided q/k/v views to attention. On FLUX.2-klein-4B at 1024² that costs
about as much as the attention itself. :func:`install_fused_attention` swaps each attention module for a subclass
that does the prologue in one kernel (:func:`~nunchaku_lite.mlx.kernels.qk_norm_rope`, head dim 128,
interleaved-pair RoPE, which all four families use) reading q/k/v straight from the projection output.

Each family needs only a small call function: where its projections, norms and rotary tables live, and how its
output is projected. Anything the kernel does not cover (KV cache, batch > 1, other head dims, missing norms or
rotary tables, float16 compute with Z-Image's value headroom) falls back to mflux's own implementation.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import mlx.core as mx

from .kernels import qk_norm_rope

__all__ = ["install_fused_attention"]

_HEAD_DIM = 128


def install_fused_attention(transformer: Any) -> int:
    """Route the transformer's supported attention modules through the fused prologue. Returns how many were
    patched (0 for unsupported models); calling it again patches nothing."""

    patched = 0
    for _, module in transformer.named_modules():
        call = _CALLS.get(_class_key(type(module)))
        if call is None or getattr(type(module), "_nl_fused", False) or _head_dim(module) != _HEAD_DIM:
            continue
        module.__class__ = _fused_class(type(module), call)
        patched += 1
    return patched


def _class_key(cls: type) -> tuple[str, str] | None:
    # Exact mflux classes only: subclasses (e.g. FLUX.1 concept attention) have other signatures.
    parts = cls.__module__.split(".")
    return (parts[2], cls.__name__) if len(parts) > 2 and parts[:2] == ["mflux", "models"] else None


def _head_dim(attn) -> int | None:
    for name in ("dim_head", "head_dimension", "head_dim"):
        value = getattr(attn, name, None)
        if isinstance(value, int):
            return value
    return None


@functools.cache
def _fused_class(base: type, call: Callable) -> type:
    def __call__(self, *args, **kwargs):
        return call(self, base, *args, **kwargs)

    return type(base.__name__, (base,), {"__call__": __call__, "_nl_fused": True, "__module__": base.__module__})


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------


def _attention(q: mx.array, k: mx.array, v: mx.array, mask: mx.array | None = None) -> mx.array:
    """``(1, H, L, 128)`` q/k/v -> ``(1, L, H * 128)``."""

    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=_HEAD_DIM**-0.5, mask=mask)
    heads, length = out.shape[1], out.shape[2]
    return out.transpose(0, 2, 1, 3).reshape(1, length, heads * _HEAD_DIM)


def _qkv_rows(attn, x: mx.array, names: tuple[str, str, str], width: int) -> tuple[mx.array, tuple[int, int, int]]:
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
    return out.reshape(-1, out.shape[-1]), (0, width, 2 * width)


def _joint_qkv(attn, x, context, width, image_names, text_names):
    """Image and text projection rows for joint attention, or ``None`` when their layouts differ."""

    img, offs = _qkv_rows(attn, x, image_names, width)
    txt, txt_offs = _qkv_rows(attn, context, text_names, width)
    return (img, txt, offs) if offs == txt_offs and img.dtype == txt.dtype else None


_IMAGE_QKV = ("to_q", "to_k", "to_v")
_TEXT_QKV = ("add_q_proj", "add_k_proj", "add_v_proj")


def _tables_fit(cos: mx.array, sin: mx.array, rows: int) -> bool:
    return cos.shape == (rows, _HEAD_DIM // 2) and sin.shape == (rows, _HEAD_DIM // 2)


# ---------------------------------------------------------------------------
# FLUX.2 (Flux2Attention, Flux2ParallelSelfAttention)
# ---------------------------------------------------------------------------


def _flux2_joint(
    self, base, hidden_states, encoder_hidden_states, image_rotary_emb, kv_cache=None, kv_cache_layer_idx=None
):
    def fallback():
        return base.__call__(
            self,
            hidden_states,
            encoder_hidden_states,
            image_rotary_emb,
            kv_cache=kv_cache,
            kv_cache_layer_idx=kv_cache_layer_idx,
        )

    if (
        kv_cache is not None
        or image_rotary_emb is None
        or encoder_hidden_states is None
        or self.added_kv_proj_dim is None
        or hidden_states.shape[0] != 1
    ):
        return fallback()
    dtype, context_dtype = hidden_states.dtype, encoder_hidden_states.dtype
    precision = self.compute_precision
    rows = _joint_qkv(
        self,
        precision.to_compute(hidden_states),
        precision.to_compute(encoder_hidden_states),
        self.inner_dim,
        _IMAGE_QKV,
        _TEXT_QKV,
    )
    if rows is None:
        return fallback()
    img, txt, offs = rows
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
    out = _attention(q, k, v)
    lt = encoder_hidden_states.shape[1]
    encoder_out = precision.to_stream(self.to_add_out(out[:, :lt]), context_dtype)
    return precision.to_stream(self.to_out(out[:, lt:]), dtype), encoder_out


def _flux2_parallel(self, base, hidden_states, image_rotary_emb, kv_cache=None, kv_cache_layer_idx=None):
    if kv_cache is not None or image_rotary_emb is None or hidden_states.shape[0] != 1:
        return base.__call__(
            self, hidden_states, image_rotary_emb, kv_cache=kv_cache, kv_cache_layer_idx=kv_cache_layer_idx
        )
    dtype, precision = hidden_states.dtype, self.compute_precision
    proj = self.to_qkv_mlp_proj(precision.to_compute(hidden_states))
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
    out = _attention(q, k, v)
    mlp_hidden = self.mlp_act(proj[..., 3 * inner :])
    out = self.to_out(mx.concatenate([out, mlp_hidden], axis=-1))
    return precision.to_stream(out, dtype)


# ---------------------------------------------------------------------------
# FLUX.1 (JointAttention, SingleBlockAttention)
# ---------------------------------------------------------------------------


def _flux1_tables(image_rotary_emb: mx.array, rows: int) -> tuple[mx.array, mx.array] | None:
    """cos / sin ``(L, 64)`` from mflux's FLUX.1 rotation matrices ``(1, 1, L, 64, 2, 2)`` = [[cos, -sin], [sin,
    cos]]."""

    if image_rotary_emb.shape != (1, 1, rows, _HEAD_DIM // 2, 2, 2):
        return None
    return image_rotary_emb[0, 0, :, :, 0, 0], image_rotary_emb[0, 0, :, :, 1, 0]


def _flux1_joint(self, base, hidden_states, encoder_hidden_states, image_rotary_emb):
    lt = encoder_hidden_states.shape[1]
    tables = _flux1_tables(image_rotary_emb, lt + hidden_states.shape[1])
    width = self.num_heads * _HEAD_DIM
    rows = None
    if hidden_states.shape[0] == 1 and tables is not None:
        rows = _joint_qkv(self, hidden_states, encoder_hidden_states, width, _IMAGE_QKV, _TEXT_QKV)
    if rows is None:
        return base.__call__(self, hidden_states, encoder_hidden_states, image_rotary_emb)
    img, txt, offs = rows
    q, k, v = qk_norm_rope(
        img,
        offs,
        self.norm_q.weight,
        self.norm_k.weight,
        *tables,
        self.num_heads,
        eps=self.norm_q.eps,
        text=txt,
        text_q_weight=self.norm_added_q.weight,
        text_k_weight=self.norm_added_k.weight,
    )
    out = _attention(q, k, v)
    return self.to_out[0](out[:, lt:]), self.to_add_out(out[:, :lt])


def _flux1_single(self, base, hidden_states, image_rotary_emb):
    tables = _flux1_tables(image_rotary_emb, hidden_states.shape[1])
    if hidden_states.shape[0] != 1 or tables is None:
        return base.__call__(self, hidden_states, image_rotary_emb)
    rows, offs = _qkv_rows(self, hidden_states, _IMAGE_QKV, self.num_heads * _HEAD_DIM)
    q, k, v = qk_norm_rope(
        rows, offs, self.norm_q.weight, self.norm_k.weight, *tables, self.num_heads, eps=self.norm_q.eps
    )
    return _attention(q, k, v)


# ---------------------------------------------------------------------------
# Qwen-Image (QwenAttention)
# ---------------------------------------------------------------------------


def _qwen(self, base, img_modulated, txt_modulated, encoder_hidden_states_mask, image_rotary_emb, block_idx=None):
    def fallback():
        return base.__call__(
            self, img_modulated, txt_modulated, encoder_hidden_states_mask, image_rotary_emb, block_idx=block_idx
        )

    norms = (self.norm_q, self.norm_k, self.norm_added_q, self.norm_added_k)
    if image_rotary_emb is None or img_modulated.shape[0] != 1 or any(n is None for n in norms):
        return fallback()
    (img_cos, img_sin), (txt_cos, txt_sin) = image_rotary_emb
    li, lt = img_modulated.shape[1], txt_modulated.shape[1]
    if not (_tables_fit(img_cos, img_sin, li) and _tables_fit(txt_cos, txt_sin, lt)):
        return fallback()
    rows = _joint_qkv(self, img_modulated, txt_modulated, self.num_heads * _HEAD_DIM, _IMAGE_QKV, _TEXT_QKV)
    if rows is None:
        return fallback()
    img, txt, offs = rows
    q, k, v = qk_norm_rope(
        img,
        offs,
        self.norm_q.weight,
        self.norm_k.weight,
        mx.concatenate([txt_cos, img_cos]),
        mx.concatenate([txt_sin, img_sin]),
        self.num_heads,
        eps=self.norm_q.eps,
        text=txt,
        text_q_weight=self.norm_added_q.weight,
        text_k_weight=self.norm_added_k.weight,
    )
    mask = self._convert_mask_for_qwen(mask=encoder_hidden_states_mask, joint_seq_len=lt + li, txt_seq_len=lt)
    out = _attention(q, k, v, mask)
    return self.attn_to_out[0](out[:, lt:]), self.to_add_out(out[:, :lt])


# ---------------------------------------------------------------------------
# Z-Image (ZImageAttention)
# ---------------------------------------------------------------------------


def _z_image(self, base, hidden_states, attention_mask=None, freqs_cis=None):
    length = hidden_states.shape[1]
    if (
        freqs_cis is None
        or hidden_states.shape[0] != 1
        or self.norm_q is None
        or self.norm_k is None
        or self.float32
        or self.compute_precision.dtype is not None  # float16 compute shrinks v, which the kernel copies as is
        or freqs_cis.shape != (length, _HEAD_DIM // 2, 2)
    ):
        return base.__call__(self, hidden_states, attention_mask=attention_mask, freqs_cis=freqs_cis)
    rows, offs = _qkv_rows(self, hidden_states, _IMAGE_QKV, self.dim)
    q, k, v = qk_norm_rope(
        rows,
        offs,
        self.norm_q.weight,
        self.norm_k.weight,
        freqs_cis[..., 0],
        freqs_cis[..., 1],
        self.n_heads,
        eps=self.norm_q.eps,
    )
    mask = None
    if attention_mask is not None:
        mask = mx.where(
            attention_mask[:, None, None, :], mx.array(0.0, dtype=q.dtype), mx.array(float("-inf"), dtype=q.dtype)
        )
    return self.to_out[0](_attention(q, k, v, mask))


_CALLS: dict[tuple[str, str], Callable] = {
    ("flux2", "Flux2Attention"): _flux2_joint,
    ("flux2", "Flux2ParallelSelfAttention"): _flux2_parallel,
    ("flux", "JointAttention"): _flux1_joint,
    ("flux", "SingleBlockAttention"): _flux1_single,
    ("qwen", "QwenAttention"): _qwen,
    ("z_image", "ZImageAttention"): _z_image,
}
