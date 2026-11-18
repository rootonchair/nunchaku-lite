"""Diffusers attention processors with the fused MPS q/k norm + RoPE prologue (Apple silicon).

Checkpoints patched from a runtime manifest keep Diffusers' own attention modules and processors. On MPS,
:func:`install_mps_fused_processors` swaps the stock processors of FLUX.1, FLUX.2, Qwen-Image and Z-Image for
subclasses that run the q/k RMSNorm, the rotary embedding and the head layout in one kernel
(:func:`~nunchaku_lite.ops.mps.qk_norm_rope_mps`, head dim 128, interleaved-pair RoPE), reading q/k/v straight from
the projections. Each call checks whether the kernel applies (MPS, batch 1, head dim 128, fp16/bf16, q/k norms
present, the family's usual rotary tables) and otherwise runs the stock processor unchanged.

:func:`fused_qk_norm_rope` is the shared entry point, also used by the Nunchaku Qwen-Image processor.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers import transformer_flux, transformer_flux2

from ..ops.fused import _rms_norm_eps, _rms_norm_weight
from ..ops.mps import _derived, qk_norm_rope_mps, qk_norm_rope_mps_available

__all__ = ["fused_qk_norm_rope", "install_mps_fused_processors"]

_QKV = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def fused_qk_norm_rope(
    image: _QKV,
    norms: Sequence,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    text: _QKV | None = None,
) -> _QKV | None:
    """q/k RMSNorm + RoPE + ``(1, L, heads, 128)`` layout in one MPS kernel, or ``None`` when it does not apply.

    Args:
        image: Image-token ``(q, k, v)`` projections ``(1, Li, heads * 128)`` (column slices of one fused
            projection are fine).
        norms: ``(norm_q, norm_k)`` for the image rows, followed by ``(norm_added_q, norm_added_k)`` with ``text``.
        cos, sin: Rotary tables for all rows, text first (see :func:`~nunchaku_lite.ops.mps.qk_norm_rope_mps`).
        heads: Number of attention heads.
        text: Optional text-token ``(q, k, v)``, placed before the image rows.
    """

    q = image[0]
    if q.device.type != "mps" or q.shape[0] != 1 or q.dtype not in (torch.float16, torch.bfloat16):
        return None
    if q.shape[-1] != heads * 128 or any(norm is None for norm in norms):  # the kernel always normalizes q and k
        return None
    if any(w is not None and w.shape != (128,) for w in map(_rms_norm_weight, norms)):
        return None
    if not qk_norm_rope_mps_available():
        return None
    weights = [_rms_norm_weight(n) for n in norms]
    return qk_norm_rope_mps(
        *image, weights[0], weights[1], cos, sin, heads,
        eps_q=_rms_norm_eps(norms[0]), eps_k=_rms_norm_eps(norms[1]),
        text=text,
        text_q_weight=weights[2] if text is not None else None,
        text_k_weight=weights[3] if text is not None else None,
    )  # fmt: skip


def complex_rope_tables(freqs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Per-pair ``(cos, sin)`` float32 tables ``(L, 64)`` from complex rotary tables ``(L_i, 64)`` (concatenated
    in order), cached while the tables live; ``None`` for other layouts."""

    if any(not torch.is_tensor(f) or not f.is_complex() or f.ndim != 2 or f.shape[-1] != 64 for f in freqs):
        return None
    head, rest = freqs[0], freqs[1:]
    key = "complex_rope:" + ",".join(f"{id(f)}.{f._version}.{f.shape[0]}" for f in rest)

    def build(_):
        table = torch.cat(list(freqs)) if rest else head
        return table.real.float().contiguous(), table.imag.float().contiguous()

    return _derived(head, key, build)


def _real_rope_tables(rotary) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Diffusers' FLUX ``(cos, sin)`` tables ``(L, 128)``, or ``None``."""

    if not (isinstance(rotary, tuple) and len(rotary) == 2):
        return None
    if any(not torch.is_tensor(t) or t.ndim != 2 or t.shape[-1] != 128 for t in rotary):
        return None
    return rotary


def _attend(processor, query, key, value, mask=None) -> torch.Tensor:
    out = dispatch_attention_fn(
        query, key, value, attn_mask=mask, backend=processor._attention_backend, parallel_config=processor._parallel_config
    )
    return out.flatten(2, 3).to(query.dtype)


# ---------------------------------------------------------------------------
# FLUX.1 and FLUX.2 (joint and single-stream / parallel blocks)
# ---------------------------------------------------------------------------


def _flux_joint(processor, attn, hidden_states, encoder_hidden_states, image_rotary_emb, get_qkv):
    """FLUX.1 / FLUX.2 joint attention (also FLUX.1 single blocks, where ``encoder_hidden_states`` is ``None``).
    Returns ``None`` to fall back."""

    tables = _real_rope_tables(image_rotary_emb)
    if tables is None or attn.head_dim != 128:
        return None
    joint = encoder_hidden_states is not None and attn.added_kv_proj_dim is not None
    norms = (attn.norm_q, attn.norm_k) + ((attn.norm_added_q, attn.norm_added_k) if joint else ())
    query, key, value, enc_query, enc_key, enc_value = get_qkv(attn, hidden_states, encoder_hidden_states)
    fused = fused_qk_norm_rope(
        (query, key, value), norms, *tables, attn.heads, text=(enc_query, enc_key, enc_value) if joint else None
    )
    if fused is None:
        return None
    hidden_states = _attend(processor, *fused)
    if encoder_hidden_states is None:
        return hidden_states
    lt = encoder_hidden_states.shape[1]
    encoder_hidden_states, hidden_states = hidden_states.split_with_sizes([lt, hidden_states.shape[1] - lt], dim=1)
    hidden_states = attn.to_out[1](attn.to_out[0](hidden_states.contiguous()))
    return hidden_states, attn.to_add_out(encoder_hidden_states.contiguous())


class MPSFusedFluxAttnProcessor(transformer_flux.FluxAttnProcessor):
    """Diffusers ``FluxAttnProcessor`` (FLUX.1 joint and single blocks) with the fused MPS prologue."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        out = None
        if attention_mask is None:
            out = _flux_joint(
                self, attn, hidden_states, encoder_hidden_states, image_rotary_emb, transformer_flux._get_qkv_projections
            )
        if out is None:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
        return out


class MPSFusedFlux2AttnProcessor(transformer_flux2.Flux2AttnProcessor):
    """Diffusers ``Flux2AttnProcessor`` (joint text + image attention) with the fused MPS prologue."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        out = None
        if attention_mask is None:
            out = _flux_joint(
                self, attn, hidden_states, encoder_hidden_states, image_rotary_emb, transformer_flux2._get_qkv_projections
            )
        if out is None:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
        return out


class MPSFusedFlux2ParallelSelfAttnProcessor(transformer_flux2.Flux2ParallelSelfAttnProcessor):
    """Diffusers ``Flux2ParallelSelfAttnProcessor`` (single-stream blocks) with the fused MPS prologue."""

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        tables = _real_rope_tables(image_rotary_emb)
        if attention_mask is None and tables is not None and attn.head_dim == 128:
            proj = attn.to_qkv_mlp_proj(hidden_states)
            inner = attn.inner_dim
            if proj.shape[-1] == 3 * inner + attn.mlp_hidden_dim * attn.mlp_mult_factor:  # not column-sharded (TP)
                qkv = (proj[..., :inner], proj[..., inner : 2 * inner], proj[..., 2 * inner : 3 * inner])
                fused = fused_qk_norm_rope(qkv, (attn.norm_q, attn.norm_k), *tables, attn.heads)
                if fused is not None:
                    hidden_states = _attend(self, *fused)
                    mlp_hidden_states = attn.mlp_act_fn(proj[..., 3 * inner :])
                    return attn.to_out(torch.cat([hidden_states, mlp_hidden_states], dim=-1))
        return super().__call__(attn, hidden_states, attention_mask, image_rotary_emb)


_REPLACEMENTS: dict[type, type] = {
    transformer_flux.FluxAttnProcessor: MPSFusedFluxAttnProcessor,
    transformer_flux2.Flux2AttnProcessor: MPSFusedFlux2AttnProcessor,
    transformer_flux2.Flux2ParallelSelfAttnProcessor: MPSFusedFlux2ParallelSelfAttnProcessor,
}


# ---------------------------------------------------------------------------
# Qwen-Image (double-stream joint attention, complex rotary tables)
# ---------------------------------------------------------------------------

try:
    from diffusers.models.transformers import transformer_qwenimage
except ImportError:  # older Diffusers
    transformer_qwenimage = None

if transformer_qwenimage is not None:

    class MPSFusedQwenDoubleStreamAttnProcessor(transformer_qwenimage.QwenDoubleStreamAttnProcessor2_0):
        """Diffusers ``QwenDoubleStreamAttnProcessor2_0`` with the fused MPS prologue."""

        def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            encoder_hidden_states_mask=None,
            attention_mask=None,
            image_rotary_emb=None,
        ):
            out = None
            if encoder_hidden_states is not None and attention_mask is None and image_rotary_emb is not None:
                out = _qwen_joint(
                    self, attn, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, image_rotary_emb
                )
            if out is None:
                return super().__call__(
                    attn, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, attention_mask, image_rotary_emb
                )
            return out

    _REPLACEMENTS[transformer_qwenimage.QwenDoubleStreamAttnProcessor2_0] = MPSFusedQwenDoubleStreamAttnProcessor


def _qwen_joint(processor, attn, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, image_rotary_emb):
    img_freqs, txt_freqs = image_rotary_emb
    tables = complex_rope_tables((txt_freqs, img_freqs))
    li, lt = hidden_states.shape[1], encoder_hidden_states.shape[1]
    if tables is None or tables[0].shape[0] != lt + li or attn.inner_dim != attn.heads * 128:
        return None
    image = (attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states))
    text = (
        attn.add_q_proj(encoder_hidden_states),
        attn.add_k_proj(encoder_hidden_states),
        attn.add_v_proj(encoder_hidden_states),
    )
    norms = (attn.norm_q, attn.norm_k, attn.norm_added_q, attn.norm_added_k)
    fused = fused_qk_norm_rope(image, norms, *tables, attn.heads, text=text)
    if fused is None:
        return None
    mask = None
    if encoder_hidden_states_mask is not None:
        image_mask = torch.ones((1, li), dtype=torch.bool, device=hidden_states.device)
        mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)[:, None, None, :]
    out = _attend(processor, *fused, mask)
    img_out = attn.to_out[0](out[:, lt:].contiguous())
    if len(attn.to_out) > 1:
        img_out = attn.to_out[1](img_out)
    return img_out, attn.to_add_out(out[:, :lt].contiguous())


# ---------------------------------------------------------------------------
# Z-Image (single-stream self-attention, complex rotary tables)
# ---------------------------------------------------------------------------

try:
    from diffusers.models.transformers import transformer_z_image
except ImportError:  # older Diffusers
    transformer_z_image = None

if transformer_z_image is not None:

    class MPSFusedZSingleStreamAttnProcessor(transformer_z_image.ZSingleStreamAttnProcessor):
        """Diffusers ``ZSingleStreamAttnProcessor`` with the fused MPS prologue."""

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, freqs_cis=None):
            out = None
            if freqs_cis is not None and freqs_cis.ndim == 3 and freqs_cis.shape[0] == 1:
                out = _z_image(self, attn, hidden_states, attention_mask, freqs_cis)
            if out is None:
                return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, freqs_cis)
            return out

    _REPLACEMENTS[transformer_z_image.ZSingleStreamAttnProcessor] = MPSFusedZSingleStreamAttnProcessor


def _z_image(processor, attn, hidden_states, attention_mask, freqs_cis):
    tables = complex_rope_tables((freqs_cis[0],))
    if tables is None or tables[0].shape[0] != hidden_states.shape[1]:
        return None
    qkv = (attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states))
    fused = fused_qk_norm_rope(qkv, (attn.norm_q, attn.norm_k), *tables, attn.heads)
    if fused is None:
        return None
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask[:, None, None, :]
    output = attn.to_out[0](_attend(processor, *fused, attention_mask))
    if len(attn.to_out) > 1:
        output = attn.to_out[1](output)
    return output


def install_mps_fused_processors(transformer: torch.nn.Module) -> int:
    """Swap stock FLUX.1 / FLUX.2 / Qwen-Image / Z-Image attention processors for the fused-MPS ones. Returns how
    many modules were updated (0 when already installed or for other models)."""

    patched = 0
    for module in transformer.modules():
        processor = getattr(module, "processor", None)
        replacement = _REPLACEMENTS.get(type(processor))
        if replacement is None or not hasattr(module, "set_processor"):
            continue
        new = replacement()
        new._attention_backend = processor._attention_backend
        new._parallel_config = processor._parallel_config
        module.set_processor(new)
        patched += 1
    return patched
