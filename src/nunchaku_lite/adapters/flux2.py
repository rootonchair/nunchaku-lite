"""Flux2 adapter for patching Diffusers Flux2 transformers with Nunchaku Lite modules."""

import inspect
import types
from typing import Any

import torch
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry, dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2AttnProcessor,
    Flux2FeedForward,
    Flux2ParallelSelfAttention,
    Flux2ParallelSelfAttnProcessor,
    _get_qkv_projections,
)
from torch import nn

from ..checkpoint_keys import convert_flux2_state_dict, flux2_state_dict_needs_conversion
from ..core import PatchOptions, register_adapter
from ..ops.fused import fused_qkv_norm_rotary
from ..ops.mps_fused import attention_mps, attention_supported, fused_qk_norm_rope_available, qk_norm_rope_mps
from .common import (
    SVDQPatchContext,
    build_svdq_context,
    finalize_svdq_checkpoint,
    fuse_linears,
    pack_rotemb,
    pad_tensor,
    patch_modules_recursively,
    prepare_transformer_dtype,
    svdq_from_linear,
)


def _pack_flux2_rotary_emb(freqs_cis: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Pack Diffusers Flux2 rotary embeddings for native kernels.

    Args:
        freqs_cis: ``(cos, sin)`` tuple produced by Diffusers Flux2 positional
            embeddings. Both tensors must have shape ``(sequence, dim)``.

    Returns:
        Float32 packed rotary tensor accepted by the native SVDQ GEMM kernels.

    Raises:
        ValueError: If the tuple tensors do not have matching 2D shapes.
    """

    cos, sin = freqs_cis
    if cos.ndim != 2 or sin.ndim != 2 or cos.shape != sin.shape:
        raise ValueError("Expected Flux.2 rotary embeddings as a (cos, sin) tuple with shape (seq_len, dim).")

    rotemb = torch.stack([sin[:, 0::2], cos[:, 0::2]], dim=-1).unsqueeze(0).unsqueeze(-2).contiguous()
    return pack_rotemb(pad_tensor(rotemb, 256, 1))


def _flux2_kv_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_txt_tokens: int,
    num_ref_tokens: int,
    kv_cache=None,
    backend=None,
) -> torch.Tensor:
    """Apply Flux2 reference-token causal attention with optional KV-cache reuse.

    Args:
        query: Query tensor for the full sequence.
        key: Key tensor for the full sequence.
        value: Value tensor for the full sequence.
        num_txt_tokens: Number of leading text tokens.
        num_ref_tokens: Number of reference tokens after the text tokens.
        kv_cache: Optional cache object exposing ``get`` and ``store``.
        backend: Optional Diffusers attention backend override.

    Returns:
        Attention output with the same sequence layout as ``query``.
    """

    if num_ref_tokens == 0 and kv_cache is None:
        return dispatch_attention_fn(query, key, value, backend=backend)

    if kv_cache is not None:
        k_ref, v_ref = kv_cache.get()
        k_all = torch.cat([key[:, :num_txt_tokens], k_ref, key[:, num_txt_tokens:]], dim=1)
        v_all = torch.cat([value[:, :num_txt_tokens], v_ref, value[:, num_txt_tokens:]], dim=1)
        return dispatch_attention_fn(query, k_all, v_all, backend=backend)

    ref_start = num_txt_tokens
    ref_end = num_txt_tokens + num_ref_tokens
    q_txt = query[:, :ref_start]
    q_ref = query[:, ref_start:ref_end]
    q_img = query[:, ref_end:]
    k_txt = key[:, :ref_start]
    k_ref = key[:, ref_start:ref_end]
    k_img = key[:, ref_end:]
    v_txt = value[:, :ref_start]
    v_ref = value[:, ref_start:ref_end]
    v_img = value[:, ref_end:]

    q_txt_img = torch.cat([q_txt, q_img], dim=1)
    k_all = torch.cat([k_txt, k_ref, k_img], dim=1)
    v_all = torch.cat([v_txt, v_ref, v_img], dim=1)
    attn_txt_img = dispatch_attention_fn(query=q_txt_img, key=k_all, value=v_all, backend=backend)
    attn_txt = attn_txt_img[:, :ref_start]
    attn_img = attn_txt_img[:, ref_start:]
    attn_ref = dispatch_attention_fn(query=q_ref, key=k_ref, value=v_ref, backend=backend)
    return torch.cat([attn_txt, attn_ref, attn_img], dim=1)


class NunchakuFlux2AttnProcessor:
    """Flux2 attention processor using patched SVDQ projections."""

    def __init__(self, processor=None):
        self._attention_backend = getattr(processor, "_attention_backend", None)
        self._parallel_config = getattr(processor, "_parallel_config", None)

    def __call__(
        self,
        attn: Flux2Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
        kv_cache=None,
        kv_cache_mode: str | None = None,
        num_ref_tokens: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        query, key, value, encoder_seq_len = self._project_qkv(
            attn, hidden_states, encoder_hidden_states, image_rotary_emb
        )

        if kv_cache_mode == "extract" and kv_cache is not None and num_ref_tokens > 0:
            ref_start = encoder_seq_len
            ref_end = encoder_seq_len + num_ref_tokens
            kv_cache.store(key[:, ref_start:ref_end].clone(), value[:, ref_start:ref_end].clone())

        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            hidden_states = _flux2_kv_causal_attention(
                query, key, value, encoder_seq_len, num_ref_tokens, backend=self._attention_backend
            )
        elif kv_cache_mode == "cached" and kv_cache is not None:
            hidden_states = _flux2_kv_causal_attention(
                query, key, value, encoder_seq_len, 0, kv_cache=kv_cache, backend=self._attention_backend
            )
        else:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_seq_len:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_seq_len, hidden_states.shape[1] - encoder_seq_len], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if encoder_seq_len:
            return hidden_states, encoder_hidden_states
        return hidden_states

    def _project_qkv(
        self,
        attn: Flux2Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        batch_size = hidden_states.shape[0]
        packed_rotary = (
            isinstance(image_rotary_emb, tuple)
            and len(image_rotary_emb) == 2
            and torch.is_tensor(image_rotary_emb[0])
            and image_rotary_emb[0].ndim == 3
        )
        if encoder_hidden_states is not None and packed_rotary:
            qkv = fused_qkv_norm_rotary(hidden_states, attn.to_qkv, attn.norm_q, attn.norm_k, image_rotary_emb[0])
            query, key, value = qkv.chunk(3, dim=-1)
            query = query.view(batch_size, -1, attn.heads, attn.head_dim)
            key = key.view(batch_size, -1, attn.heads, attn.head_dim)
            value = value.view(batch_size, -1, attn.heads, attn.head_dim)

            encoder_qkv = fused_qkv_norm_rotary(
                encoder_hidden_states,
                attn.to_added_qkv,
                attn.norm_added_q,
                attn.norm_added_k,
                image_rotary_emb[1],
            )
            encoder_query, encoder_key, encoder_value = encoder_qkv.chunk(3, dim=-1)
            encoder_query = encoder_query.view(batch_size, -1, attn.heads, attn.head_dim)
            encoder_key = encoder_key.view(batch_size, -1, attn.heads, attn.head_dim)
            encoder_value = encoder_value.view(batch_size, -1, attn.heads, attn.head_dim)
            encoder_seq_len = encoder_hidden_states.shape[1]
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)
            return query, key, value, encoder_seq_len

        query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        encoder_seq_len = 0

        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)
            encoder_seq_len = encoder_hidden_states.shape[1]
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        return query, key, value, encoder_seq_len


class NunchakuFlux2Attention(nn.Module, AttentionModuleMixin):
    """Flux2 cross-stream attention module with patched Nunchaku projections."""

    def __init__(self, other: Flux2Attention, context: SVDQPatchContext | None = None, **kwargs) -> None:
        super().__init__()
        self.head_dim = other.head_dim
        self.inner_dim = other.inner_dim
        self.query_dim = other.query_dim
        self.out_dim = other.out_dim
        self.heads = other.heads
        self.use_bias = other.use_bias
        self.dropout = other.dropout
        self.added_kv_proj_dim = other.added_kv_proj_dim
        self.added_proj_bias = other.added_proj_bias

        self.norm_q = other.norm_q
        self.norm_k = other.norm_k
        self.to_out = other.to_out
        self.to_out[0] = svdq_from_linear(self.to_out[0], context, **kwargs)
        with torch.device("meta"):
            to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
        self.to_qkv = svdq_from_linear(to_qkv, context, **kwargs)

        if self.added_kv_proj_dim is not None:
            self.norm_added_q = other.norm_added_q
            self.norm_added_k = other.norm_added_k
            self.to_add_out = svdq_from_linear(other.to_add_out, context, **kwargs)
            with torch.device("meta"):
                to_added_qkv = fuse_linears([other.add_q_proj, other.add_k_proj, other.add_v_proj])
            self.to_added_qkv = svdq_from_linear(to_added_qkv, context, **kwargs)

        self.set_processor(NunchakuFlux2AttnProcessor(getattr(other, "processor", None)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)


def _patch_flux2_feed_forward(ff: Flux2FeedForward, context: SVDQPatchContext | None = None, **kwargs):
    """Patch a Diffusers Flux2 feed-forward module in place."""

    patch_modules_recursively(
        ff,
        module_converters={nn.Linear: lambda _path, linear: svdq_from_linear(linear, context, **kwargs)},
    )
    ff.linear_out.act_unsigned = False
    return ff


class NunchakuFlux2ParallelSelfAttnProcessor:
    """Flux2 single-stream parallel attention processor using split SVDQ projections."""

    def __init__(self, processor=None):
        self._attention_backend = getattr(processor, "_attention_backend", None)
        self._parallel_config = getattr(processor, "_parallel_config", None)

    def __call__(
        self,
        attn: Flux2ParallelSelfAttention,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_cache=None,
        kv_cache_mode: str | None = None,
        num_txt_tokens: int = 0,
        num_ref_tokens: int = 0,
    ) -> torch.Tensor:
        if torch.is_tensor(image_rotary_emb) and image_rotary_emb.ndim == 3:
            batch_size = hidden_states.shape[0]
            qkv = fused_qkv_norm_rotary(hidden_states, attn.qkv_proj, attn.norm_q, attn.norm_k, image_rotary_emb)
            query, key, value = qkv.chunk(3, dim=-1)
            query = query.view(batch_size, -1, attn.heads, attn.head_dim)
            key = key.view(batch_size, -1, attn.heads, attn.head_dim)
            value = value.view(batch_size, -1, attn.heads, attn.head_dim)
        else:
            qkv = attn.qkv_proj(hidden_states)
            query, key, value = qkv.chunk(3, dim=-1)
            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))
            query = attn.norm_q(query)
            key = attn.norm_k(key)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if kv_cache_mode == "extract" and kv_cache is not None and num_ref_tokens > 0:
            ref_start = num_txt_tokens
            ref_end = num_txt_tokens + num_ref_tokens
            kv_cache.store(key[:, ref_start:ref_end].clone(), value[:, ref_start:ref_end].clone())

        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            attn_output = _flux2_kv_causal_attention(
                query, key, value, num_txt_tokens, num_ref_tokens, backend=self._attention_backend
            )
        elif kv_cache_mode == "cached" and kv_cache is not None:
            attn_output = _flux2_kv_causal_attention(
                query, key, value, num_txt_tokens, 0, kv_cache=kv_cache, backend=self._attention_backend
            )
        else:
            attn_output = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        attn_output = attn_output.flatten(2, 3).to(query.dtype)
        mlp_hidden_states = attn.mlp_act_fn(attn.mlp_fc1(hidden_states))
        return attn.out_proj(attn_output) + attn.mlp_fc2(mlp_hidden_states)


class NunchakuFlux2ParallelSelfAttention(nn.Module, AttentionModuleMixin):
    """Flux2 parallel self-attention module with split Nunchaku projections."""

    def __init__(
        self,
        other: Flux2ParallelSelfAttention,
        context: SVDQPatchContext | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.head_dim = other.head_dim
        self.inner_dim = other.inner_dim
        self.query_dim = other.query_dim
        self.out_dim = other.out_dim
        self.heads = other.heads
        self.use_bias = other.use_bias
        self.dropout = other.dropout
        self.mlp_ratio = other.mlp_ratio
        self.mlp_hidden_dim = other.mlp_hidden_dim
        self.mlp_mult_factor = other.mlp_mult_factor

        with torch.device("meta"):
            qkv_proj = nn.Linear(other.query_dim, other.inner_dim * 3, bias=other.use_bias)
            mlp_fc1 = nn.Linear(other.query_dim, other.mlp_hidden_dim * other.mlp_mult_factor, bias=other.use_bias)
            out_proj = nn.Linear(other.inner_dim, other.out_dim, bias=other.to_out.bias is not None)
            mlp_fc2 = nn.Linear(other.mlp_hidden_dim, other.out_dim, bias=other.to_out.bias is not None)
        device = other.to_qkv_mlp_proj.weight.device
        self.qkv_proj = svdq_from_linear(qkv_proj, context, device=device, **kwargs)
        self.mlp_fc1 = svdq_from_linear(mlp_fc1, context, device=device, **kwargs)
        self.mlp_act_fn = other.mlp_act_fn
        self.norm_q = other.norm_q
        self.norm_k = other.norm_k
        self.out_proj = svdq_from_linear(out_proj, context, device=device, **kwargs)
        self.mlp_fc2 = svdq_from_linear(mlp_fc2, context, device=device, **kwargs)
        self.mlp_fc2.act_unsigned = False
        self.set_processor(NunchakuFlux2ParallelSelfAttnProcessor(getattr(other, "processor", None)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, attention_mask, image_rotary_emb, **kwargs)


# ---------------------------------------------------------------------------
# Stock Diffusers processors with the fused MPS q/k norm + RoPE prologue
# ---------------------------------------------------------------------------
# Manifest checkpoints keep Diffusers' own FLUX.2 attention modules and processors. On Apple silicon these
# processors replace them (see ``core._install_mps_fused_attention``); each call checks whether the fused kernel
# applies (MPS, batch 1, head dim 128, fp16/bf16, no mask, plain rotary tables) and otherwise runs the stock
# processor unchanged.


def _eps(norm, dtype) -> float:
    eps = getattr(norm, "eps", None)
    return float(torch.finfo(dtype).eps if eps is None else eps)


def _norm_weight(norm):
    weight = getattr(norm, "weight", None)
    return weight if weight is None or weight.shape == (128,) else False


def _fusable(attn, hidden_states, attention_mask, image_rotary_emb, norms) -> bool:
    if hidden_states.device.type != "mps" or hidden_states.shape[0] != 1 or attention_mask is not None:
        return False
    if attn.head_dim != 128 or hidden_states.dtype not in (torch.float16, torch.bfloat16):
        return False
    if not (isinstance(image_rotary_emb, tuple) and len(image_rotary_emb) == 2):
        return False
    if any(not torch.is_tensor(t) or t.ndim != 2 or t.shape[-1] != 128 for t in image_rotary_emb):
        return False
    if any(_norm_weight(n) is False for n in norms):
        return False
    return fused_qk_norm_rope_available()


def _mps_attention(processor, query, key, value):
    """Neural Accelerator flash attention when Diffusers would use its native backend, else Diffusers' dispatch."""

    backend = processor._attention_backend or _AttentionBackendRegistry.get_active_backend()[0]
    if (
        backend == AttentionBackendName.NATIVE
        and processor._parallel_config is None
        and attention_supported(query, key, value)
    ):
        return attention_mps(query, key, value)
    return dispatch_attention_fn(
        query, key, value, attn_mask=None, backend=processor._attention_backend,
        parallel_config=processor._parallel_config,
    )  # fmt: skip


class MPSFusedFlux2AttnProcessor(Flux2AttnProcessor):
    """Diffusers ``Flux2AttnProcessor`` with the fused MPS prologue (joint text + image attention)."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        joint = encoder_hidden_states is not None and attn.added_kv_proj_dim is not None
        norms = (attn.norm_q, attn.norm_k) + ((attn.norm_added_q, attn.norm_added_k) if joint else ())
        if not _fusable(attn, hidden_states, attention_mask, image_rotary_emb, norms):
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)

        query, key, value, enc_query, enc_key, enc_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        cos, sin = image_rotary_emb
        query, key, value = qk_norm_rope_mps(
            query, key, value, attn.norm_q.weight, attn.norm_k.weight, cos, sin, attn.heads,
            eps_q=_eps(attn.norm_q, query.dtype), eps_k=_eps(attn.norm_k, query.dtype),
            text=(enc_query, enc_key, enc_value) if joint else None,
            text_q_weight=attn.norm_added_q.weight if joint else None,
            text_k_weight=attn.norm_added_k.weight if joint else None,
        )  # fmt: skip
        hidden_states = _mps_attention(self, query, key, value)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


class MPSFusedFlux2ParallelSelfAttnProcessor(Flux2ParallelSelfAttnProcessor):
    """Diffusers ``Flux2ParallelSelfAttnProcessor`` with the fused MPS prologue (single-stream blocks)."""

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        if not _fusable(attn, hidden_states, attention_mask, image_rotary_emb, (attn.norm_q, attn.norm_k)):
            return super().__call__(attn, hidden_states, attention_mask, image_rotary_emb)

        proj = attn.to_qkv_mlp_proj(hidden_states)
        inner = attn.inner_dim
        if proj.shape[-1] != 3 * inner + attn.mlp_hidden_dim * attn.mlp_mult_factor:  # column-sharded (TP)
            return super().__call__(attn, hidden_states, attention_mask, image_rotary_emb)
        cos, sin = image_rotary_emb
        query, key, value = qk_norm_rope_mps(
            proj[..., :inner], proj[..., inner : 2 * inner], proj[..., 2 * inner : 3 * inner],
            attn.norm_q.weight, attn.norm_k.weight, cos, sin, attn.heads,
            eps_q=_eps(attn.norm_q, proj.dtype), eps_k=_eps(attn.norm_k, proj.dtype),
        )  # fmt: skip
        hidden_states = _mps_attention(self, query, key, value)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        mlp_hidden_states = attn.mlp_act_fn(proj[..., 3 * inner :])
        return attn.to_out(torch.cat([hidden_states, mlp_hidden_states], dim=-1))


_REPLACEMENTS = {
    Flux2AttnProcessor: MPSFusedFlux2AttnProcessor,
    Flux2ParallelSelfAttnProcessor: MPSFusedFlux2ParallelSelfAttnProcessor,
}


def install_mps_fused_attention(transformer: torch.nn.Module) -> int:
    """Swap stock FLUX.2 attention processors for the fused-MPS ones. Returns how many modules were updated."""

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


def lite_flux2_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
    kv_cache=None,
    kv_cache_mode: str | None = None,
    num_ref_tokens: int = 0,
    ref_fixed_timestep: float = 0.0,
) -> torch.Tensor | Transformer2DModelOutput:
    """Run Flux2 with pre-packed RoPE tensors for Nunchaku kernels."""

    if kv_cache_mode is not None:
        return self._nunchaku_lite_flux2_original_forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=return_dict,
            kv_cache=kv_cache,
            kv_cache_mode=kv_cache_mode,
            num_ref_tokens=num_ref_tokens,
            ref_fixed_timestep=ref_fixed_timestep,
        )

    num_txt_tokens = encoder_hidden_states.shape[1]
    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    temb = self.time_guidance_embed(timestep, guidance)
    double_stream_mod_img = self.double_stream_modulation_img(temb)
    double_stream_mod_txt = self.double_stream_modulation_txt(temb)
    single_stream_mod = self.single_stream_modulation(temb)

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]

    image_rotary_emb = self.pos_embed(img_ids)
    text_rotary_emb = self.pos_embed(txt_ids)
    rotary_emb_img = _pack_flux2_rotary_emb(image_rotary_emb)
    rotary_emb_txt = _pack_flux2_rotary_emb(text_rotary_emb)
    rotary_emb_single = _pack_flux2_rotary_emb(
        (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )
    )
    kv_attn_kwargs = joint_attention_kwargs

    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                double_stream_mod_img,
                double_stream_mod_txt,
                (rotary_emb_img, rotary_emb_txt),
                kv_attn_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=double_stream_mod_img,
                temb_mod_txt=double_stream_mod_txt,
                image_rotary_emb=(rotary_emb_img, rotary_emb_txt),
                joint_attention_kwargs=kv_attn_kwargs,
            )

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    for block in self.single_transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                None,
                single_stream_mod,
                rotary_emb_single,
                kv_attn_kwargs,
            )
        else:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod=single_stream_mod,
                image_rotary_emb=rotary_emb_single,
                joint_attention_kwargs=kv_attn_kwargs,
            )

    hidden_states = hidden_states[:, num_txt_tokens:, ...]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


class Flux2Adapter:
    """Adapter for Diffusers ``Flux2Transformer2DModel`` checkpoints."""

    target = "flux2"

    def matches(self, transformer: torch.nn.Module) -> bool:
        """Return whether ``transformer`` is a Diffusers Flux2 transformer.

        Args:
            transformer: Candidate module.

        Returns:
            ``True`` when the class name and module path match Diffusers Flux2.
        """

        return (
            transformer.__class__.__name__ == "Flux2Transformer2DModel"
            and "transformer_flux2" in transformer.__class__.__module__
        )

    def patch(
        self,
        transformer: torch.nn.Module,
        checkpoint_state: dict[str, torch.Tensor],
        quantization_config: dict[str, Any],
        options: PatchOptions,
    ) -> dict[str, torch.Tensor]:
        """Patch a Flux2 transformer in place and install the lite forward wrapper.

        Args:
            transformer: Diffusers Flux2 transformer to mutate.
            checkpoint_state: Checkpoint tensors to load after patching.
            quantization_config: Quantization metadata from the checkpoint.
            options: Normalized patch options.

        Returns:
            Checkpoint state dict to load into the patched transformer.
        """

        context = build_svdq_context(transformer, quantization_config, options)
        prepare_transformer_dtype(transformer, context)
        self._patch_transformer(transformer, context)

        transformer._nunchaku_lite_flux2_original_forward = transformer.forward
        transformer.forward = types.MethodType(lite_flux2_forward, transformer)
        if flux2_state_dict_needs_conversion(checkpoint_state):
            checkpoint_state = convert_flux2_state_dict(checkpoint_state)
        finalize_svdq_checkpoint(transformer, checkpoint_state, context)
        _drop_unused_zero_bias_tensors(transformer, checkpoint_state)
        from ..lora.core.runtime import bind_transformer_lora_methods
        from ..lora.flux2 import NunchakuFlux2TransformerLoraMixin

        bind_transformer_lora_methods(transformer, NunchakuFlux2TransformerLoraMixin)
        transformer._nunchaku_lite_flux2_patched = True
        return checkpoint_state

    def patch_pipeline(
        self,
        pipeline: Any,
        *,
        component_name: str = "transformer",
        component: torch.nn.Module | None = None,
    ) -> None:
        """Attach Flux2 pipeline-level runtime APIs."""

        from ..lora.core.runtime import NunchakuPipelineLoraMixin, bind_pipeline_lora_methods

        bind_pipeline_lora_methods(pipeline, NunchakuPipelineLoraMixin)

    def _patch_transformer(self, transformer: torch.nn.Module, context: SVDQPatchContext) -> None:
        """Patch Flux2 block modules through one recursive transformer traversal.

        Args:
            transformer: Flux2 transformer whose module tree should be patched.
            context: Shared SVDQ patch settings used by lite block
                replacements.

        Returns:
            None.
        """

        patch_modules_recursively(
            transformer,
            skips=lambda _path, module: isinstance(module, nn.Linear),
            module_converters={
                Flux2Attention: lambda _path, attn: NunchakuFlux2Attention(attn, context=context),
                Flux2FeedForward: lambda _path, ff: _patch_flux2_feed_forward(ff, context=context),
                Flux2ParallelSelfAttention: lambda _path, attn: NunchakuFlux2ParallelSelfAttention(
                    attn, context=context
                ),
            },
        )


def _drop_unused_zero_bias_tensors(transformer: torch.nn.Module, checkpoint_state: dict[str, torch.Tensor]) -> None:
    """Drop zero bias tensors emitted by original Flux2 checkpoints for biasless modules."""

    expected_keys = transformer.state_dict().keys()
    for key in list(checkpoint_state):
        if key in expected_keys or not key.endswith(".bias"):
            continue
        value = checkpoint_state[key]
        if torch.count_nonzero(value) == 0:
            checkpoint_state.pop(key)


register_adapter(Flux2Adapter())
