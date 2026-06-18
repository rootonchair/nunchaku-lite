"""LTX2 adapter for patching Diffusers LTX2 transformers with Nunchaku Lite modules."""

import types
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.activations import GELU
from diffusers.models.attention import FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_ltx2 import (
    LTX2AdaLayerNormSingle,
    LTX2Attention,
    LTX2AudioVideoAttnProcessor,
    LTX2PerturbedAttnProcessor,
    LTX2VideoTransformer3DModel,
)

from ..core import PatchOptions, register_adapter
from ..linear import AWQW4A16Linear, SVDQW4A4Linear
from ..ops.fused import fused_gelu_mlp, fused_cross_head_qk_norm_rope
from .common import (
    SVDQPatchContext,
    build_svdq_context,
    finalize_svdq_checkpoint,
    patch_modules_recursively,
    prepare_transformer_dtype,
    svdq_from_linear,
)


def _patch_ltx2_feed_forward(ff: FeedForward, context: SVDQPatchContext) -> FeedForward:
    patch_modules_recursively(
        ff,
        module_converters={nn.Linear: lambda _path, linear: svdq_from_linear(linear, context)},
    )
    if len(ff.net) > 2 and isinstance(ff.net[2], SVDQW4A4Linear):
        ff.net[2].act_unsigned = False
    ff._nunchaku_lite_ltx2_original_forward = ff.forward
    ff.forward = types.MethodType(_ltx2_feed_forward, ff)
    return ff


def _ltx2_feed_forward(self: FeedForward, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    if (
        hidden_states.is_cuda
        and not torch.is_grad_enabled()
        and not args
        and not kwargs
        and len(self.net) > 2
        and isinstance(self.net[0], GELU)
        and isinstance(getattr(self.net[0], "proj", None), SVDQW4A4Linear)
        and isinstance(self.net[2], SVDQW4A4Linear)
    ):
        return fused_gelu_mlp(hidden_states, self.net[0].proj, self.net[2])
    return self._nunchaku_lite_ltx2_original_forward(hidden_states, *args, **kwargs)


def _patch_ltx2_attention(
    path: str,
    attn: LTX2Attention,
    checkpoint_state: dict[str, torch.Tensor],
    context: SVDQPatchContext,
    awq_group_sizes: dict[str, int],
) -> LTX2Attention:
    attn.to_q = svdq_from_linear(attn.to_q, context)
    attn.to_k = svdq_from_linear(attn.to_k, context)
    attn.to_v = svdq_from_linear(attn.to_v, context)
    attn.to_out[0] = svdq_from_linear(attn.to_out[0], context)

    gate_logits_prefix = f"{path}.to_gate_logits"
    if attn.to_gate_logits is not None and f"{gate_logits_prefix}.qweight" in checkpoint_state:
        attn.to_gate_logits = AWQW4A16Linear.from_linear(
            attn.to_gate_logits,
            group_size=awq_group_sizes.get(gate_logits_prefix, 64),
            torch_dtype=context.torch_dtype,
        )

    original_processor = getattr(attn, "processor", None)
    if isinstance(original_processor, LTX2PerturbedAttnProcessor):
        processor = NunchakuLTX2PerturbedAttnProcessor(original_processor)
    else:
        processor = NunchakuLTX2AudioVideoAttnProcessor(original_processor)
    attn.set_processor(processor)
    attn._nunchaku_lite_ltx2_attention_patched = True
    return attn


def _awq_group_sizes(quantization_config: dict[str, Any]) -> dict[str, int]:
    manifest = quantization_config.get("runtime_manifest")
    if not isinstance(manifest, dict):
        return {}

    group_sizes: dict[str, int] = {}
    for target in manifest.get("targets", []):
        if not isinstance(target, dict):
            continue
        if target.get("nunchaku_op") not in {"awq_w4a16", "adanorm_awq_w4a16"}:
            continue
        prefix = target.get("checkpoint_prefix")
        group_size = target.get("group_size")
        if isinstance(prefix, str) and isinstance(group_size, int):
            group_sizes[prefix] = group_size
    return group_sizes


def _patch_ltx2_adaln_single(
    path: str,
    module: LTX2AdaLayerNormSingle,
    checkpoint_state: dict[str, torch.Tensor],
    context: SVDQPatchContext,
    awq_group_sizes: dict[str, int],
) -> LTX2AdaLayerNormSingle:
    linear_prefix = f"{path}.linear"
    if f"{linear_prefix}.qweight" not in checkpoint_state:
        return module
    module.linear = AWQW4A16Linear.from_linear(
        module.linear,
        group_size=awq_group_sizes.get(linear_prefix, 64),
        torch_dtype=context.torch_dtype,
    )
    return module


def _apply_fused_cross_head_qk_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    norm_q: nn.Module,
    norm_k: nn.Module,
    query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    rope_type: str,
    heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query_rotary_emb is None:
        raise ValueError("fused cross-head Q/K norm+RoPE requires query rotary embeddings")
    if rope_type not in {"split", "interleaved"}:
        raise ValueError("fused cross-head Q/K norm+RoPE supports only split and interleaved RoPE")

    q_heads = int(heads)
    head_dim = int(head_dim)
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    k_heads = int(key.shape[-1] // head_dim)
    return fused_cross_head_qk_norm_rope(
        query,
        key,
        norm_q,
        norm_k,
        query_rotary_emb,
        key_rotary_emb if key_rotary_emb is not None else query_rotary_emb,
        q_heads=q_heads,
        k_heads=k_heads,
        head_dim=head_dim,
        rope_type=rope_type,
    )


class NunchakuLTX2AudioVideoAttnProcessor:
    """LTX2 attention processor that keeps separate Q/K/V SVDQ projections."""

    def __init__(self, processor=None):
        self._attention_backend = getattr(processor, "_attention_backend", None)
        self._parallel_config = getattr(processor, "_parallel_config", None)

    def __call__(
        self,
        attn: LTX2Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        gate_logits = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if query_rotary_emb is None:
            query = attn.norm_q(query)
            key = attn.norm_k(key)
        else:
            query, key = _apply_fused_cross_head_qk_norm_rope(
                query,
                key,
                attn.norm_q,
                attn.norm_k,
                query_rotary_emb,
                key_rotary_emb,
                attn.rope_type,
                attn.heads,
                attn.head_dim,
            )

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if gate_logits is not None:
            hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
            gates = 2.0 * torch.sigmoid(gate_logits)
            hidden_states = (hidden_states * gates.unsqueeze(-1)).flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


class NunchakuLTX2PerturbedAttnProcessor(NunchakuLTX2AudioVideoAttnProcessor):
    """LTX2 STG processor preserving perturbation masking behavior."""

    def __call__(
        self,
        attn: LTX2Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        gate_logits = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None
        value = attn.to_v(encoder_hidden_states)
        if all_perturbed is None:
            all_perturbed = torch.all(perturbation_mask == 0) if perturbation_mask is not None else False

        if all_perturbed:
            hidden_states = value
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)

            if query_rotary_emb is None:
                query = attn.norm_q(query)
                key = attn.norm_k(key)
            else:
                query, key = _apply_fused_cross_head_qk_norm_rope(
                    query,
                    key,
                    attn.norm_q,
                    attn.norm_k,
                    query_rotary_emb,
                    key_rotary_emb,
                    attn.rope_type,
                    attn.heads,
                    attn.head_dim,
                )

            query = query.unflatten(2, (attn.heads, -1))
            key = key.unflatten(2, (attn.heads, -1))
            value_heads = value.unflatten(2, (attn.heads, -1))

            hidden_states = dispatch_attention_fn(
                query,
                key,
                value_heads,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

            if perturbation_mask is not None:
                hidden_states = torch.lerp(value, hidden_states, perturbation_mask)

        if gate_logits is not None:
            hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
            gates = 2.0 * torch.sigmoid(gate_logits)
            hidden_states = (hidden_states * gates.unsqueeze(-1)).flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


class LTX2Adapter:
    """Adapter for Diffusers ``LTX2VideoTransformer3DModel`` checkpoints."""

    target = "ltx2"

    def matches(self, transformer: torch.nn.Module) -> bool:
        return isinstance(transformer, LTX2VideoTransformer3DModel)

    def patch(
        self,
        transformer: torch.nn.Module,
        checkpoint_state: dict[str, torch.Tensor],
        quantization_config: dict[str, Any],
        options: PatchOptions,
    ) -> dict[str, torch.Tensor]:
        context = build_svdq_context(transformer, quantization_config, options)
        prepare_transformer_dtype(transformer, context)
        self._patch_transformer(transformer, context, checkpoint_state, quantization_config)
        finalize_svdq_checkpoint(transformer, checkpoint_state, context)
        transformer._nunchaku_lite_ltx2_patched = True
        return checkpoint_state

    def _patch_transformer(
        self,
        transformer: torch.nn.Module,
        context: SVDQPatchContext,
        checkpoint_state: dict[str, torch.Tensor],
        quantization_config: dict[str, Any],
    ) -> None:
        awq_group_sizes = _awq_group_sizes(quantization_config)
        patch_modules_recursively(
            transformer,
            skips=lambda _path, module: isinstance(module, nn.Linear),
            module_converters={
                LTX2AdaLayerNormSingle: lambda path, module: _patch_ltx2_adaln_single(
                    path,
                    module,
                    checkpoint_state,
                    context,
                    awq_group_sizes,
                ),
                LTX2Attention: lambda path, attn: _patch_ltx2_attention(
                    path,
                    attn,
                    checkpoint_state,
                    context,
                    awq_group_sizes,
                ),
                FeedForward: lambda _path, ff: _patch_ltx2_feed_forward(ff, context),
            },
        )


register_adapter(LTX2Adapter())
