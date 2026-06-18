"""LTX2 adapter for patching Diffusers LTX2 transformers with Nunchaku Lite modules."""

import types
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.activations import GELU
from diffusers.models.attention import FeedForward
from diffusers.models.transformers.transformer_ltx2 import (
    LTX2AdaLayerNormSingle,
    LTX2Attention,
    LTX2AudioVideoAttnProcessor,
    LTX2PerturbedAttnProcessor,
    LTX2VideoTransformer3DModel,
    LTX2VideoTransformerBlock,
)

from .attention_dispatch import dispatch_lite_attention_fn
from ..core import PatchOptions, register_adapter
from ..linear import AWQW4A16Linear, SVDQW4A4Linear
from ..ops.fused import (
    fused_affine_modulate,
    fused_cross_head_qk_norm_rope,
    fused_gelu_mlp,
    fused_rms_norm_modulate,
)
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

        hidden_states = dispatch_lite_attention_fn(
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

            hidden_states = dispatch_lite_attention_fn(
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


def _modulated_norm(norm: nn.Module, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return fused_rms_norm_modulate(x, norm, scale, shift)


def _affine_modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return fused_affine_modulate(x, scale, shift)


def _broadcast_ltx2_gate(branch: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if branch.ndim == 3 and gate.ndim == 2 and gate.shape == (branch.shape[0], branch.shape[-1]):
        return gate.unsqueeze(1)
    return gate


def _residual_gate(residual: torch.Tensor, gate: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
    return residual + _broadcast_ltx2_gate(branch, gate) * branch


def _ltx2_block_forward(
    self: LTX2VideoTransformerBlock,
    hidden_states: torch.Tensor,
    audio_hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    audio_encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    temb_audio: torch.Tensor,
    temb_ca_scale_shift: torch.Tensor,
    temb_ca_audio_scale_shift: torch.Tensor,
    temb_ca_gate: torch.Tensor,
    temb_ca_audio_gate: torch.Tensor,
    temb_prompt: torch.Tensor | None = None,
    temb_prompt_audio: torch.Tensor | None = None,
    video_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    audio_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ca_video_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ca_audio_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    encoder_attention_mask: torch.Tensor | None = None,
    audio_encoder_attention_mask: torch.Tensor | None = None,
    self_attention_mask: torch.Tensor | None = None,
    audio_self_attention_mask: torch.Tensor | None = None,
    a2v_cross_attention_mask: torch.Tensor | None = None,
    v2a_cross_attention_mask: torch.Tensor | None = None,
    use_a2v_cross_attention: bool = True,
    use_v2a_cross_attention: bool = True,
    perturbation_mask: torch.Tensor | None = None,
    all_perturbed: bool | None = None,
) -> torch.Tensor:
    batch_size = hidden_states.size(0)

    video_ada_params = self.get_mod_params(self.scale_shift_table, temb, batch_size)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = video_ada_params[:6]
    if self.video_cross_attn_adaln:
        shift_text_q, scale_text_q, gate_text_q = video_ada_params[6:9]

    norm_hidden_states = _modulated_norm(self.norm1, hidden_states, scale_msa, shift_msa)
    video_self_attn_args = {
        "hidden_states": norm_hidden_states,
        "encoder_hidden_states": None,
        "query_rotary_emb": video_rotary_emb,
        "attention_mask": self_attention_mask,
    }
    if self.perturbed_attn:
        video_self_attn_args["perturbation_mask"] = perturbation_mask
        video_self_attn_args["all_perturbed"] = all_perturbed
    hidden_states = _residual_gate(hidden_states, gate_msa, self.attn1(**video_self_attn_args))

    audio_ada_params = self.get_mod_params(self.audio_scale_shift_table, temb_audio, batch_size)
    audio_shift_msa, audio_scale_msa, audio_gate_msa, audio_shift_mlp, audio_scale_mlp, audio_gate_mlp = (
        audio_ada_params[:6]
    )
    if self.audio_cross_attn_adaln:
        audio_shift_text_q, audio_scale_text_q, audio_gate_text_q = audio_ada_params[6:9]

    norm_audio_hidden_states = _modulated_norm(
        self.audio_norm1, audio_hidden_states, audio_scale_msa, audio_shift_msa
    )
    audio_self_attn_args = {
        "hidden_states": norm_audio_hidden_states,
        "encoder_hidden_states": None,
        "query_rotary_emb": audio_rotary_emb,
        "attention_mask": audio_self_attention_mask,
    }
    if self.perturbed_attn:
        audio_self_attn_args["perturbation_mask"] = perturbation_mask
        audio_self_attn_args["all_perturbed"] = all_perturbed
    audio_hidden_states = _residual_gate(audio_hidden_states, audio_gate_msa, self.audio_attn1(**audio_self_attn_args))

    if self.cross_attn_adaln:
        video_prompt_ada_params = self.get_mod_params(self.prompt_scale_shift_table, temb_prompt, batch_size)
        shift_text_kv, scale_text_kv = video_prompt_ada_params
        audio_prompt_ada_params = self.get_mod_params(
            self.audio_prompt_scale_shift_table, temb_prompt_audio, batch_size
        )
        audio_shift_text_kv, audio_scale_text_kv = audio_prompt_ada_params

    norm_hidden_states = self.norm2(hidden_states)
    if self.video_cross_attn_adaln:
        norm_hidden_states = _affine_modulate(norm_hidden_states, scale_text_q, shift_text_q)
    if self.cross_attn_adaln:
        encoder_hidden_states = _affine_modulate(encoder_hidden_states, scale_text_kv, shift_text_kv)

    attn_hidden_states = self.attn2(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        query_rotary_emb=None,
        attention_mask=encoder_attention_mask,
    )
    if self.video_cross_attn_adaln:
        attn_hidden_states = attn_hidden_states * gate_text_q
    hidden_states = hidden_states + attn_hidden_states

    norm_audio_hidden_states = self.audio_norm2(audio_hidden_states)
    if self.audio_cross_attn_adaln:
        norm_audio_hidden_states = _affine_modulate(
            norm_audio_hidden_states, audio_scale_text_q, audio_shift_text_q
        )
    if self.cross_attn_adaln:
        audio_encoder_hidden_states = _affine_modulate(
            audio_encoder_hidden_states, audio_scale_text_kv, audio_shift_text_kv
        )

    attn_audio_hidden_states = self.audio_attn2(
        norm_audio_hidden_states,
        encoder_hidden_states=audio_encoder_hidden_states,
        query_rotary_emb=None,
        attention_mask=audio_encoder_attention_mask,
    )
    if self.audio_cross_attn_adaln:
        attn_audio_hidden_states = attn_audio_hidden_states * audio_gate_text_q
    audio_hidden_states = audio_hidden_states + attn_audio_hidden_states

    if use_a2v_cross_attention or use_v2a_cross_attention:
        norm_hidden_states = self.audio_to_video_norm(hidden_states)
        norm_audio_hidden_states = self.video_to_audio_norm(audio_hidden_states)

        video_per_layer_ca_scale_shift = self.video_a2v_cross_attn_scale_shift_table[:4, :]
        video_per_layer_ca_gate = self.video_a2v_cross_attn_scale_shift_table[4:, :]
        video_ca_ada_params = self.get_mod_params(video_per_layer_ca_scale_shift, temb_ca_scale_shift, batch_size)
        video_ca_gate_param = self.get_mod_params(video_per_layer_ca_gate, temb_ca_gate, batch_size)
        video_a2v_ca_scale, video_a2v_ca_shift, video_v2a_ca_scale, video_v2a_ca_shift = video_ca_ada_params
        a2v_gate = video_ca_gate_param[0].squeeze(2)

        audio_per_layer_ca_scale_shift = self.audio_a2v_cross_attn_scale_shift_table[:4, :]
        audio_per_layer_ca_gate = self.audio_a2v_cross_attn_scale_shift_table[4:, :]
        audio_ca_ada_params = self.get_mod_params(
            audio_per_layer_ca_scale_shift, temb_ca_audio_scale_shift, batch_size
        )
        audio_ca_gate_param = self.get_mod_params(audio_per_layer_ca_gate, temb_ca_audio_gate, batch_size)
        audio_a2v_ca_scale, audio_a2v_ca_shift, audio_v2a_ca_scale, audio_v2a_ca_shift = audio_ca_ada_params
        v2a_gate = audio_ca_gate_param[0].squeeze(2)

        if use_a2v_cross_attention:
            mod_norm_hidden_states = _affine_modulate(
                norm_hidden_states, video_a2v_ca_scale.squeeze(2), video_a2v_ca_shift.squeeze(2)
            )
            mod_norm_audio_hidden_states = _affine_modulate(
                norm_audio_hidden_states, audio_a2v_ca_scale.squeeze(2), audio_a2v_ca_shift.squeeze(2)
            )
            a2v_attn_hidden_states = self.audio_to_video_attn(
                mod_norm_hidden_states,
                encoder_hidden_states=mod_norm_audio_hidden_states,
                query_rotary_emb=ca_video_rotary_emb,
                key_rotary_emb=ca_audio_rotary_emb,
                attention_mask=a2v_cross_attention_mask,
            )
            hidden_states = _residual_gate(hidden_states, a2v_gate, a2v_attn_hidden_states)

        if use_v2a_cross_attention:
            mod_norm_hidden_states = _affine_modulate(
                norm_hidden_states, video_v2a_ca_scale.squeeze(2), video_v2a_ca_shift.squeeze(2)
            )
            mod_norm_audio_hidden_states = _affine_modulate(
                norm_audio_hidden_states, audio_v2a_ca_scale.squeeze(2), audio_v2a_ca_shift.squeeze(2)
            )
            v2a_attn_hidden_states = self.video_to_audio_attn(
                mod_norm_audio_hidden_states,
                encoder_hidden_states=mod_norm_hidden_states,
                query_rotary_emb=ca_audio_rotary_emb,
                key_rotary_emb=ca_video_rotary_emb,
                attention_mask=v2a_cross_attention_mask,
            )
            audio_hidden_states = _residual_gate(audio_hidden_states, v2a_gate, v2a_attn_hidden_states)

    norm_hidden_states = _modulated_norm(self.norm3, hidden_states, scale_mlp, shift_mlp)
    hidden_states = _residual_gate(hidden_states, gate_mlp, self.ff(norm_hidden_states))

    norm_audio_hidden_states = _modulated_norm(
        self.audio_norm3, audio_hidden_states, audio_scale_mlp, audio_shift_mlp
    )
    audio_hidden_states = _residual_gate(
        audio_hidden_states, audio_gate_mlp, self.audio_ff(norm_audio_hidden_states)
    )

    return hidden_states, audio_hidden_states


def _patch_ltx2_block(block: LTX2VideoTransformerBlock) -> LTX2VideoTransformerBlock:
    block._nunchaku_lite_ltx2_original_forward = block.forward
    block.forward = types.MethodType(_ltx2_block_forward, block)
    return block


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
                LTX2VideoTransformerBlock: lambda _path, block: _patch_ltx2_block(block),
            },
        )


register_adapter(LTX2Adapter())
