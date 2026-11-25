"""Qwen-Image (and Qwen-Image-Edit) checkpoints in ComfyUI's ``comfy.ldm.qwen_image.model``.

ComfyUI's Qwen-Image model keeps the Diffusers module names, so dense tensors load unchanged. Per block:

==============================  ====================================  =========================================
checkpoint                      ComfyUI ``transformer_blocks.N``      notes
==============================  ====================================  =========================================
``attn.to_qkv``                 ``attn.to_qkv``                       replaces ComfyUI's ``to_q``/``to_k``/``to_v``
``attn.add_qkv_proj``           ``attn.add_qkv_proj``                 replaces ``add_q_proj``/``add_k_proj``/...
``attn.to_out.0``               ``attn.to_out.0``                     SVDQ
``attn.to_add_out``             ``attn.to_add_out``                   SVDQ
``img_mlp.net.0.proj``/``.2``   ``img_mlp.fc1`` / ``img_mlp.fc2``     fused GELU MLP
``txt_mlp.net.0.proj``/``.2``   ``txt_mlp.fc1`` / ``txt_mlp.fc2``     fused GELU MLP
``img_mod.1`` / ``txt_mod.1``   same                                  AWQ W4A16 (interleaved, ``+1`` folded)
==============================  ====================================  =========================================

The attention forward is ComfyUI's, with the six projections replaced by the two fused q|k|v projections.
"""

from __future__ import annotations

import json
import re
from typing import Any

import torch
from torch import nn

from ..adapters.common import SVDQPatchContext
from .common import FusedGeluMLP, make_module, meta_weight, modulation_spec, module_tensors, svdq_spec
from .core import ComfyLoadPlan, register_comfy_family

BLOCK_TARGETS = {
    "attn.to_qkv": "attn.to_qkv",
    "attn.add_qkv_proj": "attn.add_qkv_proj",
    "attn.to_out.0": "attn.to_out.0",
    "attn.to_add_out": "attn.to_add_out",
    "img_mlp.net.0.proj": "img_mlp.fc1",
    "img_mlp.net.2": "img_mlp.fc2",
    "txt_mlp.net.0.proj": "txt_mlp.fc1",
    "txt_mlp.net.2": "txt_mlp.fc2",
}
BLOCK_MODULATION = {"img_mod.1": "img_mod.1", "txt_mod.1": "txt_mod.1"}

# ComfyUI module whose dense ``.weight`` each installed module stands in for during model detection.
_DETECT_NAMES = {
    "attn.to_qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
    "attn.add_qkv_proj": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"),
    "attn.to_out.0": ("attn.to_out.0",),
    "attn.to_add_out": ("attn.to_add_out",),
    "img_mlp.fc1": ("img_mlp.net.0.proj",),
    "img_mlp.fc2": ("img_mlp.net.2",),
    "txt_mlp.fc1": ("txt_mlp.net.0.proj",),
    "txt_mlp.fc2": ("txt_mlp.net.2",),
    "img_mod.1": ("img_mod.1",),
    "txt_mod.1": ("txt_mod.1",),
}

_BLOCK_KEY = re.compile(r"^transformer_blocks\.(\d+)\.")


def _diffusers_config(metadata: dict[str, str]) -> dict[str, Any]:
    raw = metadata.get("config")
    if not raw:
        return {}
    try:
        config = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return config if isinstance(config, dict) else {}


class QwenImageComfyFamily:
    """Qwen-Image, Qwen-Image-Edit (2509, 2511) and their Lightning distillations in ComfyUI's Qwen-Image model."""

    name = "qwen_image"
    supports_input_rotation = False

    def matches(self, state_dict: dict[str, torch.Tensor], metadata: dict[str, str]) -> bool:
        return "txt_norm.weight" in state_dict and any(
            key.startswith("transformer_blocks.0.attn.to_qkv.") for key in state_dict
        )

    def plan(
        self, state_dict: dict[str, torch.Tensor], metadata: dict[str, str], context: SVDQPatchContext
    ) -> ComfyLoadPlan:
        tensors = dict(state_dict)
        block_ids = sorted({int(m.group(1)) for key in tensors if (m := _BLOCK_KEY.match(key))})
        if block_ids != list(range(len(block_ids))):
            raise ValueError("Qwen-Image checkpoint block indices are not contiguous.")

        modules: dict[str, dict[str, Any]] = {}
        quant: dict[str, torch.Tensor] = {}
        consumed: set[str] = set()

        def take(src_prefix: str, dst_prefix: str, spec: dict[str, Any]) -> None:
            for suffix, value in module_tensors(tensors, src_prefix).items():
                quant[f"{dst_prefix}.{suffix}"] = value
                consumed.add(f"{src_prefix}.{suffix}")
            modules[dst_prefix] = spec

        for index in block_ids:
            prefix = f"transformer_blocks.{index}"
            for src, dst in BLOCK_TARGETS.items():
                if f"{prefix}.{src}.qweight" not in tensors:
                    raise ValueError(f"Qwen-Image checkpoint has no quantized {prefix}.{src}.")
                spec = svdq_spec(tensors, f"{prefix}.{src}", fused_mlp_fc2=dst.endswith(".fc2"))
                take(f"{prefix}.{src}", f"{prefix}.{dst}", spec)
            for src, dst in BLOCK_MODULATION.items():
                spec = modulation_spec(tensors, f"{prefix}.{src}", chunks=6, scale_chunks=(1, 4))
                if spec is not None:
                    take(f"{prefix}.{src}", f"{prefix}.{dst}", spec)

        dense = {key: value for key, value in tensors.items() if key not in consumed}
        stray = sorted(key for key in dense if key.rsplit(".", 1)[-1] not in ("weight", "bias"))
        if stray:
            raise ValueError(f"Unrecognized tensors in Qwen-Image checkpoint: {stray[:8]}.")

        detect = dict(dense)
        for path, spec in modules.items():
            _, index, local = path.split(".", 2)
            names = _DETECT_NAMES[local]
            for name in names:
                out_features = spec["out_features"] // len(names)
                detect[f"transformer_blocks.{index}.{name}.weight"] = meta_weight(out_features, spec["in_features"])
        if _diffusers_config(metadata).get("zero_cond_t"):  # Qwen-Image-Edit-2511: ComfyUI's detection marker
            detect["__index_timestep_zero__"] = torch.empty(0, device="meta")
        return ComfyLoadPlan(dense=dense, quant=quant, detect=detect, modules=modules)

    def install(self, diffusion_model: nn.Module, plan: ComfyLoadPlan, context: SVDQPatchContext) -> None:
        from comfy.ldm.qwen_image.model import Attention, QwenImageTransformerBlock

        blocks = getattr(diffusion_model, "transformer_blocks", None)
        if blocks is None:
            raise TypeError(f"Expected ComfyUI's Qwen-Image model, got {type(diffusion_model).__name__}.")
        expected = len({path.split(".")[1] for path in plan.modules})
        if len(blocks) != expected:
            raise ValueError(f"ComfyUI's Qwen-Image model has {len(blocks)} blocks, the checkpoint {expected}.")

        attention_cls = _attention_class()
        for index, block in enumerate(blocks):
            if not isinstance(block, QwenImageTransformerBlock) or not isinstance(block.attn, Attention):
                raise TypeError("Qwen-Image blocks must be ComfyUI QwenImageTransformerBlocks.")
            prefix = f"transformer_blocks.{index}"
            attn = block.attn
            for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
                delattr(attn, name)
            attn.to_qkv = make_module(plan.modules[f"{prefix}.attn.to_qkv"], context)
            attn.add_qkv_proj = make_module(plan.modules[f"{prefix}.attn.add_qkv_proj"], context)
            if attn.to_qkv.out_features != 3 * attn.inner_dim:
                raise ValueError(f"{prefix}.attn.to_qkv width does not match ComfyUI's attention width.")
            attn.to_out[0] = make_module(plan.modules[f"{prefix}.attn.to_out.0"], context)
            attn.to_add_out = make_module(plan.modules[f"{prefix}.attn.to_add_out"], context)
            attn.__class__ = attention_cls
            for side in ("img", "txt"):
                fc1 = make_module(plan.modules[f"{prefix}.{side}_mlp.fc1"], context)
                fc2 = make_module(plan.modules[f"{prefix}.{side}_mlp.fc2"], context)
                setattr(block, f"{side}_mlp", FusedGeluMLP(fc1, fc2))
                mod_path = f"{prefix}.{side}_mod.1"
                if mod_path in plan.modules:
                    getattr(block, f"{side}_mod")[1] = make_module(plan.modules[mod_path], context, mod_path)


_ATTENTION_CLASS = None


def _attention_class():
    """ComfyUI Qwen-Image ``Attention`` subclass projecting q|k|v with one W4A4 GEMM per stream."""

    global _ATTENTION_CLASS
    if _ATTENTION_CLASS is not None:
        return _ATTENTION_CLASS

    from comfy.ldm.flux.math import apply_rope1
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention_masked
    from comfy.ldm.qwen_image.model import Attention

    class NunchakuQwenImageAttention(Attention):
        """Mirrors ComfyUI's ``Attention.forward`` with fused ``to_qkv`` / ``add_qkv_proj`` projections."""

        def forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            encoder_hidden_states_mask=None,
            attention_mask=None,
            image_rotary_emb=None,
            transformer_options={},  # noqa: B006
        ):
            batch_size, seq_img = hidden_states.shape[:2]
            seq_txt = encoder_hidden_states.shape[1]
            transformer_patches = transformer_options.get("patches", {})
            extra_options = transformer_options.copy()

            def heads(x, seq):
                return x.view(batch_size, seq, self.heads, -1).transpose(1, 2)

            img_query, img_key, img_value = self.to_qkv(hidden_states).chunk(3, dim=-1)
            txt_query, txt_key, txt_value = self.add_qkv_proj(encoder_hidden_states).chunk(3, dim=-1)
            img_query = self.norm_q(heads(img_query, seq_img).contiguous())
            img_key = self.norm_k(heads(img_key, seq_img).contiguous())
            txt_query = self.norm_added_q(heads(txt_query, seq_txt).contiguous())
            txt_key = self.norm_added_k(heads(txt_key, seq_txt).contiguous())

            joint_query = torch.cat([txt_query, img_query], dim=2)
            joint_key = torch.cat([txt_key, img_key], dim=2)
            joint_value = torch.cat([heads(txt_value, seq_txt), heads(img_value, seq_img)], dim=2)
            del img_query, img_key, img_value, txt_query, txt_key, txt_value

            if encoder_hidden_states_mask is not None:
                attn_mask = torch.zeros(
                    (batch_size, 1, seq_txt + seq_img), dtype=hidden_states.dtype, device=hidden_states.device
                )
                attn_mask[:, 0, :seq_txt] = encoder_hidden_states_mask
            else:
                attn_mask = None

            extra_options["img_slice"] = [seq_txt, joint_query.shape[2]]
            if "attn1_patch" in transformer_patches:
                for patch in transformer_patches["attn1_patch"]:
                    out = patch(
                        joint_query,
                        joint_key,
                        joint_value,
                        pe=image_rotary_emb,
                        attn_mask=encoder_hidden_states_mask,
                        extra_options=extra_options,
                    )
                    joint_query, joint_key, joint_value = (
                        out.get("q", joint_query),
                        out.get("k", joint_key),
                        out.get("v", joint_value),
                    )
                    image_rotary_emb = out.get("pe", image_rotary_emb)
                    encoder_hidden_states_mask = out.get("attn_mask", encoder_hidden_states_mask)

            joint_query = apply_rope1(joint_query, image_rotary_emb)
            joint_key = apply_rope1(joint_key, image_rotary_emb)
            joint_hidden_states = optimized_attention_masked(
                AttentionTensorContainer(joint_query),
                AttentionTensorContainer(joint_key),
                AttentionTensorContainer(joint_value),
                self.heads,
                attn_mask,
                transformer_options=transformer_options,
                skip_reshape=True,
                preferred_attention=self.comfy_attention,
            )

            txt_attn_output = joint_hidden_states[:, :seq_txt, :]
            img_attn_output = joint_hidden_states[:, seq_txt:, :]
            img_attn_output = self.to_out[1](self.to_out[0](img_attn_output.contiguous()))
            txt_attn_output = self.to_add_out(txt_attn_output.contiguous())
            return img_attn_output, txt_attn_output

    _ATTENTION_CLASS = NunchakuQwenImageAttention
    return _ATTENTION_CLASS


register_comfy_family(QwenImageComfyFamily())
