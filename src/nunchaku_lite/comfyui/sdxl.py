"""SDXL UNet checkpoints in ComfyUI's ``comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel``.

Nunchaku SDXL checkpoints use Diffusers UNet names and quantize the transformer blocks: ``attn1.to_qkv`` (fused
self-attention projection), ``attn1.to_out.0``, ``attn2.to_q``, ``attn2.to_out.0``, ``ff.net.0.proj`` (GEGLU input)
and ``ff.net.2``. Every other tensor is dense.

Module paths are translated with ComfyUI's own ``unet_to_diffusers`` map for the detected UNet config. ComfyUI's
transformer blocks use the Diffusers names below the block (``attn1``, ``attn2``, ``ff.net``), so the quantized
layers drop in, except the fused self-attention projection: ComfyUI's ``CrossAttention`` has separate
``to_q``/``to_k``/``to_v``, so self-attention layers get a forward that projects q|k|v with one W4A4 GEMM. All W4A4
layers run on the signed standalone path, as in the Diffusers adapter.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..adapters.common import SVDQPatchContext
from ..adapters.sdxl import convert_sdxl_state_dict
from .common import make_module, meta_weight, module_tensors, svdq_spec
from .core import ComfyLoadPlan, register_comfy_family

QUANTIZED_SUFFIXES = ("attn1.to_qkv", "attn1.to_out.0", "attn2.to_q", "attn2.to_out.0", "ff.net.0.proj", "ff.net.2")


def _set_submodule(root: nn.Module, path: str, module: nn.Module) -> nn.Module:
    parent_path, _, name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    old = parent[int(name)] if name.isdigit() else getattr(parent, name)
    if name.isdigit():
        parent[int(name)] = module
    else:
        setattr(parent, name, module)
    return old


class SDXLComfyFamily:
    """SDXL (base, Turbo and fine-tunes) UNet checkpoints in ComfyUI's UNet."""

    name = "sdxl"
    supports_input_rotation = False

    def matches(self, state_dict: dict[str, torch.Tensor], metadata: dict[str, str]) -> bool:
        return "conv_in.weight" in state_dict and any(
            key.endswith("attn1.to_qkv.qweight") and ".attentions." in key for key in state_dict
        )

    def plan(
        self, state_dict: dict[str, torch.Tensor], metadata: dict[str, str], context: SVDQPatchContext
    ) -> ComfyLoadPlan:
        import comfy.model_detection
        import comfy.utils

        tensors = convert_sdxl_state_dict(dict(state_dict))
        prefixes = sorted({key[: -len(".qweight")] for key in tensors if key.endswith(".qweight")})
        for prefix in prefixes:
            if not prefix.endswith(QUANTIZED_SUFFIXES) or ".transformer_blocks." not in prefix:
                raise ValueError(f"Unexpected quantized SDXL layer: {prefix}.")
        specs = {prefix: svdq_spec(tensors, prefix) for prefix in prefixes}
        consumed = {f"{prefix}.{suffix}" for prefix in prefixes for suffix in module_tensors(tensors, prefix)}
        dense_diffusers = {key: value for key, value in tensors.items() if key not in consumed}

        # Diffusers-named detection view: dense tensors plus dense-shape stand-ins for the quantized layers.
        detect_diffusers = dict(dense_diffusers)
        for prefix, spec in specs.items():
            if prefix.endswith("attn1.to_qkv"):
                base = prefix[: -len("to_qkv")]
                for name in ("to_q", "to_k", "to_v"):
                    detect_diffusers[f"{base}{name}.weight"] = meta_weight(
                        spec["out_features"] // 3, spec["in_features"]
                    )
            else:
                detect_diffusers[f"{prefix}.weight"] = meta_weight(spec["out_features"], spec["in_features"])
        model_config = comfy.model_detection.model_config_from_diffusers_unet(detect_diffusers)
        if model_config is None:
            raise ValueError("ComfyUI does not recognize the UNet architecture of this SDXL checkpoint.")
        key_map = comfy.utils.unet_to_diffusers(model_config.unet_config)  # Diffusers key -> ComfyUI key

        def comfy_key(key: str) -> str:
            if key not in key_map:
                raise ValueError(f"Unmapped tensor in SDXL checkpoint: {key}.")
            return key_map[key]

        dense = {comfy_key(key): value for key, value in dense_diffusers.items()}
        detect = {comfy_key(key): value for key, value in detect_diffusers.items()}
        modules: dict[str, dict[str, Any]] = {}
        quant: dict[str, torch.Tensor] = {}
        for prefix, spec in specs.items():
            if prefix.endswith("attn1.to_qkv"):
                path = comfy_key(f"{prefix[: -len('to_qkv')]}to_q.weight")[: -len("to_q.weight")] + "to_qkv"
            else:
                path = comfy_key(f"{prefix}.weight")[: -len(".weight")]
            modules[path] = spec
            for suffix, value in module_tensors(tensors, prefix).items():
                quant[f"{path}.{suffix}"] = value
        return ComfyLoadPlan(dense=dense, quant=quant, detect=detect, modules=modules)

    def install(self, diffusion_model: nn.Module, plan: ComfyLoadPlan, context: SVDQPatchContext) -> None:
        from comfy.ldm.modules.attention import CrossAttention

        attention_cls = _self_attention_class()
        for path, spec in plan.modules.items():
            module = make_module(spec, context, path)
            if path.endswith("attn1.to_qkv"):
                attn = diffusion_model.get_submodule(path[: -len(".to_qkv")])
                if not isinstance(attn, CrossAttention) or attn.to_q.in_features != attn.to_k.in_features:
                    raise TypeError(f"{path}: expected ComfyUI self-attention.")
                if module.out_features != 3 * attn.to_q.out_features:
                    raise ValueError(f"{path} width does not match ComfyUI's attention.")
                del attn.to_q, attn.to_k, attn.to_v
                attn.to_qkv = module
                attn.__class__ = attention_cls
                continue
            old = _set_submodule(diffusion_model, path, module)
            if (old.in_features, old.out_features) != (module.in_features, module.out_features):
                raise ValueError(f"{path}: checkpoint layer shape does not match ComfyUI's model.")


_SELF_ATTENTION_CLASS = None


def _self_attention_class():
    """ComfyUI ``CrossAttention`` subclass for self-attention with a fused W4A4 ``to_qkv`` projection."""

    global _SELF_ATTENTION_CLASS
    if _SELF_ATTENTION_CLASS is not None:
        return _SELF_ATTENTION_CLASS

    from comfy.ldm.modules.attention import (
        AttentionTensorContainer,
        CrossAttention,
        optimized_attention,
        optimized_attention_masked,
    )

    class NunchakuFusedSelfAttention(CrossAttention):
        """``CrossAttention.forward`` with q, k and v sliced from one fused projection.

        Attention patches may pass a separate ``context`` (keys/values) or ``value``; the fused projection is then
        applied to that tensor and only its k or v slice is used, which equals the separate projections.
        """

        def forward(self, x, context=None, value=None, mask=None, transformer_options={}):  # noqa: B006
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            if context is not None:
                k, v = self.to_qkv(context).chunk(3, dim=-1)[1:]
            if value is not None:
                v = self.to_qkv(value).chunk(3, dim=-1)[2]
            q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
            if mask is None:
                out = optimized_attention(
                    q,
                    k,
                    v,
                    self.heads,
                    attn_precision=self.attn_precision,
                    preferred_attention=self.comfy_attention,
                    transformer_options=transformer_options,
                )
            else:
                out = optimized_attention_masked(
                    q,
                    k,
                    v,
                    self.heads,
                    mask,
                    attn_precision=self.attn_precision,
                    preferred_attention=self.comfy_attention,
                    transformer_options=transformer_options,
                )
            return self.to_out(out)

    _SELF_ATTENTION_CLASS = NunchakuFusedSelfAttention
    return _SELF_ATTENTION_CLASS


register_comfy_family(SDXLComfyFamily())
