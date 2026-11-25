"""FLUX.1 checkpoints in ComfyUI's ``comfy.ldm.flux.model.Flux``.

Checkpoint (Nunchaku FLUX.1 layout, normalized by :func:`nunchaku_lite.adapters.flux.convert_flux_state_dict`)
to ComfyUI module mapping:

====================================  ====================================  =====================================
checkpoint (per double block)         ComfyUI ``double_blocks.N``           notes
====================================  ====================================  =====================================
``attn.to_qkv``                       ``img_attn.qkv``                      SVDQ, q|k|v rows as in ComfyUI
``attn.add_qkv_proj``                 ``txt_attn.qkv``                      SVDQ
``attn.to_out.0``                     ``img_attn.proj``                     SVDQ
``attn.to_add_out``                   ``txt_attn.proj``                     SVDQ
``ff.net.0.proj`` / ``ff.net.2``      ``img_mlp.fc1`` / ``img_mlp.fc2``     fused GELU MLP replaces the Sequential
``ff_context.net.*``                  ``txt_mlp.fc1`` / ``txt_mlp.fc2``     fused GELU MLP
``norm1.linear``                      ``img_mod.lin``                       AWQ W4A16, SVDQ or dense
``norm1_context.linear``              ``txt_mod.lin``                       AWQ W4A16, SVDQ or dense
====================================  ====================================  =====================================

Single blocks: ComfyUI fuses q|k|v with the MLP input (``linear1``) and the attention output with the MLP output
(``linear2``); the checkpoint keeps them apart (``attn.to_qkv``, ``mlp_fc1``, ``attn.to_out``, ``mlp_fc2``), so
those two layers are replaced by ``qkv_proj``, ``mlp_fc1``, ``out_proj`` and ``mlp_fc2`` and the block forward runs
``out_proj(attention) + fused_gelu_mlp(x)``. ``norm.linear`` maps to ``modulation.lin``.

AWQ modulation layers are stored with interleaved output channels (channel ``c`` of chunk ``j`` at ``c * n + j``)
and with ``+1`` folded into the scale chunks; :class:`ComfyAWQModulationLinear` returns ComfyUI's chunk-major
layout without the ``+1`` (ComfyUI applies ``1 + scale`` itself).

The int4 fc2 of every GELU MLP takes its input quantized unsigned inside fc1's GEMM epilogue
(:func:`nunchaku_lite.ops.fused.fused_gelu_mlp`), so the MLP always runs fused.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from torch import nn

from ..adapters.common import SVDQPatchContext
from ..adapters.flux import convert_flux_state_dict
from ..ops.fused import fused_gelu_mlp
from .common import (  # noqa: F401  (ComfyAWQModulationLinear, FusedGeluMLP re-exported for callers and tests)
    ComfyAWQModulationLinear,
    FusedGeluMLP,
    make_module,
    meta_weight,
    modulation_spec,
    module_tensors,
    svdq_spec,
)
from .core import ComfyLoadPlan, register_comfy_family

DOUBLE_TARGETS = {
    "attn.to_qkv": "img_attn.qkv",
    "attn.add_qkv_proj": "txt_attn.qkv",
    "attn.to_out.0": "img_attn.proj",
    "attn.to_add_out": "txt_attn.proj",
    "ff.net.0.proj": "img_mlp.fc1",
    "ff.net.2": "img_mlp.fc2",
    "ff_context.net.0.proj": "txt_mlp.fc1",
    "ff_context.net.2": "txt_mlp.fc2",
}
DOUBLE_MODULATION = {"norm1.linear": "img_mod.lin", "norm1_context.linear": "txt_mod.lin"}
SINGLE_TARGETS = {
    "attn.to_qkv": "qkv_proj",
    "attn.to_out": "out_proj",
    "mlp_fc1": "mlp_fc1",
    "mlp_fc2": "mlp_fc2",
}
SINGLE_MODULATION = {"norm.linear": "modulation.lin"}

# Dense-shape stand-ins for ComfyUI's model detection: comfy module whose ``.weight`` the quantized layer replaces.
_DOUBLE_DETECT_NAMES = {
    "img_attn.qkv": "img_attn.qkv",
    "txt_attn.qkv": "txt_attn.qkv",
    "img_attn.proj": "img_attn.proj",
    "txt_attn.proj": "txt_attn.proj",
    "img_mlp.fc1": "img_mlp.0",
    "img_mlp.fc2": "img_mlp.2",
    "txt_mlp.fc1": "txt_mlp.0",
    "txt_mlp.fc2": "txt_mlp.2",
    "img_mod.lin": "img_mod.lin",
    "txt_mod.lin": "txt_mod.lin",
}

_BLOCK_KEY = re.compile(r"^(transformer_blocks|single_transformer_blocks)\.(\d+)\.(.+)$")


class FluxComfyFamily:
    """FLUX.1 (dev, schnell, Kontext, Krea and fine-tunes) checkpoints in ComfyUI's Flux model."""

    name = "flux"
    supports_input_rotation = False

    def matches(self, state_dict: dict[str, torch.Tensor], metadata: dict[str, str]) -> bool:
        keys = state_dict.keys()
        return (
            "x_embedder.weight" in state_dict
            and "time_text_embed.timestep_embedder.linear_1.weight" in state_dict
            and any(key.startswith("single_transformer_blocks.") for key in keys)
            and not any(key.startswith("double_stream_modulation_img.") for key in keys)
        )

    def plan(
        self, state_dict: dict[str, torch.Tensor], metadata: dict[str, str], context: SVDQPatchContext
    ) -> ComfyLoadPlan:
        import comfy.model_detection

        tensors = convert_flux_state_dict(dict(state_dict))
        double_ids = sorted(
            {int(m.group(2)) for k in tensors if (m := _BLOCK_KEY.match(k)) and m.group(1) == "transformer_blocks"}
        )
        single_ids = sorted(
            {
                int(m.group(2))
                for k in tensors
                if (m := _BLOCK_KEY.match(k)) and m.group(1) == "single_transformer_blocks"
            }
        )
        if double_ids != list(range(len(double_ids))) or single_ids != list(range(len(single_ids))):
            raise ValueError("FLUX checkpoint block indices are not contiguous.")

        modules: dict[str, dict[str, Any]] = {}
        quant: dict[str, torch.Tensor] = {}
        consumed: set[str] = set()

        def take(lite_prefix: str, comfy_path: str, spec: dict[str, Any]) -> None:
            tensors_of_module = module_tensors(tensors, lite_prefix)
            for suffix, value in tensors_of_module.items():
                quant[f"{comfy_path}.{suffix}"] = value
                consumed.add(f"{lite_prefix}.{suffix}")
            modules[comfy_path] = spec

        for index in double_ids:
            lite, target = f"transformer_blocks.{index}", f"double_blocks.{index}"
            for src, dst in DOUBLE_TARGETS.items():
                if f"{lite}.{src}.qweight" not in tensors:
                    raise ValueError(f"FLUX checkpoint has no quantized {lite}.{src}.")
                spec = svdq_spec(tensors, f"{lite}.{src}", fused_mlp_fc2=dst.endswith(".fc2"))
                take(f"{lite}.{src}", f"{target}.{dst}", spec)
            for src, dst in DOUBLE_MODULATION.items():
                spec = modulation_spec(tensors, f"{lite}.{src}", chunks=6, scale_chunks=(1, 4))
                if spec is not None:
                    take(f"{lite}.{src}", f"{target}.{dst}", spec)
        for index in single_ids:
            lite, target = f"single_transformer_blocks.{index}", f"single_blocks.{index}"
            for src, dst in SINGLE_TARGETS.items():
                if f"{lite}.{src}.qweight" not in tensors:
                    raise ValueError(f"FLUX checkpoint has no quantized {lite}.{src}.")
                spec = svdq_spec(tensors, f"{lite}.{src}", fused_mlp_fc2=dst == "mlp_fc2")
                take(f"{lite}.{src}", f"{target}.{dst}", spec)
            for src, dst in SINGLE_MODULATION.items():
                spec = modulation_spec(tensors, f"{lite}.{src}", chunks=3, scale_chunks=(1,))
                if spec is not None:
                    take(f"{lite}.{src}", f"{target}.{dst}", spec)

        remaining = {key: value for key, value in tensors.items() if key not in consumed}
        stray = sorted(key for key in remaining if key.rsplit(".", 1)[-1] not in ("weight", "bias"))
        if stray:
            raise ValueError(f"Unrecognized quantized tensors in FLUX checkpoint: {stray[:8]}.")
        dense = comfy.model_detection.convert_diffusers_mmdit(remaining, "")
        if dense is None:
            raise ValueError("ComfyUI did not recognize the dense part of this FLUX checkpoint.")
        if remaining:
            raise ValueError(f"Unmapped dense tensors in FLUX checkpoint: {sorted(remaining)[:8]}.")

        detect = dict(dense)
        for path, spec in modules.items():
            block, index, local = path.split(".", 2)
            if block == "double_blocks":
                detect[f"{block}.{index}.{_DOUBLE_DETECT_NAMES[local]}.weight"] = meta_weight(
                    spec["out_features"], spec["in_features"]
                )
            elif local == "modulation.lin":
                detect[f"{path}.weight"] = meta_weight(spec["out_features"], spec["in_features"])
        for index in single_ids:
            qkv, fc1 = modules[f"single_blocks.{index}.qkv_proj"], modules[f"single_blocks.{index}.mlp_fc1"]
            out, fc2 = modules[f"single_blocks.{index}.out_proj"], modules[f"single_blocks.{index}.mlp_fc2"]
            hidden = qkv["in_features"]
            detect[f"single_blocks.{index}.linear1.weight"] = meta_weight(
                qkv["out_features"] + fc1["out_features"], hidden
            )
            detect[f"single_blocks.{index}.linear2.weight"] = meta_weight(
                out["out_features"], out["in_features"] + fc2["in_features"]
            )
        return ComfyLoadPlan(dense=dense, quant=quant, detect=detect, modules=modules)

    def install(self, diffusion_model: nn.Module, plan: ComfyLoadPlan, context: SVDQPatchContext) -> None:
        from comfy.ldm.flux.layers import DoubleStreamBlock

        if not hasattr(diffusion_model, "double_blocks") or not hasattr(diffusion_model, "single_blocks"):
            raise TypeError(f"Expected ComfyUI's Flux model, got {type(diffusion_model).__name__}.")
        if len(diffusion_model.double_blocks) != len(
            {p.split(".")[1] for p in plan.modules if p.startswith("double_blocks.")}
        ):
            raise ValueError("ComfyUI's Flux model and the checkpoint disagree on the number of double blocks.")

        for index, block in enumerate(diffusion_model.double_blocks):
            if not isinstance(block, DoubleStreamBlock) or not block.modulation:
                raise TypeError("FLUX.1 double blocks must be ComfyUI DoubleStreamBlocks with per-block modulation.")
            prefix = f"double_blocks.{index}"
            for side in ("img", "txt"):
                attn = getattr(block, f"{side}_attn")
                attn.qkv = _make(plan, f"{prefix}.{side}_attn.qkv", context)
                attn.proj = _make(plan, f"{prefix}.{side}_attn.proj", context)
                if attn.qkv.out_features != 3 * block.hidden_size:
                    raise ValueError(f"{prefix}.{side}_attn.qkv width does not match ComfyUI's hidden size.")
                mlp = getattr(block, f"{side}_mlp")
                if not (isinstance(mlp, nn.Sequential) and len(mlp) == 3 and isinstance(mlp[1], nn.GELU)):
                    raise TypeError(f"{prefix}.{side}_mlp is not ComfyUI's Linear-GELU-Linear MLP.")
                fc1 = _make(plan, f"{prefix}.{side}_mlp.fc1", context)
                fc2 = _make(plan, f"{prefix}.{side}_mlp.fc2", context)
                setattr(block, f"{side}_mlp", FusedGeluMLP(fc1, fc2))
                mod_path = f"{prefix}.{side}_mod.lin"
                if mod_path in plan.modules:
                    getattr(block, f"{side}_mod").lin = _make(plan, mod_path, context)

        _install_single_blocks(diffusion_model, plan, context, family="FLUX.1", per_block_modulation=True)


def _make(plan: ComfyLoadPlan, path: str, context: SVDQPatchContext) -> nn.Module:
    return make_module(plan.modules[path], context, path)


def _install_single_blocks(
    diffusion_model: nn.Module,
    plan: ComfyLoadPlan,
    context: SVDQPatchContext,
    *,
    family: str,
    per_block_modulation: bool,
) -> None:
    """Split every single block's ``linear1``/``linear2`` into ``qkv_proj``, ``mlp_fc1``, ``out_proj``, ``mlp_fc2``."""

    from comfy.ldm.flux.layers import SingleStreamBlock

    if len(diffusion_model.single_blocks) != len(
        {p.split(".")[1] for p in plan.modules if p.startswith("single_blocks.")}
    ):
        raise ValueError(f"ComfyUI's Flux model and the {family} checkpoint disagree on the number of single blocks.")
    single_cls = _single_block_class()
    for index, block in enumerate(diffusion_model.single_blocks):
        if not isinstance(block, SingleStreamBlock) or block.yak_mlp:
            raise TypeError(f"{family} single blocks must be ComfyUI SingleStreamBlocks.")
        if (block.modulation is not None) != per_block_modulation:
            raise TypeError(f"{family} single blocks have unexpected modulation in ComfyUI's model.")
        prefix = f"single_blocks.{index}"
        if bool(plan.modules[f"{prefix}.mlp_fc2"].get("fused_mlp_fc2")) != isinstance(block.mlp_act, nn.GELU):
            raise TypeError(f"{prefix}: the checkpoint's MLP activation does not match ComfyUI's single block.")
        del block.linear1, block.linear2
        block.qkv_proj = _make(plan, f"{prefix}.qkv_proj", context)
        block.out_proj = _make(plan, f"{prefix}.out_proj", context)
        block.mlp_fc1 = _make(plan, f"{prefix}.mlp_fc1", context)
        block.mlp_fc2 = _make(plan, f"{prefix}.mlp_fc2", context)
        if block.qkv_proj.out_features != 3 * block.hidden_size:
            raise ValueError(f"{prefix}.qkv_proj width does not match ComfyUI's hidden size.")
        if (
            block.mlp_fc1.out_features != block.mlp_hidden_dim_first
            or block.mlp_fc2.in_features != block.mlp_hidden_dim
        ):
            raise ValueError(f"{prefix} MLP widths do not match ComfyUI's single block.")
        mod_path = f"{prefix}.modulation.lin"
        if mod_path in plan.modules:
            block.modulation.lin = _make(plan, mod_path, context)
        block.__class__ = single_cls


_SINGLE_BLOCK_CLASS = None


def _single_block_class():
    """ComfyUI ``SingleStreamBlock`` subclass running the split W4A4 projections (built on first use)."""

    global _SINGLE_BLOCK_CLASS
    if _SINGLE_BLOCK_CLASS is not None:
        return _SINGLE_BLOCK_CLASS

    from comfy.ldm.flux.layers import SingleStreamBlock, apply_mod, modulated_norm
    from comfy.ldm.flux.math import attention
    from comfy.ldm.modules.attention import AttentionTensorContainer

    class NunchakuSingleStreamBlock(SingleStreamBlock):
        """Single-stream block with ``linear1``/``linear2`` split into separate W4A4 layers.

        Mirrors ComfyUI's unchunked ``SingleStreamBlock.forward`` (attention patches, modulation dims, fp16
        clamping), computing ``out_proj(attention) + mlp(x)`` instead of ``linear2(cat(...))``. A GELU MLP (FLUX.1)
        runs through the fused W4A4 MLP kernel; a gated SiLU MLP (FLUX.2) runs unfused with ComfyUI's activation.
        """

        def forward(self, x, vec, pe, attn_mask=None, modulation_dims=None, transformer_options={}):  # noqa: B006
            mod = self.modulation(vec)[0] if self.modulation else vec
            transformer_patches = transformer_options.get("patches", {})
            extra_options = transformer_options.copy()

            x_mod = modulated_norm(x, self.pre_norm, mod.scale, mod.shift, modulation_dims)
            if isinstance(self.mlp_act, nn.GELU):
                mlp = fused_gelu_mlp(x_mod, self.mlp_fc1, self.mlp_fc2)
            else:
                mlp = self.mlp_fc2(self.mlp_activation(self.mlp_fc1(x_mod)))
            qkv = self.qkv_proj(x_mod)
            del x_mod
            q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            del qkv
            q, k = self.norm(q, k, v)

            if "attn1_patch" in transformer_patches:
                for patch in transformer_patches["attn1_patch"]:
                    out = patch(q, k, v, pe=pe, attn_mask=attn_mask, extra_options=extra_options)
                    q, k, v = out.get("q", q), out.get("k", k), out.get("v", v)
                    pe, attn_mask = out.get("pe", pe), out.get("attn_mask", attn_mask)

            q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
            attn = attention(
                q,
                k,
                v,
                pe=pe,
                mask=attn_mask,
                transformer_options=transformer_options,
                preferred_attention=self.comfy_attention,
            )
            del q, k, v
            if "attn1_output_patch" in transformer_patches:
                for patch in transformer_patches["attn1_output_patch"]:
                    attn = patch(attn, extra_options)

            output = self.out_proj(attn.contiguous())
            output += mlp
            x += apply_mod(output, mod.gate, None, modulation_dims)
            if x.dtype == torch.float16:
                x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
            return x

    _SINGLE_BLOCK_CLASS = NunchakuSingleStreamBlock
    return _SINGLE_BLOCK_CLASS


FLUX2_DOUBLE_TARGETS = {
    "attn.to_qkv": "img_attn.qkv",
    "attn.to_added_qkv": "txt_attn.qkv",
    "attn.to_out.0": "img_attn.proj",
    "attn.to_add_out": "txt_attn.proj",
    "ff.linear_in": "img_mlp.0",
    "ff.linear_out": "img_mlp.2",
    "ff_context.linear_in": "txt_mlp.0",
    "ff_context.linear_out": "txt_mlp.2",
}
FLUX2_SINGLE_TARGETS = {
    "attn.qkv_proj": "qkv_proj",
    "attn.out_proj": "out_proj",
    "attn.mlp_fc1": "mlp_fc1",
    "attn.mlp_fc2": "mlp_fc2",
}
FLUX2_DENSE = {
    "x_embedder.weight": "img_in.weight",
    "context_embedder.weight": "txt_in.weight",
    "time_guidance_embed.timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
    "time_guidance_embed.timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
    "time_guidance_embed.guidance_embedder.linear_1.weight": "guidance_in.in_layer.weight",
    "time_guidance_embed.guidance_embedder.linear_2.weight": "guidance_in.out_layer.weight",
    "double_stream_modulation_img.linear.weight": "double_stream_modulation_img.lin.weight",
    "double_stream_modulation_txt.linear.weight": "double_stream_modulation_txt.lin.weight",
    "single_stream_modulation.linear.weight": "single_stream_modulation.lin.weight",
    "proj_out.weight": "final_layer.linear.weight",
}
FLUX2_BLOCK_DENSE = {
    "transformer_blocks": {
        "attn.norm_q.weight": "img_attn.norm.query_norm.weight",
        "attn.norm_k.weight": "img_attn.norm.key_norm.weight",
        "attn.norm_added_q.weight": "txt_attn.norm.query_norm.weight",
        "attn.norm_added_k.weight": "txt_attn.norm.key_norm.weight",
    },
    "single_transformer_blocks": {
        "attn.norm_q.weight": "norm.query_norm.weight",
        "attn.norm_k.weight": "norm.key_norm.weight",
    },
}


def _flux2_dense_key(key: str) -> str | None:
    if key in FLUX2_DENSE:
        return FLUX2_DENSE[key]
    if m := _BLOCK_KEY.match(key):
        group, index, local = m.groups()
        local_target = FLUX2_BLOCK_DENSE[group].get(local)
        if local_target is not None:
            comfy_group = "double_blocks" if group == "transformer_blocks" else "single_blocks"
            return f"{comfy_group}.{index}.{local_target}"
    return None


class Flux2ComfyFamily:
    """FLUX.2 (Klein and dev) checkpoints in ComfyUI's Flux model (global modulation, gated SiLU MLPs).

    Every W4A4 layer runs unfused on the signed path, so ComfyUI's double-block MLPs keep their
    ``Linear -> SiLU gate -> Linear`` form with the two linears replaced; single blocks are split as for FLUX.1.
    """

    name = "flux2"
    supports_input_rotation = False

    def matches(self, state_dict: dict[str, torch.Tensor], metadata: dict[str, str]) -> bool:
        return "double_stream_modulation_img.linear.weight" in state_dict and any(
            key.startswith("transformer_blocks.0.attn.to_qkv.") for key in state_dict
        )

    def plan(
        self, state_dict: dict[str, torch.Tensor], metadata: dict[str, str], context: SVDQPatchContext
    ) -> ComfyLoadPlan:
        from ..checkpoint_keys import convert_flux2_state_dict, flux2_state_dict_needs_conversion

        tensors = dict(state_dict)
        if flux2_state_dict_needs_conversion(tensors):
            tensors = convert_flux2_state_dict(tensors)
        # Original FLUX.2 exports carry all-zero biases for biasless layers; ComfyUI's FLUX.2 has no biases.
        for key in [k for k, v in tensors.items() if k.endswith(".bias") and not v.is_meta and not v.any()]:
            tensors.pop(key)
        groups = {"transformer_blocks": set(), "single_transformer_blocks": set()}
        for key in tensors:
            if m := _BLOCK_KEY.match(key):
                groups[m.group(1)].add(int(m.group(2)))
        for ids in groups.values():
            if sorted(ids) != list(range(len(ids))):
                raise ValueError("FLUX.2 checkpoint block indices are not contiguous.")

        modules: dict[str, dict[str, Any]] = {}
        quant: dict[str, torch.Tensor] = {}
        consumed: set[str] = set()
        for group, targets, comfy_group in (
            ("transformer_blocks", FLUX2_DOUBLE_TARGETS, "double_blocks"),
            ("single_transformer_blocks", FLUX2_SINGLE_TARGETS, "single_blocks"),
        ):
            for index in sorted(groups[group]):
                for src, dst in targets.items():
                    prefix = f"{group}.{index}.{src}"
                    if f"{prefix}.qweight" not in tensors:
                        raise ValueError(f"FLUX.2 checkpoint has no quantized {prefix}.")
                    for suffix, value in module_tensors(tensors, prefix).items():
                        quant[f"{comfy_group}.{index}.{dst}.{suffix}"] = value
                        consumed.add(f"{prefix}.{suffix}")
                    modules[f"{comfy_group}.{index}.{dst}"] = svdq_spec(tensors, prefix)

        dense: dict[str, torch.Tensor] = {}
        for key, value in tensors.items():
            if key in consumed:
                continue
            if key == "norm_out.linear.weight":
                shift, scale = value.chunk(2, dim=0)  # Diffusers (scale, shift) -> ComfyUI (shift, scale) order
                dense["final_layer.adaLN_modulation.1.weight"] = torch.cat([scale, shift], dim=0)
                continue
            target = _flux2_dense_key(key)
            if target is None:
                raise ValueError(f"Unmapped tensor in FLUX.2 checkpoint: {key}.")
            dense[target] = value

        detect = dict(dense)
        for path, spec in modules.items():
            block = path.split(".", 1)[0]
            if block == "double_blocks":
                detect[f"{path}.weight"] = meta_weight(spec["out_features"], spec["in_features"])
        for index in sorted(groups["single_transformer_blocks"]):
            qkv, fc1 = modules[f"single_blocks.{index}.qkv_proj"], modules[f"single_blocks.{index}.mlp_fc1"]
            out, fc2 = modules[f"single_blocks.{index}.out_proj"], modules[f"single_blocks.{index}.mlp_fc2"]
            detect[f"single_blocks.{index}.linear1.weight"] = meta_weight(
                qkv["out_features"] + fc1["out_features"], qkv["in_features"]
            )
            detect[f"single_blocks.{index}.linear2.weight"] = meta_weight(
                out["out_features"], out["in_features"] + fc2["in_features"]
            )
        return ComfyLoadPlan(dense=dense, quant=quant, detect=detect, modules=modules)

    def install(self, diffusion_model: nn.Module, plan: ComfyLoadPlan, context: SVDQPatchContext) -> None:
        from comfy.ldm.flux.layers import DoubleStreamBlock, SiLUActivation

        if not getattr(getattr(diffusion_model, "params", None), "global_modulation", False):
            raise TypeError("FLUX.2 checkpoints need ComfyUI's Flux model with global modulation.")
        if len(diffusion_model.double_blocks) != len(
            {p.split(".")[1] for p in plan.modules if p.startswith("double_blocks.")}
        ):
            raise ValueError("ComfyUI's Flux model and the FLUX.2 checkpoint disagree on the number of double blocks.")
        for index, block in enumerate(diffusion_model.double_blocks):
            if not isinstance(block, DoubleStreamBlock) or block.modulation:
                raise TypeError("FLUX.2 double blocks must be ComfyUI DoubleStreamBlocks with global modulation.")
            prefix = f"double_blocks.{index}"
            for side in ("img", "txt"):
                attn = getattr(block, f"{side}_attn")
                attn.qkv = _make(plan, f"{prefix}.{side}_attn.qkv", context)
                attn.proj = _make(plan, f"{prefix}.{side}_attn.proj", context)
                if attn.qkv.out_features != 3 * block.hidden_size:
                    raise ValueError(f"{prefix}.{side}_attn.qkv width does not match ComfyUI's hidden size.")
                mlp = getattr(block, f"{side}_mlp")
                if not (isinstance(mlp, nn.Sequential) and len(mlp) == 3 and isinstance(mlp[1], SiLUActivation)):
                    raise TypeError(f"{prefix}.{side}_mlp is not ComfyUI's gated SiLU MLP.")
                mlp[0] = _make(plan, f"{prefix}.{side}_mlp.0", context)
                mlp[2] = _make(plan, f"{prefix}.{side}_mlp.2", context)
        _install_single_blocks(diffusion_model, plan, context, family="FLUX.2", per_block_modulation=False)


register_comfy_family(FluxComfyFamily())
register_comfy_family(Flux2ComfyFamily())
