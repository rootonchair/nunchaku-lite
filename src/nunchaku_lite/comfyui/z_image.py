"""Z-Image checkpoints in ComfyUI's Lumina ``NextDiT`` (``comfy.ldm.lumina.model``).

Per block in ``layers``, ``noise_refiner`` and ``context_refiner``:

================================  ================================  ==============================================
checkpoint                        ComfyUI                           notes
================================  ================================  ==============================================
``attention.to_qkv``              ``attention.qkv``                 SVDQ, q|k|v rows as in ComfyUI
``attention.to_out.0``            ``attention.out``                 SVDQ
``feed_forward.net.0.proj``       ``feed_forward.fc1``              SVDQ, rows ``[hidden; gate]`` = ``[w3; w1]``
``feed_forward.net.2``            ``feed_forward.fc2``              SVDQ (``w2``)
================================  ================================  ==============================================

The feed-forward is replaced by :class:`SwiGLUMLP` computing ``fc2(hidden * silu(gate))``, which equals ComfyUI's
``w2(silu(w1 x) * w3 x)``. Checkpoints exported with ``skip_refiners`` keep the refiners dense, in the Diffusers
``FeedForward`` layout; those are split back into ComfyUI's ``w1``/``w2``/``w3``. All other dense tensors are
renamed with ComfyUI's own ``z_image_to_diffusers`` map; a tensor the map does not cover is an error.
"""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..adapters.common import SVDQPatchContext
from .common import make_module, meta_weight, module_tensors, svdq_spec
from .core import ComfyLoadPlan, register_comfy_family

BLOCK_GROUPS = ("layers", "noise_refiner", "context_refiner")
BLOCK_TARGETS = {
    "attention.to_qkv": "attention.qkv",
    "attention.to_out.0": "attention.out",
    "feed_forward.net.0.proj": "feed_forward.fc1",
    "feed_forward.net.2": "feed_forward.fc2",
}
_DETECT_NAMES = {
    "attention.qkv": ("attention.qkv",),
    "attention.out": ("attention.out",),
    "feed_forward.fc1": ("feed_forward.w3", "feed_forward.w1"),
    "feed_forward.fc2": ("feed_forward.w2",),
}
_BLOCK_KEY = re.compile(r"^(layers|noise_refiner|context_refiner)\.(\d+)\.(.+)$")


class SwiGLUMLP(nn.Module):
    """``fc2(hidden * silu(gate))`` with ``hidden, gate = fc1(x).chunk(2)`` (Diffusers' SwiGLU ``FeedForward``)."""

    def __init__(self, fc1: nn.Module, fc2: nn.Module) -> None:
        super().__init__()
        self.fc1 = fc1
        self.fc2 = fc2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(hidden * F.silu(gate))


def _rename_dense(dense: dict[str, torch.Tensor], key_map: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Apply ComfyUI's ``z_image_to_diffusers`` map (Diffusers key -> ComfyUI key or ``(key, offset)`` slice)."""

    out: dict[str, torch.Tensor] = {}
    fused: dict[str, list[tuple[int, torch.Tensor]]] = {}
    for key, value in dense.items():
        if key.endswith(("feed_forward.net.0.proj.weight", "feed_forward.net.2.weight")):
            prefix = key[: key.index("feed_forward.") + len("feed_forward.")]
            if key.endswith("net.0.proj.weight"):
                hidden, gate = value.chunk(2, dim=0)
                out[f"{prefix}w3.weight"], out[f"{prefix}w1.weight"] = hidden, gate
            else:
                out[f"{prefix}w2.weight"] = value
            continue
        target = key_map.get(key)
        if target is None:
            raise ValueError(f"Unmapped dense tensor in Z-Image checkpoint: {key}.")
        if isinstance(target, str):
            out[target] = value
            continue
        if len(target) > 2 or target[1] is None or target[1][0] != 0:
            raise ValueError(f"Unsupported ComfyUI key mapping for {key}: {target!r}.")
        fused.setdefault(target[0], []).append((target[1][1], value))
    for name, parts in fused.items():
        out[name] = torch.cat([value for _, value in sorted(parts, key=lambda part: part[0])], dim=0)
    return out


class ZImageComfyFamily:
    """Z-Image (Turbo and fine-tunes) in ComfyUI's Lumina NextDiT."""

    name = "z_image"
    supports_input_rotation = False

    def matches(self, state_dict: dict[str, torch.Tensor], metadata: dict[str, str]) -> bool:
        return (
            "cap_pad_token" in state_dict
            and "x_pad_token" in state_dict
            and "layers.0.attention.to_qkv.qweight" in state_dict
        )

    def plan(
        self, state_dict: dict[str, torch.Tensor], metadata: dict[str, str], context: SVDQPatchContext
    ) -> ComfyLoadPlan:
        import comfy.utils

        tensors = dict(state_dict)
        groups: dict[str, set[int]] = {group: set() for group in BLOCK_GROUPS}
        for key in tensors:
            if m := _BLOCK_KEY.match(key):
                groups[m.group(1)].add(int(m.group(2)))
        for group, ids in groups.items():
            if sorted(ids) != list(range(len(ids))):
                raise ValueError(f"Z-Image checkpoint {group} indices are not contiguous.")
        if len(groups["noise_refiner"]) != len(groups["context_refiner"]):
            raise ValueError("Z-Image checkpoint has different noise and context refiner depths.")

        modules: dict[str, dict[str, Any]] = {}
        quant: dict[str, torch.Tensor] = {}
        consumed: set[str] = set()
        for group, ids in groups.items():
            for index in sorted(ids):
                prefix = f"{group}.{index}"
                quantized = [src for src in BLOCK_TARGETS if f"{prefix}.{src}.qweight" in tensors]
                if not quantized and group != "layers":
                    continue  # a dense refiner (skip_refiners export)
                if len(quantized) != len(BLOCK_TARGETS):
                    missing = sorted(set(BLOCK_TARGETS) - set(quantized))
                    raise ValueError(f"Z-Image checkpoint {prefix} is partially quantized; dense: {missing}.")
                for src, dst in BLOCK_TARGETS.items():
                    spec = svdq_spec(tensors, f"{prefix}.{src}")
                    for suffix, value in module_tensors(tensors, f"{prefix}.{src}").items():
                        quant[f"{prefix}.{dst}.{suffix}"] = value
                        consumed.add(f"{prefix}.{src}.{suffix}")
                    modules[f"{prefix}.{dst}"] = spec

        remaining = {key: value for key, value in tensors.items() if key not in consumed}
        stray = sorted(
            key
            for key in remaining
            if key not in ("cap_pad_token", "x_pad_token") and key.rsplit(".", 1)[-1] not in ("weight", "bias")
        )
        if stray:
            raise ValueError(f"Unrecognized tensors in Z-Image checkpoint: {stray[:8]}.")
        dim = tensors["cap_embedder.1.weight"].shape[0]
        key_map = comfy.utils.z_image_to_diffusers(
            {"n_layers": len(groups["layers"]), "dim": dim, "n_refiner_layers": len(groups["noise_refiner"])}
        )
        dense = _rename_dense(remaining, key_map)

        detect = dict(dense)
        for path, spec in modules.items():
            group, index, local = path.split(".", 2)
            names = _DETECT_NAMES[local]
            for name in names:
                detect[f"{group}.{index}.{name}.weight"] = meta_weight(
                    spec["out_features"] // len(names), spec["in_features"]
                )
        return ComfyLoadPlan(dense=dense, quant=quant, detect=detect, modules=modules)

    def install(self, diffusion_model: nn.Module, plan: ComfyLoadPlan, context: SVDQPatchContext) -> None:
        from comfy.ldm.lumina.model import FeedForward, JointAttention

        for path in sorted({p.rsplit(".", 2)[0] for p in plan.modules}):
            block = diffusion_model.get_submodule(path)
            if not isinstance(block.attention, JointAttention) or not isinstance(block.feed_forward, FeedForward):
                raise TypeError(f"{path} is not a ComfyUI Lumina JointTransformerBlock.")
            block.attention.qkv = make_module(plan.modules[f"{path}.attention.qkv"], context)
            block.attention.out = make_module(plan.modules[f"{path}.attention.out"], context)
            expected = (block.attention.n_local_heads + 2 * block.attention.n_local_kv_heads) * block.attention.head_dim
            if block.attention.qkv.out_features != expected:
                raise ValueError(f"{path}.attention.qkv width does not match ComfyUI's attention heads.")
            fc1 = make_module(plan.modules[f"{path}.feed_forward.fc1"], context)
            fc2 = make_module(plan.modules[f"{path}.feed_forward.fc2"], context)
            block.feed_forward = SwiGLUMLP(fc1, fc2)


register_comfy_family(ZImageComfyFamily())
