"""Building blocks shared by the ComfyUI family adapters.

A family's ``plan`` describes every module it will install as a *spec* (a plain dict) derived from checkpoint
tensor shapes, including ``source``, the module's checkpoint prefix; ``install`` turns specs into meta-device modules with :func:`make_module`, so module shapes always
come from the checkpoint and the exact load in :func:`nunchaku_lite.comfyui.core.load_state_exactly` catches any
disagreement with ComfyUI's model.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..adapters.common import SVDQPatchContext, linear_class_from_checkpoint
from ..linear import AWQW4A16Linear, DenseRuntimeLoraLinear, SVDQW4A4Linear
from ..ops.fused import FUSED_MLP_FC2_ATTR, fused_gelu_mlp

AWQ_GROUP_SIZE = 64


class ComfyAWQModulationLinear(AWQW4A16Linear):
    """AWQ W4A16 modulation projection returning ComfyUI's chunk-major ``(shift, scale, gate) * n`` layout.

    Nunchaku stores AWQ modulation layers with interleaved output channels (channel ``c`` of chunk ``j`` at
    ``c * chunks + j``) and with ``+1`` folded into the scale chunks; ComfyUI applies ``1 + scale`` itself.
    """

    def __init__(self, *args, chunks: int, scale_chunks: tuple[int, ...], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.out_features % chunks:
            raise ValueError(f"AWQ modulation width {self.out_features} is not divisible by {chunks} chunks.")
        self.chunks = chunks
        self.scale_chunks = scale_chunks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        width = self.out_features // self.chunks
        out = out.unflatten(-1, (width, self.chunks)).transpose(-1, -2).contiguous()
        out[..., list(self.scale_chunks), :] -= 1.0
        return out.flatten(-2)


class FusedGeluMLP(nn.Module):
    """``fc2(gelu_tanh(fc1(x)))`` through Nunchaku's fused W4A4 MLP kernel.

    The INT4 fc2 of these checkpoints takes its input quantized unsigned inside fc1's GEMM epilogue, so the MLP
    must run fused; ComfyUI's ``Linear -> GELU -> Linear`` would quantize fc2's input on the signed path.
    """

    def __init__(self, fc1: SVDQW4A4Linear, fc2: SVDQW4A4Linear) -> None:
        super().__init__()
        self.fc1 = fc1
        self.fc2 = fc2

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return fused_gelu_mlp(x, self.fc1, self.fc2)


def svdq_spec(tensors: dict[str, torch.Tensor], prefix: str, **extra: Any) -> dict[str, Any]:
    """Spec of the SVDQ W4A4 layer stored under ``prefix`` (shapes from ``qweight`` and ``proj_down``)."""

    qweight = tensors[f"{prefix}.qweight"]
    return {
        "kind": "svdq",
        "source": prefix,
        "in_features": qweight.shape[1] * 2,
        "out_features": qweight.shape[0],
        "bias": f"{prefix}.bias" in tensors,
        "rank": tensors[f"{prefix}.proj_down"].shape[1],
        **extra,
    }


def modulation_spec(
    tensors: dict[str, torch.Tensor], prefix: str, *, chunks: int, scale_chunks: tuple[int, ...]
) -> dict[str, Any] | None:
    """Spec of a modulation projection: AWQ W4A16, SVDQ W4A4, or ``None`` when it is dense (ComfyUI's Linear)."""

    linear_cls = linear_class_from_checkpoint(tensors, prefix)
    if linear_cls is AWQW4A16Linear:
        wscales = tensors[f"{prefix}.wscales"]
        return {
            "kind": "awq_modulation",
            "source": prefix,
            "in_features": wscales.shape[0] * AWQ_GROUP_SIZE,
            "out_features": wscales.shape[1],
            "bias": f"{prefix}.bias" in tensors,
            "chunks": chunks,
            "scale_chunks": scale_chunks,
        }
    if linear_cls is SVDQW4A4Linear:
        return svdq_spec(tensors, prefix)
    if linear_cls is DenseRuntimeLoraLinear:
        return None
    raise ValueError(f"Unsupported modulation format for {prefix!r}: {linear_cls.__name__}.")


def module_tensors(tensors: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Tensors stored under ``prefix.``, keyed by their suffix."""

    return {key[len(prefix) + 1 :]: value for key, value in tensors.items() if key.startswith(f"{prefix}.")}


def meta_weight(out_features: int, in_features: int) -> torch.Tensor:
    """Meta stand-in for a dense ``(out, in)`` weight, for ComfyUI's shape-based model detection."""

    return torch.empty(out_features, in_features, dtype=torch.bfloat16, device="meta")


def make_module(spec: dict[str, Any], context: SVDQPatchContext, path: str = "") -> nn.Module:
    """Build the meta-device module a spec describes."""

    if spec["kind"] == "svdq":
        module = SVDQW4A4Linear(
            spec["in_features"],
            spec["out_features"],
            rank=spec["rank"],
            bias=spec["bias"],
            precision=context.precision,
            torch_dtype=context.torch_dtype,
            device="meta",
        )
        if spec.get("fused_mlp_fc2"):
            module.act_unsigned = context.precision != "nvfp4"
            setattr(module, FUSED_MLP_FC2_ATTR, True)
        return module
    if spec["kind"] == "awq_modulation":
        return ComfyAWQModulationLinear(
            spec["in_features"],
            spec["out_features"],
            bias=spec["bias"],
            group_size=AWQ_GROUP_SIZE,
            torch_dtype=context.torch_dtype,
            device="meta",
            chunks=spec["chunks"],
            scale_chunks=spec["scale_chunks"],
        )
    raise ValueError(f"{path}: unknown module kind {spec['kind']!r}.")
