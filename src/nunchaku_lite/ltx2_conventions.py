"""Conventions of LTX-2 / LTX-2.5 Nunchaku Lite exports, shared by the Diffusers LTX adapter and the ltx-core
integration. Torch-free; nothing here applies to other model families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

__all__ = ["GeluMlpPlan", "ltx2_gelu_mlp_plan"]


@dataclass(frozen=True)
class GeluMlpPlan:
    """How an LTX ``fc1 -> GELU -> fc2`` pair of SVDQ layers runs."""

    fused: bool  # through ops.fused.fused_gelu_mlp (fc2's input quantized in fc1's epilogue)
    fc2_act_unsigned: bool
    fc2_fused_flag: bool  # set ops.fused.FUSED_MLP_FC2_ATTR on fc2


class _QuantizedLayer(Protocol):
    precision: str
    input_rotation: str | None


def ltx2_gelu_mlp_plan(fc1: _QuantizedLayer, fc2: _QuantizedLayer) -> GeluMlpPlan:
    """The GELU-MLP rule for LTX exports (checkpoint layer specs or bound SVDQ layers).

    LTX exports quantize fc2's input signed, without the shifted-GELU convention that FLUX and Qwen-Image INT4
    exports use (their adapters set ``act_unsigned`` for INT4 instead). So the fused kernel, whose INT4 epilogue
    applies that shift, is used only for NVFP4 with neither layer rotated; otherwise fc1 -> GELU -> fc2 runs
    unfused with fc2 signed.
    """

    fused = (
        fc1.precision == "nvfp4"
        and fc2.precision == "nvfp4"
        and fc1.input_rotation is None
        and fc2.input_rotation is None
    )
    return GeluMlpPlan(fused=fused, fc2_act_unsigned=False, fc2_fused_flag=fused)
