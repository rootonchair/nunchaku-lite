"""ComfyUI integration: load Nunchaku Lite checkpoints into ComfyUI's own model classes.

ComfyUI builds its native model for the checkpoint's family (for example ``comfy.ldm.flux.model.Flux``); the
family adapter in this package swaps the quantized layers in by module name and, where ComfyUI's module layout
differs from the checkpoint's (fused projections, fused GELU MLPs), replaces the affected block forwards. The
result is a regular ComfyUI ``ModelPatcher`` that works with ComfyUI's samplers and model management.

ComfyUI is imported lazily: importing this package does not require ComfyUI. ComfyUI's custom-node entry point
lives in :mod:`nunchaku_lite.comfyui.nodes`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core import ComfyFamilyAdapter, list_comfy_families, load_comfy_diffusion_model, register_comfy_family

__all__ = ["ComfyFamilyAdapter", "list_comfy_families", "load_comfy_diffusion_model", "register_comfy_family"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import core

        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
