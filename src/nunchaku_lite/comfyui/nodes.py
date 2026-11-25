"""ComfyUI nodes for Nunchaku Lite checkpoints.

``NODE_CLASS_MAPPINGS`` and ``NODE_DISPLAY_NAME_MAPPINGS`` are re-exported by the repository's custom-node entry
point (``__init__.py`` at the repository root), so cloning nunchaku-lite into ``ComfyUI/custom_nodes`` registers
these nodes. This module imports ComfyUI and must only be imported inside a ComfyUI process.
"""

from __future__ import annotations

import folder_paths
import torch

from .core import list_comfy_families, load_comfy_diffusion_model

_DTYPES = {"default": None, "bfloat16": torch.bfloat16, "float16": torch.float16}


class NunchakuLiteDiffusionModelLoader:
    """Load a Nunchaku Lite SVDQ (INT4 / NVFP4) diffusion model from ``models/diffusion_models``."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "compute_dtype": (
                    list(_DTYPES),
                    {
                        "default": "default",
                        "tooltip": "Activation dtype. 'default' lets ComfyUI pick (bfloat16 where supported).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "Nunchaku Lite"
    DESCRIPTION = (
        "Loads a Nunchaku Lite 4-bit checkpoint (supported: " + ", ".join(list_comfy_families()) + ") into "
        "ComfyUI's native model, with W4A4 layers swapped in."
    )

    def load(self, model_name: str, compute_dtype: str = "default"):
        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        model_options = {}
        if _DTYPES[compute_dtype] is not None:
            model_options["dtype"] = _DTYPES[compute_dtype]
        return (load_comfy_diffusion_model(path, model_options=model_options),)


NODE_CLASS_MAPPINGS = {"NunchakuLiteDiffusionModelLoader": NunchakuLiteDiffusionModelLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"NunchakuLiteDiffusionModelLoader": "Nunchaku Lite Diffusion Model Loader"}
