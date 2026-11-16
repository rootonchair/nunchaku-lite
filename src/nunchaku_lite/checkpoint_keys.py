"""Torch-free checkpoint key normalization shared by the PyTorch and MLX runtimes.

Shared with the MLX runtime (:mod:`nunchaku_lite.mlx`), which must import without torch: keep this module free
of torch and diffusers imports (``tests/test_torch_free_imports.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["convert_flux2_state_dict", "flux2_state_dict_needs_conversion"]


def convert_flux2_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize original Nunchaku Flux2 checkpoint keys to lite module names."""

    if not flux2_state_dict_needs_conversion(state_dict):
        return state_dict

    converted: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("transformer_blocks.") and not any(
            marker in new_key for marker in (".attn.", ".ff.", ".ff_context.")
        ):
            replacements = (
                (".qkv_proj_context.", ".attn.to_added_qkv."),
                (".qkv_proj.", ".attn.to_qkv."),
                (".out_proj_context.", ".attn.to_add_out."),
                (".out_proj.", ".attn.to_out.0."),
                (".mlp_context_fc1.", ".ff_context.linear_in."),
                (".mlp_context_fc2.", ".ff_context.linear_out."),
                (".mlp_fc1.", ".ff.linear_in."),
                (".mlp_fc2.", ".ff.linear_out."),
                (".norm_added_q.", ".attn.norm_added_q."),
                (".norm_added_k.", ".attn.norm_added_k."),
                (".norm_q.", ".attn.norm_q."),
                (".norm_k.", ".attn.norm_k."),
            )
            for old, new in replacements:
                if old in new_key:
                    new_key = new_key.replace(old, new, 1)
                    break

        elif new_key.startswith("single_transformer_blocks.") and ".attn." not in new_key:
            replacements = (
                (".qkv_proj.", ".attn.qkv_proj."),
                (".mlp_fc1.", ".attn.mlp_fc1."),
                (".out_proj.", ".attn.out_proj."),
                (".mlp_fc2.", ".attn.mlp_fc2."),
                (".norm_q.", ".attn.norm_q."),
                (".norm_k.", ".attn.norm_k."),
            )
            for old, new in replacements:
                if old in new_key:
                    new_key = new_key.replace(old, new, 1)
                    break

        new_key = new_key.replace(".lora_down", ".proj_down")
        new_key = new_key.replace(".lora_up", ".proj_up")
        if ".smooth_orig" in new_key and ".smooth_factor_orig" not in new_key:
            new_key = new_key.replace(".smooth_orig", ".smooth_factor_orig")
        elif ".smooth" in new_key and ".smooth_factor" not in new_key:
            new_key = new_key.replace(".smooth", ".smooth_factor")

        converted[new_key] = value
    return converted


def flux2_state_dict_needs_conversion(state_dict: Mapping[str, Any]) -> bool:
    return any(_is_uncorrected_flux2_key(key) for key in state_dict)


def _is_uncorrected_flux2_key(key: str) -> bool:
    double_block_key = key.startswith("transformer_blocks.")
    single_block_key = key.startswith("single_transformer_blocks.")
    if not double_block_key and not single_block_key:
        return False

    if key.endswith((".lora_down", ".lora_up", ".smooth", ".smooth_orig")):
        return True

    if double_block_key and not any(marker in key for marker in (".attn.", ".ff.", ".ff_context.")):
        return any(
            marker in key
            for marker in (
                ".qkv_proj_context.",
                ".qkv_proj.",
                ".out_proj_context.",
                ".out_proj.",
                ".mlp_context_fc1.",
                ".mlp_context_fc2.",
                ".mlp_fc1.",
                ".mlp_fc2.",
                ".norm_added_q.",
                ".norm_added_k.",
                ".norm_q.",
                ".norm_k.",
            )
        )

    if single_block_key and ".attn." not in key:
        return any(
            marker in key
            for marker in (
                ".qkv_proj.",
                ".mlp_fc1.",
                ".out_proj.",
                ".mlp_fc2.",
                ".norm_q.",
                ".norm_k.",
            )
        )

    return False
