"""Qwen-Image in ComfyUI: the ComfyUI model must reproduce the Diffusers adapter on the same checkpoint.

ComfyUI's Qwen-Image detection only reads the input channels and the block count (the width is fixed at 3072), so
the synthetic model has the real width and a single block.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from comfyui_helpers import fill_random_quantized_state, kernel_device, relative_error, require_comfyui
from diffusers import QwenImageTransformer2DModel
from safetensors.torch import save_file

from nunchaku_lite import patch_transformer
from nunchaku_lite.adapters.qwen_image import QwenImageAdapter

RANK = 16


def make_qwen() -> QwenImageTransformer2DModel:
    return QwenImageTransformer2DModel(num_layers=1)


def write_checkpoint(tmp_path):
    torch.manual_seed(0)
    source = make_qwen().to(torch.bfloat16)
    QwenImageAdapter().patch(
        source,
        {},
        {"rank": RANK},
        SimpleNamespace(precision="int4", torch_dtype=torch.bfloat16, device=None, strict=True, adapter_options={}),
    )
    state = fill_random_quantized_state(dict(source.state_dict()))
    assert any(key.endswith("img_mod.1.wzeros") for key in state), "the Qwen adapter installs AWQ modulation"
    path = tmp_path / "qwen-tiny.safetensors"
    save_file(state, path, metadata={"quantization_config": json.dumps({"rank": RANK})})
    return path


def inputs(device):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 16, 1, 16, 16, generator=g)  # (B, C, T, H, W): 8x8 = 64 image tokens
    context = torch.randn(1, 12, 3584, generator=g)
    timestep = torch.tensor([0.5])  # exact in bf16, see test_comfyui_flux
    return x.to(device, torch.bfloat16), context.to(device, torch.bfloat16), timestep.to(device)


def pack(latents: torch.Tensor) -> torch.Tensor:
    b, c, _, h, w = latents.shape
    latents = latents[:, :, 0]
    return latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)


def test_comfy_qwen_image_matches_diffusers_adapter(tmp_path):
    require_comfyui()
    device = kernel_device()
    from nunchaku_lite.comfyui import load_comfy_diffusion_model

    checkpoint = write_checkpoint(tmp_path)
    x, context, timestep = inputs(device)

    transformer = make_qwen()
    patch_transformer(transformer, checkpoint, precision="int4", torch_dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        expected = transformer(
            hidden_states=pack(x),
            encoder_hidden_states=context,
            encoder_hidden_states_mask=torch.ones(1, context.shape[1], device=device, dtype=torch.long),
            timestep=timestep.to(torch.bfloat16),
            img_shapes=[[(1, x.shape[-2] // 2, x.shape[-1] // 2)]],
            return_dict=False,
        )[0]
    del transformer

    patcher = load_comfy_diffusion_model(
        checkpoint, model_options={"dtype": torch.bfloat16}, load_device=device, offload_device=torch.device("cpu")
    )
    assert patcher.model._nunchaku_lite_family == "qwen_image"
    diffusion_model = patcher.model.diffusion_model.to(device)
    with torch.no_grad():
        actual = pack(diffusion_model(x, timestep, context))

    assert torch.isfinite(actual).all()
    # MPS bf16 baseline ~9e-3 (one 3072-wide block of random weights); a wrong AWQ modulation layout gives ~0.4.
    assert relative_error(actual, expected) < 3e-2


def test_qwen_family_rejects_unquantized_attention():
    from nunchaku_lite.comfyui.core import build_comfy_context
    from nunchaku_lite.comfyui.qwen_image import QwenImageComfyFamily

    state = {
        "txt_norm.weight": torch.empty(8),
        "transformer_blocks.0.attn.to_qkv.weight": torch.empty(24, 8),
        "transformer_blocks.0.attn.to_qkv.qweight": torch.empty(24, 4, dtype=torch.int8),
        "transformer_blocks.0.attn.to_qkv.wscales": torch.empty(1, 24),
        "transformer_blocks.0.attn.to_qkv.proj_down": torch.empty(8, 4),
    }
    family = QwenImageComfyFamily()
    assert family.matches(state, {})
    context = build_comfy_context(state, {"rank": 4}, torch.bfloat16)
    with pytest.raises(ValueError, match="no quantized transformer_blocks.0.attn.add_qkv_proj"):
        family.plan(state, {}, context)
