"""FLUX.2 in ComfyUI: the ComfyUI model must reproduce the Diffusers adapter on the same checkpoint."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from comfyui_helpers import fill_random_quantized_state, kernel_device, relative_error, require_comfyui
from diffusers import Flux2Transformer2DModel
from safetensors.torch import save_file

from nunchaku_lite import patch_transformer
from nunchaku_lite.adapters.flux2 import Flux2Adapter

RANK = 16


def make_flux2() -> Flux2Transformer2DModel:
    # Klein-like: no guidance embedding. Two heads of 128 (ComfyUI derives heads as hidden // sum(axes_dim)).
    return Flux2Transformer2DModel(
        num_layers=1,
        num_single_layers=2,
        num_attention_heads=2,
        joint_attention_dim=128,
        guidance_embeds=False,
    )


def write_checkpoint(tmp_path):
    torch.manual_seed(0)
    source = make_flux2().to(torch.bfloat16)
    Flux2Adapter().patch(
        source,
        {},
        {"rank": RANK},
        SimpleNamespace(precision="int4", torch_dtype=torch.bfloat16, device=None, strict=True, adapter_options={}),
    )
    state = fill_random_quantized_state(dict(source.state_dict()))
    path = tmp_path / "flux2-tiny.safetensors"
    save_file(state, path, metadata={"quantization_config": json.dumps({"rank": RANK})})
    return path


def ids(height: int, width: int, text_tokens: int, device):
    img = torch.zeros(height, width, 4)
    img[..., 1] = torch.arange(height)[:, None]
    img[..., 2] = torch.arange(width)[None, :]
    txt = torch.zeros(text_tokens, 4)
    txt[:, 3] = torch.arange(text_tokens)
    return img.reshape(1, -1, 4).to(device), txt.unsqueeze(0).to(device)


def test_comfy_flux2_matches_diffusers_adapter(tmp_path):
    require_comfyui()
    device = kernel_device()
    from nunchaku_lite.comfyui import load_comfy_diffusion_model

    checkpoint = write_checkpoint(tmp_path)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 128, 8, 8, generator=g).to(device, torch.bfloat16)  # patch size 1: 64 image tokens
    context = torch.randn(1, 10, 128, generator=g).to(device, torch.bfloat16)
    timestep = torch.tensor([0.5], device=device)  # exact in bf16 after Diffusers' x1000

    transformer = make_flux2()
    patch_transformer(transformer, checkpoint, precision="int4", torch_dtype=torch.bfloat16, device=device)
    img_ids, txt_ids = ids(8, 8, context.shape[1], device)
    with torch.no_grad():
        expected = transformer(
            hidden_states=x.flatten(2).transpose(1, 2),
            encoder_hidden_states=context,
            timestep=timestep.to(torch.bfloat16),
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )[0]
    del transformer

    patcher = load_comfy_diffusion_model(
        checkpoint, model_options={"dtype": torch.bfloat16}, load_device=device, offload_device=torch.device("cpu")
    )
    assert patcher.model._nunchaku_lite_family == "flux2"
    diffusion_model = patcher.model.diffusion_model.to(device)
    with torch.no_grad():
        actual = diffusion_model(x, timestep, context).flatten(2).transpose(1, 2)

    assert torch.isfinite(actual).all()
    assert relative_error(actual, expected) < 1e-2
