"""Z-Image in ComfyUI: the ComfyUI model must reproduce the Diffusers adapter on the same checkpoint.

ComfyUI's Z-Image config has a fixed width (3840) and two refiner layers, so the synthetic model has the real width
and a single main layer; both models are built on the meta device to avoid allocating dense weights.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from comfyui_helpers import fill_random_quantized_state, kernel_device, relative_error, require_comfyui
from diffusers import ZImageTransformer2DModel
from safetensors.torch import save_file

from nunchaku_lite.adapters.z_image import ZImageAdapter
from nunchaku_lite.core import _patch_component

RANK = 16
CAPTION_TOKENS = 32  # a multiple of 32: neither side pads the caption


def make_z_image() -> ZImageTransformer2DModel:
    with torch.device("meta"):
        return ZImageTransformer2DModel(n_layers=1).to(torch.bfloat16)


def write_checkpoint(tmp_path):
    source = make_z_image()
    ZImageAdapter().patch(
        source,
        {},
        {"rank": RANK},
        SimpleNamespace(precision="int4", torch_dtype=torch.bfloat16, device=None, strict=True, adapter_options={}),
    )
    state = fill_random_quantized_state(dict(source.state_dict()))
    path = tmp_path / "z-image-tiny.safetensors"
    save_file(state, path, metadata={"quantization_config": json.dumps({"rank": RANK})})
    return path


def test_comfy_z_image_matches_diffusers_adapter(tmp_path):
    require_comfyui()
    device = kernel_device()
    from nunchaku_lite.comfyui import load_comfy_diffusion_model

    checkpoint = write_checkpoint(tmp_path)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 16, 16, 16, generator=g).to(device, torch.bfloat16)  # 8x8 = 64 image tokens
    context = torch.randn(1, CAPTION_TOKENS, 2560, generator=g).to(device, torch.bfloat16)
    sigma = torch.tensor([0.5], device=device)  # ComfyUI feeds t = 1 - sigma; 0.5 is exact in bf16

    transformer = load_diffusers(checkpoint, device)
    with torch.no_grad():
        expected = transformer([x[0].unsqueeze(1)], 1.0 - sigma, [context[0]], return_dict=False)[0][0][:, 0]
    del transformer

    patcher = load_comfy_diffusion_model(
        checkpoint, model_options={"dtype": torch.bfloat16}, load_device=device, offload_device=torch.device("cpu")
    )
    assert patcher.model._nunchaku_lite_family == "z_image"
    diffusion_model = patcher.model.diffusion_model.to(device)
    with torch.no_grad():
        actual = diffusion_model(x, sigma, context, CAPTION_TOKENS)[0]

    assert torch.isfinite(actual).all()
    # ComfyUI's NextDiT returns the negated prediction (Diffusers' Z-Image pipeline negates it instead). The bound is
    # loose because this random-weight model amplifies tiny differences: ComfyUI's own q/k-norm + RoPE differ from
    # Diffusers' by ~1% in bf16 (MPS: ~7e-2 end to end). On CUDA even Diffusers alone differs by ~0.1 between two
    # runs on the same input, and a 0.1% input change moves its output by ~7e-2 (ComfyUI vs Diffusers: ~0.3), so
    # there the bound only catches gross errors. Swapping the SwiGLU halves gives ~0.8. The exact mapping is checked
    # layer by layer in test_comfy_z_image_layers_match_diffusers_layers.
    assert relative_error(-actual, expected) < (0.15 if device.type == "mps" else 0.5)


def load_diffusers(checkpoint, device):
    return _patch_component(
        make_z_image(),
        checkpoint,
        target="auto",
        precision="int4",
        torch_dtype=torch.bfloat16,
        device=device,
        strict=True,
        adapter_options=None,
        assign=True,
    )


def test_comfy_z_image_layers_match_diffusers_layers(tmp_path):
    """Every installed W4A4 layer gives the Diffusers adapter's output on the same input (bit-exact on MPS)."""

    require_comfyui()
    device = kernel_device()
    from nunchaku_lite.comfyui import load_comfy_diffusion_model

    checkpoint = write_checkpoint(tmp_path)
    reference = load_diffusers(checkpoint, device)
    patcher = load_comfy_diffusion_model(
        checkpoint, model_options={"dtype": torch.bfloat16}, load_device=device, offload_device=torch.device("cpu")
    )
    model = patcher.model.diffusion_model.to(device)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 40, 3840, generator=g).to(device, torch.bfloat16)
    with torch.no_grad():
        for group in ("layers.0", "noise_refiner.0", "noise_refiner.1", "context_refiner.0", "context_refiner.1"):
            ours, theirs = model.get_submodule(group), reference.get_submodule(group)
            # Same kernels on the same weights: equal up to the CUDA kernels' run-to-run noise (~1e-4); a wrong
            # mapping (e.g. swapped SwiGLU halves or q/k/v order) gives errors near 1.
            assert relative_error(ours.attention.qkv(x), theirs.attention.to_qkv(x)) < 1e-3, group
            assert relative_error(ours.attention.out(x), theirs.attention.to_out[0](x)) < 1e-3, group
            assert relative_error(ours.feed_forward(x), theirs.feed_forward(x)) < 1e-3, group
