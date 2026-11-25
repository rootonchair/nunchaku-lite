"""SDXL in ComfyUI: the ComfyUI UNet must reproduce the Diffusers adapter on the same checkpoint.

ComfyUI only recognizes known UNet architectures, so this test uses the full SDXL UNet (built on the meta device,
random weights); it needs roughly 8 GB of host memory.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from comfyui_helpers import fill_random_quantized_state, kernel_device, relative_error, require_comfyui
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file

from nunchaku_lite.adapters.sdxl import SDXLAdapter
from nunchaku_lite.core import _patch_component

RANK = 16
SDXL_CONFIG = {
    "act_fn": "silu",
    "addition_embed_type": "text_time",
    "addition_time_embed_dim": 256,
    "attention_head_dim": [5, 10, 20],
    "block_out_channels": [320, 640, 1280],
    "cross_attention_dim": 2048,
    "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
    "in_channels": 4,
    "layers_per_block": 2,
    "out_channels": 4,
    "projection_class_embeddings_input_dim": 2816,
    "sample_size": 128,
    "transformer_layers_per_block": [1, 2, 10],
    "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
    "use_linear_projection": True,
}


def make_unet() -> UNet2DConditionModel:
    with torch.device("meta"):
        return UNet2DConditionModel(**SDXL_CONFIG).to(torch.bfloat16)


def write_checkpoint(tmp_path):
    source = make_unet()
    SDXLAdapter().patch(
        source,
        {},
        {"rank": RANK},
        SimpleNamespace(precision="int4", torch_dtype=torch.bfloat16, device=None, strict=True, adapter_options={}),
    )
    state = fill_random_quantized_state(dict(source.state_dict()))
    del source
    path = tmp_path / "sdxl.safetensors"
    save_file(state, path, metadata={"quantization_config": json.dumps({"rank": RANK})})
    return path


def test_comfy_sdxl_matches_diffusers_adapter(tmp_path):
    require_comfyui()
    device = kernel_device()
    from nunchaku_lite.comfyui import load_comfy_diffusion_model

    checkpoint = write_checkpoint(tmp_path)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 4, 32, 32, generator=g).to(device, torch.bfloat16)
    context = torch.randn(1, 8, 2048, generator=g).to(device, torch.bfloat16)
    pooled = torch.randn(1, 1280, generator=g).to(device, torch.bfloat16)
    time_ids = torch.tensor([[256.0, 256.0, 0.0, 0.0, 256.0, 256.0]], device=device)
    timestep = torch.tensor([500.0], device=device)  # exact in bf16

    unet = _patch_component(
        make_unet(),
        checkpoint,
        target="auto",
        precision="int4",
        torch_dtype=torch.bfloat16,
        device=device,
        strict=True,
        adapter_options=None,
        assign=True,
    )
    with torch.no_grad():
        expected = unet(
            x,
            timestep.to(torch.bfloat16),
            encoder_hidden_states=context,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids.to(torch.bfloat16)},
            return_dict=False,
        )[0]
        # ComfyUI's SDXL ``y``: pooled text embedding followed by the sinusoidal size/crop conditioning.
        y = torch.cat([pooled, unet.add_time_proj(time_ids.flatten()).reshape(1, -1).to(torch.bfloat16)], dim=-1)
    del unet

    patcher = load_comfy_diffusion_model(
        checkpoint, model_options={"dtype": torch.bfloat16}, load_device=device, offload_device=torch.device("cpu")
    )
    assert patcher.model._nunchaku_lite_family == "sdxl"
    diffusion_model = patcher.model.diffusion_model.to(device)
    with torch.no_grad():
        actual = diffusion_model(x, timestep, context, y=y)

    assert torch.isfinite(actual).all()
    assert relative_error(actual, expected) < 3e-2


def test_comfy_sdxl_attention_and_ff_match_diffusers_modules(tmp_path):
    """Each ComfyUI transformer-block module equals its Diffusers adapter counterpart on the same input.

    The input is scaled so attention is peaked: with random weights and unit inputs the softmax is nearly uniform,
    and a q/k mix-up in the fused self-attention would barely change the end-to-end output. Attention runs through
    different kernels on each side (MPS bf16: up to ~2e-2); swapping q and k in the fused projection gives > 1.
    """

    require_comfyui()
    device = kernel_device()
    from nunchaku_lite.comfyui import load_comfy_diffusion_model

    checkpoint = write_checkpoint(tmp_path)
    unet = _patch_component(
        make_unet(),
        checkpoint,
        target="auto",
        precision="int4",
        torch_dtype=torch.bfloat16,
        device=device,
        strict=True,
        adapter_options=None,
        assign=True,
    )
    patcher = load_comfy_diffusion_model(
        checkpoint, model_options={"dtype": torch.bfloat16}, load_device=device, offload_device=torch.device("cpu")
    )
    model = patcher.model.diffusion_model.to(device)
    modules = model._nunchaku_lite_modules
    g = torch.Generator().manual_seed(4)
    checked = 0
    with torch.no_grad():
        for path, spec in modules.items():
            if not path.endswith("attn1.to_qkv"):
                continue
            ours = model.get_submodule(path[: -len(".to_qkv")])
            block = path[: -len(".attn1.to_qkv")]
            theirs_block = unet.get_submodule(spec["source"][: -len(".attn1.to_qkv")])
            dim = ours.to_qkv.in_features
            x = (torch.randn(1, 24, dim, generator=g) * 8).to(device, torch.bfloat16)
            context = (torch.randn(1, 8, 2048, generator=g) * 8).to(device, torch.bfloat16)
            assert relative_error(ours(x), theirs_block.attn1(x)) < 5e-2, path
            ours_block = model.get_submodule(block)
            assert (
                relative_error(
                    ours_block.attn2(x, context=context), theirs_block.attn2(x, encoder_hidden_states=context)
                )
                < 5e-2
            ), path
            assert torch.equal(ours_block.ff(x), theirs_block.ff(x)), path
            checked += 1
    assert checked == 70  # SDXL transformer blocks: down 2x2 + 2x10, mid 10, up 3x10 + 3x2
