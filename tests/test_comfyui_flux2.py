"""FLUX.2 in ComfyUI: the ComfyUI model must reproduce the Diffusers adapter on the same checkpoint."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
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


def write_checkpoint(tmp_path, rotated: tuple[str, ...] = ()):
    torch.manual_seed(0)
    source = make_flux2().to(torch.bfloat16)
    Flux2Adapter().patch(
        source,
        {},
        {"rank": RANK},
        SimpleNamespace(precision="int4", torch_dtype=torch.bfloat16, device=None, strict=True, adapter_options={}),
    )
    state = fill_random_quantized_state(dict(source.state_dict()))
    config = {"rank": RANK}
    if rotated:
        # Rotation leaves the packed layout unchanged; only the declaration and the marker tensor are added.
        config["input_rotation"] = {"format_version": 2, "type": "hadamard", "targets": list(rotated)}
        state["nunchaku_lite.input_rotation"] = torch.empty(0)
    path = tmp_path / "flux2-tiny.safetensors"
    save_file(state, path, metadata={"quantization_config": json.dumps(config)})
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


# Checkpoint targets -> the ComfyUI modules they must bind to.
ROTATED = {
    "transformer_blocks.0.attn.to_qkv": "double_blocks.0.img_attn.qkv",
    "transformer_blocks.0.attn.to_add_out": "double_blocks.0.txt_attn.proj",
    "transformer_blocks.0.ff.linear_out": "double_blocks.0.img_mlp.2",
    "transformer_blocks.0.ff_context.linear_in": "double_blocks.0.txt_mlp.0",
    "single_transformer_blocks.1.attn.mlp_fc2": "single_blocks.1.mlp_fc2",
    "single_transformer_blocks.0.attn.qkv_proj": "single_blocks.0.qkv_proj",
}


def _rotated_modules(diffusion_model):
    from nunchaku_lite.linear import SVDQW4A4Linear

    return {
        name
        for name, module in diffusion_model.named_modules()
        if isinstance(module, SVDQW4A4Linear) and module.input_rotation is not None
    }


def test_comfy_flux2_binds_rotation_to_exactly_the_declared_layers(tmp_path):
    """Placeholders follow the key mapping onto ComfyUI module paths (CPU load: no kernels run)."""

    require_comfyui()
    from nunchaku_lite.comfyui import load_comfy_diffusion_model

    checkpoint = write_checkpoint(tmp_path, rotated=tuple(ROTATED))
    cpu = torch.device("cpu")
    patcher = load_comfy_diffusion_model(
        checkpoint, model_options={"dtype": torch.bfloat16}, load_device=cpu, offload_device=cpu
    )
    assert _rotated_modules(patcher.model.diffusion_model) == set(ROTATED.values())
    assert all(
        module.input_rotation in (None, "hadamard")
        for module in patcher.model.diffusion_model.modules()
        if hasattr(module, "input_rotation")
    )


def test_comfy_flux2_rotation_decision_table(tmp_path):
    """The Diffusers loader's refusals apply unchanged: marker without declaration, declaration without marker."""

    require_comfyui()
    from safetensors.torch import load_file

    from nunchaku_lite.comfyui import load_comfy_diffusion_model

    cpu = torch.device("cpu")
    checkpoint = write_checkpoint(tmp_path, rotated=tuple(ROTATED))
    state = load_file(checkpoint)
    for config, extra, message in (
        ({"rank": RANK}, {"nunchaku_lite.input_rotation": torch.empty(0)}, "no quantization_config.input_rotation"),
        (
            {"rank": RANK, "input_rotation": {"format_version": 2, "type": "hadamard", "targets": list(ROTATED)}},
            {},
            "marker tensor",
        ),
        (
            {"rank": RANK, "input_rotation": {"format_version": 2, "type": "hadamard", "targets": ["nope.proj"]}},
            {"nunchaku_lite.input_rotation": torch.empty(0)},
            "no packed SVDQ weight",
        ),
    ):
        tensors = {k: v for k, v in state.items() if k != "nunchaku_lite.input_rotation"} | extra
        path = tmp_path / "variant.safetensors"
        save_file(tensors, path, metadata={"quantization_config": json.dumps(config)})
        with pytest.raises(ValueError, match=message):
            load_comfy_diffusion_model(path, load_device=cpu, offload_device=cpu)


def test_comfy_flux2_rotation_is_refused_on_mps(tmp_path):
    require_comfyui()
    if not torch.backends.mps.is_available():
        pytest.skip("needs MPS")
    from nunchaku_lite.comfyui import load_comfy_diffusion_model

    checkpoint = write_checkpoint(tmp_path, rotated=tuple(ROTATED))
    with pytest.raises(NotImplementedError, match="MPS"):
        load_comfy_diffusion_model(checkpoint, load_device=torch.device("mps"), offload_device=torch.device("cpu"))


def test_comfy_flux2_rotated_matches_diffusers_adapter(tmp_path):
    """Rotated layers run the Hadamard quantize op on both sides (CUDA kernels with the op only).

    The end-to-end output of this tiny random model barely depends on the rotation (~1e-2), so each rotated layer
    is also compared directly: same output as its Diffusers counterpart, and ~1 relative change without rotation.
    """

    require_comfyui()
    if not torch.cuda.is_available():
        pytest.skip("rotation runs on CUDA only")
    from nunchaku_lite.ops.backend import get_ops

    if getattr(get_ops(), "quantize_w4a4_act_hadamard_fuse_lora", None) is None:
        pytest.skip("installed kernels have no Hadamard quantize op")
    from nunchaku_lite.comfyui import load_comfy_diffusion_model

    device = torch.device("cuda")
    checkpoint = write_checkpoint(tmp_path, rotated=tuple(ROTATED))
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 128, 8, 8, generator=g).to(device, torch.bfloat16)
    context = torch.randn(1, 10, 128, generator=g).to(device, torch.bfloat16)
    timestep = torch.tensor([0.5], device=device)

    transformer = make_flux2()
    patch_transformer(transformer, checkpoint, precision="int4", torch_dtype=torch.bfloat16, device=device)
    assert {n for n, m in transformer.named_modules() if getattr(m, "input_rotation", None)} == set(ROTATED)
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

    patcher = load_comfy_diffusion_model(
        checkpoint, model_options={"dtype": torch.bfloat16}, load_device=device, offload_device=torch.device("cpu")
    )
    diffusion_model = patcher.model.diffusion_model.to(device)
    with torch.no_grad():
        actual = diffusion_model(x, timestep, context).flatten(2).transpose(1, 2)
        assert torch.isfinite(actual).all()
        assert relative_error(actual, expected) < 1e-2

        for source, path in ROTATED.items():
            ours, theirs = diffusion_model.get_submodule(path), transformer.get_submodule(source)
            h = torch.randn(1, 64, ours.in_features, generator=g).to(device, torch.bfloat16)
            rotated = ours(h)
            assert relative_error(rotated, theirs(h)) < 1e-3, path  # CUDA measured: bit-exact
            ours.input_rotation = None
            try:
                assert relative_error(ours(h), rotated) > 0.5, path  # measured ~1.03-1.06
            finally:
                ours.input_rotation = "hadamard"
