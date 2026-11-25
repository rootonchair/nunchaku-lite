"""FLUX.1 in ComfyUI: the ComfyUI model must reproduce the Diffusers adapter on the same checkpoint."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from comfyui_helpers import fill_random_quantized_state, kernel_device, relative_error, require_comfyui
from diffusers import FluxTransformer2DModel
from safetensors.torch import save_file

from nunchaku_lite import patch_transformer
from nunchaku_lite.adapters.flux import FluxAdapter
from nunchaku_lite.linear import AWQW4A16Linear

RANK = 16
HIDDEN = 256  # two heads of 128, the head size ComfyUI's Flux detection assumes (hidden // sum(axes_dim))


def make_flux(**overrides) -> FluxTransformer2DModel:
    config = {
        "in_channels": 64,
        "num_layers": 2,
        "num_single_layers": 2,
        "attention_head_dim": 128,
        "num_attention_heads": 2,
        "joint_attention_dim": 128,
        "pooled_projection_dim": 64,
        "guidance_embeds": True,
        "axes_dims_rope": (16, 56, 56),
    }
    config.update(overrides)
    return FluxTransformer2DModel(**config)


def _awq_modulations(state: dict[str, torch.Tensor], dense: FluxTransformer2DModel) -> dict[str, torch.Tensor]:
    state = dict(state)
    modulation_suffixes = (".norm1.linear.weight", ".norm1_context.linear.weight", ".norm.linear.weight")
    prefixes = [k[: -len(".weight")] for k in list(state) if k.endswith(modulation_suffixes)]
    for prefix in prefixes:
        for suffix in ("weight", "bias"):
            state.pop(f"{prefix}.{suffix}", None)
        awq = AWQW4A16Linear.from_linear(dense.get_submodule(prefix), torch_dtype=torch.bfloat16, device="cpu")
        state.update({f"{prefix}.{suffix}": value for suffix, value in awq.state_dict().items()})
    return state


def write_checkpoint(tmp_path, *, awq_modulation: bool, precision: str = "int4"):
    torch.manual_seed(0)
    dense = make_flux().to(torch.bfloat16)
    source = make_flux().to(torch.bfloat16)
    FluxAdapter().patch(
        source,
        {},
        {"rank": RANK},
        SimpleNamespace(precision=precision, torch_dtype=torch.bfloat16, device=None, strict=True, adapter_options={}),
    )
    state = {k: v for k, v in source.state_dict().items()}
    if awq_modulation:
        state = _awq_modulations(state, dense)
    state = fill_random_quantized_state(state, precision=precision)
    path = tmp_path / "flux-tiny.safetensors"
    save_file(state, path, metadata={"quantization_config": json.dumps({"rank": RANK})})
    return path


def comfy_inputs(device):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 16, 16, 16, generator=g)  # 8x8 = 64 image tokens after 2x2 patchify
    context = torch.randn(1, 8, 128, generator=g)
    y = torch.randn(1, 64, generator=g)
    # Diffusers scales timestep and guidance by 1000 in bf16 (ComfyUI in fp32): use values whose x1000 is exact in bf16.
    timestep = torch.tensor([0.5])
    guidance = torch.tensor([4.0])
    to = lambda t: t.to(device=device, dtype=torch.bfloat16)
    return to(x), to(context), to(y), timestep.to(device), guidance.to(device)


def pack(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    return latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)


def image_ids(h: int, w: int, device) -> torch.Tensor:
    ids = torch.zeros(h // 2, w // 2, 3)
    ids[..., 1] = torch.arange(h // 2)[:, None]
    ids[..., 2] = torch.arange(w // 2)[None, :]
    return ids.reshape(-1, 3).to(device)


def run_diffusers(checkpoint, device, x, context, y, timestep, guidance, precision="int4"):
    transformer = make_flux()
    public = "fp4" if precision == "nvfp4" else precision
    patch_transformer(transformer, checkpoint, precision=public, torch_dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        out = transformer(
            hidden_states=pack(x),
            encoder_hidden_states=context,
            pooled_projections=y,
            timestep=timestep.to(torch.bfloat16),
            img_ids=image_ids(x.shape[2], x.shape[3], device),
            txt_ids=torch.zeros(context.shape[1], 3, device=device),
            guidance=guidance.to(torch.bfloat16),
            return_dict=False,
        )[0]
    return out


def run_comfy(checkpoint, device, x, context, y, timestep, guidance):
    from nunchaku_lite.comfyui import load_comfy_diffusion_model

    patcher = load_comfy_diffusion_model(
        checkpoint, model_options={"dtype": torch.bfloat16}, load_device=device, offload_device=torch.device("cpu")
    )
    diffusion_model = patcher.model.diffusion_model.to(device)
    with torch.no_grad():
        out = diffusion_model(x, timestep, context, y=y, guidance=guidance)
    return patcher, pack(out)


@pytest.mark.parametrize("precision", ["int4", "nvfp4"])
@pytest.mark.parametrize("awq_modulation", [True, False], ids=["awq-modulation", "dense-modulation"])
def test_comfy_flux_matches_diffusers_adapter(tmp_path, awq_modulation, precision):
    require_comfyui()
    device = kernel_device()
    if precision == "nvfp4" and (device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 10):
        pytest.skip("NVFP4 kernels need a Blackwell GPU")
    checkpoint = write_checkpoint(tmp_path, awq_modulation=awq_modulation, precision=precision)
    inputs = comfy_inputs(device)

    expected = run_diffusers(checkpoint, device, *inputs, precision=precision)
    patcher, actual = run_comfy(checkpoint, device, *inputs)

    assert patcher.model._nunchaku_lite_family == "flux"
    assert torch.isfinite(actual).all()
    assert expected.abs().mean() > 0
    # INT4 bf16 noise ~3e-3, NVFP4 (Blackwell CUDA) ~1.3e-2; a missing AWQ +1 offset gives ~6e-2.
    assert relative_error(actual, expected) < (1e-2 if precision == "int4" else 2e-2)


def test_comfy_flux_installs_quantized_modules(tmp_path):
    require_comfyui()
    from comfy.ldm.flux.layers import SingleStreamBlock

    from nunchaku_lite.comfyui import load_comfy_diffusion_model
    from nunchaku_lite.comfyui.flux import ComfyAWQModulationLinear, FusedGeluMLP
    from nunchaku_lite.linear import SVDQW4A4Linear

    checkpoint = write_checkpoint(tmp_path, awq_modulation=True)
    patcher = load_comfy_diffusion_model(
        checkpoint,
        model_options={"dtype": torch.bfloat16},
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )
    model = patcher.model.diffusion_model
    double, single = model.double_blocks[0], model.single_blocks[0]
    assert isinstance(double.img_attn.qkv, SVDQW4A4Linear)
    assert isinstance(double.txt_attn.proj, SVDQW4A4Linear)
    assert isinstance(double.img_mlp, FusedGeluMLP) and double.img_mlp.fc2.act_unsigned
    assert isinstance(double.img_mod.lin, ComfyAWQModulationLinear)
    assert isinstance(single, SingleStreamBlock) and type(single).__name__ == "NunchakuSingleStreamBlock"
    assert not hasattr(single, "linear1") and not hasattr(single, "linear2")
    assert isinstance(single.modulation.lin, ComfyAWQModulationLinear)
    assert not any(t.is_meta for t in model.parameters())


def test_comfy_flux_samples_and_offloads_through_model_management(tmp_path):
    """ComfyUI's sampler loads the model onto the device through ModelPatcher and unloading returns every tensor,
    including the W4A4 layers' packed weights, to the offload device."""

    require_comfyui()
    device = kernel_device()
    import comfy.model_management
    import comfy.sample

    from nunchaku_lite.comfyui import load_comfy_diffusion_model

    checkpoint = write_checkpoint(tmp_path, awq_modulation=True)
    patcher = load_comfy_diffusion_model(
        checkpoint, model_options={"dtype": torch.bfloat16}, load_device=device, offload_device=torch.device("cpu")
    )
    g = torch.Generator().manual_seed(2)
    latent = torch.zeros(1, 16, 16, 16)
    noise = torch.randn(latent.shape, generator=g)
    positive = [
        [torch.randn(1, 8, 128, generator=g), {"pooled_output": torch.randn(1, 64, generator=g), "guidance": 4.0}]
    ]
    negative = [[torch.zeros(1, 8, 128), {"pooled_output": torch.zeros(1, 64), "guidance": 4.0}]]

    samples = comfy.sample.sample(
        patcher,
        noise,
        steps=2,
        cfg=1.0,
        sampler_name="euler",
        scheduler="simple",
        positive=positive,
        negative=negative,
        latent_image=latent,
        disable_pbar=True,
        seed=0,
    )

    assert samples.shape == latent.shape
    assert torch.isfinite(samples).all()
    comfy.model_management.unload_all_models()
    devices = {t.device.type for t in patcher.model.diffusion_model.state_dict().values()}
    assert devices == {"cpu"}


def test_comfy_flux_refuses_rotated_checkpoints(tmp_path):
    """FLUX.1 MLPs run fused (unsigned fc2 input), so rotation is refused before anything is built."""

    require_comfyui()
    from safetensors.torch import load_file

    from nunchaku_lite.comfyui import load_comfy_diffusion_model

    state = load_file(write_checkpoint(tmp_path, awq_modulation=False))
    state["nunchaku_lite.input_rotation"] = torch.empty(0)
    target = "single_transformer_blocks.0.attn.to_out"
    config = {"rank": RANK, "input_rotation": {"format_version": 2, "type": "hadamard", "targets": [target]}}
    path = tmp_path / "rotated.safetensors"
    save_file(state, path, metadata={"quantization_config": json.dumps(config)})
    cpu = torch.device("cpu")
    with pytest.raises(ValueError, match="does not support"):
        load_comfy_diffusion_model(path, load_device=cpu, offload_device=cpu)
