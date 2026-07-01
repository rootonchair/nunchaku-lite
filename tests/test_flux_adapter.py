import json
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from diffusers import FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import FluxIPAdapterAttnProcessor
from nunchaku_lite import patch_transformer
from nunchaku_lite.adapters.flux import (
    FluxAdapter,
    NunchakuAdaLayerNormZero,
    NunchakuAdaLayerNormZeroSingle,
    NunchakuFluxAttention,
    NunchakuFluxAttnProcessor,
    NunchakuFluxFeedForward,
    NunchakuFluxTransformerBlock,
    convert_flux_state_dict,
)
from nunchaku_lite.linear import AWQW4A16Linear, DenseRuntimeLoraLinear, SVDQW4A4Linear


def make_tiny_flux_transformer():
    return FluxTransformer2DModel(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        pooled_projection_dim=32,
        guidance_embeds=False,
        axes_dims_rope=(4, 6, 6),
    )


def replace_flux_adanorm_awq_with_dense(state):
    state = dict(state)
    dense_state = make_tiny_flux_transformer().state_dict()
    prefixes = (
        "transformer_blocks.0.norm1.linear",
        "transformer_blocks.0.norm1_context.linear",
        "single_transformer_blocks.0.norm.linear",
    )

    for prefix in prefixes:
        for suffix in ("qweight", "wscales", "wzeros"):
            state.pop(f"{prefix}.{suffix}", None)
        state[f"{prefix}.weight"] = dense_state[f"{prefix}.weight"].clone()
        state[f"{prefix}.bias"] = dense_state[f"{prefix}.bias"].clone()
    return state


def replace_flux_adanorm_dense_with_awq(state):
    state = dict(state)
    dense = make_tiny_flux_transformer()
    prefixes = (
        "transformer_blocks.0.norm1.linear",
        "transformer_blocks.0.norm1_context.linear",
        "single_transformer_blocks.0.norm.linear",
    )

    for prefix in prefixes:
        for suffix in ("weight", "proj_down", "proj_up", "smooth_factor", "smooth_factor_orig"):
            state.pop(f"{prefix}.{suffix}", None)
        linear = dense.get_submodule(prefix)
        awq = AWQW4A16Linear.from_linear(linear, torch_dtype=torch.bfloat16)
        for suffix, value in awq.state_dict().items():
            state[f"{prefix}.{suffix}"] = value.clone()
    return state


def replace_flux_adanorm_awq_with_svdq(state, rank=4):
    state = dict(state)
    dense = make_tiny_flux_transformer()
    prefixes = (
        "transformer_blocks.0.norm1.linear",
        "transformer_blocks.0.norm1_context.linear",
        "single_transformer_blocks.0.norm.linear",
    )

    for prefix in prefixes:
        for suffix in ("qweight", "wscales", "wzeros", "weight", "bias"):
            state.pop(f"{prefix}.{suffix}", None)
        linear = dense.get_submodule(prefix)
        svdq = SVDQW4A4Linear.from_linear(linear, rank=rank, precision="int4", torch_dtype=torch.bfloat16)
        for suffix, value in svdq.state_dict().items():
            state[f"{prefix}.{suffix}"] = value.clone()
    return state


def test_flux_adapter_matches_diffusers_transformer():
    transformer = make_tiny_flux_transformer()
    assert FluxAdapter().matches(transformer)


def test_convert_flux_state_dict_maps_original_nunchaku_keys():
    state = {
        "transformer_blocks.0.qkv_proj.qweight": torch.empty(1),
        "transformer_blocks.0.qkv_proj_context.lora_down": torch.empty(1),
        "transformer_blocks.0.out_proj_context.smooth_orig": torch.empty(1),
        "single_transformer_blocks.0.out_proj.lora_up": torch.empty(1),
    }

    converted = convert_flux_state_dict(state)

    assert "transformer_blocks.0.attn.to_qkv.qweight" in converted
    assert "transformer_blocks.0.attn.add_qkv_proj.proj_down" in converted
    assert "transformer_blocks.0.attn.to_add_out.smooth_factor_orig" in converted
    assert "single_transformer_blocks.0.attn.to_out.proj_up" in converted


def test_convert_flux_state_dict_leaves_corrected_keys_unchanged():
    state = {
        "transformer_blocks.0.attn.to_qkv.qweight": torch.empty(1),
        "transformer_blocks.0.ff_context.net.0.proj.smooth_factor": torch.empty(1),
        "single_transformer_blocks.0.attn.to_out.proj_up": torch.empty(1),
    }

    converted = convert_flux_state_dict(state)

    assert converted is state
    assert set(converted) == set(state)


def test_patch_transformer_patches_flux_from_synthetic_checkpoint(tmp_path):
    rank = 4
    source = make_tiny_flux_transformer()
    adapter = FluxAdapter()
    adapter.patch(
        source,
        {},
        {"rank": rank},
        SimpleNamespace(
            precision="int4",
            torch_dtype=torch.bfloat16,
            device=None,
            strict=True,
            adapter_options={},
        ),
    )
    state = source.state_dict()
    checkpoint = tmp_path / "flux-lite.safetensors"
    save_file(state, checkpoint, metadata={"quantization_config": json.dumps({"rank": rank})})

    transformer = make_tiny_flux_transformer()
    returned = patch_transformer(transformer, checkpoint, precision="int4", torch_dtype=torch.bfloat16)

    assert returned is transformer
    assert transformer._nunchaku_lite_patched
    assert transformer._nunchaku_lite_target == "flux"
    assert isinstance(transformer.transformer_blocks[0], NunchakuFluxTransformerBlock)
    assert transformer.single_transformer_blocks[0].__class__.__name__ == "NunchakuFluxSingleTransformerBlock"
    assert isinstance(transformer.transformer_blocks[0].attn, NunchakuFluxAttention)
    assert isinstance(transformer.transformer_blocks[0].ff, NunchakuFluxFeedForward)
    assert isinstance(transformer.transformer_blocks[0].ff_context, NunchakuFluxFeedForward)
    assert isinstance(transformer.transformer_blocks[0].norm1.linear, DenseRuntimeLoraLinear)
    assert isinstance(transformer.transformer_blocks[0].norm1_context.linear, DenseRuntimeLoraLinear)
    assert isinstance(transformer.single_transformer_blocks[0].norm.linear, DenseRuntimeLoraLinear)
    assert isinstance(transformer.transformer_blocks[0].attn.to_qkv, SVDQW4A4Linear)
    assert isinstance(transformer.single_transformer_blocks[0].attn.to_out, SVDQW4A4Linear)
    assert not hasattr(transformer.single_transformer_blocks[0], "proj_out")


def test_patch_transformer_loads_flux_with_dense_adanorm_checkpoint(tmp_path):
    rank = 4
    source = make_tiny_flux_transformer()
    FluxAdapter().patch(
        source,
        {},
        {"rank": rank},
        SimpleNamespace(
            precision="int4",
            torch_dtype=torch.bfloat16,
            device=None,
            strict=True,
            adapter_options={},
        ),
    )
    state = replace_flux_adanorm_awq_with_dense(source.state_dict())
    checkpoint = tmp_path / "flux-lite-dense-adanorm.safetensors"
    save_file(state, checkpoint, metadata={"quantization_config": json.dumps({"rank": rank})})

    transformer = make_tiny_flux_transformer()
    returned = patch_transformer(transformer, checkpoint, precision="int4", torch_dtype=torch.bfloat16)

    assert returned is transformer
    assert transformer._nunchaku_lite_patched
    assert isinstance(transformer.transformer_blocks[0], NunchakuFluxTransformerBlock)
    assert isinstance(transformer.transformer_blocks[0].norm1.linear, DenseRuntimeLoraLinear)
    assert isinstance(transformer.transformer_blocks[0].norm1_context.linear, DenseRuntimeLoraLinear)
    assert isinstance(transformer.single_transformer_blocks[0].norm.linear, DenseRuntimeLoraLinear)
    assert isinstance(transformer.transformer_blocks[0].attn.to_qkv, SVDQW4A4Linear)
    assert isinstance(transformer.single_transformer_blocks[0].attn.to_out, SVDQW4A4Linear)


def test_patch_transformer_loads_flux_with_awq_adanorm_checkpoint(tmp_path):
    rank = 4
    source = make_tiny_flux_transformer()
    FluxAdapter().patch(
        source,
        {},
        {"rank": rank},
        SimpleNamespace(
            precision="int4",
            torch_dtype=torch.bfloat16,
            device=None,
            strict=True,
            adapter_options={},
        ),
    )
    state = replace_flux_adanorm_dense_with_awq(source.state_dict())
    checkpoint = tmp_path / "flux-lite-awq-adanorm.safetensors"
    save_file(state, checkpoint, metadata={"quantization_config": json.dumps({"rank": rank})})

    transformer = make_tiny_flux_transformer()
    returned = patch_transformer(transformer, checkpoint, precision="int4", torch_dtype=torch.bfloat16)

    assert returned is transformer
    assert transformer._nunchaku_lite_patched
    assert isinstance(transformer.transformer_blocks[0], NunchakuFluxTransformerBlock)
    assert isinstance(transformer.transformer_blocks[0].norm1.linear, AWQW4A16Linear)
    assert isinstance(transformer.transformer_blocks[0].norm1_context.linear, AWQW4A16Linear)
    assert isinstance(transformer.single_transformer_blocks[0].norm.linear, AWQW4A16Linear)
    assert isinstance(transformer.transformer_blocks[0].attn.to_qkv, SVDQW4A4Linear)
    assert isinstance(transformer.single_transformer_blocks[0].attn.to_out, SVDQW4A4Linear)


def test_patch_transformer_loads_flux_with_svdq_adanorm_checkpoint(tmp_path):
    rank = 4
    source = make_tiny_flux_transformer()
    FluxAdapter().patch(
        source,
        {},
        {"rank": rank},
        SimpleNamespace(
            precision="int4",
            torch_dtype=torch.bfloat16,
            device=None,
            strict=True,
            adapter_options={},
        ),
    )
    state = replace_flux_adanorm_awq_with_svdq(source.state_dict(), rank=rank)
    checkpoint = tmp_path / "flux-lite-svdq-adanorm.safetensors"
    save_file(state, checkpoint, metadata={"quantization_config": json.dumps({"rank": rank})})

    transformer = make_tiny_flux_transformer()
    returned = patch_transformer(transformer, checkpoint, precision="int4", torch_dtype=torch.bfloat16)

    assert returned is transformer
    assert transformer._nunchaku_lite_patched
    assert isinstance(transformer.transformer_blocks[0], NunchakuFluxTransformerBlock)
    assert isinstance(transformer.transformer_blocks[0].norm1.linear, SVDQW4A4Linear)
    assert isinstance(transformer.transformer_blocks[0].norm1_context.linear, SVDQW4A4Linear)
    assert isinstance(transformer.single_transformer_blocks[0].norm.linear, SVDQW4A4Linear)
    assert isinstance(transformer.transformer_blocks[0].attn.to_qkv, SVDQW4A4Linear)
    assert isinstance(transformer.single_transformer_blocks[0].attn.to_out, SVDQW4A4Linear)


def test_dense_adanorm_zero_uses_diffusers_channel_layout():
    norm = make_tiny_flux_transformer().transformer_blocks[0].norm1
    wrapped = NunchakuAdaLayerNormZero(norm, scale_shift=0.0, linear_cls=DenseRuntimeLoraLinear)
    x = torch.randn(2, 3, norm.norm.normalized_shape[0])
    emb = torch.randn(2, norm.linear.in_features)

    expected = norm(x, emb=emb)
    actual = wrapped(x, emb=emb)

    assert torch.allclose(actual[0], expected[0])
    assert torch.allclose(actual[1], expected[1])
    assert torch.allclose(actual[2], expected[2])
    assert torch.allclose(actual[3], expected[3] + 1.0)
    assert torch.allclose(actual[4], expected[4])


def test_dense_adanorm_zero_single_uses_diffusers_channel_layout():
    norm = make_tiny_flux_transformer().single_transformer_blocks[0].norm
    wrapped = NunchakuAdaLayerNormZeroSingle(norm, scale_shift=0.0, linear_cls=DenseRuntimeLoraLinear)
    x = torch.randn(2, 3, norm.norm.normalized_shape[0])
    emb = torch.randn(2, norm.linear.in_features)

    expected = norm(x, emb=emb)
    actual = wrapped(x, emb=emb)

    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        assert torch.allclose(actual_tensor, expected_tensor)


def test_patch_transformer_is_idempotent_for_flux(tmp_path):
    rank = 4
    source = make_tiny_flux_transformer()
    FluxAdapter().patch(
        source,
        {},
        {"rank": rank},
        SimpleNamespace(
            precision="int4",
            torch_dtype=torch.bfloat16,
            device=None,
            strict=True,
            adapter_options={},
        ),
    )
    checkpoint = tmp_path / "flux-lite.safetensors"
    save_file(source.state_dict(), checkpoint, metadata={"quantization_config": json.dumps({"rank": rank})})

    transformer = make_tiny_flux_transformer()
    first = patch_transformer(transformer, checkpoint, precision="int4", torch_dtype=torch.bfloat16)
    second = patch_transformer(transformer, checkpoint, precision="int4", torch_dtype=torch.bfloat16)

    assert first is second is transformer


def test_flux_attention_wraps_ip_adapter_processor():
    base = make_tiny_flux_transformer().transformer_blocks[0].attn
    ip_processor = FluxIPAdapterAttnProcessor(
        hidden_size=base.inner_dim,
        cross_attention_dim=8,
        num_tokens=(2,),
        scale=0.5,
        dtype=torch.bfloat16,
    )

    attn = NunchakuFluxAttention(base, processor=ip_processor, precision="int4", rank=4, torch_dtype=torch.bfloat16)

    assert isinstance(attn.processor, NunchakuFluxAttnProcessor)
    assert attn.processor.supports_ip_adapter
    assert len(attn.processor.to_k_ip) == 1
    assert len(attn.processor.to_v_ip) == 1
