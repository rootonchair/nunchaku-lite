import json
from types import SimpleNamespace

import torch
import torch.nn as nn
from diffusers import LTX2VideoTransformer3DModel
from safetensors.torch import save_file

from nunchaku_lite import list_adapters, patch_transformer
from nunchaku_lite.adapters.ltx2 import (
    LTX2Adapter,
    NunchakuLTX2AudioVideoAttnProcessor,
    NunchakuLTX2PerturbedAttnProcessor,
)
from nunchaku_lite.linear import AWQW4A16Linear, SVDQW4A4Linear


def make_tiny_ltx2_transformer(**kwargs):
    config = dict(
        in_channels=4,
        out_channels=4,
        audio_in_channels=4,
        audio_out_channels=4,
        num_attention_heads=1,
        attention_head_dim=64,
        audio_num_attention_heads=1,
        audio_attention_head_dim=64,
        cross_attention_dim=64,
        audio_cross_attention_dim=64,
        num_layers=1,
        caption_channels=64,
        norm_elementwise_affine=False,
        gated_attn=True,
        audio_gated_attn=True,
    )
    config.update(kwargs)
    return LTX2VideoTransformer3DModel(**config)


def _patch_options(rank: int = 4):
    return SimpleNamespace(
        precision="int4",
        torch_dtype=torch.bfloat16,
        device=None,
        strict=True,
        adapter_options={"rank": rank},
    )


def test_ltx2_adapter_is_explicit_builtin_target():
    transformer = make_tiny_ltx2_transformer()

    assert LTX2Adapter().matches(transformer)
    assert "ltx2" in list_adapters()


def test_ltx2_patch_keeps_separate_qkv_and_dense_gate_logits():
    transformer = make_tiny_ltx2_transformer(perturbed_attn=False)

    LTX2Adapter().patch(transformer, {}, {"rank": 4}, _patch_options())
    attn = transformer.transformer_blocks[0].attn1

    assert isinstance(attn.to_q, SVDQW4A4Linear)
    assert isinstance(attn.to_k, SVDQW4A4Linear)
    assert isinstance(attn.to_v, SVDQW4A4Linear)
    assert isinstance(attn.to_out[0], SVDQW4A4Linear)
    assert not hasattr(attn, "to_qkv")
    assert isinstance(attn.to_gate_logits, nn.Linear)
    assert isinstance(attn.processor, NunchakuLTX2AudioVideoAttnProcessor)
    assert isinstance(transformer.transformer_blocks[0].ff.net[0].proj, SVDQW4A4Linear)
    assert isinstance(transformer.transformer_blocks[0].audio_ff.net[0].proj, SVDQW4A4Linear)
    assert not hasattr(transformer.transformer_blocks[0], "_nunchaku_lite_ltx2_original_forward")


def test_ltx2_perturbed_attention_processor_is_preserved():
    transformer = make_tiny_ltx2_transformer(perturbed_attn=True)

    LTX2Adapter().patch(transformer, {}, {"rank": 4}, _patch_options())

    assert isinstance(transformer.transformer_blocks[0].attn1.processor, NunchakuLTX2PerturbedAttnProcessor)
    assert isinstance(transformer.transformer_blocks[0].audio_attn1.processor, NunchakuLTX2PerturbedAttnProcessor)


def test_ltx2_adapter_patches_checkpoint_declared_awq_adaln_linears():
    transformer = make_tiny_ltx2_transformer()
    checkpoint_state = {"time_embed.linear.qweight": torch.empty(1, dtype=torch.int32)}
    quantization_config = {
        "rank": 4,
        "runtime_manifest": {
            "targets": [
                {
                    "checkpoint_prefix": "time_embed.linear",
                    "nunchaku_op": "awq_w4a16",
                    "group_size": 64,
                }
            ]
        },
    }

    LTX2Adapter().patch(transformer, checkpoint_state, quantization_config, _patch_options())

    assert isinstance(transformer.time_embed.linear, AWQW4A16Linear)


def test_ltx2_adapter_patches_checkpoint_declared_awq_gate_logits():
    transformer = make_tiny_ltx2_transformer(num_attention_heads=4, attention_head_dim=16)
    prefix = "transformer_blocks.0.attn1.to_gate_logits"
    checkpoint_state = {f"{prefix}.qweight": torch.empty(1, dtype=torch.int32)}
    quantization_config = {
        "rank": 4,
        "runtime_manifest": {
            "targets": [
                {
                    "checkpoint_prefix": prefix,
                    "nunchaku_op": "awq_w4a16",
                    "group_size": 64,
                }
            ]
        },
    }

    LTX2Adapter().patch(transformer, checkpoint_state, quantization_config, _patch_options())

    assert isinstance(transformer.transformer_blocks[0].attn1.to_gate_logits, AWQW4A16Linear)


def test_patch_transformer_loads_synthetic_ltx2_checkpoint_with_separate_keys(tmp_path):
    rank = 4
    source = make_tiny_ltx2_transformer()
    LTX2Adapter().patch(source, {}, {"rank": rank}, _patch_options(rank))
    state = source.state_dict()
    checkpoint = tmp_path / "ltx2-lite.safetensors"
    save_file(state, checkpoint, metadata={"quantization_config": json.dumps({"rank": rank})})

    transformer = make_tiny_ltx2_transformer()
    returned = patch_transformer(transformer, checkpoint, target="ltx2", precision="int4", torch_dtype=torch.bfloat16)

    assert returned is transformer
    assert transformer._nunchaku_lite_patched
    assert transformer._nunchaku_lite_target == "ltx2"
    keys = transformer.state_dict().keys()
    assert "transformer_blocks.0.attn1.to_q.qweight" in keys
    assert "transformer_blocks.0.attn1.to_k.qweight" in keys
    assert "transformer_blocks.0.attn1.to_v.qweight" in keys
    assert "transformer_blocks.0.attn1.to_gate_logits.weight" in keys
    assert "transformer_blocks.0.attn1.to_qkv.qweight" not in keys
