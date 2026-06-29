import json
from types import SimpleNamespace

import torch
import torch.nn as nn
from diffusers import LTX2VideoTransformer3DModel
from safetensors.torch import save_file

from nunchaku_lite import list_adapters, patch_transformer
import nunchaku_lite.adapters.ltx2 as ltx2_adapter
from nunchaku_lite.adapters.ltx2 import (
    LTX2Adapter,
    NunchakuLTX2AudioVideoAttnProcessor,
    NunchakuLTX2PerturbedAttnProcessor,
    _ltx2_block_forward,
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
    assert hasattr(transformer.transformer_blocks[0], "_nunchaku_lite_ltx2_original_forward")
    assert transformer.transformer_blocks[0].forward.__func__ is _ltx2_block_forward


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


def test_ltx2_processor_applies_fused_qk_norm_rope_when_rotary_present(monkeypatch):
    class FakeAttention:
        heads = 2
        head_dim = 4
        rope_type = "split"
        to_gate_logits = None
        to_q = nn.Identity()
        to_k = nn.Identity()
        to_v = nn.Identity()
        norm_q = nn.Identity()
        norm_k = nn.Identity()
        to_out = [nn.Identity(), nn.Identity()]

    calls = []

    def fake_apply_fused(query, key, *args):
        calls.append((query, key, args))
        return query + 1, key + 2

    def fake_dispatch_attention_fn(query, key, value, **kwargs):
        assert torch.equal(query.flatten(2, 3), calls[0][0] + 1)
        assert torch.equal(key.flatten(2, 3), calls[0][1] + 2)
        return torch.zeros_like(value)

    monkeypatch.setattr(ltx2_adapter, "_apply_fused_cross_head_qk_norm_rope", fake_apply_fused)
    monkeypatch.setattr(ltx2_adapter, "dispatch_lite_attention_fn", fake_dispatch_attention_fn)

    processor = NunchakuLTX2AudioVideoAttnProcessor()
    hidden_states = torch.randn(1, 3, 8)
    rope = (torch.randn(1, 2, 3, 2), torch.randn(1, 2, 3, 2))

    output = processor(FakeAttention(), hidden_states, query_rotary_emb=rope)

    assert output.shape == hidden_states.shape
    assert len(calls) == 1


def test_ltx2_perturbed_attention_all_perturbed_skips_attention(monkeypatch):
    class FakeAttention:
        heads = 2
        head_dim = 4
        rope_type = "split"
        to_gate_logits = None
        to_q = nn.Identity()
        to_k = nn.Identity()
        to_v = nn.Identity()
        norm_q = nn.Identity()
        norm_k = nn.Identity()
        to_out = [nn.Identity(), nn.Identity()]

    def fail_dispatch(*args, **kwargs):
        raise AssertionError("attention should be skipped when all samples are perturbed")

    monkeypatch.setattr(ltx2_adapter, "dispatch_lite_attention_fn", fail_dispatch)
    processor = NunchakuLTX2PerturbedAttnProcessor()
    hidden_states = torch.randn(1, 3, 8)

    output = processor(FakeAttention(), hidden_states, all_perturbed=True)

    torch.testing.assert_close(output, hidden_states)


def test_ltx2_perturbed_attention_mask_lerps_value_and_attention(monkeypatch):
    class FakeAttention:
        heads = 2
        head_dim = 4
        rope_type = "split"
        to_gate_logits = None
        to_q = nn.Identity()
        to_k = nn.Identity()
        to_v = nn.Identity()
        norm_q = nn.Identity()
        norm_k = nn.Identity()
        to_out = [nn.Identity(), nn.Identity()]

    hidden_states = torch.randn(1, 3, 8)
    attended = torch.randn(1, 3, 8)
    perturbation_mask = torch.full((1, 3, 1), 0.25)

    def fake_dispatch_attention_fn(*args, **kwargs):
        return attended.unflatten(2, (2, 4))

    monkeypatch.setattr(ltx2_adapter, "dispatch_lite_attention_fn", fake_dispatch_attention_fn)
    processor = NunchakuLTX2PerturbedAttnProcessor()

    output = processor(FakeAttention(), hidden_states, perturbation_mask=perturbation_mask)

    torch.testing.assert_close(output, torch.lerp(hidden_states, attended, perturbation_mask))


def test_ltx2_residual_gate_torch_path_broadcasts_batch_channel_gate():
    residual = torch.randn(2, 3, 4)
    branch = torch.randn(2, 3, 4)
    gate = torch.randn(2, 4)

    output = ltx2_adapter._residual_gate(residual, gate, branch)

    torch.testing.assert_close(output, residual + gate.unsqueeze(1) * branch)
