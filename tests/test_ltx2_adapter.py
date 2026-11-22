import inspect
import json
import types
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from diffusers import LTX2VideoTransformer3DModel
from safetensors.torch import save_file
from torch import nn

import nunchaku_lite.adapters.ltx2 as ltx2_adapter
from nunchaku_lite import list_adapters, patch_transformer
from nunchaku_lite.adapters.ltx2 import (
    LTX2Adapter,
    NunchakuLTX2AudioVideoAttnProcessor,
    NunchakuLTX2PerturbedAttnProcessor,
    _ltx2_block_forward,
)
from nunchaku_lite.linear import AWQW4A16Linear, SVDQW4A4Linear


def make_tiny_ltx2_transformer(**kwargs):
    config = {
        "in_channels": 4,
        "out_channels": 4,
        "audio_in_channels": 4,
        "audio_out_channels": 4,
        "num_attention_heads": 1,
        "attention_head_dim": 64,
        "audio_num_attention_heads": 1,
        "audio_attention_head_dim": 64,
        "cross_attention_dim": 64,
        "audio_cross_attention_dim": 64,
        "num_layers": 1,
        "caption_channels": 64,
        "norm_elementwise_affine": False,
        "gated_attn": True,
        "audio_gated_attn": True,
    }
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
        to_out: ClassVar = [nn.Identity(), nn.Identity()]

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
        to_out: ClassVar = [nn.Identity(), nn.Identity()]

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
        to_out: ClassVar = [nn.Identity(), nn.Identity()]

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


# --- LTX-2.5 compatibility and INT4 fused-MLP routing --------------------------------------

_LTX25_SUPPORTED = {"ff_bias", "use_prompt_adaln_single", "use_keyframes_abs_pos_embedding"} <= set(
    inspect.signature(LTX2VideoTransformer3DModel.__init__).parameters
)
requires_ltx25_diffusers = pytest.mark.skipif(not _LTX25_SUPPORTED, reason="needs a diffusers version with LTX-2.5")


def _ltx25_style_config(use_prompt_adaln_single):
    # Small version of the LTX-2.5 transformer: prompt cross-attention modulation, no FF biases,
    # keyframe embedding, with or without the cacheable (static-table) prompt modulation.
    return {
        "in_channels": 4,
        "out_channels": 4,
        "patch_size": 1,
        "patch_size_t": 1,
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "cross_attention_dim": 16,
        "audio_in_channels": 4,
        "audio_out_channels": 4,
        "audio_num_attention_heads": 2,
        "audio_attention_head_dim": 4,
        "audio_cross_attention_dim": 8,
        "num_layers": 2,
        "qk_norm": "rms_norm_across_heads",
        "caption_channels": 16,
        "rope_double_precision": False,
        "cross_attn_mod": True,
        "audio_cross_attn_mod": True,
        "ff_bias": False,
        "audio_ff_bias": False,
        "use_prompt_adaln_single": use_prompt_adaln_single,
        "use_keyframes_abs_pos_embedding": True,
    }


def _ltx2_dummy_inputs(generator):
    batch, frames, height, width, audio_frames, seq = 2, 2, 16, 16, 9, 16
    return {
        "hidden_states": torch.randn((batch, frames * height * width, 4), generator=generator),
        "audio_hidden_states": torch.randn((batch, audio_frames, 4), generator=generator),
        "encoder_hidden_states": torch.randn((batch, seq, 16), generator=generator),
        "audio_encoder_hidden_states": torch.randn((batch, seq, 16), generator=generator),
        "timestep": torch.rand((batch,), generator=generator) * 1000,
        "sigma": torch.rand((batch,), generator=generator),
        "audio_sigma": torch.rand((batch,), generator=generator),
        "encoder_attention_mask": torch.ones((batch, seq)).bool(),
        "num_frames": frames,
        "height": height,
        "width": width,
        "audio_num_frames": audio_frames,
        "fps": 25.0,
    }


@requires_ltx25_diffusers
@pytest.mark.parametrize("use_prompt_adaln_single", [True, False])
def test_lite_block_forward_matches_diffusers_for_ltx25_config(use_prompt_adaln_single):
    # The adapter replaces the block forward with its own copy; it must match diffusers' block,
    # including the static prompt modulation used when `use_prompt_adaln_single=False`.
    torch.manual_seed(0)
    model = LTX2VideoTransformer3DModel(**_ltx25_style_config(use_prompt_adaln_single)).eval()
    with torch.no_grad():
        for table in ("prompt_scale_shift_table", "audio_prompt_scale_shift_table"):
            for block in model.transformer_blocks:
                getattr(block, table).normal_()  # zero-init by default; make the static table matter
    inputs = _ltx2_dummy_inputs(torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = model(**inputs, return_dict=False)
        for block in model.transformer_blocks:
            block.forward = types.MethodType(_ltx2_block_forward, block)
        actual = model(**inputs, return_dict=False)
    for exp, act in zip(expected, actual):
        torch.testing.assert_close(act, exp, rtol=1e-5, atol=1e-5)


@requires_ltx25_diffusers
def test_ltx25_bias_free_feed_forward_patches_and_loads(tmp_path):
    rank = 4
    source = make_tiny_ltx2_transformer(ff_bias=False, audio_ff_bias=False)
    LTX2Adapter().patch(source, {}, {"rank": rank}, _patch_options(rank))
    assert source.transformer_blocks[0].ff.net[0].proj.bias is None
    assert source.transformer_blocks[0].ff.net[2].bias is None
    checkpoint = tmp_path / "ltx25-lite.safetensors"
    save_file(source.state_dict(), checkpoint, metadata={"quantization_config": json.dumps({"rank": rank})})

    transformer = make_tiny_ltx2_transformer(ff_bias=False, audio_ff_bias=False)
    patch_transformer(transformer, checkpoint, target="ltx2", precision="int4", torch_dtype=torch.bfloat16)
    incompatible = transformer._nunchaku_lite_incompatible_keys
    assert not incompatible.missing_keys and not incompatible.unexpected_keys


@pytest.mark.parametrize("precision,fused", [("nvfp4", True), ("int4", False)])
def test_ltx2_feed_forward_uses_fused_mlp_only_for_nvfp4(precision, fused):
    # INT4 LTX exports are unshifted; the fused GELU epilogue assumes a +0.171875 shift folded into
    # fc2's bias, so INT4 must run unfused. NVFP4 has no shift and keeps the fused kernel.
    from diffusers.models.attention import FeedForward

    from nunchaku_lite.adapters.common import SVDQPatchContext
    from nunchaku_lite.adapters.ltx2 import _patch_ltx2_feed_forward, _use_fused_gelu_mlp

    context = SVDQPatchContext(precision=precision, rank=16, torch_dtype=torch.bfloat16)
    ff = _patch_ltx2_feed_forward(FeedForward(64, activation_fn="gelu-approximate"), context)
    cuda_like = SimpleNamespace(is_cuda=True)
    with torch.no_grad():
        assert _use_fused_gelu_mlp(ff, cuda_like, (), {}) is fused
    assert not _use_fused_gelu_mlp(ff, SimpleNamespace(is_cuda=False), (), {})
