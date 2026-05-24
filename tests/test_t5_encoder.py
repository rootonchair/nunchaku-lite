import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import T5Config, T5EncoderModel


def _tiny_t5_config() -> T5Config:
    return T5Config(
        d_model=128,
        d_ff=256,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        d_kv=32,
        vocab_size=32,
    )


def _quantized_t5_checkpoint(tmp_path):
    config = _tiny_t5_config()
    dense = T5EncoderModel(config).to(torch.bfloat16)
    state = {key: value.detach().clone() for key, value in dense.state_dict().items()}
    target = "encoder.block.0.layer.0.SelfAttention.q"
    state.pop(f"{target}.weight")
    state[f"{target}.qweight"] = torch.zeros(32, 128, dtype=torch.int16)
    state[f"{target}.scales"] = torch.ones(8, 128, dtype=torch.bfloat16)
    state[f"{target}.scaled_zeros"] = torch.zeros(8, 128, dtype=torch.bfloat16)

    checkpoint = tmp_path / "t5-awq.safetensors"
    save_file(state, checkpoint, metadata={"config": json.dumps(config.to_dict())})
    return checkpoint


def test_nunchaku_t5_encoder_imports_from_top_level():
    from nunchaku_lite import NunchakuT5EncoderModel

    assert NunchakuT5EncoderModel.__name__ == "NunchakuT5EncoderModel"


def test_nunchaku_t5_encoder_replaces_quantized_linears(tmp_path):
    from nunchaku_lite import NunchakuT5EncoderModel
    from nunchaku_lite.linear import TinyChatAWQW4A16Linear

    model = NunchakuT5EncoderModel.from_pretrained(
        _quantized_t5_checkpoint(tmp_path),
        torch_dtype=torch.bfloat16,
        device="cpu",
    )

    quantized = model.encoder.block[0].layer[0].SelfAttention.q
    dense = model.encoder.block[0].layer[0].SelfAttention.k
    assert isinstance(quantized, TinyChatAWQW4A16Linear)
    assert isinstance(dense, torch.nn.Linear)
    assert quantized.qweight.shape == (32, 128)
    assert quantized.scales.shape == (8, 128)
    assert quantized.weight.dtype is torch.bfloat16


def test_nunchaku_t5_encoder_rejects_missing_config_metadata(tmp_path):
    from nunchaku_lite import NunchakuT5EncoderModel

    checkpoint = tmp_path / "missing-config.safetensors"
    save_file({}, checkpoint)

    with pytest.raises(ValueError, match="metadata must include"):
        NunchakuT5EncoderModel.from_pretrained(checkpoint, device="cpu")


def test_nunchaku_t5_encoder_rejects_unsupported_model_type(tmp_path):
    from nunchaku_lite import NunchakuT5EncoderModel

    checkpoint = tmp_path / "qwen.safetensors"
    save_file({}, checkpoint, metadata={"config": json.dumps({"model_type": "qwen3"})})

    with pytest.raises(ValueError, match="Unsupported quantized encoder checkpoint model_type='qwen3'"):
        NunchakuT5EncoderModel.from_pretrained(checkpoint, device="cpu")
