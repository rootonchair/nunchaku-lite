"""CPU tests for data-free W4A16 text-encoder quantization (nunchaku_lite.text_encoders.w4a16)."""

import pytest
import torch
from torch import nn

from nunchaku_lite.linear import AWQW4A16Linear
from nunchaku_lite.ops.mps import _awq_g64_codes, _awq_g64_pack, dequantize_awq_g64_int32
from nunchaku_lite.text_encoders import w4a16


def test_pack_matches_the_reference_layout_and_roundtrips():
    torch.manual_seed(0)
    codes = torch.randint(0, 16, (128, 256), dtype=torch.int64)
    packed = w4a16.pack_awq_g64(codes)
    assert torch.equal(packed, _awq_g64_pack(codes))
    assert torch.equal(_awq_g64_codes(packed, 128), codes.to(torch.int32))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_quantized_weight_dequantizes_close_to_the_original(dtype):
    torch.manual_seed(0)
    weight = torch.randn(256, 512) * 0.02
    codes, scales, zeros = w4a16.quantize_weight_w4a16(weight, dtype=dtype)
    assert codes.dtype == torch.uint8 and int(codes.max()) <= 15
    assert scales.shape == zeros.shape == (512 // 64, 256) and scales.dtype == dtype
    deq = dequantize_awq_g64_int32(w4a16.pack_awq_g64(codes), scales, zeros)
    rel = ((deq - weight).norm() / weight.norm()).item()
    assert rel < 0.12, rel  # 4-bit, group 64, gaussian weights: ~0.08-0.1


def test_clip_search_never_does_worse_than_plain_min_max():
    torch.manual_seed(1)
    weight = torch.randn(128, 256) * 0.02
    weight[:, ::97] *= 20  # outliers make clipping worthwhile
    errs = {}
    for name, ratios in (("plain", (1.0,)), ("search", w4a16.CLIP_RATIOS)):
        codes, scales, zeros = w4a16.quantize_weight_w4a16(weight, clip_ratios=ratios)
        deq = dequantize_awq_g64_int32(w4a16.pack_awq_g64(codes), scales, zeros)
        errs[name] = (deq - weight).pow(2).sum().item()
    assert errs["search"] <= errs["plain"]


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(128, 256, bias=False)
        self.o_proj = nn.Linear(256, 128, bias=True)
        self.odd = nn.Linear(100, 128)  # in_features not a multiple of 64: stays dense


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([_Layer(), _Layer()])
        self.model.vision_tower = nn.Linear(128, 128)
        self.lm_head = nn.Linear(128, 512, bias=False)


def test_quantize_text_encoder_replaces_only_decoder_layer_linears():
    torch.manual_seed(0)
    model = _Model()
    ref = {
        n: (m.weight.detach().clone(), None if m.bias is None else m.bias.detach().clone())
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear)
    }
    summary = w4a16.quantize_text_encoder_w4a16_(model, device="cpu")
    assert summary["layers"] == 4
    layers = model.model.language_model.layers
    for layer, idx in zip(layers, range(2)):
        for proj in ("q_proj", "o_proj"):
            q = getattr(layer, proj)
            assert isinstance(q, AWQW4A16Linear)
            weight, bias = ref[f"model.language_model.layers.{idx}.{proj}"]
            deq = dequantize_awq_g64_int32(q.qweight, q.wscales, q.wzeros)
            assert ((deq - weight).norm() / weight.norm()) < 0.12
            if bias is not None:
                assert torch.equal(q.bias.float(), bias.to(torch.bfloat16).float())
        assert type(layer.odd) is nn.Linear
    assert type(model.model.vision_tower) is nn.Linear and type(model.lm_head) is nn.Linear
    assert 0 < summary["packed_bytes"] < summary["dense_bytes"] / 3


def test_quantize_text_encoder_respects_a_custom_include():
    model = _Model()
    summary = w4a16.quantize_text_encoder_w4a16_(model, device="cpu", include=lambda name, _m: name.endswith("q_proj"))
    assert summary["layers"] == 2
    assert type(model.model.language_model.layers[0].o_proj) is nn.Linear
