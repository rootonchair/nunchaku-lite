import pytest
import torch
from torch import nn

from nunchaku_lite.linear import AWQW4A16Linear, Int4WeightOnlyLinear, Int8WeightOnlyLinear
from nunchaku_lite.text_encoders.quantize import _fill, quantize_text_encoder


def _tiny_qwen3(tie_word_embeddings=True):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        tie_word_embeddings=tie_word_embeddings,
    )
    return Qwen3ForCausalLM(config).to(torch.bfloat16).eval()


def _hidden_states(model, device="cpu"):
    input_ids = torch.arange(1, 25, device=device).view(1, -1)
    with torch.no_grad():
        output = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    return output.hidden_states[-1].float().cpu()


def _rel_err(actual, expected):
    return ((actual - expected).norm() / expected.norm()).item()


@pytest.mark.parametrize(
    ("weights", "module_cls", "tolerance"),
    [("int4", Int4WeightOnlyLinear, 0.3), ("int8", Int8WeightOnlyLinear, 0.03)],
)
def test_quantize_text_encoder_replaces_linears_and_keeps_tied_lm_head(weights, module_cls, tolerance):
    model = _tiny_qwen3()
    expected = _hidden_states(model)

    quantize_text_encoder(model, weights, device="cpu")

    layer = model.model.layers[0]
    for linear in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.mlp.down_proj):
        assert type(linear) is module_cls
    assert type(model.lm_head) is nn.Linear
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert model._nunchaku_lite_text_encoder_quantization == weights
    assert _rel_err(_hidden_states(model), expected) < tolerance


def test_quantize_text_encoder_quantizes_untied_lm_head():
    model = _tiny_qwen3(tie_word_embeddings=False)

    quantize_text_encoder(model, "int8", device="cpu")

    assert type(model.lm_head) is Int8WeightOnlyLinear


def test_quantize_text_encoder_skips_excluded_and_ungroupable_layers():
    model = nn.Module()
    model.visual = nn.Linear(64, 64)
    model.proj = nn.Linear(64, 64)
    model.odd = nn.Linear(96, 64)  # 96 input channels do not split into groups of 64

    quantize_text_encoder(model, "int4", device="cpu")

    assert type(model.visual) is nn.Linear
    assert type(model.proj) is Int4WeightOnlyLinear
    assert type(model.odd) is nn.Linear


def test_quantize_text_encoder_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported text encoder quantization"):
        quantize_text_encoder(nn.Linear(64, 64), "int2", device="cpu")


@pytest.mark.parametrize("bias", [True, False])
def test_int4_awq_layout_matches_plain_layout(bias):
    from nunchaku_lite.ops.mps import dequantize_awq_g64_int32

    torch.manual_seed(0)
    linear = nn.Linear(256, 4104 + 8, bias=bias).to(torch.bfloat16)  # spans two row chunks
    plain = Int4WeightOnlyLinear(256, linear.out_features, bias=bias)
    awq = AWQW4A16Linear(256, linear.out_features, bias=bias, group_size=64).requires_grad_(False)

    _fill(plain, linear)
    _fill(awq, linear)

    dense = plain.dequantize(torch.float32)
    torch.testing.assert_close(dequantize_awq_g64_int32(awq.qweight, awq.wscales, awq.wzeros), dense)
    assert _rel_err(dense, linear.weight.float()) < 0.1


def test_int8_weight_only_linear_small_and_large_inputs_agree():
    torch.manual_seed(0)
    linear = nn.Linear(128, 64).to(torch.bfloat16)
    module = Int8WeightOnlyLinear(128, 64)
    _fill(module, linear)

    x = torch.randn(32, 128, dtype=torch.bfloat16)
    full = module(x)  # dequantized matmul
    rows = torch.cat([module(x[i : i + 4]) for i in range(0, 32, 4)])  # aten._weight_int8pack_mm
    torch.testing.assert_close(rows, full, rtol=0.02, atol=0.02)
    assert _rel_err(full.float(), linear(x).float()) < 0.02


def test_load_quantized_text_encoder_from_cpu_load(tmp_path):
    from nunchaku_lite import load_quantized_text_encoder

    reference = _tiny_qwen3()
    reference.save_pretrained(tmp_path / "text_encoder")

    model = load_quantized_text_encoder(tmp_path, "int8", subfolder="text_encoder", device="cpu")

    assert type(model).__name__ == "Qwen3ForCausalLM"
    assert type(model.model.layers[1].mlp.up_proj) is Int8WeightOnlyLinear
    assert getattr(model, "hf_device_map", None) is None
    assert _rel_err(_hidden_states(model), _hidden_states(reference)) < 0.02


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
@pytest.mark.parametrize("weights", ["int4", "int8"])
def test_quantized_text_encoder_on_mps_matches_cpu(weights):
    cpu = _tiny_qwen3()
    mps = _tiny_qwen3()
    quantize_text_encoder(cpu, weights, device="cpu")
    quantize_text_encoder(mps, weights, device="mps")

    assert _rel_err(_hidden_states(mps, "mps"), _hidden_states(cpu)) < 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("weights", ["int4", "int8"])
def test_quantized_text_encoder_on_cuda_matches_cpu(weights):
    cpu = _tiny_qwen3()
    cuda = _tiny_qwen3()
    quantize_text_encoder(cpu, weights, device="cpu")
    quantize_text_encoder(cuda, weights, device="cuda")

    expected_cls = AWQW4A16Linear if weights == "int4" else Int8WeightOnlyLinear
    assert type(cuda.model.layers[0].mlp.up_proj) is expected_cls
    assert _rel_err(_hidden_states(cuda, "cuda"), _hidden_states(cpu)) < 0.02


class _FakeTextEncoderPipeline:
    @classmethod
    def _get_signature_keys(cls, obj):
        return ["transformer", "text_encoder", "text_encoder_2", "tokenizer"], []

    @classmethod
    def load_config(cls, model_id, **kwargs):
        return {
            "transformer": ["test_core", "FakePipelineComponent"],
            "text_encoder": ["transformers", "Qwen3ForCausalLM"],
            "text_encoder_2": ["transformers", "T5EncoderModel"],
            "tokenizer": ["transformers", "Qwen2TokenizerFast"],
        }

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        pipe = cls()
        pipe.kwargs = kwargs
        return pipe


def test_load_nunchaku_pipeline_quantizes_text_encoders(tmp_path, monkeypatch):
    import test_core

    from nunchaku_lite import core
    from nunchaku_lite.text_encoders import quantize

    calls = {}

    def fake_load(path, weights, *, model_cls, subfolder, torch_dtype, device, **kwargs):
        calls[subfolder] = (weights, model_cls.__name__, torch_dtype, device, kwargs)
        return nn.Identity()

    test_core._install_fake_adapter(monkeypatch)
    monkeypatch.setattr(quantize, "load_quantized_text_encoder", fake_load)

    def load(**kwargs):
        calls.clear()
        return core.load_nunchaku_pipeline(
            tmp_path,
            pipeline_cls=_FakeTextEncoderPipeline,
            checkpoint=test_core._fake_checkpoint(tmp_path),
            target=test_core.FakeAdapter.target,
            device="cpu",
            **kwargs,
        )

    pipe = load(text_encoder_quantization="int4", torch_dtype=torch.float16, revision="main")
    assert calls == {
        "text_encoder": ("int4", "Qwen3ForCausalLM", torch.float16, "cpu", {"revision": "main"}),
        "text_encoder_2": ("int4", "T5EncoderModel", torch.float16, "cpu", {"revision": "main"}),
    }
    assert isinstance(pipe.kwargs["text_encoder"], nn.Identity)

    load(text_encoder_quantization={"text_encoder_2": "int8"})
    assert list(calls) == ["text_encoder_2"]
    assert calls["text_encoder_2"][2] is torch.bfloat16

    load(text_encoder_quantization="int8", text_encoder=nn.Identity())  # a passed-in encoder is left alone
    assert list(calls) == ["text_encoder_2"]

    with pytest.raises(ValueError, match="not a Transformers model"):
        load(text_encoder_quantization={"tokenizer": "int4"})
    with pytest.raises(ValueError, match="Unsupported text encoder quantization"):
        load(text_encoder_quantization="fp8")


@pytest.mark.parametrize("weights", ["int4", "int8"])
def test_quantize_text_encoder_runs_t5(weights):
    from transformers import T5Config, T5EncoderModel

    torch.manual_seed(0)
    config = T5Config(
        d_model=128, d_ff=256, num_layers=1, num_heads=4, d_kv=32, vocab_size=32, feed_forward_proj="gated-gelu"
    )
    model = T5EncoderModel(config).to(torch.bfloat16).eval()
    input_ids = torch.arange(1, 17).view(1, -1)
    with torch.no_grad():
        expected = model(input_ids=input_ids).last_hidden_state.float()

    quantize_text_encoder(model, weights, device="cpu")

    with torch.no_grad():
        actual = model(input_ids=input_ids).last_hidden_state.float()
    assert type(model.encoder.block[0].layer[1].DenseReluDense.wo) is not nn.Linear
    assert _rel_err(actual, expected) < (0.3 if weights == "int4" else 0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("rows", [5, 77])  # AWQ GEMV (< 16 rows) and GEMM paths
def test_int4_awq_kernel_matches_dequantized_weight_with_bias(rows):
    torch.manual_seed(0)
    linear = nn.Linear(512, 1024, bias=True).to(torch.bfloat16)
    model = nn.Sequential(linear)
    reference = Int4WeightOnlyLinear(512, 1024, bias=True)
    _fill(reference, linear)

    quantize_text_encoder(model, "int4", device="cuda")

    assert type(model[0]) is AWQW4A16Linear
    x = torch.randn(rows, 512, dtype=torch.bfloat16)
    expected = reference(x).float()
    actual = model(x.cuda()).float().cpu()
    assert _rel_err(actual, expected) < 0.01
