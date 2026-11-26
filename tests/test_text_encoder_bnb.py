from typing import ClassVar

import pytest
import torch

from nunchaku_lite.text_encoders import bnb as bnb_module
from nunchaku_lite.text_encoders import bnb_quantization_config, load_text_encoder_bnb


class FakeModel:
    calls: ClassVar[list] = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return cls()


@pytest.fixture(autouse=True)
def _reset():
    FakeModel.calls.clear()


@pytest.fixture
def bitsandbytes_present(monkeypatch):
    real = bnb_module.importlib.util.find_spec
    monkeypatch.setattr(
        bnb_module.importlib.util, "find_spec", lambda name, *a: object() if name == "bitsandbytes" else real(name, *a)
    )


def test_nf4_config():
    config = bnb_quantization_config("nf4")
    assert config.load_in_4bit and not config.load_in_8bit
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_compute_dtype == torch.bfloat16
    assert "lm_head" in config.llm_int8_skip_modules and "embed_vision" in config.llm_int8_skip_modules


def test_int8_config_and_custom_skip():
    config = bnb_quantization_config("int8", skip_modules=["foo"])
    assert config.load_in_8bit and not config.load_in_4bit
    assert config.llm_int8_skip_modules == ["foo"]


def test_unknown_quantization_is_refused():
    with pytest.raises(ValueError, match="nf4"):
        bnb_quantization_config("int4")


def test_load_passes_config_and_device_map(bitsandbytes_present):
    load_text_encoder_bnb(FakeModel, "some/path", quantization="int8", device="cuda:1", revision="abc")
    ((path, kwargs),) = FakeModel.calls
    assert path == "some/path"
    assert kwargs["quantization_config"].load_in_8bit
    assert kwargs["device_map"] == {"": torch.device("cuda:1")}
    assert kwargs["torch_dtype"] == torch.bfloat16 and kwargs["revision"] == "abc"


def test_missing_bitsandbytes_has_install_hint(monkeypatch):
    monkeypatch.setattr(bnb_module.importlib.util, "find_spec", lambda name, *a: None)
    with pytest.raises(ImportError, match="pip install bitsandbytes"):
        load_text_encoder_bnb(FakeModel, "p")
    assert FakeModel.calls == []


def test_non_cuda_device_and_reserved_kwargs_are_refused(bitsandbytes_present):
    with pytest.raises(ValueError, match="CUDA"):
        load_text_encoder_bnb(FakeModel, "p", device="cpu")
    with pytest.raises(TypeError, match="device_map"):
        load_text_encoder_bnb(FakeModel, "p", device_map="auto")
    assert FakeModel.calls == []
