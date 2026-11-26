"""Load a Hugging Face text encoder with bitsandbytes NF4 or 8-bit weights.

Large text encoders, such as LTX-2.5's Gemma 4 12B (22.3 GiB in bf16), are often the biggest component left once the
transformer is quantized. This loads them with bitsandbytes' weight quantization, already placed on ``device``.

Usage::

    from transformers import AutoModelForImageTextToText
    from nunchaku_lite.text_encoders import load_text_encoder_bnb

    text_encoder = load_text_encoder_bnb(
        AutoModelForImageTextToText, "path/to/ltx-2.5-distilled-diffusers/text_encoder", quantization="nf4"
    )

Requires ``pip install bitsandbytes`` (or ``pip install "nunchaku-lite[bnb]"``). The model is placed at load time; do
not call ``.to()`` on it afterwards.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

QUANTIZATIONS = ("nf4", "int8")
# vision/audio towers and the LM head stay in the load dtype; only the text decoder layers are quantized
DEFAULT_SKIP_MODULES = ("vision_embedder", "embed_vision", "embed_audio", "vision_tower", "audio_tower", "lm_head")


def bnb_quantization_config(
    quantization: str = "nf4",
    skip_modules: Sequence[str] | None = DEFAULT_SKIP_MODULES,
    compute_dtype: torch.dtype = torch.bfloat16,
):
    """``transformers.BitsAndBytesConfig`` for ``"nf4"`` (4-bit NF4) or ``"int8"`` (LLM.int8)."""
    from transformers import BitsAndBytesConfig

    skip = list(skip_modules) if skip_modules else None
    if quantization == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            llm_int8_skip_modules=skip,
        )
    if quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
    raise ValueError(f"quantization must be one of {QUANTIZATIONS}, got {quantization!r}")


def load_text_encoder_bnb(
    model_cls: Any,
    pretrained_model_name_or_path: str | Path,
    quantization: str = "nf4",
    device: str | torch.device = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    skip_modules: Sequence[str] | None = DEFAULT_SKIP_MODULES,
    **kwargs: Any,
):
    """Load ``model_cls.from_pretrained(...)`` with bitsandbytes ``"nf4"`` or ``"int8"`` weights on ``device``.

    Args:
        model_cls: Transformers model class or auto class, e.g. ``AutoModelForImageTextToText`` (LTX-2.5) or
            ``T5EncoderModel``.
        pretrained_model_name_or_path: Local folder or Hub repo id, as for ``from_pretrained``.
        quantization: ``"nf4"`` (smallest) or ``"int8"`` (closest to the original).
        device: CUDA device to place the model on.
        torch_dtype: Dtype of the modules that stay unquantized, and the NF4 compute dtype.
        skip_modules: Module names left unquantized.
        **kwargs: Passed to ``from_pretrained``.
    """
    config = bnb_quantization_config(quantization, skip_modules, torch_dtype)
    if importlib.util.find_spec("bitsandbytes") is None:
        raise ImportError(
            'bitsandbytes is not installed: pip install bitsandbytes (or pip install "nunchaku-lite[bnb]")'
        )
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"bitsandbytes text encoders need a CUDA device, got {device}")
    for key in ("quantization_config", "device_map"):
        if key in kwargs:
            raise TypeError(f"{key} is set by load_text_encoder_bnb")
    return model_cls.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        quantization_config=config,
        device_map={"": device},
        **kwargs,
    )
