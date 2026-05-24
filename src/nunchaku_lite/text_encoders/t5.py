"""Quantized T5 encoder support compatible with original Nunchaku checkpoints."""

from pathlib import Path

import torch
from accelerate import init_empty_weights
from torch import nn
from transformers import T5Config, T5EncoderModel

from ..adapters.common import patch_modules_recursively
from ..linear import TinyChatAWQW4A16Linear
from ..utils import load_state_dict_in_safetensors, parse_config_metadata


class NunchakuT5EncoderModel(T5EncoderModel):
    """T5 encoder loader for original Nunchaku AWQ INT4 checkpoints."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs):
        """Load a quantized T5 encoder from a Nunchaku safetensors checkpoint."""

        state_dict, metadata = load_state_dict_in_safetensors(pretrained_model_name_or_path, return_metadata=True)
        config_dict = parse_config_metadata(metadata)
        model_type = config_dict.get("model_type")
        if model_type not in (None, "t5"):
            raise ValueError(
                f"Unsupported quantized encoder checkpoint model_type={model_type!r}; "
                "only T5 checkpoints are supported by NunchakuT5EncoderModel."
            )
        if not config_dict:
            raise ValueError("Quantized T5 checkpoint metadata must include a JSON 'config' object.")

        config = T5Config(**config_dict)
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        with init_empty_weights():
            t5_encoder = cls(config).to(torch_dtype)
        t5_encoder.eval()

        patch_modules_recursively(
            t5_encoder,
            skips=lambda path, module: isinstance(module, nn.Linear) and f"{path}.qweight" not in state_dict,
            module_converters={nn.Linear: _quantized_t5_linear_from_linear},
        )

        device = kwargs.get("device", "cuda")
        if isinstance(device, str):
            device = torch.device(device)
        t5_encoder.to_empty(device=device)
        t5_encoder.load_state_dict(state_dict, strict=True)
        return t5_encoder


def _quantized_t5_linear_from_linear(_path: str, module: nn.Module) -> TinyChatAWQW4A16Linear:
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {module.__class__.__name__}.")
    quantized = TinyChatAWQW4A16Linear.from_linear(module, group_size=128)
    # Hugging Face T5DenseGatedActDense checks ``weight.dtype``.
    quantized.weight = torch.empty(1, dtype=module.weight.dtype, device=module.weight.device)
    return quantized


__all__ = ["NunchakuT5EncoderModel"]
