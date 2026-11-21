"""Public package API for loading Diffusers pipelines with Nunchaku Lite.

The PyTorch runtime (``load_nunchaku_pipeline``, ``patch_transformer``, submodules such as ``nunchaku_lite.linear``)
is imported lazily, on first use, so ``import nunchaku_lite.mlx`` (the MLX runtime) never loads torch.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING, Any

from .__version__ import __version__

if TYPE_CHECKING:
    from .core import TransformerAdapter, list_adapters, load_nunchaku_pipeline, patch_transformer, register_adapter
    from .text_encoders import NunchakuT5EncoderModel, load_quantized_text_encoder, quantize_text_encoder

_LAZY = {
    "TransformerAdapter": ".core",
    "list_adapters": ".core",
    "load_nunchaku_pipeline": ".core",
    "patch_transformer": ".core",
    "register_adapter": ".core",
    "NunchakuT5EncoderModel": ".text_encoders",
    "load_quantized_text_encoder": ".text_encoders",
    "quantize_text_encoder": ".text_encoders",
}

__all__ = [
    "NunchakuT5EncoderModel",
    "TransformerAdapter",
    "__version__",
    "list_adapters",
    "load_nunchaku_pipeline",
    "load_quantized_text_encoder",
    "patch_transformer",
    "quantize_text_encoder",
    "register_adapter",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        # Submodules (``nunchaku_lite.linear`` after ``import nunchaku_lite``), as when this file imported .core eagerly.
        if name.startswith("_") or importlib.util.find_spec(f"{__name__}.{name}") is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        return _import(f".{name}", name)
    value = getattr(_import(module_name, name), name)
    globals()[name] = value
    return value


def _import(module_name: str, name: str) -> Any:
    try:
        return importlib.import_module(module_name, __name__)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] in {"torch", "diffusers", "transformers", "accelerate", "peft"}:
            raise ModuleNotFoundError(
                f"nunchaku_lite.{name} needs the PyTorch runtime dependencies ({exc.name} is missing). "
                "Install them with: pip install nunchaku-lite",
                name=exc.name,
            ) from exc
        raise


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
