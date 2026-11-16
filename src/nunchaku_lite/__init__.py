"""Public package API for loading Diffusers pipelines with Nunchaku Lite.

The PyTorch runtime (``load_nunchaku_pipeline``, ``patch_transformer``, ...) needs the ``torch`` extra
(``pip install "nunchaku-lite[torch]"``); its names are imported lazily, so ``import nunchaku_lite.mlx`` (the MLX
runtime, ``pip install "nunchaku-lite[mlx]"``) works without torch installed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .__version__ import __version__

if TYPE_CHECKING:
    from .core import TransformerAdapter, list_adapters, load_nunchaku_pipeline, patch_transformer, register_adapter
    from .text_encoders import NunchakuT5EncoderModel

_LAZY = {
    "TransformerAdapter": ".core",
    "list_adapters": ".core",
    "load_nunchaku_pipeline": ".core",
    "patch_transformer": ".core",
    "register_adapter": ".core",
    "NunchakuT5EncoderModel": ".text_encoders",
}

__all__ = [
    "NunchakuT5EncoderModel",
    "TransformerAdapter",
    "__version__",
    "list_adapters",
    "load_nunchaku_pipeline",
    "patch_transformer",
    "register_adapter",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_name, __name__)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] in {"torch", "diffusers", "transformers", "accelerate", "peft"}:
            raise ModuleNotFoundError(
                f"nunchaku_lite.{name} needs the PyTorch runtime dependencies ({exc.name} is missing). "
                'Install them with: pip install "nunchaku-lite[torch]"',
                name=exc.name,
            ) from exc
        raise
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
