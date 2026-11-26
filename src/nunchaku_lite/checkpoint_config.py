"""Checkpoint-level rules shared by every loading path (Diffusers ``patch_transformer`` and ``nunchaku_lite.loader``).

Torch and the Hub client only: no Diffusers or framework imports, so framework integrations can use it.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import torch

from .utils import fetch_or_download

__all__ = [
    "check_device_precision",
    "parse_json_metadata",
    "resolve_quantization_config",
    "sibling_quantization_config",
]

_PRE_BLACKWELL = {75, 80, 86, 89}  # Turing, Ampere, Ada: INT4 only


def parse_json_metadata(metadata: dict[str, str] | None, key: str) -> dict[str, Any]:
    """Decode a JSON object from safetensors metadata; a missing key gives ``{}``.

    Raises:
        ValueError: If the value is not valid JSON.
        TypeError: If it does not decode to an object.
    """

    if not metadata:
        return {}
    raw = metadata.get(key, "{}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in checkpoint metadata field {key!r}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise TypeError(f"Checkpoint metadata field {key!r} must decode to a JSON object.")
    return parsed


def sibling_quantization_config(checkpoint: str | os.PathLike) -> dict[str, Any]:
    """``quantization_config`` from the ``config.json`` next to a checkpoint file (Diffusers-format repos that keep it
    there instead of in the safetensors metadata); ``{}`` when there is none."""

    from huggingface_hub.errors import EntryNotFoundError

    sibling = Path(checkpoint).parent / "config.json"
    try:
        if Path(checkpoint).exists():  # local file: never fall through to a Hub download
            if not sibling.is_file():
                return {}
            config_path = sibling
        else:  # Hub reference ``org/repo/.../file.safetensors``
            config_path = fetch_or_download(sibling)
        value = json.loads(Path(config_path).read_text()).get("quantization_config")
    except json.JSONDecodeError as exc:
        warnings.warn(f"Ignoring malformed {sibling} ({exc}).", stacklevel=3)
        return {}
    except (EntryNotFoundError, FileNotFoundError, ValueError):  # no config.json (ValueError: not a Hub path)
        return {}
    except Exception as exc:  # noqa: BLE001 - e.g. network or auth errors: continue without, but say so
        warnings.warn(f"Could not read {sibling} for checkpoint metadata ({exc}); continuing without it.", stacklevel=3)
        return {}
    return value if isinstance(value, dict) else {}


def resolve_quantization_config(metadata: dict[str, str] | None, checkpoint: str | os.PathLike) -> dict[str, Any]:
    """The checkpoint's ``quantization_config``: from its metadata, else from a sibling ``config.json``."""

    return parse_json_metadata(metadata, "quantization_config") or sibling_quantization_config(checkpoint)


def check_device_precision(precision: str, device: str | torch.device | None, *, stacklevel: int = 2) -> None:
    """Refuse a checkpoint precision this CUDA device cannot run; warn when INT4 runs on Blackwell.

    No-op without CUDA or for a non-CUDA ``device``.
    """

    if not torch.cuda.is_available():
        return
    target_device = torch.device(device or "cuda")
    if target_device.type != "cuda":
        return
    device_index = 0 if target_device.index is None else target_device.index
    major, minor = torch.cuda.get_device_capability(device_index)
    sm = major * 10 + minor
    if sm >= 100:
        if precision != "nvfp4":
            warnings.warn(
                "INT4 quantization on Blackwell GPUs may be slower than FP4. "
                "Use an FP4 checkpoint for best performance when one is available.",
                UserWarning,
                stacklevel=stacklevel + 1,
            )
        return
    if sm in _PRE_BLACKWELL:
        if precision != "int4":
            raise ValueError('Please use "int4" quantization for Turing, Ampere, and Ada GPUs.')
        return
    raise ValueError(f"Unsupported GPU architecture sm{sm}; Nunchaku Lite requires Turing, Ampere, Ada, or Blackwell.")
