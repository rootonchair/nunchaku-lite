"""Shared helpers for ComfyUI integration tests.

ComfyUI is not pip-installable; tests that need it import it from a checkout named by
``NUNCHAKU_LITE_COMFYUI_PATH`` and are skipped otherwise.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch


def require_comfyui():
    path = os.environ.get("NUNCHAKU_LITE_COMFYUI_PATH")
    if not path:
        pytest.skip("set NUNCHAKU_LITE_COMFYUI_PATH to a ComfyUI checkout to run ComfyUI tests")
    if path not in sys.path:
        sys.path.insert(0, path)
    pytest.importorskip("comfy.model_detection")


def kernel_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    pytest.skip("SVDQ kernels need CUDA or Apple MPS")


def fill_random_quantized_state(
    state: dict[str, torch.Tensor], *, precision: str = "int4", seed: int = 0
) -> dict[str, torch.Tensor]:
    """Replace the (empty) tensors of a patched-model state dict with random, well-scaled values."""

    g = torch.Generator().manual_seed(seed)
    out = {}
    awq_prefixes = {key[: -len(".wzeros")] for key in state if key.endswith(".wzeros")}
    for key, value in state.items():
        prefix, _, suffix = key.rpartition(".")
        if prefix in awq_prefixes:
            if suffix == "qweight":
                out[key] = torch.randint(-(2**31), 2**31 - 1, value.shape, generator=g, dtype=torch.int64).to(
                    torch.int32
                )
            elif suffix == "wscales":
                out[key] = (torch.rand(value.shape, generator=g) * 0.002 + 0.001).to(value.dtype)
            elif suffix == "wzeros":
                out[key] = (torch.randn(value.shape, generator=g) * 0.01).to(value.dtype)
            elif suffix == "bias":
                out[key] = (torch.randn(value.shape, generator=g) * 0.1).to(value.dtype)
            else:
                raise AssertionError(f"unexpected AWQ tensor {key}")
            continue
        if suffix == "qweight":
            out[key] = torch.randint(-128, 128, value.shape, generator=g, dtype=torch.int16).to(torch.int8)
        elif suffix == "wscales":
            if precision == "nvfp4":
                out[key] = (torch.rand(value.shape, generator=g) * 0.5 + 0.25).to(torch.float8_e4m3fn)
            else:
                out[key] = (torch.rand(value.shape, generator=g) * 0.002 + 0.001).to(value.dtype)
        elif suffix == "wcscales":
            out[key] = (torch.rand(value.shape, generator=g) * 0.01 + 0.005).to(value.dtype)
        elif suffix in ("smooth_factor", "smooth_factor_orig"):
            out[key] = (torch.rand(value.shape, generator=g) + 0.5).to(value.dtype)
        elif suffix in ("proj_down", "proj_up"):
            out[key] = (torch.randn(value.shape, generator=g) * 0.05).to(value.dtype)
        elif value.is_floating_point():
            scale = 1.0 if suffix == "weight" and value.ndim == 1 else 0.05
            base = 1.0 if suffix == "weight" and value.ndim == 1 else 0.0
            out[key] = (base + torch.randn(value.shape, generator=g) * scale).to(value.dtype)
        else:
            out[key] = value.clone()
    return out


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.float().cpu(), expected.float().cpu()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()
