"""Runtime backend selection for Nunchaku Lite kernel ops."""

from __future__ import annotations

from typing import Any

HF_KERNEL_REPO = "rootonchair/nunchaku-lite-kernels"
HF_KERNEL_VERSION = 2

_OPS: Any | None = None


def _load_native_ops() -> Any:
    from nunchaku_lite_kernels import ops

    return ops


def _load_hf_ops() -> Any:
    try:
        from kernels import get_kernel
    except ImportError as exc:
        raise ImportError(
            "Nunchaku Lite kernels are unavailable. Install the default dependencies for Hugging Face kernels "
            'or run "pip install ./nunchaku-lite-kernels" to build the local CUDA package.'
        ) from exc
    kernel = get_kernel(HF_KERNEL_REPO, version=HF_KERNEL_VERSION, trust_remote_code=True)
    try:
        return kernel.ops
    except AttributeError as exc:
        raise AttributeError(f"Hugging Face kernel {HF_KERNEL_REPO!r} does not expose an ops backend.") from exc


def get_ops() -> Any:
    """Return the active ops backend, preferring the local kernels package."""

    global _OPS
    if _OPS is not None:
        return _OPS
    try:
        _OPS = _load_native_ops()
    except (ImportError, ModuleNotFoundError, OSError):
        _OPS = _load_hf_ops()
    return _OPS


def _clear_ops_cache() -> None:
    """Clear the cached backend for tests."""

    global _OPS
    _OPS = None
