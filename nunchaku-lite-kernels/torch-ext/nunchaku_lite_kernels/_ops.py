from __future__ import annotations

import importlib
import pkgutil
import sys

import torch

_DISPATCH_NAMESPACE = "nunchaku_lite_kernels"


def add_op_namespace_prefix(op_name: str) -> str:
    return f"{_DISPATCH_NAMESPACE}::{op_name}"


def _load_extension_module():
    try:
        from . import _C

        return _C
    except ImportError:
        package = sys.modules[__package__]
        for module_info in pkgutil.iter_modules(package.__path__):
            if module_info.name.startswith("_nunchaku_lite_kernels_cuda"):
                return importlib.import_module(f"{__package__}.{module_info.name}")
        raise


_extension_module = _load_extension_module()
ops = getattr(torch.ops, _DISPATCH_NAMESPACE)
