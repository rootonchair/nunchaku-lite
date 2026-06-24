try:
    from ._C import ops
except ImportError:
    import importlib
    import pkgutil
    import sys

    import torch

    package = sys.modules[__package__]
    extension_module = None
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_nunchaku_lite_kernels_cuda"):
            extension_module = importlib.import_module(f"{__package__}.{module_info.name}")
            break

    if extension_module is None:
        raise

    ops = getattr(torch.ops, extension_module.__name__.rsplit(".", 1)[-1])
