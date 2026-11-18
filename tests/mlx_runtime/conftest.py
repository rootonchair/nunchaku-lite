"""MLX runtime tests: collected only where MLX is installed and has a Metal GPU (Apple silicon)."""

import importlib.util


def _metal_available() -> bool:
    if importlib.util.find_spec("mlx") is None:
        return False
    try:
        import mlx.core as mx

        return bool(mx.metal.is_available())
    except Exception:  # noqa: BLE001 - e.g. a CI VM without a Metal device
        return False


collect_ignore_glob = [] if _metal_available() else ["test_*.py"]
