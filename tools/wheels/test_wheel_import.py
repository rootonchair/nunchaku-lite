"""Smoke test for release wheels in a CPU-only CI environment."""

from __future__ import annotations

import importlib
import os


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("Release wheel smoke tests must hide GPUs.")

    package = importlib.import_module("nunchaku_lite")

    for name in ["load_nunchaku_pipeline", "patch_transformer", "register_adapter"]:
        if not hasattr(package, name):
            raise RuntimeError(f"Missing public API: {name}")


if __name__ == "__main__":
    main()
