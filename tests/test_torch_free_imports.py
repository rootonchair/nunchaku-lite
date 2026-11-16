"""``nunchaku_lite.mlx`` and the modules it shares with the PyTorch runtime must import without torch."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def _modules_loaded_after(statement: str) -> set[str]:
    code = f"import sys; {statement}; print('\\n'.join(sys.modules))"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env={"PYTHONPATH": str(SRC)}
    )
    return set(result.stdout.split())


def _heavy(modules: set[str]) -> set[str]:
    return {m for m in modules if m.split(".")[0] in {"torch", "diffusers", "transformers", "accelerate", "peft"}}


def test_shared_modules_import_without_torch():
    loaded = _modules_loaded_after("import nunchaku_lite, nunchaku_lite.manifest, nunchaku_lite.checkpoint_keys")
    assert not _heavy(loaded)


@pytest.mark.skipif(importlib.util.find_spec("mlx") is None, reason="MLX is not installed")
def test_mlx_runtime_imports_without_torch():
    loaded = _modules_loaded_after("import nunchaku_lite.mlx, nunchaku_lite.mlx.pipeline")
    assert not _heavy(loaded)
