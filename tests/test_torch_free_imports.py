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


def test_package_attributes_resolve_lazily():
    """``import nunchaku_lite`` exposes submodules and the public API as attributes, as the eager __init__ did."""

    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    import nunchaku_lite

    for name in ("core", "linear", "adapters", "ops", "text_encoders", "manifest"):
        assert getattr(nunchaku_lite, name).__name__ == f"nunchaku_lite.{name}"
    assert nunchaku_lite.load_nunchaku_pipeline is nunchaku_lite.core.load_nunchaku_pipeline
    with pytest.raises(AttributeError):
        nunchaku_lite.does_not_exist  # noqa: B018
