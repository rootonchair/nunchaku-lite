import builtins
import sys
import types

import pytest

from nunchaku_lite.ops import backend


@pytest.fixture(autouse=True)
def clear_backend_cache():
    backend._clear_ops_cache()
    yield
    backend._clear_ops_cache()


def test_get_ops_prefers_local_kernels_package(monkeypatch):
    local_ops = object()

    monkeypatch.setattr(backend, "_load_native_ops", lambda: local_ops)
    monkeypatch.setattr(backend, "_load_hf_ops", lambda: pytest.fail("HF backend should not be loaded"))

    assert backend.get_ops() is local_ops
    assert backend.get_ops() is local_ops


def test_get_ops_falls_back_to_hf_backend(monkeypatch):
    hf_ops = object()

    def fail_native():
        raise ImportError("missing extension")

    monkeypatch.setattr(backend, "_load_native_ops", fail_native)
    monkeypatch.setattr(backend, "_load_hf_ops", lambda: hf_ops)

    assert backend.get_ops() is hf_ops


def test_load_hf_ops_uses_pinned_kernel_repo(monkeypatch):
    calls = []
    hf_ops = object()
    hf_kernel = types.SimpleNamespace(ops=hf_ops)

    def get_kernel(repo, *, version, trust_remote_code):
        calls.append((repo, version, trust_remote_code))
        return hf_kernel

    monkeypatch.setitem(sys.modules, "kernels", types.SimpleNamespace(get_kernel=get_kernel))

    assert backend._load_hf_ops() is hf_ops
    assert calls == [(backend.HF_KERNEL_REPO, backend.HF_KERNEL_VERSION, True)]


def test_missing_backends_raise_actionable_install_error(monkeypatch):
    def fail_native():
        raise ImportError("missing extension")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "kernels":
            raise ImportError("missing kernels")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(backend, "_load_native_ops", fail_native)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"pip install ./nunchaku-lite-kernels"):
        backend.get_ops()
