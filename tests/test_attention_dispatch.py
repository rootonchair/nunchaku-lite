import torch

from nunchaku_lite.adapters import attention_dispatch
from nunchaku_lite.adapters.attention_dispatch import dispatch_lite_attention_fn, lite_attention_backend


def test_lite_attention_native_delegates_to_diffusers_dispatch(monkeypatch):
    calls = []

    def fake_dispatch(query, key, value, **kwargs):
        calls.append((query, key, value, kwargs))
        return value + 1

    monkeypatch.setattr(attention_dispatch, "dispatch_attention_fn", fake_dispatch)
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 3, 2, 4)
    value = torch.randn(1, 3, 2, 4)

    output = dispatch_lite_attention_fn(query, key, value, backend="native")

    torch.testing.assert_close(output, value + 1)
    assert calls[0][3]["backend"] == "native"


def test_lite_attention_sage_context_delegates_to_diffusers_sage(monkeypatch):
    calls = []

    def fake_dispatch(query, key, value, **kwargs):
        calls.append(kwargs)
        return value

    monkeypatch.setattr(attention_dispatch, "dispatch_attention_fn", fake_dispatch)
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 3, 2, 4)
    value = torch.randn(1, 3, 2, 4)

    with lite_attention_backend("sage"):
        output = dispatch_lite_attention_fn(query, key, value)

    torch.testing.assert_close(output, value)
    assert calls[0]["backend"] == "sage"


def test_lite_attention_masked_sage_backend_falls_back_to_native(monkeypatch):
    calls = []

    def fake_dispatch(query, key, value, **kwargs):
        calls.append(kwargs)
        return value + 3

    monkeypatch.setattr(attention_dispatch, "dispatch_attention_fn", fake_dispatch)
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 3, 2, 4)
    value = torch.randn(1, 3, 2, 4)
    attn_mask = torch.ones(1, 1, 3, 3)

    output = dispatch_lite_attention_fn(query, key, value, attn_mask=attn_mask, backend="sage")

    torch.testing.assert_close(output, value + 3)
    assert calls[0]["backend"] == "native"
    assert calls[0]["attn_mask"] is attn_mask
