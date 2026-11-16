"""Apple-silicon (MPS) execution paths: layout unpacking and module numerics."""

import pytest
import torch
import torch.nn.functional as F

from nunchaku_lite.adapters.common import pack_rotemb
from nunchaku_lite.linear import AWQW4A16Linear, SVDQW4A4Linear
from nunchaku_lite.lora.core.layout import pack_lowrank_weight
from nunchaku_lite.ops import mps as mps_ops

# ---------------------------------------------------------------------------
# Reference packers (port of DeepCompressor's NunchakuWeightPacker, bits=4, warp_n=128)
# ---------------------------------------------------------------------------


def pack_weight(q: torch.Tensor) -> torch.Tensor:
    n, k = q.shape
    w = q.to(torch.int64).reshape(n // 128, 8, 2, 8, 1, k // 64, 1, 2, 4, 8)
    w = w.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9) & 0xF
    words = (w << (4 * torch.arange(8, dtype=torch.int64))).sum(-1).to(torch.int64)
    words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    return words.contiguous().view(torch.int8).reshape(n, k // 2)


def pack_scale(scale: torch.Tensor) -> torch.Tensor:
    n, groups = scale.shape
    s = scale.reshape(n // 128, 1, 8, 2, 4, 2, groups).permute(0, 6, 1, 2, 4, 3, 5).contiguous()
    return s.reshape(-1) if groups == 1 else s.reshape(groups, n)


def _pack_awq_g64(codes: torch.Tensor) -> torch.Tensor:
    out_features, in_features = codes.shape
    groups = in_features // 64
    channel = torch.arange(64)
    packed_index = (channel // 32) * 4 + (channel % 8) // 2
    nibble = ((channel % 32) // 8) + 4 * (channel % 2)
    packed = torch.zeros(out_features, groups, 8, dtype=torch.int64)
    for c in range(64):
        packed[:, :, packed_index[c]] |= (codes.view(out_features, groups, 64)[:, :, c].to(torch.int64) & 0xF) << (
            4 * nibble[c]
        )
    packed = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)
    return packed.view(out_features // 4, 4, groups, 8).permute(0, 2, 1, 3).reshape(out_features // 4, groups * 32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _logical(n=256, k=512, r=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "q": torch.randint(-8, 8, (n, k), generator=g, dtype=torch.int8),
        "wscale": torch.rand(n, k // 64, generator=g) * 0.02 + 0.001,
        "smooth": torch.rand(k, generator=g) + 0.5,
        "down": torch.randn(r, k, generator=g) * 0.05,
        "up": torch.randn(n, r, generator=g) * 0.05,
        "bias": torch.randn(n, generator=g) * 0.1,
    }


def _module(w, device, act_unsigned=False, dtype=torch.bfloat16, backend="auto"):
    n, k = w["q"].shape
    r = w["down"].shape[0]
    m = SVDQW4A4Linear(k, n, rank=r, bias=True, torch_dtype=dtype, device="cpu", act_unsigned=act_unsigned)
    m.qweight.data = pack_weight(w["q"])
    m.wscales.data = pack_scale(w["wscale"]).to(dtype)
    m.smooth_factor.data = pack_scale(w["smooth"][:, None]).to(dtype)
    m.bias.data = pack_scale(w["bias"][:, None]).to(dtype)
    m.proj_down.data = pack_lowrank_weight(w["down"].to(dtype), down=True)
    m.proj_up.data = pack_lowrank_weight(w["up"].to(dtype), down=False)
    m.backend = backend
    return m.to(device)


def _rounded(w, dtype=torch.bfloat16):
    return {key: (v if key == "q" else v.to(dtype).double()) for key, v in w.items()}


def _reference(w, x, shift=0.0):
    wd = (w["q"].double().view(w["q"].shape[0], -1, 64) * w["wscale"].double()[..., None]).view_as(w["q"]).double()
    x = x.double()
    return ((x + shift) / w["smooth"]) @ wd.t() + (x @ w["down"].t()) @ w["up"].t() + w["bias"]


def _rel(out, ref):
    return float((out.detach().cpu().double() - ref.detach()).norm() / ref.detach().norm())


requires_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires an MPS device")


def _kernels_available() -> bool:
    try:
        return hasattr(mps_ops._kernel_ops(), "svdq_gemm_w8a8")
    except (ImportError, RuntimeError, AttributeError):
        return False


requires_kernels = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and mps_ops.has_neural_accelerators() and _kernels_available()),
    reason="requires nunchaku_lite_kernels MPS ops on an M5-class GPU",
)

# ---------------------------------------------------------------------------
# Layout tests (CPU)
# ---------------------------------------------------------------------------


def test_unpack_weight_and_scale_roundtrip():
    w = _logical(n=384, k=256)
    assert torch.equal(mps_ops.unpack_svdq_weight(pack_weight(w["q"])), w["q"])
    torch.testing.assert_close(mps_ops.unpack_svdq_scale(pack_scale(w["wscale"]), 384), w["wscale"])
    torch.testing.assert_close(
        mps_ops.unpack_svdq_scale(pack_scale(w["smooth"][:256, None]), 256)[:, 0], w["smooth"][:256]
    )


def test_unpack_rotemb_inverts_pack_rotemb():
    rotemb = torch.randn(2, 64, 64, 1, 2)
    unpacked = mps_ops.unpack_rotemb(pack_rotemb(rotemb))
    assert torch.equal(unpacked, rotemb.squeeze(3))


def test_dequantize_awq_g64_matches_codes():
    g = torch.Generator().manual_seed(3)
    codes = torch.randint(0, 16, (128, 256), generator=g, dtype=torch.int32)
    wscales = torch.rand(4, 128, generator=g) * 0.02 + 0.001
    wzeros = -7 * wscales
    dense = mps_ops.dequantize_awq_g64_int32(_pack_awq_g64(codes), wscales, wzeros)
    ref = codes.float().view(128, 4, 64) * wscales.t()[..., None] + wzeros.t()[..., None]
    torch.testing.assert_close(dense, ref.view(128, 256))


@pytest.mark.parametrize("brand,expected", [("Apple M5", True), ("Apple M5 Max", True), ("Apple M4 Pro", False)])
def test_neural_accelerator_detection(monkeypatch, brand, expected):
    mps_ops.has_neural_accelerators.cache_clear()
    monkeypatch.setattr(mps_ops.subprocess, "check_output", lambda *a, **k: brand)
    try:
        assert mps_ops.has_neural_accelerators() is expected
    finally:
        mps_ops.has_neural_accelerators.cache_clear()


# ---------------------------------------------------------------------------
# Module numerics on MPS
# ---------------------------------------------------------------------------

BACKENDS = ["dense", pytest.param("int8", marks=requires_kernels), pytest.param("int4", marks=requires_kernels)]


@requires_mps
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("act_unsigned", [False, True])
def test_svdq_linear_matches_reference(monkeypatch, backend, act_unsigned):
    w = _logical()
    module = _module(w, "mps", act_unsigned=act_unsigned, backend=backend)
    x = torch.randn(2, 77, 512)
    out = module(x.to("mps", torch.bfloat16))
    assert out.shape == (2, 77, 256) and out.dtype == torch.bfloat16
    ref = _reference(_rounded(w), x.bfloat16().view(-1, 512), shift=mps_ops.GELU_SHIFT if act_unsigned else 0.0)
    assert _rel(out.view(-1, 256), ref) < (2e-2 if backend == "dense" else 4e-2)


@requires_kernels
def test_svdq_cache_invalidates_on_lowrank_change(monkeypatch):
    w = _logical(seed=4)
    module = _module(w, "mps", backend="int8")
    x = torch.randn(64, 512).to("mps", torch.bfloat16)
    first = module(x.unsqueeze(0))
    w2 = dict(w, up=w["up"] * 3)
    module.proj_up = torch.nn.Parameter(pack_lowrank_weight(w2["up"].bfloat16(), down=False).to("mps"))
    second = module(x.unsqueeze(0))
    ref = _reference(_rounded(w2), x.cpu())
    assert not torch.allclose(first, second)
    assert _rel(second.view(-1, 256), ref) < 4e-2


@requires_mps
@pytest.mark.parametrize("backend", BACKENDS)
def test_gelu_mlp_and_qkv_norm_rotary(monkeypatch, backend):
    from nunchaku_lite.ops.fused import fused_gelu_mlp, fused_qkv_norm_rotary

    tol = 2e-2 if backend == "dense" else 5e-2
    # GELU MLP: fc1 (512 -> 256), fc2 (256 -> 512, act_unsigned with GELU shift)
    w1, w2 = _logical(n=256, k=512, seed=5), _logical(n=512, k=256, seed=6)
    fc1, fc2 = _module(w1, "mps", backend=backend), _module(w2, "mps", act_unsigned=True, backend=backend)
    x = torch.randn(1, 40, 512)
    out = fused_gelu_mlp(x.to("mps", torch.bfloat16), fc1, fc2)
    hidden = F.gelu(_reference(_rounded(w1), x.bfloat16().view(-1, 512)).bfloat16().double(), approximate="tanh")
    ref = _reference(_rounded(w2), hidden, shift=mps_ops.GELU_SHIFT)
    assert _rel(out.view(-1, 512), ref) < tol

    # QKV with per-head RMSNorm and packed RoPE: 2 heads x head_dim 128 -> out 768
    wq = _logical(n=768, k=256, seed=7)
    proj = _module(wq, "mps", backend=backend)
    norm_q, norm_k = torch.nn.RMSNorm(128, eps=1e-6), torch.nn.RMSNorm(128, eps=1e-6)
    norm_q.weight.data.uniform_(0.5, 1.5)
    norm_k.weight.data.uniform_(0.5, 1.5)
    wq_norm, wk_norm = norm_q.weight.detach().clone(), norm_k.weight.detach().clone()
    seq = 40
    angles = torch.randn(seq, 64)
    rotemb = torch.stack([angles.sin(), angles.cos()], dim=-1).unsqueeze(0).unsqueeze(-2)  # (1, S, 64, 1, 2)
    packed = pack_rotemb(F.pad(rotemb, (0, 0, 0, 0, 0, 0, 0, 256 - seq))).to("mps")
    x = torch.randn(1, seq, 256)
    out = fused_qkv_norm_rotary(
        x.to("mps", torch.bfloat16), proj, norm_q.to("mps", torch.bfloat16), norm_k.to("mps", torch.bfloat16), packed
    )
    qkv = _reference(_rounded(wq), x.bfloat16().view(-1, 256)).view(1, seq, 3, 2, 128)

    def norm_rope(t, weight):
        t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * weight.bfloat16().double()
        sin, cos = angles.sin().double()[None, :, None], angles.cos().double()[None, :, None]
        out = torch.empty_like(t)
        out[..., 0::2] = t[..., 0::2] * cos - t[..., 1::2] * sin
        out[..., 1::2] = t[..., 0::2] * sin + t[..., 1::2] * cos
        return out

    ref = torch.cat([norm_rope(qkv[:, :, 0], wq_norm), norm_rope(qkv[:, :, 1], wk_norm), qkv[:, :, 2]], dim=2).view(
        1, seq, 768
    )
    assert out.shape == (1, seq, 768)
    assert _rel(out, ref) < tol


@requires_mps
def test_awq_linear_on_mps():
    g = torch.Generator().manual_seed(8)
    codes = torch.randint(0, 16, (256, 128), generator=g, dtype=torch.int32)
    module = AWQW4A16Linear(128, 256, bias=True, group_size=64, torch_dtype=torch.bfloat16, device="cpu")
    module.qweight.data = _pack_awq_g64(codes)
    module.wscales.data = (torch.rand(2, 256, generator=g) * 0.02 + 0.001).bfloat16()
    module.wzeros.data = (-7 * module.wscales.data.float()).bfloat16()
    module.bias.data = torch.randn(256, generator=g).bfloat16()
    module = module.to("mps")
    x = torch.randn(3, 128).bfloat16()
    dense = mps_ops.dequantize_awq_g64_int32(_pack_awq_g64(codes), module.wscales.cpu(), module.wzeros.cpu())
    ref = x.double() @ dense.double().t() + module.bias.cpu().double()
    out = module(x.to("mps"))
    assert _rel(out, ref) < 1e-2
    assert module._nunchaku_lite_mps_meta["backend"] == "awq_int4pack"  # weights stay 4-bit
    assert "_nl_mps_weight" not in dict(module.named_buffers())


@requires_mps
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_awq_int4pack_matches_dense_fallback(monkeypatch, dtype):
    g = torch.Generator().manual_seed(18)
    n, k = 384, 256
    codes = torch.randint(0, 16, (n, k), generator=g, dtype=torch.int32)
    module = AWQW4A16Linear(k, n, bias=False, group_size=64, torch_dtype=dtype, device="cpu")
    module.qweight.data = _pack_awq_g64(codes)
    module.wscales.data = (torch.rand(k // 64, n, generator=g) * 0.02 + 0.001).to(dtype)
    module.wzeros.data = (-7 * module.wscales.data.float() + torch.randn(k // 64, n, generator=g) * 1e-3).to(dtype)
    module = module.to("mps")
    dense = mps_ops.dequantize_awq_g64_int32(module.qweight.cpu(), module.wscales.cpu(), module.wzeros.cpu())
    for m in (1, 2, 37):
        x = torch.randn(m, k, generator=g).to(dtype)
        ref = x.double() @ dense.double().t()
        assert _rel(module(x.to("mps")), ref) < (1e-5 if dtype == torch.float32 else 1e-2), m
    assert module._nunchaku_lite_mps_meta["backend"] == "awq_int4pack"

    monkeypatch.setattr(mps_ops, "_build_awq_int4pack", lambda module, dtype: None)  # e.g. older torch
    module._nunchaku_lite_mps_key = None
    x = torch.randn(2, k, generator=g).to(dtype)
    assert _rel(module(x.to("mps")), x.double() @ dense.double().t()) < (1e-5 if dtype == torch.float32 else 1e-2)
    assert module._nunchaku_lite_mps_meta["backend"] == "awq_dense"


@requires_mps
def test_svdq_cache_follows_device_moves_and_stays_out_of_state_dict(monkeypatch):
    w = _logical(seed=9)
    module = _module(w, "mps", backend="dense")
    x = torch.randn(1, 8, 512).to("mps", torch.bfloat16)
    first = module(x)
    builds = []
    original = mps_ops._build_svdq_cache
    monkeypatch.setattr(mps_ops, "_build_svdq_cache", lambda *a, **k: builds.append(1) or original(*a, **k))
    module.to("cpu")
    assert module._nl_mps_weight.device.type == "cpu"  # cache moved with the module (offload frees GPU memory)
    module.to("mps")
    second = module(x)
    assert not builds  # no rebuild after a device round trip, even though every data pointer changed
    torch.testing.assert_close(first, second)
    assert not any(key.startswith("_nl_mps_") for key in module.state_dict())


@requires_mps
def test_cache_tensors_do_not_pin_larger_freed_buffers():
    # The MPS allocator serves a new tensor from a larger cached free buffer and keeps all of it: a cache
    # built right after its float32 temporaries were freed inherited their buffers (~10 GB on FLUX.1-dev).
    def live():
        torch.mps.synchronize()
        return torch.mps.current_allocated_memory()

    torch.mps.empty_cache()
    big = torch.empty(36 * 2**20, dtype=torch.uint8, device="mps")
    del big  # leave a 36 MiB free buffer in the pool
    before = live()
    small = torch.ones(5 * 2**20, dtype=torch.uint8, device="mps")  # would land in the 36 MiB buffer
    cached = mps_ops._compact({"w": small})
    del small
    assert live() - before < 8 * 2**20
    assert torch.equal(cached["w"].cpu(), torch.ones(5 * 2**20, dtype=torch.uint8))


@requires_mps
def test_svdq_cache_rebuilds_when_storage_is_swapped(monkeypatch):
    w = _logical(seed=10)
    module = _module(w, "mps", backend="dense")
    x = torch.randn(1, 8, 512)
    module(x.to("mps", torch.bfloat16))
    w2 = dict(w, bias=w["bias"] + 1.0)
    module.bias.data = pack_scale(w2["bias"][:, None]).to("mps", torch.bfloat16)  # no version bump
    out = module(x.to("mps", torch.bfloat16))
    assert _rel(out.view(-1, 256), _reference(_rounded(w2), x.bfloat16().view(-1, 512))) < 2e-2


@requires_kernels
@pytest.mark.parametrize("backend", ["int8", "int4"])
def test_qweight_released_and_state_dict_roundtrips(monkeypatch, backend):
    w = _logical(seed=11)
    w["wscale"] = (
        torch.rand(256, 8, generator=torch.Generator().manual_seed(1)) * 0.01 + 0.01
    )  # ratio < 15.875: invertible
    module = _module(w, "mps", backend=backend)
    packed = module.qweight.detach().cpu().clone()
    x = torch.randn(1, 16, 512).to("mps", torch.bfloat16)
    out = module(x)
    assert module._nunchaku_lite_qweight_released and module.qweight.numel() == 0
    assert torch.equal(module.state_dict()["qweight"].cpu(), packed)  # exact re-pack from the kernel copy
    torch.testing.assert_close(module(x), out)  # cache still valid after release (no rebuild from the placeholder)

    # LoRA-style change after release: rebuild must recover q from the kernel copy
    w2 = dict(w, up=w["up"] * 2)
    module.proj_up = torch.nn.Parameter(pack_lowrank_weight(w2["up"].bfloat16(), down=False).to("mps"))
    assert _rel(module(x).view(-1, 256), _reference(_rounded(w2), x.cpu().view(-1, 512))) < 4e-2

    # loading a checkpoint restores a real qweight parameter
    state = {k: v.cpu() for k, v in module.state_dict().items()}
    module.load_state_dict(state)
    assert not module._nunchaku_lite_qweight_released and torch.equal(module.qweight.cpu(), packed)
    torch.testing.assert_close(module(x).float(), module(x).float())


@requires_kernels
def test_qweight_kept_when_int8_not_invertible(monkeypatch):
    w = _logical(seed=12)
    w["wscale"][:, 0] = 1e-5  # one tiny group per channel: wr << 1, int8 copy is not invertible
    module = _module(w, "mps", backend="int8")
    module(torch.randn(1, 8, 512).to("mps", torch.bfloat16))
    assert not getattr(module, "_nunchaku_lite_qweight_released", False)
    assert module.qweight.numel() == 256 * 256


def test_pack_svdq_weight_inverts_unpack():
    q = torch.randint(-8, 8, (256, 384), dtype=torch.int8)
    assert torch.equal(mps_ops.pack_svdq_weight(q), pack_weight(q))


@pytest.mark.parametrize(
    "nax,int8_bytes,expected",
    [(True, 4e9, "int8"), (True, 15e9, "int4"), (False, 4e9, "dense"), (True, None, "int8")],
)
def test_auto_backend_resolution(monkeypatch, nax, int8_bytes, expected):
    monkeypatch.setattr(mps_ops, "has_neural_accelerators", lambda: nax)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: int(19e9))  # half = 9.5 GB budget
    assert mps_ops.resolve_backend("auto", None if int8_bytes is None else int(int8_bytes)) == expected
    assert mps_ops.resolve_backend("int4", 1) == "int4"  # explicit choices are kept
    with pytest.raises(ValueError, match="backend must be one of"):
        mps_ops.resolve_backend("w8a8")


def test_configure_backend_sets_every_layer(monkeypatch):
    monkeypatch.setattr(mps_ops, "has_neural_accelerators", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    w = _logical(n=128, k=256, seed=13)
    model = torch.nn.Sequential(_module(w, "cpu"), torch.nn.LayerNorm(128), _module(w, "cpu"))
    estimate = mps_ops.estimate_int8_resident_bytes(model)
    assert (
        estimate
        == 2 * (128 * 256 + sum(p.numel() * p.element_size() for n, p in model[0].named_parameters() if n != "qweight"))
        + 2 * 128 * 4
    )
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 4 * estimate)  # fits -> int8
    assert mps_ops.configure_backend(model) == "int8"
    assert model[0].backend == model[2].backend == "int8"
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: estimate)  # exceeds half -> int4
    assert mps_ops.configure_backend(model) == "int4" and model[2].backend == "int4"
    assert mps_ops.configure_backend(model, "dense") == "dense"
