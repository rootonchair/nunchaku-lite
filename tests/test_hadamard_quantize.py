import pytest
import torch

from nunchaku_lite.ops import quantize as quantize_mod
from nunchaku_lite.ops.quantize import (
    hadamard_matrix,
    hadamard_rotate,
    svdq_quantize_w4a4_act_fuse_lora_cuda,
    svdq_quantize_w4a4_act_hadamard_fuse_lora_cuda,
)


def _has_native_hadamard_op() -> bool:
    try:
        from nunchaku_lite.ops.backend import get_ops

        return hasattr(get_ops(), "quantize_w4a4_act_hadamard_fuse_lora")
    except Exception:  # noqa: BLE001 - feature probe: any failure means "unavailable", so skip
        return False


requires_native_hadamard = pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_native_hadamard_op(),
    reason="needs CUDA and a kernel build with quantize_w4a4_act_hadamard_fuse_lora",
)


def _skip_unsupported(dtype=None, fp4=False):
    """Skip kernel paths the GPU cannot run: they trap on the device and poison the CUDA context."""

    major = torch.cuda.get_device_capability()[0]
    if fp4 and major < 12:
        pytest.skip("NVFP4 needs a Blackwell GPU (compute capability 12.x)")
    if dtype is torch.bfloat16 and major < 8:
        pytest.skip("bf16 W4A4 kernels need compute capability 8.0+")


# --- reference matrix (CPU) -------------------------------------------------------------


def test_hadamard_matrix_is_orthonormal_symmetric_sylvester():
    h = hadamard_matrix(64, dtype=torch.float64)
    assert torch.allclose(h @ h.T, torch.eye(64, dtype=torch.float64))
    assert torch.equal(h, h.T)
    h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    sylvester = h2
    for _ in range(5):
        sylvester = torch.kron(sylvester, h2)
    assert torch.equal(h, sylvester / 8)
    # Sylvester order has one all-ones column (the kernel's FWHT relies on this exact order).
    assert torch.equal((h * 8).sum(dim=0)[0], torch.tensor(64.0, dtype=torch.float64))


def test_hadamard_matrix_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        hadamard_matrix(48)


def test_hadamard_rotate_is_blockwise_and_self_inverse():
    x = torch.randn(5, 192, dtype=torch.float64)
    y = hadamard_rotate(x)
    assert torch.allclose(hadamard_rotate(y), x)
    h = hadamard_matrix(64, dtype=torch.float64)
    assert torch.allclose(y[:, 64:128], x[:, 64:128] @ h)
    # Rotating both operands of a GEMM leaves the product unchanged (the export contract).
    w = torch.randn(7, 192, dtype=torch.float64)
    assert torch.allclose(hadamard_rotate(x) @ hadamard_rotate(w).T, x @ w.T)


def test_hadamard_rotate_rejects_partial_block():
    with pytest.raises(ValueError, match="multiple of block size"):
        hadamard_rotate(torch.randn(2, 100))


# --- Python wrapper with a fake backend (CPU) -------------------------------------------


class _RecordingOps:
    def __init__(self):
        self.calls = []

    def quantize_w4a4_act_fuse_lora(self, *args):
        self.calls.append(("plain", args))

    def quantize_w4a4_act_hadamard_fuse_lora(self, *args):
        self.calls.append(("hadamard", args))


def test_hadamard_wrapper_calls_hadamard_op_with_allocated_buffers(monkeypatch):
    ops = _RecordingOps()
    monkeypatch.setattr(quantize_mod, "get_ops", lambda: ops)
    x = torch.randn(300, 128)
    lora_down = torch.zeros(128, 16)
    smooth = torch.ones(128)

    out, oscales, lora_act = svdq_quantize_w4a4_act_hadamard_fuse_lora_cuda(x, lora_down=lora_down, smooth=smooth)

    assert [name for name, _ in ops.calls] == ["hadamard"]
    args = ops.calls[0][1]
    assert args[0] is x and args[3] is lora_down and args[5] is smooth
    assert args[1] is out and args[2] is oscales and args[4] is lora_act
    assert args[6:] == (False, False)
    assert out.shape == (512, 64) and out.dtype == torch.uint8
    assert oscales.shape == (2, 512)
    assert lora_act.shape == (512, 16) and lora_act.dtype == torch.float32


def test_plain_wrapper_keeps_original_op_and_signature(monkeypatch):
    ops = _RecordingOps()
    monkeypatch.setattr(quantize_mod, "get_ops", lambda: ops)
    svdq_quantize_w4a4_act_fuse_lora_cuda(torch.randn(8, 64), lora_down=torch.zeros(64, 16))
    assert [name for name, _ in ops.calls] == ["plain"]
    assert len(ops.calls[0][1]) == 8  # input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4


def test_hadamard_wrapper_reports_old_backend(monkeypatch):
    class OldOps:
        def quantize_w4a4_act_fuse_lora(self, *args):
            raise AssertionError("must not fall back to the unrotated op")

    monkeypatch.setattr(quantize_mod, "get_ops", lambda: OldOps())
    with pytest.raises(NotImplementedError, match="quantize_w4a4_act_hadamard_fuse_lora"):
        svdq_quantize_w4a4_act_hadamard_fuse_lora_cuda(torch.randn(8, 64), lora_down=torch.zeros(64, 16))


def test_hadamard_wrapper_refuses_fuse_glu(monkeypatch):
    monkeypatch.setattr(quantize_mod, "get_ops", lambda: _RecordingOps())
    with pytest.raises(NotImplementedError, match="fuse_glu"):
        svdq_quantize_w4a4_act_hadamard_fuse_lora_cuda(
            torch.randn(8, 128), lora_down=torch.zeros(128, 16), fuse_glu=True
        )


def test_hadamard_wrapper_rejects_channels_not_multiple_of_64(monkeypatch):
    monkeypatch.setattr(quantize_mod, "get_ops", lambda: _RecordingOps())
    with pytest.raises(ValueError, match="divisible by 64"):
        svdq_quantize_w4a4_act_hadamard_fuse_lora_cuda(torch.randn(8, 48), lora_down=torch.zeros(48, 16))


# --- native kernel (CUDA) ----------------------------------------------------------------


@requires_native_hadamard
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens,channels", [(256, 3072), (300, 1024)])
def test_native_hadamard_op_matches_plain_op_on_prerotated_input(dtype, tokens, channels):
    _skip_unsupported(dtype)
    # With a uniform smooth factor (layout-independent), rotating inside the kernel must
    # equal feeding the plain op an input rotated in PyTorch. Small differences come only
    # from fp16/bf16 rounding of the in-register FWHT vs the fp32 reference.
    device = torch.device("cuda")
    torch.manual_seed(0)
    x = torch.randn(tokens, channels, device=device, dtype=torch.float32)
    x[:, 7] *= 30  # a channel outlier, so rotation actually changes the scales
    smooth = torch.full((channels,), 2.0, device=device, dtype=dtype)
    lora_down = torch.zeros(channels, 16, device=device, dtype=dtype)

    q_h, s_h, _ = svdq_quantize_w4a4_act_hadamard_fuse_lora_cuda(x.to(dtype), lora_down=lora_down, smooth=smooth)
    x_rot = hadamard_rotate(x).to(dtype)
    q_r, s_r, _ = svdq_quantize_w4a4_act_fuse_lora_cuda(x_rot, lora_down=lora_down, smooth=smooth)
    _, s_n, _ = svdq_quantize_w4a4_act_fuse_lora_cuda(x.to(dtype), lora_down=lora_down, smooth=smooth)

    torch.testing.assert_close(s_h.float(), s_r.float(), rtol=2e-2, atol=1e-3)
    assert (q_h == q_r).float().mean().item() > 0.97
    # Control: the rotation really did something (otherwise this test would be vacuous).
    assert not torch.allclose(s_h.float(), s_n.float(), rtol=2e-2, atol=1e-3)


def _pack_smooth(smooth: torch.Tensor) -> torch.Tensor:
    """Logical per-channel smooth factor -> Nunchaku's packed vector layout, which the kernels read."""

    n = smooth.numel()
    return smooth.reshape(n // 128, 1, 8, 2, 4, 2, 1).permute(0, 6, 1, 2, 4, 3, 5).reshape(-1).contiguous()


def test_pack_smooth_inverts_the_runtime_unpacker():
    from nunchaku_lite.ops.mps import unpack_svdq_scale

    smooth = torch.rand(1024)
    assert torch.equal(unpack_svdq_scale(_pack_smooth(smooth), 1024).reshape(-1), smooth)


@requires_native_hadamard
@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_hadamard_op_rotates_after_per_channel_smoothing(dtype, fp4):
    _skip_unsupported(dtype, fp4=fp4)
    # With a random per-channel smooth factor, smoothing and rotation do not commute, so this
    # pins the order the weights were packed for: quantize(H64/8 @ (x / smooth)). The reference
    # rotates the smoothed input in PyTorch and feeds the plain op with smooth = 1.
    device = torch.device("cuda")
    torch.manual_seed(1)
    tokens, channels = 256, 1024
    x = torch.randn(tokens, channels, device=device, dtype=torch.float32)
    x[:, 7] *= 30
    smooth = torch.empty(channels, device=device).uniform_(0.25, 4.0)
    lora_down = torch.zeros(channels, 16, device=device, dtype=dtype)

    q_h, s_h, _ = svdq_quantize_w4a4_act_hadamard_fuse_lora_cuda(
        x.to(dtype), lora_down=lora_down, smooth=_pack_smooth(smooth.to(dtype)), fp4=fp4
    )
    x_ref = hadamard_rotate(x / smooth.to(dtype).float()).to(dtype)
    ones = torch.ones(channels, device=device, dtype=dtype)
    q_r, s_r, _ = svdq_quantize_w4a4_act_fuse_lora_cuda(x_ref, lora_down=lora_down, smooth=ones, fp4=fp4)
    # Wrong order (rotate, then smooth) as a control: it must be clearly worse.
    x_bad = (hadamard_rotate(x) / smooth.to(dtype).float()).to(dtype)
    q_b, _, _ = svdq_quantize_w4a4_act_fuse_lora_cuda(x_bad, lora_down=lora_down, smooth=ones, fp4=fp4)

    match = (q_h == q_r).float().mean().item()
    assert match > (0.93 if fp4 else 0.97)
    assert (q_h == q_b).float().mean().item() < match - 0.1
    if fp4:  # FP8 E4M3 scales: allow one ULP (12.5%)
        torch.testing.assert_close(s_h.float(), s_r.float(), rtol=0.13, atol=1e-3)
    else:
        torch.testing.assert_close(s_h.float(), s_r.float(), rtol=2e-2, atol=1e-3)


@requires_native_hadamard
def test_native_hadamard_op_does_not_overflow_fp16_on_coherent_blocks():
    # 64 equal channels of 1100: the rotated result (8800 in one channel) fits fp16, but unscaled
    # butterfly partial sums would reach 70400 > 65504 and overflow to inf.
    device = torch.device("cuda")
    x = torch.randn(256, 1024, device=device, dtype=torch.float16)
    x[:, :64] = 1100.0
    smooth = torch.ones(1024, device=device, dtype=torch.float16)
    lora_down = torch.zeros(1024, 16, device=device, dtype=torch.float16)

    q_h, s_h, _ = svdq_quantize_w4a4_act_hadamard_fuse_lora_cuda(x, lora_down=lora_down, smooth=smooth)
    q_r, s_r, _ = svdq_quantize_w4a4_act_fuse_lora_cuda(
        hadamard_rotate(x.float()).half(), lora_down=lora_down, smooth=smooth
    )
    assert torch.isfinite(s_h.float()).all()
    torch.testing.assert_close(s_h.float(), s_r.float(), rtol=2e-2, atol=1e-3)
    assert (q_h == q_r).float().mean().item() > 0.97


@requires_native_hadamard
def test_native_hadamard_op_lora_branch_sees_unrotated_input():
    device = torch.device("cuda")
    torch.manual_seed(0)
    x = torch.randn(256, 1024, device=device, dtype=torch.float16)
    smooth = torch.ones(1024, device=device, dtype=torch.float16)
    lora_down = torch.randn(1024, 16, device=device, dtype=torch.float16) * 0.05
    _, _, lora_h = svdq_quantize_w4a4_act_hadamard_fuse_lora_cuda(x, lora_down=lora_down, smooth=smooth)
    _, _, lora_n = svdq_quantize_w4a4_act_fuse_lora_cuda(x, lora_down=lora_down, smooth=smooth)
    torch.testing.assert_close(lora_h, lora_n)


@requires_native_hadamard
def test_rotated_linear_forward_compiles_fullgraph():
    _skip_unsupported(torch.bfloat16)
    from nunchaku_lite.linear import SVDQW4A4Linear

    torch.manual_seed(0)
    layer = SVDQW4A4Linear(256, 256, rank=16, torch_dtype=torch.bfloat16, device="cuda", input_rotation="hadamard")
    with torch.no_grad():
        layer.qweight.copy_(torch.randint(-128, 128, layer.qweight.shape, dtype=torch.int8, device="cuda"))
        layer.wscales.fill_(0.01)
        layer.smooth_factor.fill_(1.0)
        layer.proj_down.normal_(0, 0.05)
        layer.proj_up.normal_(0, 0.05)
        if layer.bias is not None:
            layer.bias.zero_()
    x = torch.randn(1, 64, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        eager = layer(x)
        compiled = torch.compile(layer, fullgraph=True)(x)
    assert torch.isfinite(eager).all()
    torch.testing.assert_close(compiled, eager)


@requires_native_hadamard
def test_native_hadamard_op_compiles_fullgraph():
    _skip_unsupported(torch.bfloat16)
    device = torch.device("cuda")
    x = torch.randn(256, 1024, device=device, dtype=torch.bfloat16)
    smooth = torch.ones(1024, device=device, dtype=torch.bfloat16)
    lora_down = torch.zeros(1024, 16, device=device, dtype=torch.bfloat16)

    def fn(inp):
        return svdq_quantize_w4a4_act_hadamard_fuse_lora_cuda(inp, lora_down=lora_down, smooth=smooth)

    eager = fn(x)
    compiled = torch.compile(fn, fullgraph=True)(x)
    for a, b in zip(eager, compiled):
        assert torch.equal(a, b)
