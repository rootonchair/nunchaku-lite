"""MPS (Metal) SVDQ kernels: correctness against float64 references."""

import nunchaku_lite_kernels
import pytest
import torch
from nunchaku_lite_kernels._ops import ops
from torch._subclasses.fake_tensor import FakeTensorMode

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires an Apple-silicon MPS device")


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _weights(n: int, k: int, g: torch.Generator):
    """Random SVDQ INT4 weights in kernel layout: w8 = rint(q * wr), w4 packed low nibble first."""

    q = torch.randint(-8, 8, (n, k), generator=g, dtype=torch.int32)
    wr = (torch.rand(k // 64, n, generator=g) * (127 / 8)).half()
    w8 = torch.round(q.float().view(n, k // 64, 64) * wr.float().t()[..., None]).clamp(-127, 127).to(torch.int8)
    nib = (q & 0xF).to(torch.uint8).view(n, k // 2, 2)
    w4 = nib[..., 0] | (nib[..., 1] << 4)
    return w8.view(n, k), w4, wr


def _case(m, n, k, r, dtype, bias=True, seed=0):
    g = _gen(seed)
    w8, w4, wr = _weights(n, k, g)
    a = torch.randint(-127, 128, (m, k), generator=g, dtype=torch.int8)
    ascale = torch.rand(m, generator=g) * 0.01 + 1e-3
    wsc = torch.rand(n, generator=g) * 0.01 + 1e-3
    lact = torch.randn(m, r, generator=g).to(dtype)
    lup = (torch.randn(n, r, generator=g) * 0.1).to(dtype)
    b = torch.randn(n, generator=g) if bias else None
    ref = (a.double() @ w8.double().t()) * ascale.double()[:, None] * wsc.double()[None, :]
    ref = ref + lact.double() @ lup.double().t()
    if b is not None:
        ref = ref + b.double()
    mps = {
        "a": a.to("mps"),
        "w8": w8.to("mps"),
        "w4": w4.to("mps"),
        "wr": wr.to("mps"),
        "ascale": ascale.to("mps"),
        "wsc": wsc.to("mps"),
        "lact": lact.to("mps"),
        "lup": lup.to("mps"),
        "bias": None if b is None else b.to("mps"),
    }
    return mps, ref


def _rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.cpu().double() - ref).abs().max() / ref.abs().max())


def test_ops_are_exported():
    for name in ("svdq_quantize_act_int8", "svdq_gemm_w8a8", "svdq_gemm_w4a8"):
        assert callable(getattr(nunchaku_lite_kernels, name))
        assert name in nunchaku_lite_kernels.__all__


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shift", [0.0, 0.171875])
def test_quantize_act_int8_matches_reference(dtype, shift):
    g = _gen(1)
    x = (torch.randn(37, 1536, generator=g) * 3).to(dtype)
    inv_smooth = torch.rand(1536, generator=g) + 0.5
    q, scale = ops.svdq_quantize_act_int8(x.to("mps"), inv_smooth.to("mps"), shift)
    assert q.dtype == torch.int8 and scale.dtype == torch.float32
    xs = (x.double() + shift) * inv_smooth.double()
    ref_scale = xs.abs().amax(dim=1) / 127
    torch.testing.assert_close(scale.cpu().double(), ref_scale, rtol=1e-6, atol=0)
    ref_q = torch.round(xs / ref_scale[:, None]).clamp(-127, 127)
    diff = (q.cpu().double() - ref_q).abs()
    assert diff.max() <= 1 and (diff > 0).double().mean() < 1e-3


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,n,k,r", [(256, 256, 512, 32), (128, 384, 1024, 64), (64, 128, 192, 32)])
@pytest.mark.parametrize("bias", [True, False])
def test_gemm_w8a8_matches_reference(dtype, m, n, k, r, bias):
    t, ref = _case(m, n, k, r, dtype, bias=bias)
    out = ops.svdq_gemm_w8a8(t["a"], t["w8"], t["ascale"], t["wsc"], t["lact"], t["lup"], t["bias"])
    assert out.dtype == dtype and out.shape == (m, n)
    assert _rel(out, ref) < 1e-2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,n,k", [(256, 256, 512), (128, 128, 8192), (64, 192, 192)])
def test_gemm_w4a8_matches_w8a8(dtype, m, n, k):
    t, ref = _case(m, n, k, 32, dtype, seed=2)
    out4 = ops.svdq_gemm_w4a8(t["a"], t["w4"], t["wr"], t["ascale"], t["wsc"], t["lact"], t["lup"], t["bias"])
    out8 = ops.svdq_gemm_w8a8(t["a"], t["w8"], t["ascale"], t["wsc"], t["lact"], t["lup"], t["bias"])
    assert _rel(out4, ref) < 1e-2
    # identical int8 math; only the fp low-rank summation order may differ between tile shapes
    torch.testing.assert_close(out4.float().cpu(), out8.float().cpu(), rtol=1e-2, atol=1e-2 * float(ref.abs().max()))


def test_gemm_rejects_bad_inputs():
    t, _ = _case(128, 128, 256, 32, torch.bfloat16)
    with pytest.raises(RuntimeError, match="not aligned"):
        ops.svdq_gemm_w8a8(t["a"][:100], t["w8"], t["ascale"][:100], t["wsc"], t["lact"][:100], t["lup"], t["bias"])
    with pytest.raises(RuntimeError, match="share a dtype"):
        ops.svdq_gemm_w8a8(t["a"], t["w8"], t["ascale"], t["wsc"], t["lact"], t["lup"].half(), t["bias"])
    with pytest.raises(RuntimeError, match="MPS tensor"):
        ops.svdq_gemm_w8a8(t["a"].cpu(), t["w8"], t["ascale"], t["wsc"], t["lact"], t["lup"], t["bias"])


def test_fake_kernels():
    with FakeTensorMode():
        x = torch.empty(256, 512, device="mps", dtype=torch.bfloat16)
        q, scale = ops.svdq_quantize_act_int8(x, torch.empty(512, device="mps"), 0.0)
        assert q.shape == x.shape and q.dtype == torch.int8 and scale.shape == (256,)
        a = torch.empty(256, 512, device="mps", dtype=torch.int8)
        lact = torch.empty(256, 32, device="mps", dtype=torch.bfloat16)
        lup = torch.empty(384, 32, device="mps", dtype=torch.bfloat16)
        s = torch.empty(256, device="mps")
        wsc = torch.empty(384, device="mps")
        out = ops.svdq_gemm_w8a8(a, torch.empty(384, 512, device="mps", dtype=torch.int8), s, wsc, lact, lup, None)
        assert out.shape == (256, 384) and out.dtype == torch.bfloat16
        w4 = torch.empty(384, 256, device="mps", dtype=torch.uint8)
        wr = torch.empty(8, 384, device="mps", dtype=torch.float16)
        out = ops.svdq_gemm_w4a8(a, w4, wr, s, wsc, lact, lup, None)
        assert out.shape == (256, 384)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("norms,rope", [(True, True), (False, True), (True, False)])
def test_qk_norm_rope_matches_reference(dtype, norms, rope):
    g = _gen(11)
    batch, seq, heads, d = 2, 48, 3, 128
    qkv = torch.randn(batch * seq, 3 * heads * d, generator=g).to(dtype)
    wq = (torch.rand(d, generator=g) + 0.5).to(dtype) if norms else None
    wk = (torch.rand(d, generator=g) + 0.5).to(dtype) if norms else None
    angles = torch.randn(1, seq + 16, d // 2, generator=g)  # longer than seq: padded rotary table
    table = torch.stack([angles.sin(), angles.cos()], dim=-1) if rope else None
    out = qkv.to("mps")
    ops.svdq_qk_norm_rope_(
        out,
        None if wq is None else wq.to("mps"),
        None if wk is None else wk.to("mps"),
        None if table is None else table.to("mps"),
        1e-6,
        1e-6,
        heads,
        seq,
    )

    ref = qkv.double().view(batch, seq, 3, heads, d)
    for part, w in ((0, wq), (1, wk)):
        x = ref[:, :, part]
        if w is not None:
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w.double()
        if table is not None:
            sin = table[0, :seq, :, 0].double()[None, :, None]
            cos = table[0, :seq, :, 1].double()[None, :, None]
            x0, x1 = x[..., 0::2].clone(), x[..., 1::2].clone()
            x = x.clone()
            x[..., 0::2] = x0 * cos - x1 * sin
            x[..., 1::2] = x0 * sin + x1 * cos
        ref[:, :, part] = x
    assert _rel(out, ref.view(batch * seq, -1)) < 1e-2
    # V third is untouched
    torch.testing.assert_close(
        out.view(batch, seq, 3, heads, d)[:, :, 2].cpu(), qkv.view(batch, seq, 3, heads, d)[:, :, 2]
    )
