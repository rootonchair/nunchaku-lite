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
    for name in ("svdq_quantize_act_int8", "svdq_gemm_w8a8", "svdq_gemm_w4a8", "qk_norm_rope_mps_", "attention_mps"):
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
@pytest.mark.parametrize(
    "m,n,k,r", [(256, 256, 512, 32), (128, 384, 1024, 64), (64, 128, 192, 32), (128, 256, 8192, 32)]
)
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


def _norm_rope_ref(x, w, cos, sin, eps):
    """(rows, heads, 128) float64: per-head RMSNorm, then interleaved RoPE with per-pair (rows, 64) tables."""
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * cos[:, None] - x1 * sin[:, None]
    out[..., 1::2] = x0 * sin[:, None] + x1 * cos[:, None]
    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_qk_norm_rope_mps_joint_head_major(dtype):
    """Text + image rows from column slices of fused projections, Diffusers' doubled tables, (L, H, 128) output."""
    g = _gen(21)
    lt, li, heads = 7, 33, 3
    width = heads * 128
    proj_t = torch.randn(lt, 3 * width, generator=g).to(dtype)
    proj_i = torch.randn(li, 3 * width + 64, generator=g).to(dtype)  # padded rows: row stride != 3 * width
    w = [torch.rand(128, generator=g) + 0.5 for _ in range(4)]  # wqt, wkt, wqi, wki
    angles = torch.randn(lt + li, 64, generator=g).double()
    cos = angles.cos().repeat_interleave(2, -1).float()
    sin = angles.sin().repeat_interleave(2, -1).float()
    mps_t, mps_i = proj_t.to("mps"), proj_i.to("mps")
    outs = [torch.empty(lt + li, heads, 128, dtype=dtype, device="mps") for _ in range(3)]
    ops.qk_norm_rope_mps_(
        mps_i[:, :width], mps_i[:, width : 2 * width], mps_i[:, 2 * width : 3 * width],
        mps_t[:, :width], mps_t[:, width : 2 * width], mps_t[:, 2 * width :],
        w[2].to("mps"), w[3].to("mps"), w[0].to("mps"), w[1].to("mps"), cos.to("mps"), sin.to("mps"),
        *outs, 1e-6, 1e-5, heads, lt + li, 2, 128, width,
    )  # fmt: skip
    src = torch.cat([proj_t[:, : 3 * width], proj_i[:, : 3 * width]]).double().view(lt + li, 3, heads, 128)
    for part, (wt, wi, eps) in enumerate(((w[0], w[2], 1e-6), (w[1], w[3], 1e-5))):
        ref = torch.cat(
            [
                _norm_rope_ref(src[:lt, part], wt.double(), angles[:lt].cos(), angles[:lt].sin(), eps),
                _norm_rope_ref(src[lt:, part], wi.double(), angles[lt:].cos(), angles[lt:].sin(), eps),
            ]
        )
        assert _rel(outs[part], ref) < 1e-2
    torch.testing.assert_close(outs[2].cpu(), src[:, 2].to(dtype))


def test_qk_norm_rope_mps_in_place_leaves_v():
    """In place on one fused (rows, 3 * width) buffer with per-pair tables reused across a flattened batch."""
    g = _gen(22)
    batch, seq, heads = 2, 24, 2
    width = heads * 128
    qkv = torch.randn(batch * seq, 3 * width, generator=g).to(torch.bfloat16)
    w = [torch.rand(128, generator=g) + 0.5 for _ in range(2)]
    angles = torch.randn(seq, 64, generator=g).double()
    buf = qkv.to("mps")
    q, k, v = buf[:, :width], buf[:, width : 2 * width], buf[:, 2 * width :]
    wq, wk = w[0].to("mps"), w[1].to("mps")
    ops.qk_norm_rope_mps_(
        q, k, v, None, None, None, wq, wk, wq, wk,
        angles.cos().float().to("mps"), angles.sin().float().to("mps"),
        q, k, v, 1e-6, 1e-6, heads, seq, 1, 128, 3 * width,
    )  # fmt: skip
    src = qkv.double().view(batch * seq, 3, heads, 128)
    cos, sin = angles.cos().repeat(batch, 1), angles.sin().repeat(batch, 1)
    for part in (0, 1):
        ref = _norm_rope_ref(src[:, part], w[part].double(), cos, sin, 1e-6)
        assert _rel(buf.view(batch * seq, 3, heads, 128)[:, part], ref) < 1e-2
    torch.testing.assert_close(buf[:, 2 * width :].cpu(), qkv[:, 2 * width :])


def _has_neural_accelerators() -> bool:
    import subprocess

    result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], check=False, capture_output=True, text=True)
    brand = result.stdout.split()
    return len(brand) > 1 and brand[0] == "Apple" and brand[1][1:].isdigit() and int(brand[1][1:]) >= 5


needs_nax = pytest.mark.skipif(not _has_neural_accelerators(), reason="requires an M5-class GPU")


def _attention_ref(q, k, v, mask, scale):
    q, k, v = (t.double().transpose(1, 2) for t in (q, k, v))  # (B, H, L, D)
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    s = q @ k.transpose(-1, -2) * scale
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else s + mask.double()
    return (s.softmax(-1) @ v).transpose(1, 2)


@needs_nax
@pytest.mark.parametrize(
    "batch,q_len,k_len,heads,kv_heads,dim,dtype",
    [
        (1, 256, 256, 4, 4, 128, torch.bfloat16),  # aligned
        (2, 77, 133, 4, 2, 64, torch.float16),  # unaligned, grouped-query
        (1, 100, 64, 2, 2, 96, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("mask_kind", [None, "bool", "add"])
def test_attention_mps_matches_reference(batch, q_len, k_len, heads, kv_heads, dim, dtype, mask_kind):
    g = _gen(31)
    q = torch.randn(batch, q_len, heads, dim, generator=g).to(dtype)
    k = torch.randn(batch, k_len, kv_heads, dim, generator=g).to(dtype)
    v = torch.randn(batch, k_len, kv_heads, dim, generator=g).to(dtype)
    mask = None
    if mask_kind is not None:
        keep = torch.rand(batch, 1, 1, k_len, generator=g) > 0.3
        keep[..., 0] = True
        mask = keep if mask_kind == "bool" else torch.zeros(keep.shape).masked_fill(~keep, -1e4).to(dtype)
        mask = mask.expand(batch, heads, q_len, k_len)
    scale = dim**-0.5
    out = ops.attention_mps(q.to("mps"), k.to("mps"), v.to("mps"), None if mask is None else mask.to("mps"), scale)
    assert out.shape == q.shape and out.dtype == dtype
    assert _rel(out, _attention_ref(q, k, v, mask, scale)) < 2e-2


@needs_nax
def test_attention_mps_strided_views():
    """q/k/v as head slices of one fused (B, L, 3 * H, D) projection, read in place."""
    g = _gen(32)
    qkv = torch.randn(1, 130, 3 * 4, 128, generator=g).to(torch.bfloat16)
    m = qkv.to("mps")
    out = ops.attention_mps(m[:, :, :4], m[:, :, 4:8], m[:, :, 8:], None, 128**-0.5)
    ref = _attention_ref(qkv[:, :, :4], qkv[:, :, 4:8], qkv[:, :, 8:], None, 128**-0.5)
    assert _rel(out, ref) < 2e-2


def test_attention_mps_fake():
    with FakeTensorMode():
        q = torch.empty(1, 64, 4, 128, device="mps", dtype=torch.bfloat16)
        assert ops.attention_mps(q, q, q, None, 0.1).shape == q.shape
