import platform

import nunchaku_lite_kernels
import pytest
import torch
from nunchaku_lite_kernels._ops import ops
from torch._subclasses.fake_tensor import FakeTensorMode


def _awq_code_order(device: torch.device) -> torch.Tensor:
    order = []
    for packed_index in range(8):
        for nibble in range(8):
            candidates = [
                channel
                for channel in range(64)
                if ((channel // 32) * 4 + (channel % 8) // 2) == packed_index
                and (((channel % 32) // 8) + 4 * (channel % 2)) == nibble
            ]
            if len(candidates) != 1:
                raise RuntimeError("Internal AWQ W4A16 channel order construction failed")
            order.append(candidates[0])
    return torch.tensor(order, dtype=torch.long, device=device)


def _pack_awq_codes(codes: torch.Tensor) -> torch.Tensor:
    out_features, in_features = codes.shape
    groups = in_features // 64
    ordered = codes.view(out_features, groups, 64).index_select(dim=2, index=_awq_code_order(codes.device))
    ordered = ordered.view(out_features, groups, 8, 8)
    packed_groups = torch.zeros((out_features, groups, 8), dtype=torch.int32, device=codes.device)
    for nibble in range(8):
        packed_groups.bitwise_or_((ordered[:, :, :, nibble].bitwise_and(0xF)) << (4 * nibble))
    return packed_groups.view(out_features // 4, 4, groups, 8).permute(0, 2, 1, 3).reshape(out_features // 4, groups * 32)


def _dequantize_awq(qweight: torch.Tensor, wscales: torch.Tensor, wzeros: torch.Tensor) -> torch.Tensor:
    packed = qweight.to(torch.int32).cpu()
    groups, out_features = wscales.shape
    rows = packed.shape[0]
    codes = torch.empty((rows, 4, groups, 64), dtype=torch.float32)
    packed = packed.view(rows, groups, 4, 8)
    for channel in range(64):
        packed_index = (channel // 32) * 4 + (channel % 8) // 2
        nibble = ((channel % 32) // 8) + 4 * (channel % 2)
        codes[:, :, :, channel] = (
            packed[:, :, :, packed_index].bitwise_right_shift(4 * nibble).bitwise_and(0xF).permute(0, 2, 1).float()
        )
    scale = wscales.float().cpu().t().contiguous().view(out_features, groups, 1)
    zeros = wzeros.float().cpu().t().contiguous().view(out_features, groups, 1)
    return (codes.view(out_features, groups, 64) * scale + zeros).view(out_features, groups * 64)


def test_exports_expected_wrappers():
    if platform.system() == "Darwin":
        pytest.skip("CUDA kernels are not available on Darwin")

    for name in nunchaku_lite_kernels.__all__:
        assert callable(getattr(nunchaku_lite_kernels, name))


def test_local_backend_prefers_dispatcher_ops():
    assert ops is torch.ops.nunchaku_lite_kernels
    assert str(ops.gemm_w4a4) == "nunchaku_lite_kernels.gemm_w4a4"
    assert str(ops.quantize_w4a4_act_fuse_lora) == "nunchaku_lite_kernels.quantize_w4a4_act_fuse_lora"
    assert str(ops.fused_cross_head_qk_norm_rope) == "nunchaku_lite_kernels.fused_cross_head_qk_norm_rope"


def test_top_level_dispatcher_aliases():
    assert str(nunchaku_lite_kernels.gemm_w4a4) == "nunchaku_lite_kernels.gemm_w4a4"
    assert (
        str(nunchaku_lite_kernels.quantize_w4a4_act_fuse_lora)
        == "nunchaku_lite_kernels.quantize_w4a4_act_fuse_lora"
    )
    assert (
        str(nunchaku_lite_kernels.fused_cross_head_qk_norm_rope)
        == "nunchaku_lite_kernels.fused_cross_head_qk_norm_rope"
    )


def test_dispatcher_ops_have_fake_kernels():
    with FakeTensorMode():
        q = torch.empty(1, 4, 8, device="cuda", dtype=torch.bfloat16)
        k = torch.empty(1, 4, 8, device="cuda", dtype=torch.bfloat16)
        q_out, k_out = ops.fused_cross_head_qk_norm_rope(
            q, k, None, None, None, None, None, None, 1, 1, 8, 1e-6, False
        )
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        assert q_out.dtype == q.dtype
        assert k_out.dtype == k.dtype

        x = torch.empty(256, 16, device="cuda", dtype=torch.bfloat16)
        output = torch.empty(256, 8, device="cuda", dtype=torch.uint8)
        oscales = torch.empty(1, 256, device="cuda", dtype=torch.float8_e4m3fn)
        lora_down = torch.empty(16, 4, device="cuda", dtype=torch.bfloat16)
        lora_act_out = torch.empty(256, 4, device="cuda", dtype=torch.float32)
        assert (
            ops.quantize_w4a4_act_fuse_lora(x, output, oscales, lora_down, lora_act_out, None, False, True)
            is None
        )
        scale = torch.empty(8, device="cuda", dtype=torch.bfloat16)
        shift = torch.empty(8, device="cuda", dtype=torch.bfloat16)
        modulated = ops.fused_rms_norm_modulate(q, None, scale, shift, 1e-6)
        assert modulated.shape == q.shape
        affine = ops.fused_affine_modulate(q, scale, shift)
        assert affine.shape == q.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AWQ GEMM correctness requires CUDA")
def test_awq_gemm_w4a16_g64_int32_matches_dequantized_reference():
    m, k, n = 16, 1536, 3072
    dtype = torch.float16
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)

    codes = torch.randint(0, 16, (n, k), dtype=torch.int32, device=device, generator=generator)
    qweight = _pack_awq_codes(codes).contiguous()
    wscales = (torch.rand(k // 64, n, dtype=torch.float32, device=device, generator=generator) * 0.02 + 0.001).to(dtype)
    wzeros = (-7 * wscales.float()).to(dtype)
    x = (torch.randn(m, k, dtype=torch.float32, device=device, generator=generator) * 0.1).to(dtype)

    actual = nunchaku_lite_kernels.awq_gemm_w4a16_g64_int32(x, qweight, wscales, wzeros)
    weight = _dequantize_awq(qweight, wscales, wzeros).to(device=x.device)
    expected = torch.matmul(x.float(), weight.t()).to(dtype)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_hadamard_quantize_op_is_exported_from_package():
    assert "svdq_quantize_w4a4_act_hadamard_fuse_lora_cuda" in nunchaku_lite_kernels.__all__
    assert "quantize_w4a4_act_hadamard_fuse_lora" in nunchaku_lite_kernels.__all__
    from nunchaku_lite_kernels import (  # noqa: F401
        quantize_w4a4_act_hadamard_fuse_lora,
        svdq_quantize_w4a4_act_hadamard_fuse_lora_cuda,
    )

    assert hasattr(ops, "quantize_w4a4_act_hadamard_fuse_lora")


def _rope3d_split(head_dim: int) -> tuple[int, int, int]:
    # LTX2VideoVaeRotaryPosEmbed3D: a quarter to T, the rest halved between H and W, all even.
    dim_t = (head_dim // 4) // 2 * 2
    dim_hw = (head_dim - dim_t) // 2
    if dim_hw % 2 != 0:
        dim_t -= 2
        dim_hw = (head_dim - dim_t) // 2
    return dim_t, dim_hw, dim_hw


def _rope3d_tables(split, lengths, device, base=10000.0):
    tables = []
    for dim, length in zip(split, lengths):
        inv = (1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim)).to(torch.float32)
        ang = torch.arange(length, dtype=torch.float32, device=device)[:, None] * inv[None, :]
        tables += [ang.cos().contiguous(), ang.sin().contiguous()]
    return tables


def _rope3d_reference(x, tables, split):
    """The diffusers LTX2VideoVaeRotaryPosEmbed3D rotation, given the same fp32 tables."""
    out, start = [], 0
    for axis, dim in enumerate(split):
        cos, sin = tables[2 * axis], tables[2 * axis + 1]
        shape = [1, 1, 1, 1, 1, cos.shape[1]]
        shape[axis + 1] = cos.shape[0]
        cos, sin = cos.reshape(shape), sin.reshape(shape)
        chunk = x[..., start : start + dim]
        pairs = chunk.reshape(*chunk.shape[:-1], dim // 2, 2)
        even, odd = pairs[..., 0].float(), pairs[..., 1].float()
        rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
        out.append(rotated.reshape(chunk.shape).to(x.dtype))
        start += dim
    return torch.cat(out, dim=-1)


def test_per_head_qk_norm_rope3d_has_fake_kernel():
    with FakeTensorMode():
        q = torch.empty(1, 3, 4, 5, 2, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.empty_like(q)
        t = [torch.empty(n, p, device="cuda") for n, p in ((3, 8), (3, 8), (4, 12), (4, 12), (5, 12), (5, 12))]
        assert ops.per_head_qk_norm_rope3d_(q, k, None, None, *t, 0.125, 1e-6) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="per_head_qk_norm_rope3d_ correctness requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim,heads,grid", [(64, 4, (5, 6, 7)), (64, 2, (9, 3, 11)), (128, 2, (3, 4, 5))])
@pytest.mark.parametrize("with_weight", [True, False])
def test_per_head_qk_norm_rope3d_matches_eager_chain(dtype, head_dim, heads, grid, with_weight):
    torch.manual_seed(0)
    device = torch.device("cuda")
    t, h, w = grid
    split = _rope3d_split(head_dim)
    tables = _rope3d_tables(split, grid, device)
    q = torch.randn(2, t, h, w, heads, head_dim, device=device, dtype=dtype) * 3
    k = torch.randn_like(q) * 0.5
    norm_q = torch.nn.RMSNorm(head_dim, eps=1e-6, elementwise_affine=with_weight).to(device, dtype)
    norm_k = torch.nn.RMSNorm(head_dim, eps=1e-6, elementwise_affine=with_weight).to(device, dtype)
    if with_weight:
        with torch.no_grad():
            norm_q.weight.uniform_(0.5, 1.5)
            norm_k.weight.uniform_(0.5, 1.5)
    scale = head_dim**-0.5
    with torch.no_grad():
        ref_q = _rope3d_reference(norm_q(q) * scale, tables, split)
        ref_k = _rope3d_reference(norm_k(k), tables, split)
    got_q, got_k = q.clone(), k.clone()
    wq = norm_q.weight.detach() if with_weight else None
    wk = norm_k.weight.detach() if with_weight else None
    assert ops.per_head_qk_norm_rope3d_(got_q, got_k, wq, wk, *tables, scale, 1e-6) is None
    for got, ref in ((got_q, ref_q), (got_k, ref_k)):
        diff = (got.float() - ref.float()).abs()
        # Only the sum-of-squares reduction order differs from eager PyTorch: at most one-ulp flips, rarely.
        ulp = ref.float().abs().clamp_min(1e-3) * (2**-7 if dtype == torch.bfloat16 else 2**-10)
        assert (diff <= ulp * 1.01).all(), diff.max()
        assert (diff == 0).float().mean() > 0.97


@pytest.mark.skipif(not torch.cuda.is_available(), reason="per_head_qk_norm_rope3d_ correctness requires CUDA")
def test_per_head_qk_norm_rope3d_axis_order_matters():
    # Mutation guard: tables for swapped H/W axes must not reproduce the reference.
    device = torch.device("cuda")
    grid, split = (4, 5, 7), _rope3d_split(64)
    q = torch.randn(1, *grid, 2, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    good = _rope3d_tables(split, grid, device)
    ref = _rope3d_reference(torch.nn.functional.rms_norm(q, (64,), eps=1e-6), good, split)
    bad = good[:2] + good[4:] + good[2:4]
    got = q.clone()
    with pytest.raises(RuntimeError):  # shape check rejects tables for the wrong axis lengths
        ops.per_head_qk_norm_rope3d_(got, k.clone(), None, None, *bad, 1.0, 1e-6)
    ops.per_head_qk_norm_rope3d_(got, k.clone(), None, None, *good, 1.0, 1e-6)
    assert (got.float() - ref.float()).abs().max() < 0.05
