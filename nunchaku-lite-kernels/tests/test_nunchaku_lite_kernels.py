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
