import pytest
import torch

from nunchaku_lite.ops.gemm import awq_gemm_w4a16_g64_int32


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


def _make_awq_case(m: int, k: int, n: int, dtype: torch.dtype):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(m + k + n)
    codes = torch.randint(0, 16, (n, k), dtype=torch.int32, device=device, generator=generator)
    qweight = _pack_awq_codes(codes).contiguous()
    wscales = (torch.rand(k // 64, n, dtype=torch.float32, device=device, generator=generator) * 0.02 + 0.001).to(dtype)
    wzeros = (-7 * wscales.float()).to(dtype)
    x = (torch.randn(m, k, dtype=torch.float32, device=device, generator=generator) * 0.1).to(dtype)
    return x, qweight, wscales, wzeros


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AWQ GEMM correctness requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("m", "k", "n"),
    [
        (16, 1536, 3072),
        (32, 1536, 9216),
        (64, 4608, 3072),
        (128, 1536, 18432),
        (256, 1536, 3072),
    ],
)
def test_awq_gemm_w4a16_g64_int32_matches_dequantized_reference(dtype, m, k, n):
    if dtype is torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("bf16 AWQ GEMM requires sm80 or newer")

    x, qweight, wscales, wzeros = _make_awq_case(m, k, n, dtype)

    actual = awq_gemm_w4a16_g64_int32(x, qweight, wscales, wzeros)
    weight = _dequantize_awq(qweight, wscales, wzeros).to(device=x.device)
    expected = torch.matmul(x.float(), weight.t()).to(dtype)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
