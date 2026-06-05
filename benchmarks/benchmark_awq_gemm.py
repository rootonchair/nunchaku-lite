import argparse
import statistics
import time

import torch

from nunchaku_lite.ops.gemm import awq_gemm_w4a16_g64_int32
from nunchaku_lite.ops.gemv import awq_gemv_w4a16_cuda


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


def _make_case(m: int, k: int, n: int, dtype: torch.dtype):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(m + k + n)
    x = torch.randn(m, k, dtype=torch.float32, device=device, generator=generator).to(dtype)
    qweight = _pack_awq_codes(torch.randint(0, 16, (n, k), dtype=torch.int32, device=device, generator=generator))
    wscales = (torch.rand(k // 64, n, dtype=torch.float32, device=device, generator=generator) * 0.02 + 0.001).to(dtype)
    wzeros = (-7 * wscales.float()).to(dtype)
    return x, qweight.contiguous(), wscales.contiguous(), wzeros.contiguous()


def _time_cuda(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return statistics.median(timings)


def _gemv_chunked(x, qweight, wscales, wzeros):
    outputs = []
    for start in range(0, x.shape[0], 8):
        chunk = x[start : start + 8]
        outputs.append(
            awq_gemv_w4a16_cuda(
                chunk,
                qweight,
                wscales,
                wzeros,
                chunk.shape[0],
                wscales.shape[1],
                x.shape[1],
                64,
            )
        )
    return torch.cat(outputs, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AWQ W4A16 g64/int32 GEMM against chunked GEMV.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--m-values",
        type=str,
        default="16,32,64,128,256,512,1024,2048",
        help="Comma-separated flattened M values to benchmark.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    if dtype is torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("bfloat16 benchmark requires sm80 or newer.")

    m_values = tuple(int(value) for value in args.m_values.split(",") if value)
    if not m_values:
        raise RuntimeError("--m-values must contain at least one integer.")

    shapes = [
        (m, k, n)
        for m in m_values
        for k, n in ((1536, 3072), (1536, 9216), (4608, 3072), (1536, 18432))
    ]

    print(f"device={torch.cuda.get_device_name(0)} dtype={dtype}")
    print("| M | K | N | GEMV ms | GEMM ms | speedup |")
    print("| --- | --- | --- | ---: | ---: | ---: |")
    speedups = []
    for m, k, n in shapes:
        x, qweight, wscales, wzeros = _make_case(m, k, n, dtype)
        gemv_ms = _time_cuda(lambda: _gemv_chunked(x, qweight, wscales, wzeros), args.warmup, args.repeat)
        gemm_ms = _time_cuda(lambda: awq_gemm_w4a16_g64_int32(x, qweight, wscales, wzeros), args.warmup, args.repeat)
        speedup = gemv_ms / gemm_ms
        speedups.append(speedup)
        print(f"| {m} | {k} | {n} | {gemv_ms:.4f} | {gemm_ms:.4f} | {speedup:.3f}x |")

    geometric_mean = statistics.geometric_mean(speedups)
    print(f"\ngeomean_speedup={geometric_mean:.3f}x")
    if any(speedup <= 1.0 for speedup in speedups) or geometric_mean < 1.25:
        raise SystemExit("Benchmark gate failed: GEMM must beat chunked GEMV on every shape and geomean >= 1.25x.")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"elapsed={time.perf_counter() - start:.2f}s")
