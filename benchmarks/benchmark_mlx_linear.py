"""Benchmark SVDQW4A4Linear backends against dense MLX projections.

Shapes default to FLUX.1 projections at 1024x1024 (4096 image + 512 text tokens).

    python benchmarks/benchmark_mlx_linear.py --tokens 4608
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

from nunchaku_lite.mlx import SVDQLogicalWeights, SVDQW4A4Linear, has_neural_accelerators, pack_svdq_int4

SHAPES = {
    "qkv_proj": (3072, 9216),
    "mlp_fc1": (3072, 12288),
    "mlp_fc2": (12288, 3072),
    "out_proj": (3072, 3072),
}


def bench(fn, iters: int = 10, reps: int = 5) -> float:
    mx.eval(fn())
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        best = min(best, (time.perf_counter() - start) / iters * 1e3)
    return best


def random_weights(k: int, n: int, rank: int, rng: np.random.Generator) -> SVDQLogicalWeights:
    return SVDQLogicalWeights(
        q=rng.integers(-8, 8, size=(n, k), dtype=np.int8),
        wscale=rng.uniform(0.001, 0.02, size=(n, k // 64)).astype(np.float32),
        smooth=rng.uniform(0.5, 2.0, size=k).astype(np.float32),
        down=(rng.standard_normal((rank, k)) * 0.02).astype(np.float32),
        up=(rng.standard_normal((n, rank)) * 0.02).astype(np.float32),
        bias=rng.standard_normal(n).astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4608)
    parser.add_argument("--rank", type=int, default=32)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    dtype = mx.bfloat16
    backends = ["dense"] + (["int4", "int8"] if has_neural_accelerators() else [])
    print(f"device: {mx.device_info()['device_name']}  tokens={args.tokens}  rank={args.rank}  dtype=bf16")
    print(f"{'layer':10s} {'K->N':>12s} {'bf16 dense':>11s} {'mlx q4':>9s} " + " ".join(f"{b:>15s}" for b in backends))
    totals = {"bf16": 0.0, "q4": 0.0, **{b: 0.0 for b in backends}}
    for name, (k, n) in SHAPES.items():
        x = mx.random.normal((args.tokens, k)).astype(dtype)
        w_dense = (mx.random.normal((n, k)) * 0.02).astype(dtype)
        bias = mx.zeros((n,), dtype=dtype)
        t_dense = bench(lambda x=x, w=w_dense, b=bias: x @ w.T + b)
        wq, sc, bi = mx.quantize(w_dense, group_size=64, bits=4)
        t_q4 = bench(
            lambda x=x, q=(wq, sc, bi), b=bias: mx.quantized_matmul(x, *q, transpose=True, group_size=64, bits=4) + b
        )
        weights = random_weights(k, n, args.rank, rng)
        row = f"{name:10s} {f'{k}->{n}':>12s} {t_dense:9.2f}ms {t_q4:7.2f}ms "
        totals["bf16"] += t_dense
        totals["q4"] += t_q4
        layer = SVDQW4A4Linear(k, n, rank=args.rank, dtype=dtype)
        layer.load_weights([(key, mx.array(v)) for key, v in pack_svdq_int4(weights).items()])
        for backend in backends:
            layer.backend = backend
            t = bench(lambda layer=layer, x=x: layer(x))
            totals[backend] += t
            row += f" {t:7.2f}ms ({t_dense / t:4.2f}x)"
        print(row)
    summary = f"{'total':10s} {'':>12s} {totals['bf16']:9.2f}ms {totals['q4']:7.2f}ms "
    summary += " ".join(f" {totals[b]:7.2f}ms ({totals['bf16'] / totals[b]:4.2f}x)" for b in backends)
    print(summary)


if __name__ == "__main__":
    main()
