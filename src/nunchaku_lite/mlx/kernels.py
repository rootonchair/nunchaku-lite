"""Metal kernels for SVDQ linear layers on Apple-silicon Neural Accelerators.

Apple's M5-generation GPUs expose Neural Accelerators through Metal 4
``mpp::tensor_ops::matmul2d``. On these GPUs, int8 x int8 -> int32 matmuls run
at roughly 2x the fp16 rate. Any fp16/bf16 operand drops back to the fp16 rate.

Nunchaku SVDQ checkpoints store 4-bit weights with one scale per 64 input
channels. Reading the int32 accumulator back once per 64-channel group
serialises the accelerator and costs more than the int8 speedup, so these
kernels never rescale inside the K loop:

* Each output channel gets one float scale ``wsc[n]``. Each 64-channel group
  becomes an integer multiplier ratio ``wr[g, n] = wscale[g, n] / wsc[n]``
  (at most 127 / 8). The int8 weight is ``rint(q * wr)``, so ``|w8| <= 127``.
* Activations get one int8 scale per token, after SVDQ smoothing.
* The int32 accumulator runs through the whole of K inside the accelerator.
  The scales and bias are applied once, in the epilogue. The SVD low-rank
  branch runs as a second accelerator matmul (K = rank) whose accumulator
  shares the main accumulator's layout.

Two weight storage modes are provided:

``"int8"``
    Weights are expanded to int8 when the layer is loaded. Both operands
    stream straight from device memory into one matmul over the whole of K.
    This is the fastest mode (about 2x the MLX fp16 matmul on an M5) and uses
    half the memory of bf16.
``"int4"``
    Weights stay packed at 4 bits, the same footprint as the Nunchaku
    checkpoint. Each K step is expanded to int8 in threadgroup memory, with
    double buffering.
"""

from __future__ import annotations

import functools
import re

import mlx.core as mx

__all__ = [
    "TILE_M",
    "default_tile",
    "has_neural_accelerators",
    "qk_norm_rope",
    "svdq_gemm_w4a8",
    "svdq_gemm_w8a8",
    "svdq_quantize_act_int8",
]

_HEADER = """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

_PRE = r"""
    const int M = dims[0], N = dims[1], K = dims[2], R = dims[3], has_bias = dims[4];
    // Threadgroup swizzle: walk SWZ tile-rows at a time so concurrently running
    // threadgroups share weight (B) tiles in the L2 cache.
    const int tiles_n = N / BN, tiles_m = M / BM;
    int tile_m = threadgroup_position_in_grid.y, tile_n = threadgroup_position_in_grid.x;
    if (SWZ > 1) {
        const int lin = tile_m * tiles_n + tile_n;
        const int first_m = (lin / (SWZ * tiles_n)) * SWZ;
        const int group_m = min(SWZ, tiles_m - first_m);
        const int in_group = lin % (SWZ * tiles_n);
        tile_m = first_m + in_group % group_m;
        tile_n = in_group / group_m;
    }
    const int m0 = tile_m * BM, n0 = tile_n * BN;
    const uint tid = thread_index_in_threadgroup;
    constexpr int NT = 32 * SG;
    constexpr int CAP = BM * BN / NT;
"""

# Epilogue shared by both kernels. ``acc`` is the int32 cooperative accumulator.
# Computes out = acc * ascale[m] * wsc[n] + bias[n] + lact[m, :] . lup[n, :]
_EPILOGUE = r"""
    // Scheduling fence. Without it the compiler may move epilogue work (scale/bias loads)
    // up against the cooperative matmul. Whether it does depends on incidental code shape,
    // and costs up to ~35% throughput (measured 29.5 vs 17-20 TOPS on M5). A simdgroup
    // barrier or a volatile compiler barrier does not help; a threadgroup barrier does.
    threadgroup_barrier(mem_flags::mem_none);
    auto outc = op.template get_destination_cooperative_tensor<TA, TB, OUT_T>();
    float vals[CAP];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < CAP; ++i) {
        auto idx = acc.get_multidimensional_index(i);
        const int n = n0 + idx[0], m = m0 + idx[1];
        float v = float(acc[i]) * ascale[m] * wsc[n];
        if (has_bias) v += bias[n];
        vals[i] = v;
    }
    // Low-rank branch as a second Neural Accelerator matmul: lact (M, R) @ lup (N, R)^T.
    // Its destination uses the same per-thread layout as ``acc``, so it can be added element-wise.
    if (R > 0) {
        constexpr auto ldesc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, true,
                                                   matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<ldesc, execution_simdgroups<SG>> lop;
        auto tL = tensor<device LR_T, dextents<int32_t, 2>, tensor_inline>((device LR_T*)lact, dextents<int32_t, 2>(R, M));
        auto tU = tensor<device LR_T, dextents<int32_t, 2>, tensor_inline>((device LR_T*)lup, dextents<int32_t, 2>(R, N));
        auto mL = tL.slice(0, m0);
        auto mU = tU.slice(0, n0);
        auto lr = lop.template get_destination_cooperative_tensor<decltype(mL), decltype(mU), float>();
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < CAP; ++i) lr[i] = 0.f;
        lop.run(mL, mU, lr);
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < CAP; ++i) vals[i] += lr[i];
    }
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < CAP; ++i) outc[i] = OUT_T(vals[i]);
    auto tC = tensor<device OUT_T, dextents<int32_t, 2>, tensor_inline>((device OUT_T*)out, dextents<int32_t, 2>(N, M));
    auto mC = tC.slice(n0, m0);
    outc.store(mC);
"""

# int8 weights: both operands stream from device memory into one matmul over K.
_SRC_W8 = (
    _PRE
    + r"""
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, true,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto tAf = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>((device int8_t*)a, dextents<int32_t, 2>(K, M));
    auto tBf = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>((device int8_t*)b, dextents<int32_t, 2>(K, N));
    auto mA = tAf.slice(0, m0);
    auto mB = tBf.slice(0, n0);
    using TA = decltype(mA);
    using TB = decltype(mB);
    auto acc = op.template get_destination_cooperative_tensor<TA, TB, int32_t>();
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < CAP; ++i) acc[i] = 0;
    op.run(mA, mB, acc);
"""
    + _EPILOGUE
)

# Packed int4 weights (low nibble = even k), expanded to int8 in threadgroup memory.
_SRC_W4 = (
    _PRE
    + r"""
    constexpr int GPK = KC / 64;
    threadgroup int8_t Bs0[2 * BN * KC];
    constexpr int BV = KC / 32;
    constexpr int B_PER = (BN * BV + NT - 1) / NT;
    uint4 rb[B_PER];
    float rm[B_PER];
    auto gload = [&](int ks) {
        const int k0 = ks * KC;
        const device uint4* gb = (const device uint4*)(b + ((size_t)n0 * K + k0) / 2);
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < B_PER; ++j) {
            const int i = tid + j * NT;
            if (BN * BV % NT == 0 || i < BN * BV) {
                const int r = i / BV, c = i % BV;
                rb[j] = gb[(size_t)r * (K / 32) + c];
                rm[j] = float(wr[(size_t)(ks * GPK + (c * 32) / 64) * N + n0 + r]);
            }
        }
    };
    auto sstore = [&](int buf) {
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < B_PER; ++j) {
            const int i = tid + j * NT;
            if (BN * BV % NT == 0 || i < BN * BV) {
                const int r = i / BV, c = i % BV;
                const float mul = rm[j];
                threadgroup char4* dst = (threadgroup char4*)(Bs0 + buf * BN * KC + r * KC + c * 32);
                const uint words[4] = {rb[j].x, rb[j].y, rb[j].z, rb[j].w};
                _Pragma("clang loop unroll(full)")
                for (int w = 0; w < 4; ++w) {
                    const uint v = words[w];
                    _Pragma("clang loop unroll(full)")
                    for (int h = 0; h < 2; ++h) {
                        char4 o;
                        _Pragma("clang loop unroll(full)")
                        for (int t = 0; t < 4; ++t) {
                            const int q = int((v >> (4 * (h * 4 + t))) & 0xFu);
                            o[t] = (char)rint(float((q ^ 8) - 8) * mul);
                        }
                        dst[w * 2 + h] = o;
                    }
                }
            }
        }
    };
    constexpr auto desc = matmul2d_descriptor(BM, BN, KC, false, true, true,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    using TA = tensor<device int8_t, extents<int32_t, KC, BM>, tensor_inline>;
    using TB = tensor<threadgroup int8_t, extents<int32_t, KC, BN>, tensor_inline>;
    auto acc = op.template get_destination_cooperative_tensor<TA, TB, int32_t>();
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < CAP; ++i) acc[i] = 0;
    const int KS = K / KC;
    gload(0);
    sstore(0);
    for (int ks = 0; ks < KS; ++ks) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ks + 1 < KS) gload(ks + 1);
        TA gA((device int8_t*)a + (size_t)m0 * K + ks * KC, extents<int32_t, KC, BM>(), array<int32_t, 2>{1, K});
        TB gB(Bs0 + (ks & 1) * BN * KC, extents<int32_t, KC, BN>());
        op.run(gA, gB, acc);
        if (ks + 1 < KS) sstore((ks + 1) & 1);
    }
"""
    + _EPILOGUE
)

# Per-token int8 activation quantization of (x + shift) * inv_smooth: one threadgroup per row.
_SRC_QUANT = r"""
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint K = dims[0];
    constexpr uint NT = 256;
    const device T* xr = x + (size_t)row * K;
    const float sh = shift[0];
    float amax = 0.f;
    for (uint i = tid * 4; i < K; i += NT * 4) {
        const vec<T, 4> v = *(const device vec<T, 4>*)(xr + i);
        const float4 s = (float4(v) + sh) * *(const device float4*)(inv_smooth + i);
        amax = max(amax, max(max(fabs(s.x), fabs(s.y)), max(fabs(s.z), fabs(s.w))));
    }
    amax = simd_max(amax);
    threadgroup float red[NT / 32];
    if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = red[0];
    for (uint i = 1; i < NT / 32; ++i) amax = max(amax, red[i]);
    const float scale = max(amax, 1e-8f) / 127.f;
    const float inv = 1.f / scale;
    if (tid == 0) ascale[row] = scale;
    device char4* qr = (device char4*)(q + (size_t)row * K);
    for (uint i = tid * 4; i < K; i += NT * 4) {
        const vec<T, 4> v = *(const device vec<T, 4>*)(xr + i);
        const float4 s = (float4(v) + sh) * *(const device float4*)(inv_smooth + i);
        qr[i / 4] = char4(clamp(rint(s * inv), -127.f, 127.f));
    }
"""

# Attention prologue for FLUX.2-style blocks: per-head RMSNorm of q and k, interleaved-pair RoPE, and the
# (tokens, heads * 128) -> (1, heads, tokens, 128) layout change, in one pass over q/k/v. One simdgroup per
# (token, head): 32 lanes x 4 channels. Text rows (if any) come first, then image rows (joint attention).
_SRC_QK_NORM_ROPE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint head = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.y;
    const int Lt = dims[0], L = dims[1], st_t = dims[2], st_i = dims[3], qo = dims[4], ko = dims[5], vo = dims[6];
    const float eps = epsb[0];
    const bool txt = int(row) < Lt;
    const device T* src = txt ? xt + (size_t)row * st_t : xi + (size_t)(row - Lt) * st_i;
    const device float* wq = txt ? wqt : wqi;
    const device float* wk = txt ? wkt : wki;
    const uint c = head * 128 + lane * 4;
    const size_t o = ((size_t)head * L + row) * 128 + lane * 4;
    const float2 cs = *(const device float2*)(cosb + (size_t)row * 64 + lane * 2);
    const float2 sn = *(const device float2*)(sinb + (size_t)row * 64 + lane * 2);
    float4 q = float4(*(const device vec<T, 4>*)(src + qo + c));
    float4 k = float4(*(const device vec<T, 4>*)(src + ko + c));
    q *= rsqrt(simd_sum(dot(q, q)) / 128.f + eps) * *(const device float4*)(wq + lane * 4);
    k *= rsqrt(simd_sum(dot(k, k)) / 128.f + eps) * *(const device float4*)(wk + lane * 4);
    const float4 qr = float4(q.x * cs.x - q.y * sn.x, q.y * cs.x + q.x * sn.x,
                             q.z * cs.y - q.w * sn.y, q.w * cs.y + q.z * sn.y);
    const float4 kr = float4(k.x * cs.x - k.y * sn.x, k.y * cs.x + k.x * sn.x,
                             k.z * cs.y - k.w * sn.y, k.w * cs.y + k.z * sn.y);
    *(device vec<T, 4>*)(q_out + o) = vec<T, 4>(qr);
    *(device vec<T, 4>*)(k_out + o) = vec<T, 4>(kr);
    *(device vec<T, 4>*)(v_out + o) = *(const device vec<T, 4>*)(src + vo + c);
"""

_OUT_TYPES = {mx.float16: "half", mx.bfloat16: "bfloat", mx.float32: "float"}


@functools.cache
def _quant_kernel():
    return mx.fast.metal_kernel(
        name="nunchaku_lite_quantize_act_int8",
        input_names=["x", "inv_smooth", "shift", "dims"],
        output_names=["q", "ascale"],
        source=_SRC_QUANT,
    )


def svdq_quantize_act_int8(x: mx.array, inv_smooth: mx.array, shift: mx.array) -> tuple[mx.array, mx.array]:
    """Per-token symmetric int8 quantization of ``(x + shift) * inv_smooth`` (same op as the PyTorch MPS backend).

    Args:
        x: Activations ``(M, K)`` in float16, bfloat16 or float32. ``K % 4 == 0``.
        inv_smooth: float32 ``(K,)`` reciprocal SVDQ smoothing factors.
        shift: float32 ``(1,)`` additive input shift.

    Returns:
        ``(q, scale)``: int8 ``(M, K)`` and float32 ``(M,)`` with ``x ~= q * scale``.
    """

    m, k = x.shape
    if k % 4:
        raise ValueError(f"Activation width must be a multiple of 4, got {k}.")
    return _quant_kernel()(
        inputs=[x, inv_smooth, shift, mx.array([k], dtype=mx.uint32)],
        template=[("T", x.dtype)],
        grid=(256, m, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, k), (m,)],
        output_dtypes=[mx.int8, mx.float32],
    )


@functools.cache
def _qk_norm_rope_kernel():
    return mx.fast.metal_kernel(
        name="nunchaku_lite_qk_norm_rope",
        input_names=["xt", "xi", "wqt", "wkt", "wqi", "wki", "cosb", "sinb", "epsb", "dims"],
        output_names=["q_out", "k_out", "v_out"],
        source=_SRC_QK_NORM_ROPE,
    )


def qk_norm_rope(
    x: mx.array,
    offsets: tuple[int, int, int],
    q_weight: mx.array,
    k_weight: mx.array,
    cos: mx.array,
    sin: mx.array,
    heads: int,
    *,
    eps: float = 1e-5,
    text: mx.array | None = None,
    text_q_weight: mx.array | None = None,
    text_k_weight: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Fused attention prologue: q/k RMSNorm, RoPE and head-major layout (head dim 128).

    Reads q, k and v straight from a projection output (no slicing copies) and returns ``(1, heads, L, 128)``
    tensors ready for ``mx.fast.scaled_dot_product_attention``. Norm and rotation run in float32; the result is
    rounded once to ``x.dtype``.

    Args:
        x: Image-token rows ``(Li, C)`` holding q, k and v at column ``offsets`` (each ``heads * 128`` wide).
        offsets: Column offsets of q, k and v in ``x`` (and in ``text``).
        q_weight, k_weight: RMSNorm weights ``(128,)`` for the image rows.
        cos, sin: Rotary tables ``(L, 64)`` for all ``L = Lt + Li`` rows (text first); pair ``i`` is channels
            ``2i, 2i + 1``.
        heads: Number of attention heads.
        eps: RMSNorm epsilon.
        text: Optional text-token rows ``(Lt, C')``, same offsets, placed before the image rows.
        text_q_weight, text_k_weight: RMSNorm weights for the text rows.
    """

    li, st_i = x.shape
    if text is None:
        text, text_q_weight, text_k_weight, lt, st_t = x, q_weight, k_weight, 0, st_i
    else:
        lt, st_t = text.shape
    total = lt + li
    width = heads * 128
    if text.dtype != x.dtype or x.dtype not in _OUT_TYPES:
        raise ValueError(f"Unsupported activation dtypes {text.dtype} / {x.dtype}.")
    if max(offsets) + width > min(st_i, st_t) or min(offsets) < 0 or any(o % 4 for o in offsets):
        raise ValueError(f"q/k/v offsets {offsets} do not fit rows of width {min(st_i, st_t)} (heads={heads}).")
    if cos.shape != (total, 64) or sin.shape != (total, 64):
        raise ValueError(f"Rotary tables must be ({total}, 64), got {cos.shape} and {sin.shape}.")
    group = next(g for g in (8, 4, 2, 1) if heads % g == 0)
    f32 = lambda t: t.astype(mx.float32)
    dims = mx.array([lt, total, st_t, st_i, *offsets], dtype=mx.int32)
    shape = (1, heads, total, 128)
    return tuple(
        _qk_norm_rope_kernel()(
            inputs=[
                text,
                x,
                f32(text_q_weight),
                f32(text_k_weight),
                f32(q_weight),
                f32(k_weight),
                f32(cos),
                f32(sin),
                mx.array([eps], dtype=mx.float32),
                dims,
            ],
            template=[("T", x.dtype)],
            grid=(heads * 32, total, 1),
            threadgroup=(32 * group, 1, 1),
            output_shapes=[shape] * 3,
            output_dtypes=[x.dtype] * 3,
        )
    )


_INPUTS = ["a", "b", "wr", "ascale", "wsc", "lact", "lup", "bias", "dims"]


@functools.cache
def has_neural_accelerators() -> bool:
    """Return whether the default Metal GPU has M5-class Neural Accelerators."""

    if not mx.metal.is_available():
        return False
    arch = mx.device_info().get("architecture", "")
    match = re.match(r"applegpu_g(\d+)", arch)
    return bool(match) and int(match.group(1)) >= 17


@functools.cache
def _kernel(kind: str, bm: int, bn: int, sg: int, kc: int, swz: int, out_t: str, lr_t: str):
    source = _SRC_W8 if kind == "w8" else _SRC_W4
    return mx.fast.metal_kernel(
        name=f"nunchaku_lite_svdq_{kind}a8_{bm}_{bn}_{sg}_{kc}_{swz}_{out_t}_{lr_t}",
        input_names=_INPUTS,
        output_names=["out"],
        source=(
            f"constexpr int BM = {bm}, BN = {bn}, SG = {sg}, KC = {kc}, SWZ = {swz};\n"
            f"using OUT_T = {out_t};\nusing LR_T = {lr_t};\n" + source
        ),
        header=_HEADER,
    )


TILE_M = 128
"""Row tile used by every default tile; callers pad tokens to a multiple of this."""


def default_tile(kind: str, m: int, n: int, k: int) -> tuple[int, int, int, int, int]:
    """Return ``(BM, BN, simdgroups, KC, swizzle)`` tuned on an M5 (AC power).

    Long-K projections (for example FLUX MLP ``fc2``, K = 12288) favour 128x128
    tiles. Everything else favours 128x64 tiles with threadgroup swizzling.
    """

    if m % TILE_M:
        return (64, 64, 4, 64, 4)
    if k >= 8192 and n % 128 == 0:
        return (TILE_M, 128, 8, 32, 2) if kind == "w8" else (TILE_M, 128, 8, 64, 4)
    return (TILE_M, 64, 4, 64, 8 if kind == "w8" else 4)


def _launch(kind, a, b, wr, ascale, wsc, lact, lup, bias, n, tile, out_dtype):
    m, k = a.shape
    bm, bn, sg, kc, swz = (*tile, 1) if len(tile) == 4 else tile
    if m % bm or n % bn or k % 64 or (kind == "w4" and k % kc):
        raise ValueError(f"SVDQ kernel shape M={m}, N={n}, K={k} is not aligned to tile {tile}.")
    rank = lup.shape[1]
    if rank % 32:
        raise ValueError(f"Low-rank dimension must be a multiple of 32, got {rank}.")
    if out_dtype not in _OUT_TYPES:
        raise ValueError(f"Unsupported output dtype {out_dtype}.")
    if lact.dtype != lup.dtype or lact.dtype not in (mx.float16, mx.bfloat16):
        raise ValueError(f"lact and lup must share dtype float16 or bfloat16, got {lact.dtype} and {lup.dtype}.")
    has_bias = bias is not None
    if bias is None:
        bias = mx.zeros((1,), dtype=mx.float32)
    if wr is None:
        wr = mx.zeros((1,), dtype=mx.float16)
    dims = mx.array([m, n, k, rank, int(has_bias)], dtype=mx.int32)
    kernel = _kernel(kind, bm, bn, sg, kc, swz, _OUT_TYPES[out_dtype], _OUT_TYPES[lact.dtype])
    return kernel(
        inputs=[a, b, wr, ascale, wsc, lact, lup, bias, dims],
        grid=((n // bn) * 32 * sg, m // bm, 1),
        threadgroup=(32 * sg, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[out_dtype],
    )[0]


def svdq_gemm_w8a8(
    a: mx.array,
    w8: mx.array,
    ascale: mx.array,
    wsc: mx.array,
    lact: mx.array,
    lup: mx.array,
    bias: mx.array | None,
    *,
    out_dtype=mx.float16,
    tile: tuple[int, ...] | None = None,
) -> mx.array:
    """Fused SVDQ projection with int8 weights.

    Args:
        a: int8 activations, shape ``(M, K)``.
        w8: int8 weights with group multipliers folded in, shape ``(N, K)``.
        ascale: float32 per-token activation scales, shape ``(M,)``.
        wsc: float32 per-channel weight scales, shape ``(N,)``.
        lact: Low-rank activations ``x @ proj_down``, shape ``(M, R)``, float16 or bfloat16.
        lup: Low-rank up projection, shape ``(N, R)``, same dtype as ``lact``.
        bias: Optional float32 bias, shape ``(N,)``.
        out_dtype: Output dtype.
        tile: Optional ``(BM, BN, simdgroups, KC[, swizzle])`` override.
    """

    n = w8.shape[0]
    if tile is None:
        tile = default_tile("w8", a.shape[0], n, a.shape[1])
    return _launch("w8", a, w8, None, ascale, wsc, lact, lup, bias, n, tile, out_dtype)


def svdq_gemm_w4a8(
    a: mx.array,
    w4: mx.array,
    wr: mx.array,
    ascale: mx.array,
    wsc: mx.array,
    lact: mx.array,
    lup: mx.array,
    bias: mx.array | None,
    *,
    out_dtype=mx.float16,
    tile: tuple[int, ...] | None = None,
) -> mx.array:
    """Fused SVDQ projection with packed int4 weights.

    Args:
        a: int8 activations, shape ``(M, K)``.
        w4: uint8 packed signed int4 weights, low nibble first, shape ``(N, K // 2)``.
        wr: float16 per-group integer multiplier ratios, shape ``(K // 64, N)``.
        ascale: float32 per-token activation scales, shape ``(M,)``.
        wsc: float32 per-channel weight scales, shape ``(N,)``.
        lact: Low-rank activations, shape ``(M, R)``, float16 or bfloat16.
        lup: Low-rank up projection, shape ``(N, R)``, same dtype as ``lact``.
        bias: Optional float32 bias, shape ``(N,)``.
        out_dtype: Output dtype.
        tile: Optional ``(BM, BN, simdgroups, KC[, swizzle])`` override.
    """

    n = w4.shape[0]
    if tile is None:
        tile = default_tile("w4", a.shape[0], n, a.shape[1])
    return _launch("w4", a, w4, wr, ascale, wsc, lact, lup, bias, n, tile, out_dtype)
