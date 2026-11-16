#pragma once

// Metal source for the SVDQ MPS kernels, compiled at runtime (first use) by svdq_metal.mm.
// Kept as a string so the same sources work with setup.py and kernel-builder builds.
namespace nunchaku_lite::metal {

static const char* kSvdqMetalSource = R"METAL(// Generated from nunchaku-lite-mlx/nunchaku_lite_mlx/kernels.py. Keep the two in sync.
//
// SVDQ W8A8 / W4A8 kernels for Apple-silicon GPUs with Neural Accelerators (M5 class).
// The int32 accumulator runs through the whole K dimension inside mpp::tensor_ops::matmul2d;
// per-group weight scales are folded into per-channel scales plus integer multipliers, and the
// scales, bias and SVD low-rank branch are applied once in the epilogue.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// dims = {M, N, K, R, has_bias}
template <typename OUT_T, int BM, int BN, int SG, int KC, int SWZ>
[[kernel]] void svdq_gemm_w8a8(
    const device int8_t* a [[buffer(0)]],
    const device int8_t* b [[buffer(1)]],
    const device float* ascale [[buffer(2)]],
    const device float* wsc [[buffer(3)]],
    const device OUT_T* lact [[buffer(4)]],
    const device OUT_T* lup [[buffer(5)]],
    const device float* bias [[buffer(6)]],
    device OUT_T* out [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
    using LR_T = OUT_T;

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
}

template <typename OUT_T, int BM, int BN, int SG, int KC, int SWZ>
[[kernel]] void svdq_gemm_w4a8(
    const device int8_t* a [[buffer(0)]],
    const device uchar* b [[buffer(1)]],
    const device float* ascale [[buffer(2)]],
    const device float* wsc [[buffer(3)]],
    const device OUT_T* lact [[buffer(4)]],
    const device OUT_T* lup [[buffer(5)]],
    const device float* bias [[buffer(6)]],
    device OUT_T* out [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    const device half* wr [[buffer(9)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
    using LR_T = OUT_T;

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
}

// dims = {K}
template <typename T>
[[kernel]] void svdq_quantize_act_int8(
    const device T* x [[buffer(0)]],
    const device float* inv_smooth [[buffer(1)]],
    constant float* shift [[buffer(2)]],
    device int8_t* q [[buffer(3)]],
    device float* ascale [[buffer(4)]],
    constant uint* dims [[buffer(5)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {

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
}

template [[host_name("svdq_gemm_w8a8_half_128_64_4_64_8")]] [[kernel]] void svdq_gemm_w8a8<half, 128, 64, 4, 64, 8>(
    const device int8_t* a [[buffer(0)]],
    const device int8_t* b [[buffer(1)]],
    const device float* ascale [[buffer(2)]],
    const device float* wsc [[buffer(3)]],
    const device half* lact [[buffer(4)]],
    const device half* lup [[buffer(5)]],
    const device float* bias [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]);

template [[host_name("svdq_gemm_w8a8_half_128_128_8_64_1")]] [[kernel]] void svdq_gemm_w8a8<half, 128, 128, 8, 64, 1>(
    const device int8_t* a [[buffer(0)]],
    const device int8_t* b [[buffer(1)]],
    const device float* ascale [[buffer(2)]],
    const device float* wsc [[buffer(3)]],
    const device half* lact [[buffer(4)]],
    const device half* lup [[buffer(5)]],
    const device float* bias [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]);

template [[host_name("svdq_gemm_w8a8_half_64_64_4_64_4")]] [[kernel]] void svdq_gemm_w8a8<half, 64, 64, 4, 64, 4>(
    const device int8_t* a [[buffer(0)]],
    const device int8_t* b [[buffer(1)]],
    const device float* ascale [[buffer(2)]],
    const device float* wsc [[buffer(3)]],
    const device half* lact [[buffer(4)]],
    const device half* lup [[buffer(5)]],
    const device float* bias [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]);

template [[host_name("svdq_gemm_w8a8_bfloat_128_64_4_64_8")]] [[kernel]] void svdq_gemm_w8a8<bfloat, 128, 64, 4, 64, 8>(
    const device int8_t* a [[buffer(0)]],
    const device int8_t* b [[buffer(1)]],
    const device float* ascale [[buffer(2)]],
    const device float* wsc [[buffer(3)]],
    const device bfloat* lact [[buffer(4)]],
    const device bfloat* lup [[buffer(5)]],
    const device float* bias [[buffer(6)]],
    device bfloat* out [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]);

template [[host_name("svdq_gemm_w8a8_bfloat_128_128_8_64_1")]] [[kernel]] void svdq_gemm_w8a8<bfloat, 128, 128, 8, 64, 1>(
    const device int8_t* a [[buffer(0)]],
    const device int8_t* b [[buffer(1)]],
    const device float* ascale [[buffer(2)]],
    const device float* wsc [[buffer(3)]],
    const device bfloat* lact [[buffer(4)]],
    const device bfloat* lup [[buffer(5)]],
    const device float* bias [[buffer(6)]],
    device bfloat* out [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]);

template [[host_name("svdq_gemm_w8a8_bfloat_64_64_4_64_4")]] [[kernel]] void svdq_gemm_w8a8<bfloat, 64, 64, 4, 64, 4>(
    const device int8_t* a [[buffer(0)]],
    const device int8_t* b [[buffer(1)]],
    const device float* ascale [[buffer(2)]],
    const device float* wsc [[buffer(3)]],
    const device bfloat* lact [[buffer(4)]],
    const device bfloat* lup [[buffer(5)]],
    const device float* bias [[buffer(6)]],
    device bfloat* out [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]);

template [[host_name("svdq_gemm_w4a8_half_128_64_4_64_4")]] [[kernel]] void svdq_gemm_w4a8<half, 128, 64, 4, 64, 4>(
    const device int8_t* a [[buffer(0)]],
    const device uchar* b [[buffer(1)]],
    const device float* ascale [[buffer(2)]],
    const device float* wsc [[buffer(3)]],
    const device half* lact [[buffer(4)]],
    const device half* lup [[buffer(5)]],
    const device float* bias [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    const device half* wr [[buffer(9)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]);

template [[host_name("svdq_gemm_w4a8_half_128_128_8_64_4")]] [[kernel]] void svdq_gemm_w4a8<half, 128, 128, 8, 64, 4>(
    const device int8_t* a [[buffer(0)]],
    const device uchar* b [[buffer(1)]],
    const device float* ascale [[buffer(2)]],
    const device float* wsc [[buffer(3)]],
    const device half* lact [[buffer(4)]],
    const device half* lup [[buffer(5)]],
    const device float* bias [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    const device half* wr [[buffer(9)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]);

template [[host_name("svdq_gemm_w4a8_half_64_64_4_64_4")]] [[kernel]] void svdq_gemm_w4a8<half, 64, 64, 4, 64, 4>(
    const device int8_t* a [[buffer(0)]],
    const device uchar* b [[buffer(1)]],
    const device float* ascale [[buffer(2)]],
    const device float* wsc [[buffer(3)]],
    const device half* lact [[buffer(4)]],
    const device half* lup [[buffer(5)]],
    const device float* bias [[buffer(6)]],
    device half* out [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    const device half* wr [[buffer(9)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]);

template [[host_name("svdq_gemm_w4a8_bfloat_128_64_4_64_4")]] [[kernel]] void svdq_gemm_w4a8<bfloat, 128, 64, 4, 64, 4>(
    const device int8_t* a [[buffer(0)]],
    const device uchar* b [[buffer(1)]],
    const device float* ascale [[buffer(2)]],
    const device float* wsc [[buffer(3)]],
    const device bfloat* lact [[buffer(4)]],
    const device bfloat* lup [[buffer(5)]],
    const device float* bias [[buffer(6)]],
    device bfloat* out [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    const device half* wr [[buffer(9)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]);

template [[host_name("svdq_gemm_w4a8_bfloat_128_128_8_64_4")]] [[kernel]] void svdq_gemm_w4a8<bfloat, 128, 128, 8, 64, 4>(
    const device int8_t* a [[buffer(0)]],
    const device uchar* b [[buffer(1)]],
    const device float* ascale [[buffer(2)]],
    const device float* wsc [[buffer(3)]],
    const device bfloat* lact [[buffer(4)]],
    const device bfloat* lup [[buffer(5)]],
    const device float* bias [[buffer(6)]],
    device bfloat* out [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    const device half* wr [[buffer(9)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]);

template [[host_name("svdq_gemm_w4a8_bfloat_64_64_4_64_4")]] [[kernel]] void svdq_gemm_w4a8<bfloat, 64, 64, 4, 64, 4>(
    const device int8_t* a [[buffer(0)]],
    const device uchar* b [[buffer(1)]],
    const device float* ascale [[buffer(2)]],
    const device float* wsc [[buffer(3)]],
    const device bfloat* lact [[buffer(4)]],
    const device bfloat* lup [[buffer(5)]],
    const device float* bias [[buffer(6)]],
    device bfloat* out [[buffer(7)]],
    constant int* dims [[buffer(8)]],
    const device half* wr [[buffer(9)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]);

template [[host_name("svdq_quantize_act_int8_half")]] [[kernel]] void svdq_quantize_act_int8<half>(
    const device half* x [[buffer(0)]], const device float* inv_smooth [[buffer(1)]], constant float* shift [[buffer(2)]],
    device int8_t* q [[buffer(3)]], device float* ascale [[buffer(4)]], constant uint* dims [[buffer(5)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]], uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]);

template [[host_name("svdq_quantize_act_int8_bfloat")]] [[kernel]] void svdq_quantize_act_int8<bfloat>(
    const device bfloat* x [[buffer(0)]], const device float* inv_smooth [[buffer(1)]], constant float* shift [[buffer(2)]],
    device int8_t* q [[buffer(3)]], device float* ascale [[buffer(4)]], constant uint* dims [[buffer(5)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]], uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]);

template [[host_name("svdq_quantize_act_int8_float")]] [[kernel]] void svdq_quantize_act_int8<float>(
    const device float* x [[buffer(0)]], const device float* inv_smooth [[buffer(1)]], constant float* shift [[buffer(2)]],
    device int8_t* q [[buffer(3)]], device float* ascale [[buffer(4)]], constant uint* dims [[buffer(5)]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]], uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]);
)METAL";

}  // namespace nunchaku_lite::metal
