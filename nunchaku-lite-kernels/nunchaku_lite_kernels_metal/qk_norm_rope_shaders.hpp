#pragma once

// In-place per-head RMSNorm + interleaved RoPE on the Q and K thirds of a fused QKV projection
// output, matching the CUDA SVDQ GEMM epilogue (norm_q / norm_k / rotary_emb). Compiled together
// with kSvdqMetalSource by svdq_metal.mm.
namespace nunchaku_lite::metal {

static const char* kQkNormRopeMetalSource = R"METAL(
// qkv: (M, 3 * H * 128), rows are tokens. One simdgroup handles one (token, head, q|k) row of 128
// elements; each lane owns 4 consecutive elements = 2 interleaved RoPE pairs.
// rope: (rope_batch, rope_seq, 64, 2) float32 holding (sin, cos) per pair.
// dims = {M, H, seq_len, rope_batch, rope_seq, has_norm_q, has_norm_k, has_rope}
template <typename T>
[[kernel]] void svdq_qk_norm_rope(
    device T* qkv [[buffer(0)]],
    const device T* norm_q [[buffer(1)]],
    const device T* norm_k [[buffer(2)]],
    const device float* rope [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    constant float* eps [[buffer(5)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgi [[simdgroup_index_in_threadgroup]],
    uint sgs [[simdgroups_per_threadgroup]]) {
    constexpr int D = 128;
    const int M = dims[0], H = dims[1], seq_len = dims[2], rope_batch = dims[3], rope_seq = dims[4];
    const int row = tgp.y;
    const int idx = tgp.x * sgs + sgi;  // 0 .. 2H-1: q heads then k heads
    if (row >= M || idx >= 2 * H) return;
    const bool is_k = idx >= H;
    const int head = is_k ? idx - H : idx;
    device T* x = qkv + (size_t)row * (3 * H * D) + (is_k ? H * D : 0) + head * D + lane * 4;

    float4 v = float4(*(device vec<T, 4>*)x);
    const bool has_norm = is_k ? dims[6] != 0 : dims[5] != 0;
    if (has_norm) {
        const float ss = simd_sum(dot(v, v));
        const float inv = rsqrt(ss / D + (is_k ? eps[1] : eps[0]));
        const float4 w = float4(*(const device vec<T, 4>*)((is_k ? norm_k : norm_q) + lane * 4));
        v *= inv * w;
    }
    if (dims[7] != 0) {
        const int b = row / seq_len, s = row % seq_len;
        if (s < rope_seq) {
            const device float4* sc =
                (const device float4*)(rope + ((size_t)(rope_batch > 1 ? b : 0) * rope_seq + s) * D + lane * 4);
            const float4 r = *sc;  // (sin0, cos0, sin1, cos1) for pairs (v.x, v.y) and (v.z, v.w)
            v = float4(v.x * r.y - v.y * r.x, v.x * r.x + v.y * r.y,
                       v.z * r.w - v.w * r.z, v.z * r.z + v.w * r.w);
        }
    }
    *(device vec<T, 4>*)x = vec<T, 4>(v);
}

template [[host_name("svdq_qk_norm_rope_half")]] [[kernel]] void svdq_qk_norm_rope<half>(
    device half*, const device half*, const device half*, const device float*, constant int*, constant float*,
    uint3, uint, uint, uint);
template [[host_name("svdq_qk_norm_rope_bfloat")]] [[kernel]] void svdq_qk_norm_rope<bfloat>(
    device bfloat*, const device bfloat*, const device bfloat*, const device float*, constant int*, constant float*,
    uint3, uint, uint, uint);
)METAL";

}  // namespace nunchaku_lite::metal
