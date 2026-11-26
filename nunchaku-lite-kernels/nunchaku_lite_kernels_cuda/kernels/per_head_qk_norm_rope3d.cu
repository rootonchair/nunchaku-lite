// Fused per-head Q/K RMSNorm + scale + axis-factorized 3D RoPE, in place.
//
// Built for the LTX-2.5 diffusion decoder's neighborhood attention (diffusers
// LTX2VideoVaeNeighborhoodAttention.project_qkv), where the unfused chain is
// norm_q / norm_k (per head) -> query * scale -> LTX2VideoVaeRotaryPosEmbed3D.
//
// q, k: (B, T, H, W, heads, head_dim), contiguous, fp16/bf16, updated in place.
// head_dim is split into consecutive (t, h, w) chunks of rotation pairs; pair j
// of chunk a is rotated by angle pos_a * inv_freq_a[j] with interleaved
// (2j, 2j+1) layout: (e, o) -> (e*cos - o*sin, e*sin + o*cos).
// cos_t/sin_t: (T, dt/2), cos_h/sin_h: (H, dh/2), cos_w/sin_w: (W, dw/2), fp32,
// computed by the caller exactly as the diffusers RoPE does (fp32 angles).
//
// Rounding mirrors the eager diffusers chain: the normalized value is rounded
// to the storage dtype, the query is multiplied by `q_scale` and rounded again,
// and the rotation runs in fp32 with separate (non-contracted) multiplies and
// add/sub before one final rounding. The only expected difference from eager
// PyTorch is the order of the sum-of-squares reduction.
//
// One warp per (token, head); lane l owns pairs l, l + 32, ... (head_dim <= 256).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <optional>

namespace nunchaku_lite::ops {
namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kMaxPairsPerLane = 4;  // head_dim <= 256

template <typename T>
__device__ __forceinline__ float to_f(T v);
template <>
__device__ __forceinline__ float to_f<c10::BFloat16>(c10::BFloat16 v) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(&v));
}
template <>
__device__ __forceinline__ float to_f<c10::Half>(c10::Half v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

template <typename T>
__device__ __forceinline__ T from_f(float v);
template <>
__device__ __forceinline__ c10::BFloat16 from_f<c10::BFloat16>(float v) {
    __nv_bfloat16 r = __float2bfloat16(v);
    return *reinterpret_cast<c10::BFloat16 *>(&r);
}
template <>
__device__ __forceinline__ c10::Half from_f<c10::Half>(float v) {
    __half r = __float2half_rn(v);
    return *reinterpret_cast<c10::Half *>(&r);
}

template <typename T>
__device__ __forceinline__ float round_trip(float v) {
    return to_f<T>(from_f<T>(v));
}

template <typename scalar_t, typename weight_t>
__global__ void per_head_qk_norm_rope3d_kernel(scalar_t *__restrict__ q,
                                               scalar_t *__restrict__ k,
                                               const weight_t *__restrict__ q_weight,
                                               const weight_t *__restrict__ k_weight,
                                               const float *__restrict__ cos_t,
                                               const float *__restrict__ sin_t,
                                               const float *__restrict__ cos_h,
                                               const float *__restrict__ sin_h,
                                               const float *__restrict__ cos_w,
                                               const float *__restrict__ sin_w,
                                               int64_t rows,  // B*T*H*W*heads
                                               int64_t T,
                                               int64_t H,
                                               int64_t W,
                                               int64_t heads,
                                               int head_dim,
                                               int pairs_t,
                                               int pairs_h,
                                               float q_scale,
                                               float eps) {
    const bool is_q = blockIdx.y == 0;
    const int lane = threadIdx.x & 31;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    if (row >= rows) {
        return;
    }
    scalar_t *x = (is_q ? q : k) + row * head_dim;
    const weight_t *wt = is_q ? q_weight : k_weight;
    const int pairs = head_dim / 2;

    float e[kMaxPairsPerLane], o[kMaxPairsPerLane];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kMaxPairsPerLane; ++i) {
        const int p = lane + 32 * i;
        e[i] = o[i] = 0.0f;
        if (p < pairs) {
            e[i] = to_f(x[2 * p]);
            o[i] = to_f(x[2 * p + 1]);
            sum += e[i] * e[i] + o[i] * o[i];
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    const float inv_rms = rsqrtf(sum / static_cast<float>(head_dim) + eps);

    // (b, t, h, w, head) from the row index; rows are laid out (B, T, H, W, heads).
    const int64_t token = row / heads;
    const int64_t w_idx = token % W;
    const int64_t h_idx = (token / W) % H;
    const int64_t t_idx = (token / (W * H)) % T;

#pragma unroll
    for (int i = 0; i < kMaxPairsPerLane; ++i) {
        const int p = lane + 32 * i;
        if (p >= pairs) {
            continue;
        }
        float w0 = 1.0f, w1 = 1.0f;
        if (wt != nullptr) {
            w0 = static_cast<float>(wt[2 * p]);
            w1 = static_cast<float>(wt[2 * p + 1]);
        }
        float v0 = round_trip<scalar_t>(e[i] * inv_rms * w0);
        float v1 = round_trip<scalar_t>(o[i] * inv_rms * w1);
        if (is_q) {
            v0 = round_trip<scalar_t>(v0 * q_scale);
            v1 = round_trip<scalar_t>(v1 * q_scale);
        }
        float c, s;
        if (p < pairs_t) {
            c = cos_t[t_idx * pairs_t + p];
            s = sin_t[t_idx * pairs_t + p];
        } else if (p < pairs_t + pairs_h) {
            const int j = p - pairs_t;
            c = cos_h[h_idx * pairs_h + j];
            s = sin_h[h_idx * pairs_h + j];
        } else {
            const int pairs_w = pairs - pairs_t - pairs_h;
            const int j = p - pairs_t - pairs_h;
            c = cos_w[w_idx * pairs_w + j];
            s = sin_w[w_idx * pairs_w + j];
        }
        x[2 * p] = from_f<scalar_t>(__fsub_rn(__fmul_rn(v0, c), __fmul_rn(v1, s)));
        x[2 * p + 1] = from_f<scalar_t>(__fadd_rn(__fmul_rn(v0, s), __fmul_rn(v1, c)));
    }
}

void check_table(const torch::Tensor &t, int64_t len, int64_t pairs, const char *name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous(), name,
                " must be a contiguous fp32 CUDA tensor");
    TORCH_CHECK(t.dim() == 2 && t.size(0) == len && t.size(1) == pairs, name, " must have shape (", len, ", ",
                pairs, ")");
}

}  // namespace

void per_head_qk_norm_rope3d_(torch::Tensor q,
                              torch::Tensor k,
                              std::optional<torch::Tensor> q_weight,
                              std::optional<torch::Tensor> k_weight,
                              torch::Tensor cos_t,
                              torch::Tensor sin_t,
                              torch::Tensor cos_h,
                              torch::Tensor sin_h,
                              torch::Tensor cos_w,
                              torch::Tensor sin_w,
                              double q_scale,
                              double eps) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && q.device() == k.device(), "q and k must be on the same CUDA device");
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must share a dtype");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 || q.scalar_type() == torch::kHalf, "q/k must be fp16 or bf16");
    TORCH_CHECK(q.dim() == 6 && q.sizes() == k.sizes(), "q and k must both be (B, T, H, W, heads, head_dim)");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous(), "q and k must be contiguous");
    const int64_t T = q.size(1), H = q.size(2), W = q.size(3), heads = q.size(4), head_dim = q.size(5);
    TORCH_CHECK(head_dim % 2 == 0 && head_dim <= 64 * kMaxPairsPerLane, "head_dim must be even and <= 256");
    const int64_t pairs_t = cos_t.size(1), pairs_h = cos_h.size(1), pairs_w = cos_w.size(1);
    TORCH_CHECK(pairs_t + pairs_h + pairs_w == head_dim / 2, "rope tables must cover head_dim / 2 pairs");
    check_table(cos_t, T, pairs_t, "cos_t");
    check_table(sin_t, T, pairs_t, "sin_t");
    check_table(cos_h, H, pairs_h, "cos_h");
    check_table(sin_h, H, pairs_h, "sin_h");
    check_table(cos_w, W, pairs_w, "cos_w");
    check_table(sin_w, W, pairs_w, "sin_w");
    for (auto *w : {&q_weight, &k_weight}) {
        if (w->has_value()) {
            TORCH_CHECK(w->value().is_cuda() && w->value().numel() == head_dim && w->value().is_contiguous() &&
                            w->value().scalar_type() == q.scalar_type(),
                        "norm weights must be contiguous (head_dim,) CUDA tensors in the q/k dtype");
        }
    }
    TORCH_CHECK(q_weight.has_value() == k_weight.has_value(), "pass both norm weights or neither");

    const int64_t rows = q.numel() / head_dim;
    if (rows == 0) {
        return;
    }
    const dim3 block(32 * kWarpsPerBlock);
    const dim3 grid(static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock), 2);
    auto stream = at::cuda::getCurrentCUDAStream();

#define NL_LAUNCH(scalar_t)                                                                                        \
    per_head_qk_norm_rope3d_kernel<scalar_t, scalar_t><<<grid, block, 0, stream>>>(                               \
        q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),                                                            \
        q_weight.has_value() ? q_weight->data_ptr<scalar_t>() : nullptr,                                           \
        k_weight.has_value() ? k_weight->data_ptr<scalar_t>() : nullptr, cos_t.data_ptr<float>(),                 \
        sin_t.data_ptr<float>(), cos_h.data_ptr<float>(), sin_h.data_ptr<float>(), cos_w.data_ptr<float>(),       \
        sin_w.data_ptr<float>(), rows, T, H, W, heads, static_cast<int>(head_dim), static_cast<int>(pairs_t),     \
        static_cast<int>(pairs_h), static_cast<float>(q_scale), static_cast<float>(eps))

    if (q.scalar_type() == torch::kBFloat16) {
        NL_LAUNCH(c10::BFloat16);
    } else {
        NL_LAUNCH(c10::Half);
    }
#undef NL_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace nunchaku_lite::ops
