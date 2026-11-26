// Elementwise fusions for the LTX-2.5 diffusion decoder's stage-5 block (diffusers
// LTX2VideoVaeDiffusionNABlock), each reproducing the eager diffusers rounding:
//
// add_rmsnorm_modulate_(x, residual, weight, one_plus_scale, shift, eps) -> out
//   x <- x + residual                                  (in place; rounded once, like a bf16 add)
//   out = ((rmsnorm(x) * weight) * one_plus_scale) + shift
//   with rmsnorm(x) * weight rounded once (as torch's fused rms_norm does), then the
//   modulation multiply and the shift add rounded separately, as two eager ops would.
//   one_plus_scale and shift are per batch: (B, C). The caller computes 1 + scale in the
//   storage dtype, exactly as the eager `(1 + scale)` does.
//   x, residual: (B, ..., C) contiguous, C % 8 == 0, C <= 1024. One warp per row.
//
// silu_mul(gate_up) -> out
//   gate_up: (..., 2H) contiguous, gate in [0, H), up in [H, 2H) (one merged matmul).
//   out = silu(gate) * up, silu rounded to the storage dtype before the multiply, as
//   `F.silu(gate) * up` does. silu(x) = x / (1 + expf(-x)) in fp32, as torch's kernel.
//
// The only expected difference from eager PyTorch is the order of the sum-of-squares
// reduction in the norm.

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <c10/cuda/CUDAException.h>
#include <torch/types.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <optional>

namespace nunchaku_lite::ops {
namespace {

constexpr int kVec = 8;  // 16-byte vectors of fp16/bf16
constexpr int kMaxVecsPerLane = 4;  // C <= 32 * 8 * 4 = 1024
constexpr int kWarpsPerBlock = 8;

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

template <typename T>
struct alignas(16) Vec {
    T v[kVec];
};

template <typename scalar_t>
__global__ void add_rmsnorm_modulate_kernel(scalar_t *__restrict__ x,
                                            const scalar_t *__restrict__ residual,
                                            const scalar_t *__restrict__ weight,
                                            const scalar_t *__restrict__ one_plus_scale,
                                            const scalar_t *__restrict__ shift,
                                            scalar_t *__restrict__ out,
                                            int64_t rows,
                                            int64_t rows_per_batch,
                                            int channels,
                                            float eps) {
    using V = Vec<scalar_t>;
    const int lane = threadIdx.x & 31;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    if (row >= rows) {
        return;
    }
    const int vecs = channels / kVec;
    V *xr = reinterpret_cast<V *>(x + row * channels);
    const V *rr = reinterpret_cast<const V *>(residual + row * channels);

    float h[kMaxVecsPerLane][kVec];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kMaxVecsPerLane; ++i) {
        const int j = lane + 32 * i;
        if (j < vecs) {
            V a = xr[j];
            const V b = rr[j];
#pragma unroll
            for (int e = 0; e < kVec; ++e) {
                const scalar_t s = from_f<scalar_t>(__fadd_rn(to_f(a.v[e]), to_f(b.v[e])));
                a.v[e] = s;
                h[i][e] = to_f(s);
                sum += h[i][e] * h[i][e];
            }
            xr[j] = a;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    const float inv_rms = rsqrtf(sum / static_cast<float>(channels) + eps);

    const int64_t b = row / rows_per_batch;
    const V *sc = reinterpret_cast<const V *>(one_plus_scale + b * channels);
    const V *sh = reinterpret_cast<const V *>(shift + b * channels);
    const V *wv = reinterpret_cast<const V *>(weight);
    V *orow = reinterpret_cast<V *>(out + row * channels);
#pragma unroll
    for (int i = 0; i < kMaxVecsPerLane; ++i) {
        const int j = lane + 32 * i;
        if (j < vecs) {
            const V s = sc[j];
            const V t = sh[j];
            V w{};
            if (weight != nullptr) {
                w = wv[j];
            }
            V o;
#pragma unroll
            for (int e = 0; e < kVec; ++e) {
                float n = __fmul_rn(h[i][e], inv_rms);
                if (weight != nullptr) {
                    n = __fmul_rn(n, to_f(w.v[e]));
                }
                n = round_trip<scalar_t>(n);
                const float m = round_trip<scalar_t>(__fmul_rn(n, to_f(s.v[e])));
                o.v[e] = from_f<scalar_t>(__fadd_rn(m, to_f(t.v[e])));
            }
            orow[j] = o;
        }
    }
}

template <typename scalar_t>
__global__ void silu_mul_kernel(const scalar_t *__restrict__ gate_up,
                                scalar_t *__restrict__ out,
                                int64_t total_vecs,
                                int64_t hidden_vecs) {
    using V = Vec<scalar_t>;
    const V *in = reinterpret_cast<const V *>(gate_up);
    V *o = reinterpret_cast<V *>(out);
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total_vecs;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int64_t row = idx / hidden_vecs;
        const int64_t col = idx - row * hidden_vecs;
        const V g = in[row * 2 * hidden_vecs + col];
        const V u = in[row * 2 * hidden_vecs + hidden_vecs + col];
        V r;
#pragma unroll
        for (int e = 0; e < kVec; ++e) {
            const float gf = to_f(g.v[e]);
            const float s = round_trip<scalar_t>(__fdiv_rn(gf, __fadd_rn(1.0f, expf(-gf))));
            r.v[e] = from_f<scalar_t>(__fmul_rn(s, to_f(u.v[e])));
        }
        o[idx] = r;
    }
}

bool aligned16(const torch::Tensor &t) {
    return reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16 == 0;
}

void check_io(const torch::Tensor &t, const torch::Tensor &ref, const char *name) {
    TORCH_CHECK(t.is_cuda() && t.device() == ref.device(), name, " must be on the same CUDA device as x");
    TORCH_CHECK(t.scalar_type() == ref.scalar_type(), name, " must have the dtype of x");
    TORCH_CHECK(t.is_contiguous() && aligned16(t), name, " must be contiguous and 16-byte aligned");
}

}  // namespace

torch::Tensor add_rmsnorm_modulate_(torch::Tensor x,
                                    torch::Tensor residual,
                                    std::optional<torch::Tensor> weight,
                                    torch::Tensor one_plus_scale,
                                    torch::Tensor shift,
                                    double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kHalf, "x must be fp16 or bf16");
    TORCH_CHECK(x.dim() >= 2, "x must be (B, ..., C)");
    check_io(x, x, "x");
    check_io(residual, x, "residual");
    TORCH_CHECK(residual.sizes() == x.sizes(), "residual must have the shape of x");
    const int64_t channels = x.size(-1);
    const int64_t batch = x.size(0);
    TORCH_CHECK(channels % kVec == 0 && channels <= 32 * kVec * kMaxVecsPerLane,
                "channels must be a multiple of 8 and <= 1024");
    check_io(one_plus_scale, x, "one_plus_scale");
    check_io(shift, x, "shift");
    TORCH_CHECK(one_plus_scale.numel() == batch * channels && shift.numel() == batch * channels,
                "one_plus_scale and shift must have batch * channels elements, (B, C)");
    if (weight.has_value()) {
        check_io(*weight, x, "weight");
        TORCH_CHECK(weight->numel() == channels, "weight must have C elements");
    }
    auto out = at::empty_like(x);
    const int64_t rows = x.numel() / channels;
    if (rows == 0) {
        return out;
    }
    const dim3 block(32 * kWarpsPerBlock);
    const dim3 grid(static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock));
    auto stream = at::cuda::getCurrentCUDAStream();

#define NL_LAUNCH(scalar_t)                                                                                        \
    add_rmsnorm_modulate_kernel<scalar_t><<<grid, block, 0, stream>>>(                                             \
        x.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(),                                                     \
        weight.has_value() ? weight->data_ptr<scalar_t>() : nullptr, one_plus_scale.data_ptr<scalar_t>(),         \
        shift.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), rows, rows / batch, static_cast<int>(channels),     \
        static_cast<float>(eps))

    if (x.scalar_type() == torch::kBFloat16) {
        NL_LAUNCH(c10::BFloat16);
    } else {
        NL_LAUNCH(c10::Half);
    }
#undef NL_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor silu_mul(torch::Tensor gate_up) {
    TORCH_CHECK(gate_up.is_cuda(), "gate_up must be a CUDA tensor");
    TORCH_CHECK(gate_up.scalar_type() == torch::kBFloat16 || gate_up.scalar_type() == torch::kHalf,
                "gate_up must be fp16 or bf16");
    TORCH_CHECK(gate_up.dim() >= 1 && gate_up.is_contiguous() && aligned16(gate_up),
                "gate_up must be contiguous and 16-byte aligned");
    const int64_t two_h = gate_up.size(-1);
    TORCH_CHECK(two_h % (2 * kVec) == 0, "the last dim of gate_up must be 2 * H with H a multiple of 8");
    auto sizes = gate_up.sizes().vec();
    sizes.back() = two_h / 2;
    auto out = at::empty(sizes, gate_up.options());
    const int64_t total_vecs = out.numel() / kVec;
    if (total_vecs == 0) {
        return out;
    }
    const int threads = 256;
    const int64_t max_blocks = static_cast<int64_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * 16;
    const int64_t blocks = std::min<int64_t>((total_vecs + threads - 1) / threads, max_blocks);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (gate_up.scalar_type() == torch::kBFloat16) {
        silu_mul_kernel<c10::BFloat16><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
            gate_up.data_ptr<c10::BFloat16>(), out.data_ptr<c10::BFloat16>(), total_vecs, two_h / 2 / kVec);
    } else {
        silu_mul_kernel<c10::Half><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
            gate_up.data_ptr<c10::Half>(), out.data_ptr<c10::Half>(), total_vecs, two_h / 2 / kVec);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

}  // namespace nunchaku_lite::ops
