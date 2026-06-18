#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <optional>
#include <vector>

namespace nunchaku_lite::ops {
namespace {

template <typename T>
__device__ __forceinline__ float load_as_float(const T *ptr, int64_t offset) {
    return static_cast<float>(ptr[offset]);
}

template <>
__device__ __forceinline__ float load_as_float<c10::Half>(const c10::Half *ptr, int64_t offset) {
    const __half *half_ptr = reinterpret_cast<const __half *>(ptr);
    return __half2float(half_ptr[offset]);
}

template <>
__device__ __forceinline__ float load_as_float<c10::BFloat16>(const c10::BFloat16 *ptr, int64_t offset) {
    const __nv_bfloat16 *bf16_ptr = reinterpret_cast<const __nv_bfloat16 *>(ptr);
    return __bfloat162float(bf16_ptr[offset]);
}

template <typename T>
__device__ __forceinline__ T float_to_scalar(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __forceinline__ c10::Half float_to_scalar<c10::Half>(float value) {
    __half half_value = __float2half_rn(value);
    return *reinterpret_cast<c10::Half *>(&half_value);
}

template <>
__device__ __forceinline__ c10::BFloat16 float_to_scalar<c10::BFloat16>(float value) {
    __nv_bfloat16 bf16_value = __float2bfloat16(value);
    return *reinterpret_cast<c10::BFloat16 *>(&bf16_value);
}

template <typename T>
__device__ __forceinline__ float round_trip_dtype(float value) {
    T rounded = float_to_scalar<T>(value);
    return load_as_float(&rounded, 0);
}

template <typename scalar_t, typename q_weight_t, typename k_weight_t, bool interleaved>
__global__ void cross_head_qk_norm_rope_kernel(const scalar_t *__restrict__ q,
                                                const scalar_t *__restrict__ k,
                                                scalar_t *__restrict__ q_out,
                                                scalar_t *__restrict__ k_out,
                                                const q_weight_t *__restrict__ q_weight,
                                                const k_weight_t *__restrict__ k_weight,
                                                const float *__restrict__ q_cos,
                                                const float *__restrict__ q_sin,
                                                const float *__restrict__ k_cos,
                                                const float *__restrict__ k_sin,
                                                int64_t batch,
                                                int64_t q_sequence,
                                                int64_t k_sequence,
                                                int64_t q_channels,
                                                int64_t k_channels,
                                                int64_t q_heads,
                                                int64_t k_heads,
                                                int64_t head_dim,
                                                float eps,
                                                bool has_q_weight,
                                                bool has_k_weight,
                                                bool has_q_rope,
                                                bool has_k_rope) {
    extern __shared__ unsigned char shared_storage[];
    float *shared_sum = reinterpret_cast<float *>(shared_storage);
    scalar_t *shared_norm = reinterpret_cast<scalar_t *>(shared_sum + blockDim.x);

    bool is_q = blockIdx.y == 0;
    int64_t sequence = is_q ? q_sequence : k_sequence;
    int64_t token = blockIdx.x;
    if (token >= batch * sequence) {
        return;
    }

    const scalar_t *x = is_q ? q : k;
    scalar_t *out = is_q ? q_out : k_out;
    const q_weight_t *qw = q_weight;
    const k_weight_t *kw = k_weight;
    const float *cos = is_q ? q_cos : k_cos;
    const float *sin = is_q ? q_sin : k_sin;
    int64_t channels = is_q ? q_channels : k_channels;
    int64_t heads = is_q ? q_heads : k_heads;
    bool has_weight = is_q ? has_q_weight : has_k_weight;
    bool has_rope = is_q ? has_q_rope : has_k_rope;
    int64_t base = token * channels;

    float sum = 0.0f;
    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        float value = load_as_float(x, base + c);
        sum += value * value;
    }
    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(shared_sum[0] / static_cast<float>(channels) + eps);
    int64_t b = token / sequence;
    int64_t t = token - b * sequence;

    if constexpr (interleaved) {
        for (int64_t pair = threadIdx.x; pair < channels / 2; pair += blockDim.x) {
            int64_t c0 = pair * 2;
            int64_t c1 = c0 + 1;
            float v0 = load_as_float(x, base + c0) * inv_rms;
            float v1 = load_as_float(x, base + c1) * inv_rms;
            if (has_weight) {
                if (is_q) {
                    v0 *= load_as_float(qw, c0);
                    v1 *= load_as_float(qw, c1);
                } else {
                    v0 *= load_as_float(kw, c0);
                    v1 *= load_as_float(kw, c1);
                }
            }
            v0 = round_trip_dtype<scalar_t>(v0);
            v1 = round_trip_dtype<scalar_t>(v1);
            if (has_rope) {
                int64_t rope_base = (b * sequence + t) * channels;
                float cos0 = cos[rope_base + c0];
                float cos1 = cos[rope_base + c1];
                float sin0 = sin[rope_base + c0];
                float sin1 = sin[rope_base + c1];
                out[base + c0] = float_to_scalar<scalar_t>(v0 * cos0 - v1 * sin0);
                out[base + c1] = float_to_scalar<scalar_t>(v1 * cos1 + v0 * sin1);
            } else {
                out[base + c0] = float_to_scalar<scalar_t>(v0);
                out[base + c1] = float_to_scalar<scalar_t>(v1);
            }
        }
        return;
    }

    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        float value = load_as_float(x, base + c) * inv_rms;
        if (has_weight) {
            if (is_q) {
                value *= load_as_float(qw, c);
            } else {
                value *= load_as_float(kw, c);
            }
        }
        shared_norm[c] = float_to_scalar<scalar_t>(round_trip_dtype<scalar_t>(value));
    }
    __syncthreads();

    if (!has_rope) {
        for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
            out[base + c] = shared_norm[c];
        }
        return;
    }

    int64_t half_head = head_dim / 2;
    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        int64_t head = c / head_dim;
        int64_t local = c - head * head_dim;
        int64_t rope_dim = local < half_head ? local : local - half_head;
        int64_t partner = head * head_dim + (local < half_head ? local + half_head : local - half_head);
        float value = load_as_float(shared_norm, c);
        float other = load_as_float(shared_norm, partner);
        int64_t rope_base = ((b * heads + head) * sequence + t) * half_head;
        float cos_value = cos[rope_base + rope_dim];
        float sin_value = sin[rope_base + rope_dim];
        float sign = local < half_head ? -1.0f : 1.0f;
        out[base + c] = float_to_scalar<scalar_t>(value * cos_value + sign * other * sin_value);
    }
}

void check_half_bfloat_cuda(const torch::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kHalf || tensor.scalar_type() == at::kBFloat16,
                name,
                " must be float16 or bfloat16");
}

void check_optional_weight(const std::optional<torch::Tensor> &weight,
                           const torch::Tensor &x,
                           int64_t channels,
                           const char *name) {
    if (!weight.has_value() || !weight->defined()) {
        return;
    }
    TORCH_CHECK(weight->is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(weight->get_device() == x.get_device(), name, " must be on the same CUDA device as input");
    TORCH_CHECK(weight->scalar_type() == x.scalar_type() || weight->scalar_type() == at::kFloat,
                name,
                " must have input dtype or float32 dtype");
    TORCH_CHECK(weight->dim() == 1 && weight->size(0) == channels, name, " must have shape [channels]");
}

void check_optional_rope(const std::optional<torch::Tensor> &cos,
                         const std::optional<torch::Tensor> &sin,
                         const torch::Tensor &x,
                         int64_t heads,
                         int64_t head_dim,
                         bool interleaved,
                         const char *name) {
    bool has_cos = cos.has_value() && cos->defined();
    bool has_sin = sin.has_value() && sin->defined();
    TORCH_CHECK(has_cos == has_sin, name, " cos/sin must both be provided or both be omitted");
    if (!has_cos) {
        return;
    }
    TORCH_CHECK(cos->is_cuda(), name, " cos must be a CUDA tensor");
    TORCH_CHECK(sin->is_cuda(), name, " sin must be a CUDA tensor");
    TORCH_CHECK(cos->get_device() == x.get_device(), name, " cos must be on the same CUDA device as input");
    TORCH_CHECK(sin->get_device() == x.get_device(), name, " sin must be on the same CUDA device as input");
    TORCH_CHECK(cos->scalar_type() == at::kFloat, name, " cos must be float32");
    TORCH_CHECK(sin->scalar_type() == at::kFloat, name, " sin must be float32");
    TORCH_CHECK(cos->is_contiguous(), name, " cos must be contiguous");
    TORCH_CHECK(sin->is_contiguous(), name, " sin must be contiguous");
    TORCH_CHECK(cos->sizes() == sin->sizes(), name, " cos and sin must have matching shapes");
    if (interleaved) {
        TORCH_CHECK(cos->dim() == 3, name, " interleaved cos/sin must have shape [batch, sequence, channels]");
        TORCH_CHECK(cos->size(0) == x.size(0) && cos->size(1) == x.size(1) && cos->size(2) == x.size(2),
                    name,
                    " interleaved cos/sin must have shape [batch, sequence, channels]");
    } else {
        TORCH_CHECK(cos->dim() == 4, name, " split cos/sin must have shape [batch, heads, sequence, head_dim / 2]");
        TORCH_CHECK(cos->size(0) == x.size(0) && cos->size(1) == heads && cos->size(2) == x.size(1) &&
                        cos->size(3) == head_dim / 2,
                    name,
                    " split cos/sin must have shape [batch, heads, sequence, head_dim / 2]");
    }
}

} // namespace

std::vector<torch::Tensor> fused_cross_head_qk_norm_rope(torch::Tensor q,
                                                         torch::Tensor k,
                                                         std::optional<torch::Tensor> q_weight,
                                                         std::optional<torch::Tensor> k_weight,
                                                         std::optional<torch::Tensor> q_cos,
                                                         std::optional<torch::Tensor> q_sin,
                                                         std::optional<torch::Tensor> k_cos,
                                                         std::optional<torch::Tensor> k_sin,
                                                         int64_t q_heads,
                                                         int64_t k_heads,
                                                         int64_t head_dim,
                                                         double eps,
                                                         bool interleaved) {
    check_half_bfloat_cuda(q, "q");
    check_half_bfloat_cuda(k, "k");
    TORCH_CHECK(q.dim() == 3, "q must have shape [batch, sequence, channels]");
    TORCH_CHECK(k.dim() == 3, "k must have shape [batch, sequence, channels]");
    TORCH_CHECK(q.get_device() == k.get_device(), "q and k must be on the same CUDA device");
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must have the same dtype");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(q.size(0) == k.size(0), "q and k batch sizes must match");
    TORCH_CHECK(q_heads > 0 && k_heads > 0 && head_dim > 0, "heads and head_dim must be positive");
    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even for RoPE");
    TORCH_CHECK(q.size(2) == q_heads * head_dim, "q channels must equal q_heads * head_dim");
    TORCH_CHECK(k.size(2) == k_heads * head_dim, "k channels must equal k_heads * head_dim");

    check_optional_weight(q_weight, q, q.size(2), "q_weight");
    check_optional_weight(k_weight, k, k.size(2), "k_weight");
    check_optional_rope(q_cos, q_sin, q, q_heads, head_dim, interleaved, "q");
    check_optional_rope(k_cos, k_sin, k, k_heads, head_dim, interleaved, "k");

    auto q_out = torch::empty_like(q);
    auto k_out = torch::empty_like(k);
    if (q.numel() == 0 && k.numel() == 0) {
        return {q_out, k_out};
    }

    int64_t q_tokens = q.size(0) * q.size(1);
    int64_t k_tokens = k.size(0) * k.size(1);
    int64_t max_tokens = std::max(q_tokens, k_tokens);
    int64_t max_channels = std::max(q.size(2), k.size(2));
    int threads = 256;
    if (max_channels / 2 > 256) {
        threads = 512;
    }
    if (max_channels / 2 > 512) {
        threads = 1024;
    }
    dim3 grid(static_cast<unsigned int>(max_tokens), 2);
    size_t shared_mem = static_cast<size_t>(threads) * sizeof(float);
    if (!interleaved) {
        shared_mem += static_cast<size_t>(max_channels) * q.element_size();
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bool has_q_weight = q_weight.has_value() && q_weight->defined();
    bool has_k_weight = k_weight.has_value() && k_weight->defined();
    bool has_q_rope = q_cos.has_value() && q_cos->defined();
    bool has_k_rope = k_cos.has_value() && k_cos->defined();

#define LAUNCH_CROSS_HEAD_QK_KERNEL(Q_WEIGHT_T, K_WEIGHT_T, INTERLEAVED)                                               \
    cross_head_qk_norm_rope_kernel<scalar_t, Q_WEIGHT_T, K_WEIGHT_T, INTERLEAVED>                                      \
        <<<grid, threads, shared_mem, stream>>>(q.data_ptr<scalar_t>(),                                                \
                                               k.data_ptr<scalar_t>(),                                                \
                                               q_out.data_ptr<scalar_t>(),                                            \
                                               k_out.data_ptr<scalar_t>(),                                            \
                                               has_q_weight ? q_weight->data_ptr<Q_WEIGHT_T>() : nullptr,             \
                                               has_k_weight ? k_weight->data_ptr<K_WEIGHT_T>() : nullptr,             \
                                               has_q_rope ? q_cos->data_ptr<float>() : nullptr,                       \
                                               has_q_rope ? q_sin->data_ptr<float>() : nullptr,                       \
                                               has_k_rope ? k_cos->data_ptr<float>() : nullptr,                       \
                                               has_k_rope ? k_sin->data_ptr<float>() : nullptr,                       \
                                               q.size(0),                                                             \
                                               q.size(1),                                                             \
                                               k.size(1),                                                             \
                                               q.size(2),                                                             \
                                               k.size(2),                                                             \
                                               q_heads,                                                               \
                                               k_heads,                                                               \
                                               head_dim,                                                              \
                                               static_cast<float>(eps),                                               \
                                               has_q_weight,                                                          \
                                               has_k_weight,                                                          \
                                               has_q_rope,                                                            \
                                               has_k_rope)

#define DISPATCH_CROSS_HEAD_QK_BY_ROPE(Q_WEIGHT_T, K_WEIGHT_T)                                                        \
    do {                                                                                                               \
        if (interleaved) {                                                                                             \
            LAUNCH_CROSS_HEAD_QK_KERNEL(Q_WEIGHT_T, K_WEIGHT_T, true);                                                 \
        } else {                                                                                                       \
            LAUNCH_CROSS_HEAD_QK_KERNEL(Q_WEIGHT_T, K_WEIGHT_T, false);                                                \
        }                                                                                                              \
    } while (false)

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "cross_head_qk_norm_rope", [&] {
            if (has_q_weight && q_weight->scalar_type() == at::kFloat) {
                if (has_k_weight && k_weight->scalar_type() == at::kFloat) {
                    DISPATCH_CROSS_HEAD_QK_BY_ROPE(float, float);
                } else {
                    DISPATCH_CROSS_HEAD_QK_BY_ROPE(float, scalar_t);
                }
            } else if (has_k_weight && k_weight->scalar_type() == at::kFloat) {
                DISPATCH_CROSS_HEAD_QK_BY_ROPE(scalar_t, float);
            } else {
                DISPATCH_CROSS_HEAD_QK_BY_ROPE(scalar_t, scalar_t);
            }
        });

#undef DISPATCH_CROSS_HEAD_QK_BY_ROPE
#undef LAUNCH_CROSS_HEAD_QK_KERNEL

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q_out, k_out};
}

} // namespace nunchaku_lite::ops
