#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <optional>

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

template <typename scalar_t, typename weight_t>
__global__ void rms_norm_modulate_kernel(const scalar_t *__restrict__ x,
                                         scalar_t *__restrict__ out,
                                         const weight_t *__restrict__ weight,
                                         const scalar_t *__restrict__ scale,
                                         const scalar_t *__restrict__ shift,
                                         int64_t batch,
                                         int64_t sequence,
                                         int64_t channels,
                                         int64_t scale_elements,
                                         int64_t shift_elements,
                                         float eps,
                                         bool has_weight) {
    extern __shared__ float shared_sum[];

    int64_t token = blockIdx.x;
    if (token >= batch * sequence) {
        return;
    }
    int64_t b = token / sequence;
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
    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        int64_t bc_offset = b * channels + c;
        int64_t scale_offset = scale_elements == channels ? c : bc_offset;
        int64_t shift_offset = shift_elements == channels ? c : bc_offset;
        float value = load_as_float(x, base + c) * inv_rms;
        if (has_weight) {
            value *= load_as_float(weight, c);
        }
        value = value * (1.0f + load_as_float(scale, scale_offset)) + load_as_float(shift, shift_offset);
        out[base + c] = float_to_scalar<scalar_t>(value);
    }
}

template <typename scalar_t>
__global__ void affine_modulate_kernel(const scalar_t *__restrict__ x,
                                       const scalar_t *__restrict__ scale,
                                       const scalar_t *__restrict__ shift,
                                       scalar_t *__restrict__ out,
                                       int64_t numel,
                                       int64_t batch,
                                       int64_t sequence,
                                       int64_t channels,
                                       int64_t scale_elements,
                                       int64_t shift_elements) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        int64_t c = i % channels;
        int64_t token = i / channels;
        int64_t b = token / sequence;
        int64_t scale_offset = i;
        int64_t shift_offset = i;
        if (scale_elements == channels) {
            scale_offset = c;
        } else if (scale_elements == batch * channels) {
            scale_offset = b * channels + c;
        }
        if (shift_elements == channels) {
            shift_offset = c;
        } else if (shift_elements == batch * channels) {
            shift_offset = b * channels + c;
        }
        float value = load_as_float(x, i) * (1.0f + load_as_float(scale, scale_offset)) +
                      load_as_float(shift, shift_offset);
        out[i] = float_to_scalar<scalar_t>(value);
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

} // namespace

torch::Tensor fused_rms_norm_modulate(torch::Tensor x,
                                      std::optional<torch::Tensor> norm_weight,
                                      torch::Tensor scale,
                                      torch::Tensor shift,
                                      double eps) {
    check_half_bfloat_cuda(x, "x");
    TORCH_CHECK(x.dim() == 3, "x must have shape [batch, sequence, channels]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    check_half_bfloat_cuda(scale, "scale");
    check_half_bfloat_cuda(shift, "shift");
    TORCH_CHECK(scale.get_device() == x.get_device(), "scale must be on the same CUDA device as x");
    TORCH_CHECK(shift.get_device() == x.get_device(), "shift must be on the same CUDA device as x");
    TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");
    TORCH_CHECK(shift.is_contiguous(), "shift must be contiguous");

    int64_t batch = x.size(0);
    int64_t sequence = x.size(1);
    int64_t channels = x.size(2);
    check_optional_weight(norm_weight, x, channels, "norm_weight");
    TORCH_CHECK(scale.numel() == channels || scale.numel() == batch * channels,
                "scale must be broadcastable as [channels] or [batch, channels]");
    TORCH_CHECK(shift.numel() == channels || shift.numel() == batch * channels,
                "shift must be broadcastable as [channels] or [batch, channels]");

    auto out = torch::empty_like(x);
    if (x.numel() == 0) {
        return out;
    }

    int threads = 256;
    dim3 grid(static_cast<unsigned int>(batch * sequence));
    size_t shared_mem = threads * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "rms_norm_modulate", [&] {
            if (norm_weight.has_value() && norm_weight->defined() && norm_weight->scalar_type() == at::kFloat) {
                rms_norm_modulate_kernel<scalar_t, float><<<grid, threads, shared_mem, stream>>>(
                    x.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    norm_weight->data_ptr<float>(),
                    scale.data_ptr<scalar_t>(),
                    shift.data_ptr<scalar_t>(),
                    batch,
                    sequence,
                    channels,
                    scale.numel(),
                    shift.numel(),
                    static_cast<float>(eps),
                    true);
            } else {
                const scalar_t *weight_ptr =
                    norm_weight.has_value() && norm_weight->defined() ? norm_weight->data_ptr<scalar_t>() : nullptr;
                rms_norm_modulate_kernel<scalar_t, scalar_t><<<grid, threads, shared_mem, stream>>>(
                    x.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    weight_ptr,
                    scale.data_ptr<scalar_t>(),
                    shift.data_ptr<scalar_t>(),
                    batch,
                    sequence,
                    channels,
                    scale.numel(),
                    shift.numel(),
                    static_cast<float>(eps),
                    weight_ptr != nullptr);
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_affine_modulate(torch::Tensor x, torch::Tensor scale, torch::Tensor shift) {
    check_half_bfloat_cuda(x, "x");
    TORCH_CHECK(x.dim() == 3, "x must have shape [batch, sequence, channels]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    check_half_bfloat_cuda(scale, "scale");
    check_half_bfloat_cuda(shift, "shift");
    TORCH_CHECK(scale.get_device() == x.get_device(), "scale must be on the same CUDA device as x");
    TORCH_CHECK(shift.get_device() == x.get_device(), "shift must be on the same CUDA device as x");
    TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");
    TORCH_CHECK(shift.is_contiguous(), "shift must be contiguous");

    int64_t batch = x.size(0);
    int64_t sequence = x.size(1);
    int64_t channels = x.size(2);
    int64_t scale_elements = scale.numel();
    int64_t shift_elements = shift.numel();
    TORCH_CHECK(scale_elements == channels || scale_elements == batch * channels || scale_elements == x.numel(),
                "scale must be broadcastable as [channels], [batch, channels], or x shape");
    TORCH_CHECK(shift_elements == channels || shift_elements == batch * channels || shift_elements == x.numel(),
                "shift must be broadcastable as [channels], [batch, channels], or x shape");

    auto out = torch::empty_like(x);
    if (x.numel() == 0) {
        return out;
    }

    int threads = 256;
    int64_t blocks = std::min<int64_t>((x.numel() + threads - 1) / threads, 65535);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "affine_modulate", [&] {
            affine_modulate_kernel<scalar_t><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                scale.data_ptr<scalar_t>(),
                shift.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                x.numel(),
                batch,
                sequence,
                channels,
                scale_elements,
                shift_elements);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

} // namespace nunchaku_lite::ops
