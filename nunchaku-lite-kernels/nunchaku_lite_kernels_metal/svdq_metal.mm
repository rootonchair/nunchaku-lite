// Host launchers for the SVDQ MPS kernels. Kernels are encoded on PyTorch's current MPS stream,
// so they are ordered with (and batched alongside) regular MPS ops.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <mutex>
#include <string>
#include <unordered_map>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <c10/util/Exception.h>

#include "svdq_metal.h"
#include "qk_norm_rope_shaders.h"
#include "svdq_shaders.h"

namespace nunchaku_lite::metal {
namespace {

constexpr int kTileM = 128;

struct Tile {
    int bm, bn, sg, kc, swz;
};

id<MTLLibrary> library() {
    static id<MTLLibrary> lib = nil;
    static std::once_flag once;
    std::call_once(once, [] {
        id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        if (@available(macOS 26.0, *)) {
            options.languageVersion = MTLLanguageVersion4_0;
        }
        NSError *error = nil;
        const std::string source = std::string(kSvdqMetalSource) + kQkNormRopeMetalSource;
        lib = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                   options:options
                                     error:&error];
        [options release];
        TORCH_CHECK(lib != nil,
                    "nunchaku_lite_kernels: failed to compile SVDQ Metal kernels (requires macOS 26+ with Metal 4): ",
                    error ? error.localizedDescription.UTF8String : "unknown error");
    });
    return lib;
}

id<MTLComputePipelineState> pipeline(const std::string &name) {
    static std::mutex mutex;
    static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(name);
    if (it != cache.end()) {
        return it->second;
    }
    id<MTLFunction> fn = [library() newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
    TORCH_CHECK(fn != nil, "nunchaku_lite_kernels: missing Metal kernel ", name);
    NSError *error = nil;
    id<MTLComputePipelineState> pso =
        [at::mps::MPSDevice::getInstance()->device() newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    TORCH_CHECK(pso != nil, "nunchaku_lite_kernels: failed to build pipeline ", name, ": ",
                error ? error.localizedDescription.UTF8String : "unknown error");
    cache.emplace(name, pso);
    return pso;
}

void set_tensor(id<MTLComputeCommandEncoder> encoder, const at::Tensor &t, int index) {
    [encoder setBuffer:__builtin_bit_cast(id<MTLBuffer>, t.storage().data())
                offset:t.storage_offset() * t.element_size()
               atIndex:index];
}

const char *type_name(at::ScalarType dtype) {
    switch (dtype) {
    case at::kHalf:
        return "half";
    case at::kBFloat16:
        return "bfloat";
    case at::kFloat:
        return "float";
    default:
        TORCH_CHECK(false, "nunchaku_lite_kernels: unsupported dtype ", dtype);
    }
}

void check_mps(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.device().is_mps(), name, " must be an MPS tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// Tuned on an M5 (AC power); mirrors nunchaku_lite_mlx.kernels.default_tile.
Tile default_tile(bool w8, int64_t m, int64_t n, int64_t k) {
    if (m % kTileM) {
        return {64, 64, 4, 64, 4};
    }
    if (k >= 8192 && n % 128 == 0) {
        return {kTileM, 128, 8, 64, w8 ? 1 : 4};
    }
    return {kTileM, 64, 4, 64, w8 ? 8 : 4};
}

at::Tensor launch_gemm(bool w8,
                       const at::Tensor &a,
                       const at::Tensor &b,
                       const at::Tensor *wr,
                       const at::Tensor &ascale,
                       const at::Tensor &wsc,
                       const at::Tensor &lact,
                       const at::Tensor &lup,
                       const std::optional<at::Tensor> &bias) {
    check_mps(a, "a");
    check_mps(b, "weight");
    check_mps(ascale, "ascale");
    check_mps(wsc, "wsc");
    check_mps(lact, "lact");
    check_mps(lup, "lup");
    TORCH_CHECK(a.dim() == 2 && a.scalar_type() == at::kChar, "a must be a 2-D int8 tensor");
    TORCH_CHECK(ascale.scalar_type() == at::kFloat && wsc.scalar_type() == at::kFloat, "ascale and wsc must be float32");
    const at::ScalarType out_dtype = lact.scalar_type();
    TORCH_CHECK(out_dtype == at::kHalf || out_dtype == at::kBFloat16, "lact must be float16 or bfloat16");
    TORCH_CHECK(lup.scalar_type() == out_dtype, "lact and lup must share a dtype");

    const int64_t m = a.size(0), k = a.size(1), n = b.size(0), r = lup.size(1);
    if (w8) {
        TORCH_CHECK(b.scalar_type() == at::kChar && b.size(1) == k, "w8 must be int8 with shape (N, K)");
    } else {
        TORCH_CHECK(b.scalar_type() == at::kByte && b.size(1) * 2 == k, "w4 must be uint8 with shape (N, K / 2)");
        check_mps(*wr, "wr");
        TORCH_CHECK(wr->scalar_type() == at::kHalf && wr->numel() == (k / 64) * n, "wr must be float16 (K / 64, N)");
    }
    TORCH_CHECK(ascale.numel() == m && wsc.numel() == n, "ascale / wsc sizes do not match (M, N)");
    TORCH_CHECK(lact.size(0) == m && lact.size(1) == r && lup.size(0) == n, "lact / lup shapes do not match");
    TORCH_CHECK(r > 0 && r % 32 == 0, "low-rank dimension must be a positive multiple of 32, got ", r);
    if (bias.has_value()) {
        check_mps(*bias, "bias");
        TORCH_CHECK(bias->scalar_type() == at::kFloat && bias->numel() == n, "bias must be float32 (N,)");
    }

    const Tile t = default_tile(w8, m, n, k);
    TORCH_CHECK(m % t.bm == 0 && n % t.bn == 0 && k % t.kc == 0,
                "SVDQ kernel shape M=", m, ", N=", n, ", K=", k, " is not aligned to tile (", t.bm, ", ", t.bn,
                "); pad tokens to a multiple of 128 and use N, K multiples of 64");

    at::Tensor out = at::empty({m, n}, lact.options());
    const std::string name = std::string("svdq_gemm_") + (w8 ? "w8a8_" : "w4a8_") + type_name(out_dtype) + "_" +
                             std::to_string(t.bm) + "_" + std::to_string(t.bn) + "_" + std::to_string(t.sg) + "_" +
                             std::to_string(t.kc) + "_" + std::to_string(t.swz);
    id<MTLComputePipelineState> pso = pipeline(name);
    const std::array<int, 5> dims = {int(m), int(n), int(k), int(r), bias.has_value() ? 1 : 0};
    const at::Tensor &bias_or_dummy = bias.has_value() ? *bias : wsc;  // never read when has_bias == 0

    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            [encoder setComputePipelineState:pso];
            set_tensor(encoder, a, 0);
            set_tensor(encoder, b, 1);
            set_tensor(encoder, ascale, 2);
            set_tensor(encoder, wsc, 3);
            set_tensor(encoder, lact, 4);
            set_tensor(encoder, lup, 5);
            set_tensor(encoder, bias_or_dummy, 6);
            set_tensor(encoder, out, 7);
            [encoder setBytes:dims.data() length:sizeof(int) * dims.size() atIndex:8];
            if (!w8) {
                set_tensor(encoder, *wr, 9);
            }
            [encoder dispatchThreadgroups:MTLSizeMake(n / t.bn, m / t.bm, 1)
                    threadsPerThreadgroup:MTLSizeMake(32 * t.sg, 1, 1)];
        }
    });
    return out;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> svdq_quantize_act_int8(const at::Tensor &x,
                                                          const at::Tensor &inv_smooth,
                                                          double shift) {
    check_mps(x, "x");
    check_mps(inv_smooth, "inv_smooth");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D (M, K)");
    const int64_t m = x.size(0), k = x.size(1);
    TORCH_CHECK(k % 4 == 0, "activation width must be a multiple of 4, got ", k);
    TORCH_CHECK(inv_smooth.scalar_type() == at::kFloat && inv_smooth.numel() == k, "inv_smooth must be float32 (K,)");

    at::Tensor q = at::empty({m, k}, x.options().dtype(at::kChar));
    at::Tensor scale = at::empty({m}, x.options().dtype(at::kFloat));
    if (m == 0) {
        return {q, scale};
    }
    id<MTLComputePipelineState> pso = pipeline(std::string("svdq_quantize_act_int8_") + type_name(x.scalar_type()));
    const float shift_f = static_cast<float>(shift);
    const uint32_t k32 = uint32_t(k);

    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            [encoder setComputePipelineState:pso];
            set_tensor(encoder, x, 0);
            set_tensor(encoder, inv_smooth, 1);
            [encoder setBytes:&shift_f length:sizeof(shift_f) atIndex:2];
            set_tensor(encoder, q, 3);
            set_tensor(encoder, scale, 4);
            [encoder setBytes:&k32 length:sizeof(k32) atIndex:5];
            [encoder dispatchThreadgroups:MTLSizeMake(1, m, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    });
    return {q, scale};
}

void svdq_qk_norm_rope_(at::Tensor &qkv,
                        const std::optional<at::Tensor> &norm_q,
                        const std::optional<at::Tensor> &norm_k,
                        const std::optional<at::Tensor> &rope,
                        double eps_q,
                        double eps_k,
                        int64_t heads,
                        int64_t seq_len) {
    constexpr int64_t kHeadDim = 128;
    check_mps(qkv, "qkv");
    TORCH_CHECK(qkv.dim() == 2 && qkv.size(1) == 3 * heads * kHeadDim,
                "qkv must be (M, 3 * heads * 128), got ", qkv.sizes());
    const at::ScalarType dtype = qkv.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "qkv must be float16 or bfloat16");
    for (const auto *w : {&norm_q, &norm_k}) {
        if (w->has_value()) {
            check_mps(**w, "norm weight");
            TORCH_CHECK((*w)->scalar_type() == dtype && (*w)->numel() == kHeadDim, "norm weights must be (128,) in qkv dtype");
        }
    }
    int64_t rope_batch = 1, rope_seq = 0;
    if (rope.has_value()) {
        check_mps(*rope, "rope");
        TORCH_CHECK(rope->scalar_type() == at::kFloat && rope->dim() == 4 && rope->size(2) == kHeadDim / 2 &&
                        rope->size(3) == 2,
                    "rope must be float32 (batch, seq, 64, 2)");
        rope_batch = rope->size(0);
        rope_seq = rope->size(1);
    }
    const int64_t m = qkv.size(0);
    TORCH_CHECK(seq_len > 0 && m % seq_len == 0, "seq_len must divide the number of rows");
    if (m == 0 || (!norm_q.has_value() && !norm_k.has_value() && !rope.has_value())) {
        return;
    }
    id<MTLComputePipelineState> pso = pipeline(std::string("svdq_qk_norm_rope_") + type_name(dtype));
    const std::array<int, 8> dims = {int(m), int(heads), int(seq_len), int(rope_batch), int(rope_seq),
                                     norm_q.has_value() ? 1 : 0, norm_k.has_value() ? 1 : 0, rope.has_value() ? 1 : 0};
    const std::array<float, 2> eps = {float(eps_q), float(eps_k)};
    const at::Tensor &wq = norm_q.has_value() ? *norm_q : qkv;  // dummies are never read
    const at::Tensor &wk = norm_k.has_value() ? *norm_k : qkv;
    const at::Tensor &rp = rope.has_value() ? *rope : qkv;
    constexpr int kSimdgroups = 4;
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            [encoder setComputePipelineState:pso];
            set_tensor(encoder, qkv, 0);
            set_tensor(encoder, wq, 1);
            set_tensor(encoder, wk, 2);
            set_tensor(encoder, rp, 3);
            [encoder setBytes:dims.data() length:sizeof(int) * dims.size() atIndex:4];
            [encoder setBytes:eps.data() length:sizeof(float) * eps.size() atIndex:5];
            [encoder dispatchThreadgroups:MTLSizeMake((2 * heads + kSimdgroups - 1) / kSimdgroups, m, 1)
                    threadsPerThreadgroup:MTLSizeMake(32 * kSimdgroups, 1, 1)];
        }
    });
}

at::Tensor svdq_gemm_w8a8(const at::Tensor &a,
                          const at::Tensor &w8,
                          const at::Tensor &ascale,
                          const at::Tensor &wsc,
                          const at::Tensor &lact,
                          const at::Tensor &lup,
                          const std::optional<at::Tensor> &bias) {
    return launch_gemm(true, a, w8, nullptr, ascale, wsc, lact, lup, bias);
}

at::Tensor svdq_gemm_w4a8(const at::Tensor &a,
                          const at::Tensor &w4,
                          const at::Tensor &wr,
                          const at::Tensor &ascale,
                          const at::Tensor &wsc,
                          const at::Tensor &lact,
                          const at::Tensor &lup,
                          const std::optional<at::Tensor> &bias) {
    return launch_gemm(false, a, w4, &wr, ascale, wsc, lact, lup, bias);
}

}  // namespace nunchaku_lite::metal
