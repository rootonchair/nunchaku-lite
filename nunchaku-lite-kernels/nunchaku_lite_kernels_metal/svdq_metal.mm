// Host launchers for the SVDQ MPS kernels. Kernels are encoded on PyTorch's current MPS stream,
// so they are ordered with (and batched alongside) regular MPS ops.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <c10/util/Exception.h>

#include "svdq_metal.hpp"
#include "lite_shaders.hpp"
#include "qk_norm_rope_shaders.hpp"
#include "svdq_shaders.hpp"

namespace nunchaku_lite::metal {
namespace {

constexpr int kTileM = 128;

struct Tile {
    int bm, bn, sg, kc, swz;
};

enum class Lib { kSvdq, kQkNormRope, kAttention };

// Each library compiles on first use, so the 120 KB attention source costs nothing until attention runs.
id<MTLLibrary> library(Lib which) {
    static std::mutex mutex;
    static std::unordered_map<int, id<MTLLibrary>> libs;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = libs.find(int(which));
    if (it != libs.end()) {
        return it->second;
    }
    std::string source;
    switch (which) {
    case Lib::kSvdq:
        source = std::string(kSvdqMetalSource) + kQkNormRopeMetalSource;
        break;
    case Lib::kQkNormRope:
        source = kQkNormRopeLiteSource;
        break;
    case Lib::kAttention:
        source = kAttentionNaxSource;
        break;
    }
    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    if (@available(macOS 26.0, *)) {
        // MTLLanguageVersion4_0 ((4 << 16) + 0) as a raw value: the enum constant exists only in the
        // macOS 26 SDK, and kernel-builder currently builds with Xcode 16 (macOS 15 SDK).
        options.languageVersion = static_cast<MTLLanguageVersion>((4 << 16) + 0);
    }
    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                              options:options
                                                error:&error];
    [options release];
    TORCH_CHECK(lib != nil,
                "nunchaku_lite_kernels: failed to compile Metal kernels (requires macOS 26+ with Metal 4): ",
                error ? error.localizedDescription.UTF8String : "unknown error");
    libs.emplace(int(which), lib);
    return lib;
}

// Bool Metal function constants (index, value) used to specialize a kernel.
using BoolConstants = std::vector<std::pair<int, bool>>;

id<MTLComputePipelineState> pipeline(const std::string &name, Lib which = Lib::kSvdq,
                                     const BoolConstants &constants = {}) {
    static std::mutex mutex;
    static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
    std::string key = std::to_string(int(which)) + ":" + name;
    for (const auto &[index, value] : constants) {
        key += "," + std::to_string(index) + "=" + (value ? "1" : "0");
    }
    id<MTLLibrary> lib = library(which);
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    NSString *fn_name = [NSString stringWithUTF8String:name.c_str()];
    id<MTLFunction> fn = nil;
    NSError *error = nil;
    if (constants.empty()) {
        fn = [lib newFunctionWithName:fn_name];
    } else {
        MTLFunctionConstantValues *values = [[MTLFunctionConstantValues alloc] init];
        for (const auto &[index, value] : constants) {
            const bool v = value;
            [values setConstantValue:&v type:MTLDataTypeBool atIndex:index];
        }
        fn = [lib newFunctionWithName:fn_name constantValues:values error:&error];
        [values release];
    }
    TORCH_CHECK(fn != nil, "nunchaku_lite_kernels: missing Metal kernel ", name,
                error ? std::string(": ") + error.localizedDescription.UTF8String : std::string());
    id<MTLComputePipelineState> pso =
        [at::mps::MPSDevice::getInstance()->device() newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    TORCH_CHECK(pso != nil, "nunchaku_lite_kernels: failed to build pipeline ", name, ": ",
                error ? error.localizedDescription.UTF8String : "unknown error");
    cache.emplace(key, pso);
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

// An MPS tensor read through explicit strides: only the last dim must be contiguous.
void check_mps_rows(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.device().is_mps(), name, " must be an MPS tensor");
    TORCH_CHECK(t.stride(-1) == 1, name, " must have a contiguous last dimension");
}

// Tuned on an M5 (AC power); mirrors nunchaku_lite_mlx.kernels.default_tile.
Tile default_tile(bool w8, int64_t m, int64_t n, int64_t k) {
    if (m % kTileM) {
        return {64, 64, 4, 64, 4};
    }
    if (k >= 8192 && n % 128 == 0) {
        return w8 ? Tile{kTileM, 128, 8, 32, 2} : Tile{kTileM, 128, 8, 64, 4};
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
    id<MTLComputePipelineState> pso = pipeline(std::string("svdq_qk_norm_rope_") + type_name(dtype), Lib::kSvdq);
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

void qk_norm_rope_mps_(const at::Tensor &qi,
                       const at::Tensor &ki,
                       const at::Tensor &vi,
                       const std::optional<at::Tensor> &qt,
                       const std::optional<at::Tensor> &kt,
                       const std::optional<at::Tensor> &vt,
                       const at::Tensor &wqi,
                       const at::Tensor &wki,
                       const at::Tensor &wqt,
                       const at::Tensor &wkt,
                       const at::Tensor &cos,
                       const at::Tensor &sin,
                       at::Tensor &q_out,
                       at::Tensor &k_out,
                       at::Tensor &v_out,
                       double eps_q,
                       double eps_k,
                       int64_t heads,
                       int64_t rot_rows,
                       int64_t cs_pair,
                       int64_t os_h,
                       int64_t os_l) {
    constexpr int64_t kHeadDim = 128;
    const bool has_text = qt.has_value();
    TORCH_CHECK(has_text == kt.has_value() && has_text == vt.has_value(), "pass all of qt, kt, vt or none");
    const at::Tensor &qt_ = has_text ? *qt : qi, &kt_ = has_text ? *kt : ki, &vt_ = has_text ? *vt : vi;
    const at::ScalarType dtype = qi.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat,
                "q/k/v must be float16, bfloat16 or float32");
    const int64_t width = heads * kHeadDim;
    for (const at::Tensor *t : {&qi, &ki, &vi, &qt_, &kt_, &vt_}) {
        check_mps_rows(*t, "q/k/v");
        TORCH_CHECK(t->dim() == 2 && t->size(1) == width && t->scalar_type() == dtype,
                    "q/k/v must be (rows, heads * 128) views of one dtype");
    }
    TORCH_CHECK(qi.stride(0) == ki.stride(0) && qi.stride(0) == vi.stride(0) && qi.size(0) == ki.size(0) &&
                    qi.size(0) == vi.size(0),
                "image q/k/v must share rows and row stride");
    TORCH_CHECK(qt_.stride(0) == kt_.stride(0) && qt_.stride(0) == vt_.stride(0) && qt_.size(0) == kt_.size(0) &&
                    qt_.size(0) == vt_.size(0),
                "text q/k/v must share rows and row stride");
    for (const at::Tensor *w : {&wqi, &wki, &wqt, &wkt}) {
        check_mps(*w, "norm weight");
        TORCH_CHECK(w->scalar_type() == at::kFloat && w->numel() == kHeadDim, "norm weights must be float32 (128,)");
    }
    for (const at::Tensor *t : {&cos, &sin}) {
        check_mps_rows(*t, "cos/sin");
        TORCH_CHECK(t->scalar_type() == at::kFloat && t->dim() == 2, "cos/sin must be 2-D float32 tables");
    }
    TORCH_CHECK(cos.strides() == sin.strides(), "cos and sin must share strides");
    TORCH_CHECK(cs_pair == 1 || cs_pair == 2, "cs_pair must be 1 or 2");
    TORCH_CHECK(cos.size(1) >= 64 * cs_pair, "rotary tables are too narrow");
    const int64_t lt = has_text ? qt_.size(0) : 0;
    const int64_t total = lt + qi.size(0);
    TORCH_CHECK(rot_rows > 0 && cos.size(0) >= std::min(rot_rows, total), "rotary tables have too few rows");
    const int64_t span = (total - 1) * os_l + (heads - 1) * os_h + kHeadDim;
    for (const at::Tensor *o : {&q_out, &k_out, &v_out}) {
        TORCH_CHECK(o->device().is_mps() && o->scalar_type() == dtype, "outputs must be MPS tensors in the input dtype");
        TORCH_CHECK(o->storage().nbytes() >= size_t(o->storage_offset() + span) * o->element_size(),
                    "outputs are too small for the given strides");
    }
    if (total == 0) {
        return;
    }
    const bool copy_v = v_out.data_ptr() != vi.data_ptr();
    const std::array<int, 12> dims = {int(lt), int(qt_.stride(0)), int(qi.stride(0)), 0, 0, 0,
                                      int(rot_rows), int(cos.stride(0)), int(cs_pair), int(os_h), int(os_l),
                                      copy_v ? 1 : 0};
    const std::array<float, 2> eps = {float(eps_q), float(eps_k)};
    int group = 1;
    for (int g : {8, 4, 2}) {
        if (heads % g == 0) {
            group = g;
            break;
        }
    }
    id<MTLComputePipelineState> pso = pipeline(std::string("qk_norm_rope_") + type_name(dtype), Lib::kQkNormRope);
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            [encoder setComputePipelineState:pso];
            int index = 0;
            for (const at::Tensor *t : {&qt_, &kt_, &vt_, &qi, &ki, &vi, &wqt, &wkt, &wqi, &wki, &cos, &sin}) {
                set_tensor(encoder, *t, index++);
            }
            [encoder setBytes:eps.data() length:sizeof(float) * eps.size() atIndex:index++];
            [encoder setBytes:dims.data() length:sizeof(int) * dims.size() atIndex:index++];
            set_tensor(encoder, q_out, index++);
            set_tensor(encoder, k_out, index++);
            set_tensor(encoder, v_out, index++);
            [encoder dispatchThreadgroups:MTLSizeMake(heads / group, total, 1)
                    threadsPerThreadgroup:MTLSizeMake(32 * group, 1, 1)];
        }
    });
}

namespace {

// Mirrors steel::AttnParams / steel::AttnMaskParams in the MLX source.
struct AttnParams {
    int B, H, D, qL, kL, gqa_factor;
    float scale;
    int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
    int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};
struct AttnMaskParams {
    int64_t M_strides[3];
};
static_assert(sizeof(AttnParams) == 14 * 4 + 12 * 8, "AttnParams layout must match the Metal struct");

// MLX's tile for head dims 64-128 (nunchaku_lite.shaders.ATTENTION_TILE) and its function constant indices.
constexpr int kAttnBQ = 64, kAttnBK = 32, kAttnWM = 4;
constexpr int kAlignQ = 200, kAlignK = 201, kHasMask = 300, kDoCausal = 301, kHasSinks = 302, kSaveLse = 303;

void fill_strides(int64_t *dst, const at::Tensor &t) {  // (B, L, H, D) -> the kernel's (B, H, L)
    dst[0] = t.stride(0);
    dst[1] = t.stride(2);
    dst[2] = t.stride(1);
}

}  // namespace

at::Tensor attention_mps(const at::Tensor &q,
                         const at::Tensor &k,
                         const at::Tensor &v,
                         const std::optional<at::Tensor> &mask,
                         double scale) {
    check_mps_rows(q, "q");
    check_mps_rows(k, "k");
    check_mps_rows(v, "v");
    const at::ScalarType dtype = q.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "q must be float16 or bfloat16");
    TORCH_CHECK(k.scalar_type() == dtype && v.scalar_type() == dtype, "q, k and v must share a dtype");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k and v must be (B, L, H, D)");
    const int64_t batch = q.size(0), q_len = q.size(1), heads = q.size(2), dim = q.size(3);
    const int64_t k_len = k.size(1), kv_heads = k.size(2);
    TORCH_CHECK(dim == 64 || dim == 96 || dim == 128, "head dim must be 64, 96 or 128, got ", dim);
    TORCH_CHECK(k.size(0) == batch && v.sizes() == k.sizes() && k.size(3) == dim, "k and v must be (B, Lk, Hkv, D)");
    TORCH_CHECK(kv_heads > 0 && heads % kv_heads == 0, "query heads must be a multiple of key/value heads");
    const char *mask_kind = "add";
    if (mask.has_value()) {
        check_mps_rows(*mask, "mask");
        TORCH_CHECK(mask->dim() == 4 && mask->size(0) == batch && mask->size(1) == heads && mask->size(2) == q_len &&
                        mask->size(3) == k_len,
                    "mask must be (B, H, Lq, Lk) (expanded views are fine)");
        TORCH_CHECK(mask->scalar_type() == at::kBool || mask->scalar_type() == dtype,
                    "mask must be bool or the query dtype");
        if (mask->scalar_type() == at::kBool) {
            mask_kind = "bool";
        }
    }
    at::Tensor out = at::empty({batch, q_len, heads, dim}, q.options());
    if (out.numel() == 0) {
        return out;
    }
    AttnParams params{};
    params.B = int(batch);
    params.H = int(heads);
    params.D = int(dim);
    params.qL = int(q_len);
    params.kL = int(k_len);
    params.gqa_factor = int(heads / kv_heads);
    params.scale = float(scale);
    params.NQ = int((q_len + kAttnBQ - 1) / kAttnBQ);
    params.NK = int((k_len + kAttnBK - 1) / kAttnBK);
    params.NQ_aligned = int(q_len / kAttnBQ);
    params.NK_aligned = int(k_len / kAttnBK);
    params.qL_rem = int(q_len - params.NQ_aligned * kAttnBQ);
    params.kL_rem = int(k_len - params.NK_aligned * kAttnBK);
    params.qL_off = int(k_len - q_len);
    fill_strides(params.Q_strides, q);
    fill_strides(params.K_strides, k);
    fill_strides(params.V_strides, v);
    fill_strides(params.O_strides, out);
    AttnMaskParams mask_params{};
    if (mask.has_value()) {
        for (int i = 0; i < 3; ++i) {
            mask_params.M_strides[i] = mask->stride(i);
        }
    }
    const std::string name =
        std::string("attention_") + type_name(dtype) + "_" + std::to_string(dim) + "_" + mask_kind;
    id<MTLComputePipelineState> pso = pipeline(name, Lib::kAttention,
                                               {{kAlignQ, q_len % kAttnBQ == 0},
                                                {kAlignK, k_len % kAttnBK == 0},
                                                {kHasMask, mask.has_value()},
                                                {kDoCausal, false},
                                                {kHasSinks, false},
                                                {kSaveLse, false}});
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^() {
        @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
            [encoder setComputePipelineState:pso];
            set_tensor(encoder, q, 0);
            set_tensor(encoder, k, 1);
            set_tensor(encoder, v, 2);
            set_tensor(encoder, out, 3);
            [encoder setBytes:&params length:sizeof(params) atIndex:4];
            if (mask.has_value()) {
                [encoder setBytes:&mask_params length:sizeof(mask_params) atIndex:5];
                set_tensor(encoder, *mask, 6);
            }
            [encoder dispatchThreadgroups:MTLSizeMake(params.NQ, heads, batch)
                    threadsPerThreadgroup:MTLSizeMake(32, kAttnWM, 1)];
        }
    });
    return out;
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
