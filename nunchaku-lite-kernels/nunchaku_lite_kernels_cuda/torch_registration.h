#pragma once

#include <optional>
#include <vector>

#include <torch/library.h>

#include "ops_lite.h"

namespace {

using OptionalTensor = std::optional<torch::Tensor>;

std::vector<float> to_float_vector(const c10::List<double> &values) {
    std::vector<float> result;
    result.reserve(values.size());
    for (double value : values) {
        result.push_back(static_cast<float>(value));
    }
    return result;
}

void dispatch_gemm_w4a4(OptionalTensor act,
                        OptionalTensor wgt,
                        OptionalTensor out,
                        OptionalTensor qout,
                        OptionalTensor ascales,
                        OptionalTensor wscales,
                        OptionalTensor oscales,
                        OptionalTensor poolout,
                        OptionalTensor lora_act_in,
                        OptionalTensor lora_up,
                        OptionalTensor lora_down,
                        OptionalTensor lora_act_out,
                        OptionalTensor norm_q,
                        OptionalTensor norm_k,
                        OptionalTensor rotary_emb,
                        OptionalTensor bias,
                        OptionalTensor smooth_factor,
                        OptionalTensor out_vk,
                        OptionalTensor out_linearattn,
                        bool act_unsigned,
                        c10::List<double> lora_scales,
                        bool fuse_silu,
                        bool fp4,
                        OptionalTensor alpha,
                        OptionalTensor wcscales,
                        OptionalTensor out_q,
                        OptionalTensor out_k,
                        OptionalTensor out_v,
                        int64_t attn_tokens) {
    nunchaku_lite::ops::gemm_w4a4(act,
                                  wgt,
                                  out,
                                  qout,
                                  ascales,
                                  wscales,
                                  oscales,
                                  poolout,
                                  lora_act_in,
                                  lora_up,
                                  lora_down,
                                  lora_act_out,
                                  norm_q,
                                  norm_k,
                                  rotary_emb,
                                  bias,
                                  smooth_factor,
                                  out_vk,
                                  out_linearattn,
                                  act_unsigned,
                                  to_float_vector(lora_scales),
                                  fuse_silu,
                                  fp4,
                                  alpha.has_value() ? alpha.value().item<float>() : 1.0f,
                                  wcscales,
                                  out_q,
                                  out_k,
                                  out_v,
                                  static_cast<int>(attn_tokens));
}

void dispatch_quantize_w4a4_act_fuse_lora(OptionalTensor input,
                                          OptionalTensor output,
                                          OptionalTensor oscales,
                                          OptionalTensor lora_down,
                                          OptionalTensor lora_act_out,
                                          OptionalTensor smooth,
                                          bool fuse_glu,
                                          bool fp4) {
    nunchaku_lite::ops::quantize_w4a4_act_fuse_lora(
        input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4);
}

torch::Tensor dispatch_fused_rms_norm_modulate(torch::Tensor x,
                                               OptionalTensor norm_weight,
                                               torch::Tensor scale,
                                               torch::Tensor shift,
                                               double eps) {
    return nunchaku_lite::ops::fused_rms_norm_modulate(x, norm_weight, scale, shift, eps);
}

torch::Tensor dispatch_fused_affine_modulate(torch::Tensor x, torch::Tensor scale, torch::Tensor shift) {
    return nunchaku_lite::ops::fused_affine_modulate(x, scale, shift);
}

torch::Tensor dispatch_gemv_awq(torch::Tensor in_feats,
                                torch::Tensor qweight,
                                torch::Tensor scaling_factors,
                                torch::Tensor zeros,
                                int64_t m,
                                int64_t n,
                                int64_t k,
                                int64_t group_size) {
    return nunchaku_lite::ops::gemv_awq(in_feats, qweight, scaling_factors, zeros, m, n, k, group_size);
}

void dispatch_attention_fp16(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, double scale) {
    nunchaku_lite::ops::attention_fp16(q, k, v, o, scale);
}

std::vector<torch::Tensor> dispatch_fused_cross_head_qk_norm_rope(torch::Tensor q,
                                                                  torch::Tensor k,
                                                                  OptionalTensor q_weight,
                                                                  OptionalTensor k_weight,
                                                                  OptionalTensor q_cos,
                                                                  OptionalTensor q_sin,
                                                                  OptionalTensor k_cos,
                                                                  OptionalTensor k_sin,
                                                                  int64_t q_heads,
                                                                  int64_t k_heads,
                                                                  int64_t head_dim,
                                                                  double eps,
                                                                  bool interleaved) {
    return nunchaku_lite::ops::fused_cross_head_qk_norm_rope(q,
                                                             k,
                                                             q_weight,
                                                             k_weight,
                                                             q_cos,
                                                             q_sin,
                                                             k_cos,
                                                             k_sin,
                                                             q_heads,
                                                             k_heads,
                                                             head_dim,
                                                             eps,
                                                             interleaved);
}

} // namespace

TORCH_LIBRARY(nunchaku_lite_kernels, ops) {
    ops.def("gemm_w4a4(Tensor? act, Tensor? wgt, Tensor(a!)? out, Tensor(b!)? qout, "
            "Tensor? ascales, Tensor? wscales, Tensor(c!)? oscales, Tensor(d!)? "
            "poolout, Tensor? lora_act_in, Tensor? lora_up, Tensor? lora_down, "
            "Tensor(e!)? lora_act_out, Tensor? norm_q, Tensor? norm_k, Tensor? "
            "rotary_emb, Tensor? bias, Tensor? smooth_factor, Tensor(f!)? out_vk, "
            "Tensor(g!)? out_linearattn, bool act_unsigned, float[] lora_scales, "
            "bool fuse_silu, bool fp4, Tensor? alpha, Tensor? wcscales, Tensor(h!)? "
            "out_q, Tensor(i!)? out_k, Tensor(j!)? out_v, int attn_tokens) -> ()");
    ops.def("quantize_w4a4_act_fuse_lora(Tensor? input, Tensor(a!)? output, "
            "Tensor(b!)? oscales, Tensor? lora_down, Tensor(c!)? lora_act_out, "
            "Tensor? smooth, bool fuse_glu, bool fp4) -> ()");
    ops.def("fused_rms_norm_modulate(Tensor x, Tensor? norm_weight, Tensor scale, "
            "Tensor shift, float eps) -> Tensor");
    ops.def("fused_affine_modulate(Tensor x, Tensor scale, Tensor shift) -> Tensor");
    ops.def("awq_gemm_w4a16_g128_int16(Tensor in_feats, Tensor qweight, "
            "Tensor scaling_factors, Tensor zeros) -> Tensor");
    ops.def("awq_gemm_w4a16_g64_int32(Tensor in_feats, Tensor qweight, "
            "Tensor scaling_factors, Tensor zeros) -> Tensor");
    ops.def("gemv_awq(Tensor in_feats, Tensor qweight, Tensor scaling_factors, "
            "Tensor zeros, int m, int n, int k, int group_size) -> Tensor");
    ops.def("attention_fp16(Tensor q, Tensor k, Tensor v, Tensor o, "
            "float scale) -> ()");
    ops.def("fused_cross_head_qk_norm_rope(Tensor q, Tensor k, Tensor? q_weight, "
            "Tensor? k_weight, Tensor? q_cos, Tensor? q_sin, Tensor? k_cos, "
            "Tensor? k_sin, int q_heads, int k_heads, int head_dim, float eps, "
            "bool interleaved) -> Tensor[]");

#if defined(NUNCHAKU_LITE_REGISTER_CUDA_DISPATCH) || defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
    ops.impl("gemm_w4a4", torch::kCUDA, &dispatch_gemm_w4a4);
    ops.impl("quantize_w4a4_act_fuse_lora", torch::kCUDA, &dispatch_quantize_w4a4_act_fuse_lora);
    ops.impl("fused_rms_norm_modulate", torch::kCUDA, &dispatch_fused_rms_norm_modulate);
    ops.impl("fused_affine_modulate", torch::kCUDA, &dispatch_fused_affine_modulate);
    ops.impl("awq_gemm_w4a16_g128_int16", torch::kCUDA, &nunchaku_lite::ops::awq_gemm_w4a16_g128_int16);
    ops.impl("awq_gemm_w4a16_g64_int32", torch::kCUDA, &nunchaku_lite::ops::awq_gemm_w4a16_g64_int32);
    ops.impl("gemv_awq", torch::kCUDA, &dispatch_gemv_awq);
    ops.impl("attention_fp16", torch::kCUDA, &dispatch_attention_fp16);
    ops.impl("fused_cross_head_qk_norm_rope", torch::kCUDA, &dispatch_fused_cross_head_qk_norm_rope);
#endif
}
