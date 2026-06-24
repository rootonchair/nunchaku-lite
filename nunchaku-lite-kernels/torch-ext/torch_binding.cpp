#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

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

void gemm_w4a4(OptionalTensor act, OptionalTensor wgt, OptionalTensor out,
               OptionalTensor qout, OptionalTensor ascales,
               OptionalTensor wscales, OptionalTensor oscales,
               OptionalTensor poolout, OptionalTensor lora_act_in,
               OptionalTensor lora_up, OptionalTensor lora_down,
               OptionalTensor lora_act_out, OptionalTensor norm_q,
               OptionalTensor norm_k, OptionalTensor rotary_emb,
               OptionalTensor bias, OptionalTensor smooth_factor,
               OptionalTensor out_vk, OptionalTensor out_linearattn,
               bool act_unsigned, c10::List<double> lora_scales,
               bool fuse_silu, bool fp4, double alpha, OptionalTensor wcscales,
               OptionalTensor out_q, OptionalTensor out_k,
               OptionalTensor out_v, int64_t attn_tokens) {
  nunchaku_lite::ops::gemm_w4a4(
      act, wgt, out, qout, ascales, wscales, oscales, poolout, lora_act_in,
      lora_up, lora_down, lora_act_out, norm_q, norm_k, rotary_emb, bias,
      smooth_factor, out_vk, out_linearattn, act_unsigned,
      to_float_vector(lora_scales), fuse_silu, fp4, static_cast<float>(alpha),
      wcscales, out_q, out_k, out_v, static_cast<int>(attn_tokens));
}

void quantize_w4a4_act_fuse_lora(OptionalTensor input, OptionalTensor output,
                                 OptionalTensor oscales,
                                 OptionalTensor lora_down,
                                 OptionalTensor lora_act_out,
                                 OptionalTensor smooth, bool fuse_glu,
                                 bool fp4) {
  nunchaku_lite::ops::quantize_w4a4_act_fuse_lora(
      input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4);
}

torch::Tensor gemv_awq_dispatch(torch::Tensor in_feats, torch::Tensor qweight,
                                torch::Tensor scaling_factors,
                                torch::Tensor zeros, int64_t m, int64_t n,
                                int64_t k, int64_t group_size) {
  return nunchaku_lite::ops::gemv_awq(in_feats, qweight, scaling_factors, zeros,
                                      m, n, k, group_size);
}

void attention_fp16(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                    torch::Tensor o, double scale) {
  nunchaku_lite::ops::attention_fp16(q, k, v, o, scale);
}

} // namespace

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("gemm_w4a4(Tensor? act, Tensor? wgt, Tensor? out, Tensor? qout, "
          "Tensor? ascales, Tensor? wscales, Tensor? oscales, Tensor? "
          "poolout, Tensor? lora_act_in, Tensor? lora_up, Tensor? lora_down, "
          "Tensor? lora_act_out, Tensor? norm_q, Tensor? norm_k, Tensor? "
          "rotary_emb, Tensor? bias, Tensor? smooth_factor, Tensor? out_vk, "
          "Tensor? out_linearattn, bool act_unsigned, float[] lora_scales, "
          "bool fuse_silu, bool fp4, float alpha, Tensor? wcscales, Tensor? "
          "out_q, Tensor? out_k, Tensor? out_v, int attn_tokens) -> ()");
  ops.def("quantize_w4a4_act_fuse_lora(Tensor? input, Tensor? output, "
          "Tensor? oscales, Tensor? lora_down, Tensor? lora_act_out, "
          "Tensor? smooth, bool fuse_glu, bool fp4) -> ()");
  ops.def("awq_gemm_w4a16_g128_int16(Tensor in_feats, Tensor qweight, "
          "Tensor scaling_factors, Tensor zeros) -> Tensor");
  ops.def("awq_gemm_w4a16_g64_int32(Tensor in_feats, Tensor qweight, "
          "Tensor scaling_factors, Tensor zeros) -> Tensor");
  ops.def("gemv_awq(Tensor in_feats, Tensor qweight, Tensor scaling_factors, "
          "Tensor zeros, int m, int n, int k, int group_size) -> Tensor");
  ops.def("attention_fp16(Tensor q, Tensor k, Tensor v, Tensor o, "
          "float scale) -> ()");

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("gemm_w4a4", torch::kCUDA, &gemm_w4a4);
  ops.impl("quantize_w4a4_act_fuse_lora", torch::kCUDA,
           &quantize_w4a4_act_fuse_lora);
  ops.impl("awq_gemm_w4a16_g128_int16", torch::kCUDA,
           &nunchaku_lite::ops::awq_gemm_w4a16_g128_int16);
  ops.impl("awq_gemm_w4a16_g64_int32", torch::kCUDA,
           &nunchaku_lite::ops::awq_gemm_w4a16_g64_int32);
  ops.impl("gemv_awq", torch::kCUDA, &gemv_awq_dispatch);
  ops.impl("attention_fp16", torch::kCUDA, &attention_fp16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
