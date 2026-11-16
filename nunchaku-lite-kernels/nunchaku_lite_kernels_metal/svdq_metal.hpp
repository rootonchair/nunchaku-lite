#pragma once

#include <optional>
#include <tuple>

#include <ATen/ATen.h>

namespace nunchaku_lite::metal {

// Per-token symmetric int8 quantization of (x + shift) * inv_smooth.
// x: (M, K) half/bfloat/float on MPS, K % 4 == 0. Returns (q int8 (M, K), scale float (M,)).
std::tuple<at::Tensor, at::Tensor> svdq_quantize_act_int8(const at::Tensor &x,
                                                          const at::Tensor &inv_smooth,
                                                          double shift);

// out = (a @ w8^T) * ascale[m] * wsc[n] + bias[n] + lact @ lup^T, int8 x int8 on the Neural Accelerators.
// a: (M, K) int8; w8: (N, K) int8; lact (M, R) / lup (N, R) half or bfloat (output dtype).
at::Tensor svdq_gemm_w8a8(const at::Tensor &a,
                          const at::Tensor &w8,
                          const at::Tensor &ascale,
                          const at::Tensor &wsc,
                          const at::Tensor &lact,
                          const at::Tensor &lup,
                          const std::optional<at::Tensor> &bias);

// Same as svdq_gemm_w8a8 with packed signed int4 weights w4 (N, K / 2) uint8 (low nibble = even k)
// expanded in-kernel as rint(q * wr[g, n]); wr: (K / 64, N) half.
at::Tensor svdq_gemm_w4a8(const at::Tensor &a,
                          const at::Tensor &w4,
                          const at::Tensor &wr,
                          const at::Tensor &ascale,
                          const at::Tensor &wsc,
                          const at::Tensor &lact,
                          const at::Tensor &lup,
                          const std::optional<at::Tensor> &bias);

// In-place per-head RMSNorm (optional weights) + interleaved RoPE on the Q/K thirds of a fused
// QKV output (M, 3 * heads * 128). rope: float32 (batch, seq, 64, 2) holding (sin, cos) pairs.
void svdq_qk_norm_rope_(at::Tensor &qkv,
                        const std::optional<at::Tensor> &norm_q,
                        const std::optional<at::Tensor> &norm_k,
                        const std::optional<at::Tensor> &rope,
                        double eps_q,
                        double eps_k,
                        int64_t heads,
                        int64_t seq_len);

}  // namespace nunchaku_lite::metal
