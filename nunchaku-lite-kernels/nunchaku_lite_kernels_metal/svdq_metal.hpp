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

// Per-head RMSNorm of q and k + interleaved RoPE + layout change for 128-wide heads, in one pass (the kernel in
// nunchaku_lite/shaders.py). Rows: the optional text rows qt/kt/vt, then the image rows qi/ki/vi; each group is
// (rows, heads * 128) views sharing a row stride (column slices of one projection are fine). Weights: float32
// (128,). cos/sin: float32 tables, row r uses table row r % rot_rows; cs_pair = 1 (one value per pair) or 2
// (Diffusers' tables). Element (row, head, c) is written at row * os_l + head * os_h + c of each output; outputs
// may alias the inputs (in place). When v_out is vi itself, v is left untouched.
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
                       int64_t os_l);

// Softmax attention on the Neural Accelerators (MLX's flash attention, M5-class GPUs). q: (B, Lq, H, D), k / v:
// (B, Lk, Hkv, D) with any strides and a contiguous head dim; D in {64, 96, 128}; H % Hkv == 0. mask: optional
// (B, H, Lq, Lk) view with a contiguous last dim, bool (true = attend) or additive in q's dtype.
// Returns (B, Lq, H, D).
at::Tensor attention_mps(const at::Tensor &q,
                         const at::Tensor &k,
                         const at::Tensor &v,
                         const std::optional<at::Tensor> &mask,
                         double scale);

}  // namespace nunchaku_lite::metal
