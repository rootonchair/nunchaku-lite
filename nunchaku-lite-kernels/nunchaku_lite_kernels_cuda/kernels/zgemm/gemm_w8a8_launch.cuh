#include "gemm_w8a8.cuh"
#include "epilogues.cuh"

namespace nunchaku::kernels {

// Host-side launch wrapper for GEMM_W8A8<Config>, mirroring GEMM_W4A4_Launch
// but without the FP4 axis: W8A8 always uses the same (single) kernel variant.
template<typename Config>
class GEMM_W8A8_Launch {
    using GEMM      = GEMM_W8A8<Config>;
    using Epilogues = Epilogues<Config>;
    using Lora      = Lora<Config>;

    using packed_act_t    = typename GEMM::packed_act_t;
    using packed_wgt_t    = typename GEMM::packed_wgt_t;
    using packed_ascale_t = typename GEMM::packed_ascale_t;
    using packed_wscale_t = typename GEMM::packed_wscale_t;
    using packed_fpsum_t  = typename GEMM::packed_fpsum_t;
    using half_t          = typename GEMM::half_t;

public:
    static void gemm_w8a8(Tensor act,            // packed act [M, K]
                          Tensor wgt,            // packed wgt [N, K]
                          Tensor out,            // linear     [M, N]
                          Tensor qout,           // packed act [M, N]
                          Tensor ascales,        // packed as  [K / 32, M]
                          Tensor wscales,        // packed ws  [K / 32, N]
                          Tensor oscales,        // packed as  [N / 32, M]
                          Tensor lora_act_in,    // packed lora_act [M, R]
                          Tensor lora_up,        // packed lora_wgt [N, R]
                          Tensor lora_down,      // packed lora_wgt [N, R]
                          Tensor lora_act_out,   // packed lora_act [M, R]
                          Tensor bias,           // packed ws  [N]
                          Tensor smooth_factor,  // packed ws  [N], for quantization of the next layer
                          Tensor wcscales,       // packed ws  [N]
                          std::vector<float> lora_scales, // [R / 16]
                          bool fuse_silu);
    static void quantize_w8a8_act_fuse_lora(Tensor input,
                                            Tensor output,
                                            Tensor oscales,
                                            Tensor lora_down,
                                            Tensor lora_act_out,
                                            Tensor smooth,
                                            bool fuse_glu);
};

}; // namespace nunchaku::kernels
