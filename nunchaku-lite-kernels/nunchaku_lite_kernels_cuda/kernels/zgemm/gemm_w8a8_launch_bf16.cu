#include "gemm_w8a8_launch_impl.cuh"

namespace nunchaku::kernels {
template class GEMM_W8A8_Launch<GEMMConfig_W8A8_BF16>;
};
