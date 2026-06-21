#pragma once

#include "common.h"
#include "Tensor.h"

Tensor awq_gemm_w4a16_g128_int16(Tensor _in_feats, Tensor _kernel, Tensor _scales, Tensor _zeros);
Tensor awq_gemm_w4a16_g64_int32(Tensor _in_feats, Tensor _kernel, Tensor _scales, Tensor _zeros);
