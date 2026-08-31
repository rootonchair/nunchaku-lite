#include <torch/extension.h>

#include "ops_lite.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def_submodule("ops")
        .def("gemm_w4a4", nunchaku_lite::ops::gemm_w4a4)
        .def("quantize_w4a4_act_fuse_lora",
             nunchaku_lite::ops::quantize_w4a4_act_fuse_lora,
             py::arg("input"),
             py::arg("output"),
             py::arg("oscales"),
             py::arg("lora_down"),
             py::arg("lora_act_out"),
             py::arg("smooth"),
             py::arg("fuse_glu"),
             py::arg("fp4"),
             py::arg("hadamard") = false)
        .def("awq_gemm_w4a16_g128_int16", nunchaku_lite::ops::awq_gemm_w4a16_g128_int16)
        .def("awq_gemm_w4a16_g64_int32", nunchaku_lite::ops::awq_gemm_w4a16_g64_int32)
        .def("gemv_awq", nunchaku_lite::ops::gemv_awq)
        .def("fused_cross_head_qk_norm_rope", nunchaku_lite::ops::fused_cross_head_qk_norm_rope)
        .def("fused_rms_norm_modulate", nunchaku_lite::ops::fused_rms_norm_modulate)
        .def("fused_affine_modulate", nunchaku_lite::ops::fused_affine_modulate)
        .def("attention_fp16", nunchaku_lite::ops::attention_fp16);
}
