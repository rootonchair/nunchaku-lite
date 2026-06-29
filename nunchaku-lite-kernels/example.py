# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "kernels",
#     "numpy",
#     "torch",
# ]
# ///

import platform
from pathlib import Path

import kernels
import torch

# Load the locally built kernel
kernel = kernels.get_local_kernel(Path("build"), "nunchaku_lite_kernels")

# Select device
if platform.system() == "Darwin":
    device = torch.device("mps")
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
elif torch.version.cuda is not None and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

if device.type != "cuda":
    raise RuntimeError("nunchaku_lite_kernels currently exposes CUDA native kernels only")

available = [
    "attention_fp16",
    "attention_fp16_cuda",
    "awq_gemm_w4a16_g128_int16",
    "awq_gemm_w4a16_g64_int32",
    "awq_gemv_w4a16_cuda",
    "fused_affine_modulate",
    "fused_cross_head_qk_norm_rope",
    "fused_rms_norm_modulate",
    "gemm_w4a4",
    "gemv_awq",
    "quantize_w4a4_act_fuse_lora",
    "svdq_gemm_w4a4_cuda",
    "svdq_quantize_w4a4_act_fuse_lora_cuda",
]

for name in available:
    assert hasattr(kernel, name), f"Missing exported function: {name}"

print("Available functions:")
for name in available:
    print(f"- {name}")
print("Success!")
