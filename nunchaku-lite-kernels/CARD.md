---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}tags:
- kernels
- diffusion
- quantization
- svdquant
- metal
- cuda
---

# nunchaku-lite-kernels

Prebuilt kernels for [Nunchaku Lite](https://github.com/rootonchair/nunchaku-lite), which runs SVDQuant (Nunchaku)
INT4 / NVFP4 diffusion checkpoints (FLUX.1, FLUX.2 Klein, Qwen-Image, Z-Image, SDXL) in Diffusers
pipelines. Nunchaku Lite loads this repository automatically through the
[`kernels` library](https://github.com/huggingface/kernels); you only need it directly to call the ops yourself.

- **NVIDIA GPUs:** W4A4 GEMMs (INT4 and NVFP4) with the fused SVD low-rank branch, AWQ W4A16 GEMM/GEMV, and fused
  modulation, q/k norm + RoPE and attention kernels.
- **Apple silicon (PyTorch MPS):** W8A8 / W4A8 GEMMs on the M5 Neural Accelerators with the low-rank branch in the
  epilogue, plus fused q/k norm + RoPE and Neural Accelerator flash attention.

## Builds

| Backend | PyTorch | Variants |
|---|---|---|
| CUDA (Linux x86_64) | 2.11-2.15 | CUDA 12.6-13.4; compute capability 7.5, 8.0, 8.6, 8.9, 12.0a, 12.1a |
| Metal (macOS, arm64) | 2.10-2.15 | macOS 26+ (Metal 4); the MPS GEMMs and attention need an M5-class GPU |

## How to use

With Nunchaku Lite nothing is needed: `load_nunchaku_pipeline(...)` fetches the matching build. To call the ops
directly:

```python
# pip install -U kernels
from kernels import get_kernel

kernels_module = get_kernel("{{ repo_id }}", version={{ version }}, trust_remote_code=True)
ops = kernels_module.ops

# Apple silicon (M5): Neural Accelerator flash attention on (B, L, H, D) tensors
import torch

q, k, v = (torch.randn(1, 4096, 24, 128, device="mps", dtype=torch.bfloat16) for _ in range(3))
out = ops.attention_mps(q, k, v, None, 128**-0.5)  # (1, 4096, 24, 128)
```

## Available functions

All ops are registered with `torch.library` (with fake implementations for `torch.compile`).

CUDA:

| Op | Purpose |
|---|---|
| `gemm_w4a4` | SVDQuant W4A4 GEMM (INT4 or NVFP4) with the low-rank branch and fused epilogues (q/k norm, RoPE, GELU) |
| `quantize_w4a4_act_fuse_lora` | Activation quantization fused with the low-rank down projection |
| `awq_gemm_w4a16_g128_int16`, `awq_gemm_w4a16_g64_int32`, `gemv_awq` | AWQ W4A16 GEMM / GEMV |
| `fused_rms_norm_modulate`, `fused_affine_modulate` | AdaLN modulation |
| `fused_cross_head_qk_norm_rope` | q/k RMSNorm + RoPE |
| `attention_fp16` | FP16 attention |

Python helpers: `svdq_gemm_w4a4_cuda`, `svdq_quantize_w4a4_act_fuse_lora_cuda`, `awq_gemv_w4a16_cuda`,
`attention_fp16_cuda`.

Metal (MPS):

| Op | Purpose |
|---|---|
| `svdq_quantize_act_int8` | Per-token int8 activation quantization with smoothing and shift |
| `svdq_gemm_w8a8` | int8 x int8 GEMM on the Neural Accelerators; scales, bias and low-rank branch in the epilogue |
| `svdq_gemm_w4a8` | Same with 4-bit weights expanded in the kernel (about half the weight memory) |
| `qk_norm_rope_mps_` | q/k RMSNorm + RoPE + layout change in one pass; reads column slices of fused projections, works in place |
| `attention_mps` | Flash attention on the Neural Accelerators (head dim 64/96/128, grouped-query, bool or additive masks) |
| `svdq_qk_norm_rope_` | Earlier in-place q/k norm + RoPE, kept for older Nunchaku Lite releases |

Shapes and dtypes are documented in `svdq_metal.hpp` in the
[source](https://github.com/rootonchair/nunchaku-lite/tree/main/nunchaku-lite-kernels).

## Versions

Releases stay on the `v{{ version }}` branch while changes are backward compatible (new ops, faster kernels). Nunchaku
Lite pins the version it was tested with.

## Benchmarks
{% if has_benchmark %}

Run `kernels benchmark {{ repo_id }} --version {{ version }}`.
{% else %}

No benchmark available yet.
{% endif %}

## License and credits

Apache-2.0. The CUDA kernels derive from [Nunchaku](https://github.com/nunchaku-tech/nunchaku) (Apache-2.0).
`attention_mps` is MLX's Neural Accelerator flash attention from [MLX](https://github.com/ml-explore/mlx)
(MIT, Copyright © 2023 Apple Inc.; the notice ships inside the compiled kernel source).
