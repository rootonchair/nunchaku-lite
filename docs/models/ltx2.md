# LTX-2.5 NVFP4 / INT4

> **Status: preview.** The LTX-2.5 checkpoints below are not published yet (the paths are placeholders), and the
> decoder kernels are not yet in a released `nunchaku-lite-kernels` build. Until they are, the decoder speedup is
> skipped with a warning. LTX-2.5 needs a Diffusers version that includes `LTX2VideoTransformer3DModel` with the
> LTX-2.5 options and the `LTX25AutoBlocks` modular pipeline.

Quantized transformer for the LTX-2.5 distilled model (22B), plus faster kernels for its diffusion video decoder.

| GPU | Precision | Notes |
|---|---|---|
| Blackwell (RTX PRO 6000, RTX 50xx) | `"fp4"` (NVFP4, rank 32) | fastest; ~1.8x the bf16 transformer speed on an RTX PRO 6000 |
| Other CUDA GPUs (A100, ...) | `"int4"` (rank 32, Hadamard input rotation) | rotation is stored in the checkpoint and applied automatically |

## Transformer

```python
import torch
from diffusers import LTX2VideoTransformer3DModel

from nunchaku_lite import patch_transformer

model_dir = "path/to/ltx-2.5-distilled-diffusers"  # converted Diffusers folder
precision = "fp4"  # "fp4" on Blackwell, "int4" elsewhere
checkpoint = f"path/to/svdq-{precision}_r32-ltx-2.5-distilled.safetensors"  # placeholder: not published yet

transformer = LTX2VideoTransformer3DModel.from_pretrained(
    model_dir, subfolder="transformer", torch_dtype=torch.bfloat16
)
patch_transformer(transformer, checkpoint, target="ltx2", precision=precision, device="cuda")
```

`from_pretrained` loads the bf16 weights first (about 44 GB of CPU memory) before they are replaced by the quantized
ones. Pass the patched `transformer` to the LTX-2.5 pipeline as its `transformer` component.

## Diffusion decoder

LTX-2.5 decodes latents with a diffusion decoder whose last stage dominates decode time. Install the nunchaku-lite
kernels on it once the pipeline is built:

```python
from nunchaku_lite.adapters.ltx2_decoder import optimize_ltx2_diffusion_decoder

optimize_ltx2_diffusion_decoder(pipe.diffusion_decoder)  # returns False (with a warning) if the kernels are missing
pipe.diffusion_decoder.disable_tiling()  # untiled is faster; needs ~13.5 GB extra VRAM at 768x512x121
```

It replaces the decoder's per-head q/k RMSNorm + RoPE chain and the stage-5 block glue (residual add + RMSNorm +
modulation, SiLU gating) with fused kernels, keeping NATTEN for the attention. The output matches the unfused decoder
up to reduction order: 56.6 dB PSNR, max pixel difference 7/255.

Measured on an RTX PRO 6000, untiled decode of a 121-frame 768x512 clip: 2.77 s → 1.84 s.

Tiled and untiled decodes differ from each other (~30 dB) by themselves, so compare outputs only between runs that
decode the same way.
