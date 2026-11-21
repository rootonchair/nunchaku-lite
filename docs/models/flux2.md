# FLUX.2 Klein INT4 / FP4

Example for `tonera/FLUX.2-klein-9B-Nunchaku`.

Set `precision = "int4"` or `precision = "fp4"` in the script. On pre-Blackwell GPUs such as RTX A5000, prefer the INT4 checkpoint.

Run from the repository root:

```python
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline

from nunchaku_lite import load_nunchaku_pipeline


model_id = "tonera/FLUX.2-klein-9B-Nunchaku"
precision = "fp4"  # "int4" or "fp4"
checkpoints = {
    "int4": "tonera/FLUX.2-klein-9B-Nunchaku/svdq-int4_r32-FLUX.2-klein-9B-Nunchaku.safetensors",
    "fp4": "tonera/FLUX.2-klein-9B-Nunchaku/svdq-fp4_r32-FLUX.2-klein-9B-Nunchaku.safetensors",
}
checkpoint = checkpoints[precision]
output_path = Path(f"outputs/flux2_klein_nunchaku_lite_{precision}.png")

pipe = load_nunchaku_pipeline(
    model_id,
    pipeline_cls=Flux2KleinPipeline,
    checkpoint=checkpoint,
    target="flux2",
    precision=precision,
    torch_dtype=torch.bfloat16,
    device="cuda",
)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=1.0,
    generator=torch.Generator(device="cuda").manual_seed(12345),
).images[0]

output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"saved {output_path}")
```

## FLUX.2 Klein Runtime LoRA

Patched FLUX.2 transformers expose the same runtime LoRA API as FLUX.1 and Qwen-Image. ComfyUI-format FLUX.2 LoRAs with `diffusion_model.double_blocks.*` and `diffusion_model.single_blocks.*` keys are converted to patched Nunchaku module names at load time.

```python
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline

from nunchaku_lite import load_nunchaku_pipeline


model_id = "tonera/FLUX.2-klein-9B-Nunchaku"
precision = "int4"
checkpoint = "tonera/FLUX.2-klein-9B-Nunchaku/svdq-int4_r32-FLUX.2-klein-9B-Nunchaku.safetensors"
output_path = Path("outputs/flux2_klein_pixelart_lora_int4.png")

pipe = load_nunchaku_pipeline(
    model_id,
    pipeline_cls=Flux2KleinPipeline,
    checkpoint=checkpoint,
    target="flux2",
    precision=precision,
    torch_dtype=torch.bfloat16,
    device="cuda",
)
pipe.enable_model_cpu_offload()

pipe.load_lora_weights(
    "artificialguybr/PIXELART-REDMOND-FLUXKLEIN9B",
    weight_name="[FLUX.2.Klein]PixelArt_Redmond.safetensors",
    adapter_name="pixelart",
)
pipe.set_adapters("pixelart", adapter_weights=0.8)

image = pipe(
    prompt="Pixel Art, PixArFK, a tiny knight beside a glowing arcade machine in a forest",
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=1.0,
    generator=torch.Generator(device="cuda").manual_seed(12345),
).images[0]

output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)
print(f"saved {output_path}")
```

## Rotated (Hadamard) Checkpoints

FLUX.2 Klein INT4 checkpoints can be exported with an online Hadamard rotation, which spreads
activation outliers before 4-bit quantization. On FLUX.2-klein INT4 it lowers quantization error per
layer (median +1.7 dB output SQNR; about 0.82x whole-model output error) for roughly 0.1% extra
transformer time. At 4 sampling steps, end-to-end image metrics were within run-to-run noise.
Rotated checkpoints load like any other; nothing changes in your code.
They need the CUDA kernels that include `quantize_w4a4_act_hadamard_fuse_lora`. MPS and MLX
refuse them at load time.

A rotated checkpoint must carry a runtime manifest with `nunchaku_format_version: 2` in its
`quantization_config`, declaring the rotation by name on each rotated target:

```json
"op_options": {"rotation": {"type": "hadamard"}}
```

Versions of Nunchaku Lite older than this format refuse version 2 manifests, so they can't
silently load a rotated checkpoint without its rotation. An optional model-level declaration
`"input_rotation": {"format_version": 2, "type": "hadamard", "targets": [...]}` must list the same
targets. A declaration without such a manifest is refused, because older versions would ignore it.
The manifest's rotated targets also load through `target="flux2"`.

`"hadamard"` has exactly one meaning. For each block of 64 contiguous input channels, in logical
channel order, the smoothed activations `x / smooth` are multiplied by the orthonormal Sylvester
Hadamard matrix `H64 / 8`. The low-rank branch sees the unrotated input, and the residual weight
is packed as `Q(R @ blockdiag(H64 / 8))`. `nunchaku_lite.ops.hadamard_rotate` is the reference
transform. A different transform needs a new name.

Rotation is refused on NVFP4 by default, because it measured slightly worse than no rotation there.
Pass `adapter_options={"allow_fp4_rotation": True}` to load such a checkpoint anyway. Rotation is
also refused on fc2 layers whose input is quantized unsigned or inside the fused GELU MLP.
