# LTX-2.5 NVFP4 / INT4

> **Status: preview.** LTX-2.5 needs Diffusers 0.43 or newer. The decoder kernels ship in `nunchaku-lite-kernels`
> v3; with an older kernel build the decoder speedup is skipped with a warning.
>
> The checkpoints are derivatives of LTX-2.5 and are distributed under the
> [LTX-2.x Community License](https://huggingface.co/rootonchair/nunchaku-lite-ltx-2.5-distilled/blob/main/LICENSE),
> which has its own terms, including for commercial use.

Quantized transformer for the LTX-2.5 distilled model (22B), plus faster kernels for its diffusion video decoder.

| GPU | Precision | Notes |
|---|---|---|
| Blackwell (RTX PRO 6000, RTX 50xx) | `"fp4"` (NVFP4, rank 32) | fastest; ~1.8x the bf16 transformer speed on an RTX PRO 6000 |
| Other CUDA GPUs (A100, ...) | `"int4"` (rank 32, Hadamard input rotation) | rotation is stored in the checkpoint and applied automatically |

## Base model

The example loads every component except the quantized transformer from the LTX-2.5 distilled model in the Diffusers
folder layout. Lightricks ships LTX-2.5 as single files; convert them once with the script in this repository
(accept the license of the gated [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) repo and log in
first):

```bash
hf auth login
python tools/convert_ltx25_distilled.py --out ltx-2.5-distilled-diffusers
```

It downloads 72 GB, writes about 70 GB and needs about 85 GB of CPU memory. `--meta` checks the conversion from the
file headers without downloading the weights.

## Full example

Text to video with audio, 768x512, 121 frames, 8 distilled steps. Set `precision` for your GPU.

```python
import json
import os

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import diffusers
import diffusers.pipelines.ltx2 as ltx2
from diffusers import FlowMatchEulerDiscreteScheduler, LTX2Guidance, LTX2VideoTransformer3DModel, LTX25AutoBlocks
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video

from nunchaku_lite import patch_transformer
from nunchaku_lite.adapters.ltx2_decoder import optimize_ltx2_diffusion_decoder

model_dir = "ltx-2.5-distilled-diffusers"  # from tools/convert_ltx25_distilled.py
precision = "fp4"  # "fp4" on Blackwell, "int4" elsewhere
checkpoint = f"rootonchair/nunchaku-lite-ltx-2.5-distilled/svdq-{precision}_r32-ltx-2.5-distilled.safetensors"
device = "cuda"

# 1. quantized transformer
transformer = LTX2VideoTransformer3DModel.from_pretrained(
    model_dir, subfolder="transformer", torch_dtype=torch.bfloat16
)
patch_transformer(transformer, checkpoint, target="ltx2", precision=precision, device=device)


# 2. the other components, bf16
def load_component(name):
    path = os.path.join(model_dir, name)
    class_name = json.load(open(os.path.join(path, "config.json")))["_class_name"]
    cls = getattr(diffusers, class_name, None) or getattr(ltx2, class_name)
    return cls.from_pretrained(path, torch_dtype=torch.bfloat16)


guidance = dict(guidance_scale=1.0, stg_scale=0.0, modality_scale=1.0, guidance_rescale=0.0,
                spatio_temporal_guidance_blocks=None)  # distilled model: no guidance
components = {
    "transformer": transformer,
    "text_encoder": AutoModelForImageTextToText.from_pretrained(
        os.path.join(model_dir, "text_encoder"), torch_dtype=torch.bfloat16
    ),
    "tokenizer": AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer")),
    "scheduler": FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(model_dir, "scheduler")),
    "guider": LTX2Guidance(**guidance),
    "audio_guider": LTX2Guidance(**guidance),
    **{name: load_component(name) for name in
       ("connectors", "vae", "audio_vae", "vocoder", "diffusion_decoder", "duration_head")},
}

# 3. pipeline, all on the GPU (~55 GB peak at this size with the NVFP4 transformer)
pipe = LTX25AutoBlocks().init_pipeline()
pipe.update_components(**components)
for component in pipe.components.values():
    if isinstance(component, torch.nn.Module):
        component.to(device)

# 4. faster diffusion decoder (needs a nunchaku-lite-kernels build with the decoder ops)
optimize_ltx2_diffusion_decoder(pipe.diffusion_decoder)
pipe.diffusion_decoder.disable_tiling()

# 5. generate
steps = 8
state = pipe(
    prompt="A golden retriever runs through a sunlit meadow of tall grass, ears flapping, birds chirping.",
    width=768,
    height=512,
    num_frames=121,
    frame_rate=24.0,
    num_inference_steps=steps,
    sigmas=DISTILLED_SIGMA_VALUES[:steps],
    use_cross_timestep=True,
    enable_prompt_enhancement=False,
    generator=torch.Generator(device).manual_seed(0),
    output_type="np",
)
video, audio = state.get("videos"), state.get("audio")
encode_video(
    video[0],
    fps=24.0,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx25_nunchaku_lite.mp4",
)
```

`from_pretrained` on the transformer loads the bf16 weights first (about 44 GB of CPU memory) before they are replaced
by the quantized ones. The script was run as written on an RTX PRO 6000 with both checkpoints: 768x512, 121 frames
with audio, peak VRAM 55.8 GiB (NVFP4) and 55.3 GiB (INT4).

## Quality and speed

RTX PRO 6000, 6 prompts with people (seeds 47-52), 768x512x121, 8 steps. PSNR and SSIM are against bf16 on the same
GPU with the same decoder; s/clip is warm generation time per clip including decoding.

| Transformer | Decoder | PSNR vs bf16 | SSIM | s/clip | Peak VRAM |
|---|---|---|---|---|---|
| bf16 | nunchaku-lite kernels, untiled | - | - | 9.4 | 79.4 GB |
| NVFP4 r32 (this repo) | nunchaku-lite kernels, untiled | 20.91 dB | 0.710 | 6.2 | 55.2 GB |
| INT4 r32 + rotation (this repo) | nunchaku-lite kernels, untiled | 19.16 dB | 0.663 | 24.6 | 54.7 GB |
| bf16 | Diffusers default, tiled | - | - | 10.9 | 76.8 GB |
| NVFP4 r32 (this repo) | Diffusers default, tiled | 21.12 dB | 0.722 | 8.0 | 52.6 GB |
| Lightricks NVFP4 | Diffusers default, tiled | 18.86 dB | 0.671 | 10.2 | 55.2 GB |
| Lightricks INT8 | Diffusers default, tiled | 26.09 dB | 0.864 | 9.2 | 57.7 GB |

Each group is compared with the bf16 run that decodes the same way. Lightricks' INT8 checkpoint is the most accurate
option here; NVFP4 r32 is the fastest. INT4 is meant for GPUs without NVFP4 and is not fast on Blackwell. On an
A100 80GB (Diffusers default decoder) INT4 r32 + rotation takes 15.3 s/clip at 52.1 GB peak, vs 18.5 s for bf16;
Lightricks INT8 there scores 21.67 dB vs INT4's 17.15 dB.

## Diffusion decoder

LTX-2.5 decodes latents with a diffusion decoder whose last stage dominates decode time. Step 4 of the example
installs the nunchaku-lite kernels on it:

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
