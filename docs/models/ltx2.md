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

All rows run in the same Diffusers pipeline with the same (default, tiled) decoder: 6 prompts with people (seeds
47-52), 768x512, 121 frames, 8 steps. PSNR and SSIM compare each video with the bf16 video for the same GPU and seed;
s/clip is the full generation time per clip.

**Blackwell (RTX PRO 6000): `"fp4"`**

| Transformer | File size | Transformer per step | s/clip | Peak VRAM | PSNR vs bf16 | SSIM |
|---|---|---|---|---|---|---|
| bf16 | 39.1 GiB | 870 ms | 10.9 | 76.8 GB | - | - |
| NVFP4 r32 (nunchaku-lite) | 11.2 GiB | 475 ms | 8.0 | 52.6 GB | 21.12 dB | 0.722 |
| Lightricks INT8 | 20.0 GiB | 658 ms | 9.2 | 57.7 GB | 26.09 dB | 0.864 |
| Lightricks NVFP4 | 17.4 GiB | 786 ms | 10.2 | 55.2 GB | 18.86 dB | 0.671 |

**Other GPUs (A100 80GB): `"int4"`.** NVFP4 runs only on Blackwell.

| Transformer | File size | s/clip | Peak VRAM | PSNR vs bf16 | SSIM |
|---|---|---|---|---|---|
| bf16 | 39.1 GiB | 18.5 | 76.8 GB | - | - |
| INT4 r32 + rotation (nunchaku-lite) | 10.6 GiB | 15.3 | 52.1 GB | 17.15 dB | 0.659 |
| Lightricks INT8 | 20.0 GiB | 20.7 | 57.7 GB | 21.67 dB | 0.792 |

Lightricks' INT8 stays closest to bf16; the nunchaku-lite files are the fastest and smallest. With the decoder kernels
below and untiled decoding, an NVFP4 clip on the RTX PRO 6000 takes 6.2 s (55.2 GB peak) and a bf16 clip 9.4 s.
Lightricks' files run through ComfyUI's `comfy-kitchen` kernels in this pipeline.

## Lower VRAM: quantized text encoder

After the transformer, the biggest component is the Gemma 4 12B text encoder (22.3 GiB in bf16).
`load_text_encoder_bnb` loads it with bitsandbytes NF4 or 8-bit weights, already on the GPU (`pip install
"nunchaku-lite[bnb]"`). Replace the `"text_encoder"` entry of the example, and skip it when moving components to the
GPU:

```python
from nunchaku_lite.text_encoders import load_text_encoder_bnb

components["text_encoder"] = load_text_encoder_bnb(
    AutoModelForImageTextToText, os.path.join(model_dir, "text_encoder"), quantization="nf4", device=device
)  # or quantization="int8"

# step 3: move everything else
for name, component in pipe.components.items():
    if isinstance(component, torch.nn.Module) and name != "text_encoder":
        component.to(device)
```

Only the text decoder layers are quantized; the vision and audio towers and the LM head stay bf16.

RTX PRO 6000, NVFP4 transformer, everything on the GPU, 768x512x121:

| Text encoder | Size | Error vs bf16 encoder | Peak VRAM | s/clip |
|---|---|---|---|---|
| bf16 | 22.3 GiB | - | 55.2 GB | 6.2 |
| bitsandbytes 8-bit | 12.1 GiB | 1.1% | 46.4 GB | 6.4 |
| bitsandbytes NF4 | 7.0 GiB | 2.9% | 41.9 GB | 7.0 |

The error is the relative error of the encoder's hidden states on 6 prompts. On the generated clips, PSNR against the
full bf16 pipeline is 20.2 dB with NF4 and 19.5 dB with 8-bit, vs 20.9 dB with the bf16 encoder; the spread across
prompts (2-5 dB) is larger than these differences, so compare a few videos for your prompts.

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
