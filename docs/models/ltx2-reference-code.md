# LTX-2.5 in Lightricks' reference code

Nunchaku Lite LTX-2.5 transformers also run in Lightricks' own code,
[`ltx-core` / `ltx-pipelines`](https://github.com/Lightricks/LTX-2), with no Diffusers involved. The checkpoint plugs in
the same way as Lightricks' own `nvfp4-prequant`: it is an ltx-core `QuantizationPolicy`.

## Checkpoint

The published file is Lightricks' transformer file
(`ltx-2.5-22b-distilled-transformer-bf16.safetensors`) with every quantized layer replaced by its SVDQ W4A4 tensors:

```
ltx-2.5-22b-distilled-transformer-nunchaku-nvfp4-r32.safetensors          # Blackwell
ltx-2.5-22b-distilled-transformer-nunchaku-int4-r32-e0-hadall.safetensors # other GPUs
```

Every other tensor (norms, AdaLN, embeddings, the text connectors) and the file metadata are byte-identical to the
official file, so it works anywhere the official transformer file does. `tools/ltx2_native_splice.py` writes one from a
Diffusers-layout export and verifies it:

```bash
python tools/ltx2_native_splice.py \
    --official  ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    --quantized nvfp4-r32/transformer/diffusion_pytorch_model.safetensors \
    --out       ltx-2.5-22b-distilled-transformer-nunchaku-nvfp4-r32.safetensors
```

`--targets-regex` splices only the quantized layers whose full name matches; the others keep the official bf16
weights, and the file's `quantization_config` names only the spliced layers.

## Command line

Any `ltx_pipelines` entry point runs through the wrapper; pass our file as `--transformer-path` and leave out
`--quantization`:

```bash
python -m nunchaku_lite.integrations.ltx_core distilled \
    --transformer-path       ltx-2.5-22b-distilled-transformer-nunchaku-nvfp4-r32.safetensors \
    --text-encoder-path      text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path         vae/ltx-2.5-video-vae-bf16.safetensors \
    --audio-vae-path         vae/ltx-2.5-audio-vae-bf16.safetensors \
    --spatial-upsampler-path latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --num-frames 121 --seed 42 --output-path output.mp4 --prompt "..."
```

`--nunchaku-stock-attention` keeps ltx-core's own Q/K RMSNorm + RoPE instead of the fused kernel.

## Python

```python
from ltx_pipelines.distilled import DistilledPipeline
from nunchaku_lite.integrations.ltx_core import build_nunchaku_policy

policy = build_nunchaku_policy("ltx-2.5-22b-distilled-transformer-nunchaku-nvfp4-r32.safetensors")
pipeline = DistilledPipeline(model_paths=..., spatial_upsampler_path=..., loras=(), quantization=policy)
```

The policy swaps the quantized `nn.Linear`s of ltx-core's meta model for SVDQ layers before ltx-core binds the weights,
so ltx-core's loader, its exact key check, streaming and offloading work as for any checkpoint. It also

* runs the NVFP4 feed-forward through the fused GELU-MLP kernel (never for INT4 or rotated layers);
* fuses each quantized attention's Q/K RMSNorm + split RoPE into one CUDA kernel, falling back to ltx-core's code for
  anything that kernel does not cover. If Lightricks changes `Attention.forward` or `FeedForward.forward`, the
  replacement is skipped with a warning and the stock forward runs.

Not supported yet: LoRAs on quantized layers (refused with an error), Apple silicon for rotated checkpoints.

!!! note "`HF_HUB_DISABLE_TELEMETRY`"
    With `kernels` 0.14.1, setting `HF_HUB_DISABLE_TELEMETRY=1` makes the kernel download send an empty `User-Agent`
    header, and loading the Nunchaku Lite kernels fails with "Illegal header value". Unset the variable when the
    kernels are first downloaded.

## Results (RTX PRO 6000 Blackwell)

Lightricks' `DistilledPipeline` (two stages, default settings, convolutional VAE), 768x512, 121 frames, 16 prompts
(seeds 42-52), each clip scored against the bf16 clip of the same seed in the same pipeline. The ancestral sampler
amplifies any numerical difference, so absolute PSNR is lower than in single-stage comparisons; the comparison
between rows is what matters.

| transformer | PSNR (dB) | SSIM | flicker (bf16: 7.52) | transformer s/clip | s/clip | peak VRAM | file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bf16 (Lightricks) | - | - | - | 4.46 | 11.38 | 36.2 GB | 39.1 GB |
| NVFP4 (Lightricks `nvfp4-prequant`) | 18.43 | 0.570 | 8.14 | 2.87 | 9.38 | 24.2 GB | 17.4 GB |
| **Nunchaku Lite NVFP4 r32** | 18.09 | 0.558 | 9.68 | 3.01 | 8.96 | 24.2 GB | 14.9 GB |

The fused Q/K norm + RoPE kernel matches ltx-core's own path to bf16 rounding (relative difference 1.7e-3) and saves
about 3% of transformer time; `--nunchaku-stock-attention` gives the same quality.

## For integrators: `nunchaku_lite.loader`

The framework-agnostic core used here (and meant for other runtimes such as ComfyUI) needs only torch:

| function | does |
| --- | --- |
| `read_checkpoint_spec(path)` | reads and validates the header (no tensor data): layout, precision, rank, per-layer shapes and rotation |
| `replace_linears(root, spec, name_map=...)` | swaps each target module by exact name for an empty SVDQ layer (meta allowed) |
| `bind_quantized_state(root, spec)` | loads the target tensors and returns every file key it consumed |
| `gelu_mlp_plan(fc1, fc2)` | the one rule for fusing a GELU MLP (NVFP4, neither layer rotated) |
| `check_runtime_support(spec, device)` | refuses NVFP4 below Blackwell, rotation where unsupported |

### Frozen checkpoint rules

| input | accepted | refused |
| --- | --- | --- |
| layout | `model.diffusion_model.*` (native), or no prefix (Diffusers) | — |
| `quantization_config` | safetensors metadata; Diffusers layout may use a sibling `config.json` | native file without metadata |
| targets | `svdq_w4a4` target list or runtime manifest | other ops, structural patches |
| per layer | `qweight` int8 `(out, in/2)`, `wscales` `(in/g, out)` (bf16/fp16 INT4, fp8 NVFP4), `smooth_factor`, `proj_down`, `proj_up`, `bias` iff declared, NVFP4 `wcscales` (+ optional one-element `wtscale`) | any missing, extra, nested, mis-shaped or mis-typed tensor; group size other than 64 (INT4) / 16 (NVFP4); rank not a multiple of 16 or disagreeing with the declaration |
| rotation | `input_rotation` declaration + root `nunchaku_lite.input_rotation` marker (or a version 2 manifest declaring it, which must agree) | marker without declaration, declaration without marker, rotated non-targets, marker under the native prefix, NVFP4 rotation unless `allow_fp4_rotation`, legacy `.input_transform` tensors |
| `smooth_factor_orig` | dropped (reported as consumed) | — |

Invariants (tested): every target is swapped exactly once and atomically; bound tensors are bit-exact (int8/fp8) or
cast once to the layer dtype; `wtscale` is a float32 buffer that a module-wide `.to(dtype)` does not change, and the
GEMM reads it once per load; nothing stays on the meta device; casts never change `int8`/`fp8` tensors.
