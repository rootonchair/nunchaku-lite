# Benchmarks

The benchmark scripts compare unmodified Diffusers pipelines against
`nunchaku_lite` pipelines loaded with quantized transformer weights.
FLUX.1, Z-Image, and Qwen-Image can also benchmark the original Nunchaku
package with `--run-original-nunchaku`. The scripts import `nunchaku` lazily
only when that optional run is requested.

Outputs are written under `outputs/benchmark_*/` and include generated images
plus a `summary.json` file with timing and CUDA memory statistics. Benchmarks
that run both original Diffusers and `nunchaku_lite` also write
`comparison.png` with latency and peak CUDA memory charts. When original
Nunchaku is enabled, it is included as a third comparison series.

## Z-Image

```bash
python benchmarks/benchmark_z_image.py \
  --model-id Tongyi-MAI/Z-Image-Turbo \
  --checkpoint nunchaku-ai/nunchaku-z-image-turbo/svdq-fp4_r128-z-image-turbo.safetensors \
  --precision fp4 \
  --dtype bf16 \
  --runs 3 \
  --warmup-runs 1
```

## FLUX.1

```bash
python benchmarks/benchmark_flux.py \
  --model-id black-forest-labs/FLUX.1-schnell \
  --checkpoint nunchaku-ai/nunchaku-flux.1-schnell/svdq-fp4_r32-flux.1-schnell.safetensors \
  --precision fp4 \
  --dtype bf16 \
  --runs 3 \
  --warmup-runs 1
```

## FLUX.2

```bash
python benchmarks/benchmark_flux2.py \
  --model-id tonera/FLUX.2-klein-9B-Nunchaku \
  --checkpoint tonera/FLUX.2-klein-9B-Nunchaku/svdq-fp4_r32-FLUX.2-klein-9B-Nunchaku.safetensors \
  --precision fp4 \
  --dtype bf16 \
  --runs 3 \
  --warmup-runs 1
```

## Qwen-Image

```bash
python benchmarks/benchmark_qwen_image.py \
  --model-id Qwen/Qwen-Image \
  --checkpoint nunchaku-tech/nunchaku-qwen-image/svdq-fp4_r32-qwen-image.safetensors \
  --precision fp4 \
  --dtype bf16 \
  --runs 3 \
  --warmup-runs 1
```
