# nunchaku_lite

<p align="center">
  <img src="assets/logo.svg" alt="nunchaku_lite" width="640">
</p>

`nunchaku_lite` loads Nunchaku-quantized image generation components into
standard Diffusers pipelines. It keeps the integration surface small: construct
the target Diffusers pipeline, replace only the quantized transformer or UNet,
and keep scheduling, prompting, LoRA loading, and image generation in regular
Diffusers code.

## Measured Gains

On an NVIDIA RTX PRO 6000, the current benchmark set
shows up to `1.79x` lower latency, up to `46%` lower peak CUDA memory, and up to
`71%` smaller transformer storage versus unmodified Diffusers pipelines.

| Model | Speedup | Peak CUDA memory | Transformer storage |
| --- | ---: | ---: | ---: |
| FLUX.2 Klein 9B | `1.79x` | `34.73 GB -> 23.24 GB` | `16.91 GiB -> 5.40 GiB` |
| Qwen-Image | `1.59x` | `57.94 GB -> 31.21 GB` | `38.05 GiB -> 11.13 GiB` |
| Z-Image Turbo | `1.68x` | `21.67 GB -> 14.07 GB` | `11.46 GiB -> 3.91 GiB` |

See [Benchmarks](benchmarks.md) for charts, generated samples, run settings,
and reproducibility details.

## Start Here

- [Benchmarks](benchmarks.md) shows measured speed, memory, and model-size
  reductions against original Diffusers pipelines.
- [API Reference](api.md) covers the public loading and adapter APIs.
- [Development Guide](development.md) covers local validation, adapter authoring,
  and runtime LoRA implementation.
- [Roadmap](roadmap.md) tracks model support and remaining feature work.
- [Documentation Deployment](deployment.md) explains how to update and publish
  this documentation site.

## Supported Families

| Model family | Adapter target | Runtime LoRA |
| --- | --- | --- |
| FLUX.1 | `flux` | Yes |
| FLUX.2 Klein | `flux2` | Yes |
| Qwen-Image and Qwen-Image-Edit | `qwen_image` | Yes |
| SDXL and SDXL-Turbo | `sdxl` | Not yet |
| Z-Image Turbo | `z_image` | Yes |

Runnable model guides are stored under [Supported models](models/flux.md).
