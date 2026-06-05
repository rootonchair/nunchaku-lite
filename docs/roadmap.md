# Roadmap

This document tracks supported model coverage and the feature backlog from the
full `nunchaku` package.

## Design Goals

- Keep the public integration surface model-agnostic.
- Load pipelines without first materializing unused dense transformer or UNet
  weights.
- Keep low-level in-place patching available for advanced use.
- Use a registry of small adapters for model-specific graph rewrites.
- Package only the native kernels and Python code required for the lite runtime.
- Avoid a hard dependency on the original `nunchaku` Python package.

## Supported Models

| Model family | Diffusers component | Adapter target | Runtime LoRA |
| --- | --- | --- | --- |
| FLUX.1 | `FluxTransformer2DModel` | `flux` | Yes |
| FLUX.2 Klein | `Flux2Transformer2DModel` | `flux2` | Yes |
| Qwen-Image and Qwen-Image-Edit | `QwenImageTransformer2DModel` | `qwen_image` | Yes |
| SDXL and SDXL-Turbo | `UNet2DConditionModel` | `sdxl` | Not yet |
| Z-Image Turbo | `ZImageTransformer2DModel` | `z_image` | Yes |

Additional model families should be added through the common adapter registry
rather than through pipeline-specific subclasses.

## Feature Backlog from Original `nunchaku`

The full `nunchaku` package in this repository exposes a broader set of
model-specific loaders and workflow integrations. Use this checklist as the
current porting backlog for `nunchaku_lite`:

- [x] FLUX.1 transformer adapter for Diffusers `FluxTransformer2DModel`.
- [x] Flux2 transformer adapter for Diffusers `Flux2Transformer2DModel`.
- [x] Qwen-Image transformer adapter based on
  `NunchakuQwenImageTransformer2DModel`, covering Qwen-Image,
  Qwen-Image-Lightning, Qwen-Image-Edit, Qwen-Image-Edit-2509, and Qwen-Image
  ControlNet examples.
- [x] Z-Image transformer adapter for Diffusers `ZImageTransformer2DModel`.
- [ ] Sana transformer adapter based on `NunchakuSanaTransformer2DModel`,
  covering Sana 1.6B and Sana PAG examples.
- [x] SDXL UNet adapter based on `NunchakuSDXLUNet2DConditionModel`, covering
  SDXL and SDXL-Turbo examples.
- [x] Quantized T5 text encoder support based on `NunchakuT5EncoderModel`.
- [ ] Benchmark quantized text encoder memory and latency impact in end-to-end
  pipelines.
- [ ] Quantized Qwen3 text encoder support, covering FLUX.2 Klein and Z-Image.
- [ ] Quantized Qwen2.5-VL text encoder support, covering Qwen-Image and
  Qwen-Image-Edit.
- [x] FLUX runtime LoRA support, including Diffusers-format conversion,
  Nunchaku-format loading, strength control, reset, and multi-LoRA composition.
- [x] Qwen-Image runtime LoRA support, covering Qwen-Image and Qwen-Image-Edit
  families.
- [x] Flux2 runtime LoRA support, including ComfyUI Flux2 LoRA key conversion.
- [ ] SDXL runtime LoRA support for quantized UNet attention and MLP
  projections.
- [x] Z-Image runtime LoRA support, including dense AdaLN modulation LoRA
  branches.
- [ ] FLUX IP-Adapter integration.
- [ ] Full inference test coverage for FLUX IP-Adapter.
- [ ] Full inference test coverage for FLUX.2 IP-Adapter.
- [ ] FLUX PuLID pipeline or patching support.
- [ ] FLUX ControlNet workflow coverage for Canny, Depth, Fill, and
  ControlNet-Union variants.
- [ ] Caching integrations equivalent to TeaCache, first-block cache,
  double-block cache, and DiT cache examples.
- [ ] Async/offload paths for lower-VRAM inference where supported by the
  original implementation.
- [ ] Remove monkey-patched transformer forward overrides in favor of module
  wrappers or inherited transformer implementations.

## Runtime Manifest Adapter Backlog

The generic `manifest` adapter currently covers checkpoint-driven linear
replacement and simple structural rewrites. The remaining work is to let
manifest-declared checkpoints opt into the same model-level fast paths that the
handwritten adapters use:

- [ ] Declare fused attention groups so separate `to_q`, `to_k`, and `to_v`
  projections can be replaced by a model-specific attention wrapper.
- [ ] Route manifest-created attention wrappers through
  `fused_qkv_norm_rotary` when the model exposes compatible Q/K norms and
  rotary embeddings.
- [ ] Express two-layer MLP patterns that can use `fused_gelu_mlp` instead of
  running the projections and activation as separate modules.
- [ ] Expose manifest-driven attention backend selection, including
  `nunchaku-fp16` where the underlying model and kernels support it.
- [ ] Add manifest hooks for Flux-style cache integrations, including
  TeaCache, first-block cache, and double-block cache behavior.
- [x] Add a general AWQ GEMM path for large W4A16 group-size-64/int32
  projections. The TinyChat AWQ GEMM path remains separate for
  group-size-128/int16 weights, while compatible group-size-64/int32 weights now
  dispatch to GEMM instead of chunked GEMV for larger batches.
- [x] Bind model-family runtime APIs, such as LoRA loading and adapter
  management, when a manifest target corresponds to an existing family adapter.

## Notes

- `nunchaku_lite` does not import or require the full `nunchaku` Python package.
- Generated artifacts, local outputs, compiled extensions, caches, and virtual
  environments are intentionally ignored by git.
