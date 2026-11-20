# API Reference

`nunchaku_lite` exposes a small public API for loading quantized Nunchaku
components into Diffusers pipelines and for registering model-specific adapters.

```python
from nunchaku_lite import (
    NunchakuT5EncoderModel,
    TransformerAdapter,
    list_adapters,
    load_nunchaku_pipeline,
    load_quantized_text_encoder,
    patch_transformer,
    quantize_text_encoder,
    register_adapter,
)
```

## `load_nunchaku_pipeline`

```python
pipe = load_nunchaku_pipeline(
    "black-forest-labs/FLUX.1-schnell",
    pipeline_cls=FluxPipeline,
    checkpoint="nunchaku-ai/nunchaku-flux.1-schnell/svdq-fp4_r32-flux.1-schnell.safetensors",
    target="flux",
    precision="fp4",
    torch_dtype=torch.bfloat16,
    device="cuda",
)
```

`load_nunchaku_pipeline` is the preferred public API. It reads the pipeline
config, constructs the selected `transformer` or `unet` on the meta device,
patches it with the Nunchaku adapter, loads the quantized checkpoint with
`assign=True`, and passes the patched component into
`pipeline_cls.from_pretrained(...)`. Diffusers then loads the rest of the
pipeline normally while skipping the original dense component. Adapters that
provide pipeline runtime APIs patch the loaded pipeline automatically.

Arguments:

- `pretrained_model_name_or_path`: Diffusers pipeline model id or local path.
- `pipeline_cls`: Diffusers pipeline class, such as `FluxPipeline` or
  `StableDiffusionXLPipeline`.
- `checkpoint`: local or Hugging Face `.safetensors` checkpoint path.
- `target`: adapter name, or `"auto"` to select the only matching adapter.
- `component`: optional `"transformer"` or `"unet"` override. Auto-selection
  prefers `transformer`, then `unet`.
- `precision`, `torch_dtype`, `device`, `strict`, and `adapter_options`: same
  patching controls as `patch_transformer`.
- `quantized_encoder_checkpoints`: optional mapping from Diffusers component
  name to a quantized encoder checkpoint. Currently supports T5 checkpoints.
- `text_encoder_quantization`: quantize the pipeline's Transformers text
  encoders at load time, weights only: `"int4"` or `"int8"` for every
  `text_encoder*` component, or a dict such as `{"text_encoder_2": "int4"}`.
  No quantized checkpoint is needed. See
  [Text encoder quantization](#text-encoder-quantization).
- Additional keyword arguments are forwarded to `pipeline_cls.from_pretrained(...)`.

Quantized text encoders can be injected up front so Diffusers skips loading the
dense encoder:

```python
pipe = load_nunchaku_pipeline(
    "black-forest-labs/FLUX.1-dev",
    pipeline_cls=FluxPipeline,
    checkpoint="nunchaku-tech/nunchaku-flux.1-dev/svdq-int4_r32-flux.1-dev.safetensors",
    target="flux",
    quantized_encoder_checkpoints={
        "text_encoder_2": "mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors",
    },
    torch_dtype=torch.bfloat16,
    device="cuda",
)
```

## Text encoder quantization

The text encoder is often the largest part of a pipeline after the transformer
is quantized: FLUX.2 Klein 4B ships an 8 GB bf16 Qwen3-4B encoder, Qwen-Image a
16 GB Qwen2.5-VL. `text_encoder_quantization` quantizes it while loading:

```python
pipe = load_nunchaku_pipeline(
    "black-forest-labs/FLUX.2-klein-4B",
    pipeline_cls=Flux2KleinPipeline,
    checkpoint="lite-infer/flux.2-klein-4b-nunchaku-lite-int4-r32",
    torch_dtype=torch.bfloat16,
    device="cuda",  # or "mps"
    text_encoder_quantization="int4",  # or "int8"
)
```

- Weights are quantized round-to-nearest; activations stay in bf16/fp16.
  `"int4"` is asymmetric with groups of 64 input channels, `"int8"` symmetric
  per output channel.
- The bf16 encoder is loaded onto the CPU first (Transformers memory-maps it
  when the checkpoint dtype matches, so this costs almost no RAM) and each
  linear layer is quantized onto `device`, so the dense encoder is never
  resident on the GPU.
- On CUDA, `"int4"` layers run the AWQ W4A16 kernels (`AWQW4A16Linear`). On
  MPS and CPU they are `Int4WeightOnlyLinear`, which dequantizes one layer at a
  time per call and runs a dense matmul. `"int8"` layers
  (`Int8WeightOnlyLinear`) do the same.
- Kept dense: layers whose weight is tied to another parameter (`lm_head` tied
  to the input embeddings, as in Qwen3-4B), vision towers (`visual` in
  Qwen2.5-VL, used by Qwen-Image-Edit), and, for `"int4"`, layers whose input
  features are not a multiple of 64. Embeddings stay in bf16.
- Diffusers' text encoder LoRA loading (PEFT) does not support the quantized
  layers.

On an M5 MacBook Pro, FLUX.2 Klein's Qwen3-4B encoder (512-token prompts,
hidden states of layers 9/18/27 as the pipeline uses them):

| Encoder | GPU memory | Encode | Hidden-state error vs bf16 |
| --- | ---: | ---: | --- |
| bf16 | 8.11 GB | 0.70 s | |
| `"int8"` | 4.45 GB | 0.91 s | 0.85% (cosine 0.9996) |
| `"int4"` | 2.85 GB | 1.27 s | 6.0% (cosine 0.977-0.983) |

Qwen-Image's Qwen2.5-VL encoder (last hidden state, the prompt tokens the
pipeline keeps; vision tower left in bf16):

| Encoder | GPU memory | Encode | Hidden-state error vs bf16 |
| --- | ---: | ---: | --- |
| bf16 | about 16 GB | | |
| `"int8"` | 9.67 GB | 1.02 s | 3.0% (cosine 0.9995) |
| `"int4"` | 6.52 GB | 1.35 s | 21% (cosine 0.978) |

`"int8"` is close to lossless on both. `"int4"` gives FLUX.2 Klein images with
the same composition as bf16, differing in fine detail. On Qwen-Image its error
is much larger and it has not been checked on images yet; prefer `"int8"` there.

On an RTX PRO 6000 (FLUX.2 Klein 4B pipeline with the FP4 transformer, 1024x1024, 4 steps; real prompt tokens
compared):

| Encoder | Encoder size | VRAM after load | Generation peak | Embedding error vs bf16 |
| --- | ---: | ---: | ---: | --- |
| bf16 | 8.04 GB | 10.90 GB | 13.49 GB | |
| `"int8"` | 4.41 GB | 7.24 GB | 9.83 GB | 0.74% (cosine 0.9996) |
| `"int4"` (AWQ kernels) | 2.82 GB | 5.65 GB | 8.23 GB | 5.9% (cosine 0.977) |

For a model you load yourself, use the functions directly:

```python
from nunchaku_lite import load_quantized_text_encoder, quantize_text_encoder

text_encoder = load_quantized_text_encoder(
    "black-forest-labs/FLUX.2-klein-4B", "int4", subfolder="text_encoder", device="cuda"
)
quantize_text_encoder(model, "int8", device="mps")  # any model, in place
```

## `NunchakuT5EncoderModel`

```python
text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
    "mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors",
    torch_dtype=torch.bfloat16,
    device="cuda",
)
```

`NunchakuT5EncoderModel` loads original Nunchaku AWQ INT4 T5 encoder
checkpoints and can be passed manually as a Diffusers pipeline component, such
as FLUX.1 `text_encoder_2`. The quantized forward path uses CUDA native kernels.

## `patch_transformer`

```python
patch_transformer(
    transformer,
    checkpoint,
    target="auto",
    precision="auto",
    torch_dtype=None,
    device=None,
    strict=True,
    adapter_options=None,
)
```

Arguments:

- `transformer`: the Diffusers transformer or UNet module to patch.
- `checkpoint`: local or Hugging Face `.safetensors` checkpoint path.
- `target`: adapter name, or `"auto"` to select the only matching adapter.
- `precision`: `"auto"`, `"fp4"`, or `"int4"`. Internally, `"fp4"` maps to
  NVFP4 kernels.
- `torch_dtype`: optional model dtype, typically `torch.bfloat16` or
  `torch.float16`.
- `device`: optional destination device after patching.
- `strict`: forwarded to `load_state_dict`.
- `adapter_options`: model-specific adapter options.

This is the low-level compatibility API for callers that already constructed a
component. Prefer `load_nunchaku_pipeline` for normal pipeline loading because it
avoids loading dense weights that are immediately replaced. The function is
idempotent for the same target. A transformer patched once will be returned
unchanged if patched again with the same target.

## Runtime LoRA

Pipelines loaded through `load_nunchaku_pipeline` expose Diffusers-style runtime
LoRA methods when their adapter provides pipeline support:

```python
pipe = load_nunchaku_pipeline(
    model_id,
    pipeline_cls=FluxPipeline,
    checkpoint=checkpoint,
    target="flux",
    torch_dtype=torch.bfloat16,
)

pipe.load_lora_weights("artist-style.safetensors", adapter_name="artist")
pipe.set_adapters("artist", adapter_weights=0.5)
pipe.unload_lora_weights()
```

Runtime LoRA loading accepts Diffusers-format LoRAs and Nunchaku-format low-rank
tensors for supported adapters. Use pipeline-level `load_lora_weights` on
pipelines loaded with `load_nunchaku_pipeline`, or transformer-level `load_lora`
on directly patched components. Multiple LoRAs can be active at once; they are
recomposed from the original checkpoint low-rank state when strengths change or
one LoRA is reset.

Advanced callers using `patch_transformer` directly can bind pipeline LoRA
methods manually:

```python
from nunchaku_lite.lora.core.runtime import NunchakuPipelineLoraMixin, bind_pipeline_lora_methods

bind_pipeline_lora_methods(pipe, NunchakuPipelineLoraMixin)
```

## Adapter Registry

Adapters implement a small protocol:

```python
class MyAdapter:
    target = "my_model"

    def matches(self, transformer):
        return transformer.__class__.__name__ == "MyTransformer"

    def patch(self, transformer, checkpoint_state, quantization_config, options):
        # Rewrite modules, install hooks, or normalize checkpoint keys.
        return checkpoint_state

    def patch_pipeline(self, pipeline, *, component_name="transformer", component=None):
        # Optional: attach pipeline-level runtime APIs.
        return None
```

Register an adapter before calling `load_nunchaku_pipeline` or
`patch_transformer`:

```python
from nunchaku_lite import register_adapter

register_adapter(MyAdapter())
```

Model-specific code should stay inside adapters. Pipeline construction,
scheduling, prompting, and image generation should remain standard Diffusers
code.
