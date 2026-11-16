# MLX runtime (Apple silicon)

`nunchaku_lite.mlx` is the MLX runtime for Nunchaku SVDQ INT4 checkpoints on Apple silicon, with fused Metal
kernels that run on the M5 GPU's Neural Accelerators.

## Installation

```bash
pip install "nunchaku-lite[mlx]" mflux
```

The `mlx` extra installs MLX only; torch and Diffusers are not needed (they come with the `torch` extra for the
PyTorch runtime). Pipelines are built with [mflux](https://github.com/filipstrand/mflux).

## Why W4A8 instead of W4A4

Nunchaku's CUDA speedup comes from quantizing activations to 4 bits and using INT4
tensor cores. Apple GPUs have no INT4 matmul. Instead, M5-class GPUs expose
Neural Accelerators through Metal 4 `mpp::tensor_ops::matmul2d`. Compute-only
throughput measured on an M5 (AC power):

| Operands (accumulator) | Throughput |
| --- | --- |
| int8 x int8 (int32) | ~30 TOPS |
| int8 x int4 (int32) | ~30 TOPS |
| fp16/bf16 x anything | ~15 TFLOPS |

So the fast path keeps the checkpoint's 4-bit weights and its SVD low-rank branch,
and quantizes activations to **int8**, one scale per token, after SVDQ smoothing.
Nunchaku's 64-channel weight groups are folded into a per-channel scale plus
integer group multipliers. The int32 accumulator then runs through the whole of
K inside the accelerator, with no readouts. Reading the accumulator back once per
group costs about 2.5x throughput. The scales and bias are applied once, in the
kernel epilogue. The rank-R correction runs as a second, small accelerator matmul
in bf16/fp16. Its accumulator shares the main accumulator's per-thread layout, so
the two are summed in registers. On the ALUs, the correction alone cost about 30%
of kernel time. Threadgroup swizzling keeps weight tiles in L2. The fused GEMM
reaches about 29 TOPS for K = 3072, about 95% of the int8 compute ceiling.

## Interface

The API mirrors `nunchaku_lite` (the PyTorch runtime):

| `nunchaku_lite` (PyTorch, CUDA / MPS) | `nunchaku_lite.mlx` (MLX) |
| --- | --- |
| `load_nunchaku_pipeline(model, pipeline_cls=..., checkpoint=..., target=..., backend=...)` (Diffusers) | same signature (mflux pipelines) |
| `patch_transformer(transformer, checkpoint, ...)` | `patch_transformer(model, checkpoint, ...)` |
| `register_adapter`, `list_adapters` | same |
| `SVDQW4A4Linear.from_linear(linear, rank=..., precision="int4")` | same |
| `AWQW4A16Linear` (AdaNorm / modulation layers) | same (`adanorm_splits=` restores the original layer order) |
| parameters `qweight`, `wscales`, `smooth_factor`, `proj_down`, `proj_up`, `bias` | same names, loaded with `module.load_weights(...)` |
| `act_unsigned` applies the INT4 post-GELU shift | same |
| `backend="auto"\|"int8"\|"int4"\|"dense"` | same values and `auto` rule |
| op `svdq_quantize_act_int8`, `svdq_gemm_w8a8`, `svdq_gemm_w4a8` | same op names |

### Load a pipeline (mflux)

```python
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.variants import Flux2Klein
from nunchaku_lite.mlx import load_nunchaku_pipeline

model = load_nunchaku_pipeline(
    "mflux-community/flux2-klein-9b-mflux-q4",  # base model: text encoder, VAE, tokenizer, scheduler
    pipeline_cls=Flux2Klein,
    checkpoint="lite-infer/flux.2-klein-9b-nunchaku-lite-int4-r32",  # only transformer/ is downloaded
    backend="auto",
    model_config=ModelConfig.flux2_klein_9b(),  # forwarded to Flux2Klein
)
image = model.generate_image(seed=1, prompt="A cat holding a sign that says hello world", num_inference_steps=4)
```

FLUX.1-dev works the same way with mflux's `Flux1`:

```python
from mflux.models.flux.variants.txt2img.flux import Flux1

model = load_nunchaku_pipeline(
    "mflux-community/flux-1-dev-mflux-q4",
    pipeline_cls=Flux1,
    checkpoint="lite-infer/flux.1-dev-nunchaku-lite-int4-r32",
    model_config=ModelConfig.dev(),
)
image = model.generate_image(seed=0, prompt="A red fox in a misty forest", num_inference_steps=20, guidance=3.5)
```

`checkpoint` accepts a Diffusers repo on the Hub (`org/repo`, as above), a local
repo or checkpoint folder (sharded files are merged), a `.safetensors` file, or
`org/repo/path/file.safetensors`.

Two checkpoint layouts are supported, as in `nunchaku_lite`:

- **Nunchaku Lite checkpoints with a runtime manifest** (the `lite-infer`
  repos). The manifest in the safetensors metadata lists every quantized layer
  (SVDQ W4A4, AWQ W4A16, AdaNorm AWQ) and any structural splits. Its module
  names are Diffusers names. The adapter maps them to mflux's modules with a
  name map built from mflux's own Diffusers weight mapping, so no per-model
  table is needed. The manifest parser is the same code the PyTorch runtime
  uses (`nunchaku_lite.manifest`). Older lite checkpoints without a manifest
  (`lite-infer/flux.1-dev-nunchaku-lite-int4-r32`) list their quantized layers
  in `config.json` (`svdq_w4a4.targets`, `awq_w4a16.targets`); a manifest is
  built from those lists.
- **Original Nunchaku checkpoints** (for example
  `tonera/FLUX.2-klein-4B-Nunchaku/svdq-int4_r32-FLUX.2-klein-4B-Nunchaku.safetensors`).
  Keys are normalized with the same `convert_flux2_state_dict` as
  `nunchaku_lite`. Fused checkpoint layers such as `to_qkv` then serve mflux's
  separate `to_q`/`to_k`/`to_v` with one matmul. Supported for FLUX.2 only;
  original Nunchaku FLUX.1 checkpoints are not supported on MLX.

The base pipeline can be a bf16 repo or a pre-quantized mflux repo (`-q4`,
`-q8`). For a pre-quantized base, the transformer is rebuilt from the
pipeline's `model_config` and every weight comes from the Nunchaku checkpoint.
Loading fails if the checkpoint lacks any. This uses less memory than a bf16
base: the 4B peaks at 5.9 GB with `int4`.

On 24 GB Macs, cap MLX's buffer cache for the 9B (`mx.set_cache_limit(1 << 30)`).
Otherwise freed buffers (up to about 9 GB) stay resident, and the first step of
every image swaps.

Registered adapters: `flux1` (FLUX.1, mflux `Flux1`), `flux2` (FLUX.2 Klein),
and `manifest`, a generic adapter
for any transformer whose module names match the manifest's. `target="auto"`
picks the model-family adapter, else `manifest` when the checkpoint has one.
Requires `pip install mflux`.

### Patch a model

```python
import mlx.core as mx
from nunchaku_lite.mlx import patch_transformer

model = MyFluxTransformer(...)  # any MLX model whose projections are nn.Linear
patch_transformer(
    model,
    "svdq-int4_r32-flux.1-dev.safetensors",
    # FLUX.1 / Qwen-Image INT4 MLP fc2 inputs are quantized with Nunchaku's unsigned GELU shift
    act_unsigned=lambda prefix: prefix.endswith(("mlp_fc2", "ff.net.2")),
    key_map=lambda prefix: prefix,  # map checkpoint names to model paths if they differ
    backend="auto",  # or "int8" / "int4" / "dense"
)
```

With a runtime-manifest checkpoint, the manifest decides the layers, and
`key_map` (a callable, or a `PrefixMap` of Diffusers to model module patterns)
maps its names. Otherwise every checkpoint prefix with SVDQ tensors replaces the `nn.Linear` at that path
with an `SVDQW4A4Linear`; the remaining tensors (norms, embeddings, unquantized
layers) load as regular weights. Both lite (`proj_down`, `proj_up`,
`smooth_factor`) and original Nunchaku (`lora_down`, `lora_up`, `smooth`) key
names are accepted. AWQ W4A16 layers (e.g. FLUX.1 AdaNorm `norm1.linear`) become
`AWQW4A16Linear` (4-bit `mx.quantized_matmul`, lossless); pass
`adanorm_splits={prefix: 6, ...}` for layers stored in Nunchaku's AdaNorm layout.

### Single layer

```python
from mlx import nn
from nunchaku_lite.mlx import SVDQW4A4Linear

fc2 = SVDQW4A4Linear.from_linear(nn.Linear(12288, 3072), rank=32, act_unsigned=True)
fc2.load_weights(list(tensors.items()))  # this layer's Nunchaku checkpoint tensors
y = fc2(x)  # x: (..., 12288) bf16; converted to the backend layout on first call
```

The weights are converted to the backend layout on the first call, and again
whenever `load_weights` / `update` replaces them (for example a backend switch
or new low-rank weights).

## Backends

| `backend` | Weights in memory | Hardware | Notes |
| --- | --- | --- | --- |
| `int8` | int8 (0.5x bf16) | M5+ | Fastest (W8A8 Neural Accelerator kernel). |
| `int4` | int4 (same as checkpoint) | M5+ | Expanded to int8 inside the kernel; same output as `int8`. Use when memory is tight. |
| `dense` | int4 | Any Apple GPU | Exact W4A16 fallback via `mx.quantized_matmul` (lossless INT4 mapping). |

`auto` (default) selects `int8` on GPUs with Neural Accelerators and `dense`
elsewhere; `patch_transformer` resolves it once for the whole model and falls back
to `int4` when the model would exceed half of the GPU working set with int8
weights. Change a layer's backend later by setting `layer.backend`.

After conversion the packed `qweight` parameter is released when the backend copy
reproduces it exactly (always for `int4` and `dense`, per layer for `int8`): it
becomes a lazy re-pack expression that costs no memory until evaluated (for
example when saving weights) and then yields the exact checkpoint bytes.
Evaluating it explicitly (for example `mx.eval(model.parameters())`)
materializes the copy again.

## Results (Apple M5, MacBook Pro, AC power)

FLUX.1 projections, bf16, including activation quantization and the low-rank
branch. Run with `python benchmarks/benchmark_mlx_linear.py --tokens 4608`.

At 1024x1024 (4608 tokens):

| Layer | bf16 dense | `int4` | `int8` |
| --- | --- | --- | --- |
| qkv_proj 3072->9216 | 18.1 ms | 12.2 ms (1.48x) | 10.3 ms (1.76x) |
| mlp_fc1 3072->12288 | 23.8 ms | 16.1 ms (1.49x) | 15.1 ms (1.58x) |
| mlp_fc2 12288->3072 | 33.4 ms | 19.5 ms (1.72x) | 19.8 ms (1.69x) |
| out_proj 3072->3072 | 6.2 ms | 4.8 ms (1.29x) | 3.8 ms (1.62x) |
| **total** | 81.5 ms | 52.5 ms (1.55x) | 49.0 ms (1.66x) |

At 512x512 (1024 tokens), `int8` is 1.77x faster than bf16 overall and `int4`
1.40x. MLX's own 4-bit `quantized_matmul` (W4A16) runs at about bf16 speed on
these shapes.

Benchmark on AC power with Low Power Mode off. Low Power Mode roughly halves
GPU throughput, and the results are not representative.

Accuracy on three real FLUX.1-dev SVDQ INT4 layers, as relative L2 error against
the exact W4A16 output, with synthetic heavy-tailed activations:

| Layer | `int8` / `int4` | Nunchaku CUDA W4A4 (emulated) |
| --- | --- | --- |
| transformer_blocks.5.qkv_proj | 2.0-3.6% | 13-22% |
| single_transformer_blocks.10.mlp_fc1 | 2.0-5.3% | 14-29% |
| transformer_blocks.5.mlp_fc2 | 2.0-2.1% | 4.9-5.0% |

Re-expanding the weights to int8 contributes about 1.2% of that error. The int8
activations account for the rest. Running bf16 end to end (activations, output
and low-rank branch) does not change these figures.

## Requirements and limits

- macOS 26.2+, MLX >= 0.30. The `int8`/`int4` backends need an M5-class GPU
  (`applegpu_g17+`). Kernels are JIT-compiled on first use.
- INT4 SVDQ checkpoints only. NVFP4 is not supported yet.
- `N % 64 == 0` and `K % 64 == 0`. Tokens are padded internally. The low-rank
  dimension is zero-padded to a multiple of 32.
- Pipelines: adapters exist for FLUX.1 (`flux1`) and FLUX.2 Klein (`flux2`).
  Other families can use the `manifest` adapter or `patch_transformer` with a
  `key_map` (for example `PrefixMap.from_mflux(<mflux WeightMapping>.get_transformer_mapping())`),
  or register an adapter.

## Tests

```bash
pip install -e ".[mlx]" pytest
pytest tests/mlx_runtime tests/test_torch_free_imports.py
```

The packing tests check that the layouts round-trip through a NumPy port of
DeepCompressor's `NunchakuWeightPacker`. That port was verified byte-identical to
the original, and the kernel tests compare against float64 references.
