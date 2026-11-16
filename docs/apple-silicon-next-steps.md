# Apple Silicon: Status and Next Steps

## Current state

All Apple-silicon work is uncommitted on `rootonchair/macos`:

- `nunchaku-lite-kernels`: Metal MPS ops for the W8A8/W4A8 SVDQ GEMM, int8
  activation quantization, and a fused QK RMSNorm + RoPE.
- `nunchaku_lite`: the PyTorch MPS runtime, selected with `backend=`.
- `nunchaku_lite.mlx` (the `mlx` extra): MLX layers, `patch_transformer`, AWQ W4A16, and
  `load_nunchaku_pipeline` on mflux.

Both test suites pass. FLUX.2 Klein 4B is verified end to end on PyTorch MPS
and on MLX.

## Model checks on PyTorch MPS (M5, 24 GB, 1024 px)

| Model | Checkpoint | Steps | Step time | Peak GPU memory | Result |
| --- | --- | --- | --- | --- | --- |
| FLUX.1-dev | `nunchaku-tech/nunchaku-flux.1-dev` INT4 r32 | 20 | 8.5 s | 9.3 GB | correct |
| FLUX.2 Klein 9B | `lite-infer/flux.2-klein-9b-nunchaku-lite-int4-r32` | 4 | 7.6 s (int8), 8.1 s (int4) | 11.5 GB (int8), 6.3 GB (int4) | correct |
| FLUX.2 Klein 4B + LoRA | `lite-infer/flux.2-klein-4b-nunchaku-lite-int4-r32` + `Barath/flux2-klein-yarn-art-lora` | 4 | about 3.5 s | 7.8 GB with text encoder | LoRA applies and scales; unload restores the base image bit-exactly |

Step times exclude the text encoder, which was freed after encoding: on 24 GB
the 9B pipeline swaps with its 8B text encoder resident.

The checks found and fixed two bugs:

- **FLUX.1 RoPE used float64,** which MPS does not support. It now uses float32
  on MPS, as Diffusers does.
- **The MPS allocator wasted 5-10 GB.** It serves a new tensor from any larger
  free buffer and keeps the whole buffer allocated. Each layer's cache, built
  from float32 temporaries, pinned buffers several times its size. FLUX.1 grew
  to 24 GB and swapped at 85 s per step. Caches are now re-allocated from an
  empty pool. Cache keys are also refreshed in `_apply`, so CPU offload round
  trips no longer rebuild them.

## Model checks on MLX (M5, 24 GB, 1024 px, 4 steps)

`nunchaku_lite.mlx` loads the `lite-infer` Diffusers repos from their runtime
manifest. The base pipeline here is mflux's q4 repo, used only for its text
encoder and VAE; the transformer is rebuilt and filled from the Nunchaku
checkpoint. Step times are per denoising step; peaks include the q4 text
encoder.

On 24 GB, 9B timings are dominated by swapping unless MLX's buffer cache is
capped. MLX keeps freed buffers resident: 4.7-8.9 GB in the 9B pipeline, on top
of 9.2 GB live with int4. Without a cap, the first step of every image swapped
8-12 GB and took 30-100 s. That made an early int4 run look 50% slower than
int8; it was swap noise. With `mx.set_cache_limit(1 << 30)`, int4 stops
swapping. int8 (14.5 GB live) still swaps on the first step after each VAE
decode. In steady state int8 is 5-10% faster than int4, consistent with the
int4 kernel being 16% slower in isolation. `backend="auto"` picks int4 for the
9B on 24 GB, which is the right choice.

| Model | Checkpoint | Step time | Peak memory | mflux q4 | Result |
| --- | --- | --- | --- | --- | --- |
| FLUX.2 Klein 9B | `lite-infer/flux.2-klein-9b-nunchaku-lite-int4-r32` | 4.6-5.7 s (int8), 5.1-5.9 s (int4) | 16.4 GB (int8), 11.2 GB (int4) | 6.97 s, 10.3 GB | correct |
| FLUX.2 Klein 4B | `lite-infer/flux.2-klein-4b-nunchaku-lite-int4-r32` | 2.69 s (int8), 2.87 s (int4) | 7.8 GB (int8), 5.9 GB (int4) | 3.25 s, 5.5 GB | correct |
| FLUX.1-dev | `lite-infer/flux.1-dev-nunchaku-lite-int4-r32` | 7.45-7.9 s (int4, 20 steps) | 22.7 GB | 11.48 s, 21.8 GB | correct |

The FLUX.1-dev repo has no runtime manifest; the manifest is built from the
target lists in its `config.json`. Its row uses 20 steps and guidance 3.5, with
a 1 GB cache cap. `backend="auto"` picked int4. Both FLUX.1 peaks come from the
VAE decode at 1024 px, which adds about 12.3 GB in both pipelines. Our
transformer step adds less than mflux's (1.5 vs 1.8 GB). The remaining gap is
resident: 10.4 GB vs 9.6 GB. Of that, about 0.35 GB is the checkpoint itself
(rank-64 low-rank branch), and about 0.5 GB is copies still held twice: SVDQ
`wscales` next to the kernel's fp16 group multipliers, and `proj_up` next to
its fp16 kernel copy wherever fp16 can't hold the bf16 values exactly.

Earlier the peak was 25.1 GB. Quantized layers kept their checkpoint arrays
next to the kernel copies: 1.8 GB for the 76 AWQ layers, 0.5 GB of SVDQ
low-rank weights. mflux runs FLUX.1 activations in float32, which also made the
low-rank caches float32. The layers now release those arrays as exact lazy
re-packs of the kernel copies, and keep low-rank caches in the checkpoint
dtype; images are bit-identical.

Layers convert their checkpoint tensors to the kernel layouts on first use.
That conversion now runs as MLX ops on the GPU instead of NumPy
(`nunchaku_lite/mlx/mx_layout.py`, tested bit-identical against the NumPy
references): 53 s -> 7.6 s on FLUX.1-dev (SVDQ 42 -> 5.9 s, AWQ 11 -> 1.7 s), so
the two-step warmup went from 87 s to 23 s.

## Next steps, in priority order

### 1. Lock in what works

1. **Commit and open a PR as three commits:** the Metal kernels, the PyTorch MPS
   runtime, and the MLX package. Each can be reviewed on its own.
2. **Add macOS CI.** All CI runs on Ubuntu today, so none of the MPS or MLX tests
   run. GitHub's macOS runners have no M5 GPUs. CI can cover packing, layout,
   the `dense` fallback, and the fake-kernel tests. The int8 kernels still need
   a local check on M5 hardware.

### 2. Broaden model coverage (highest user value)

3. **Verify Qwen-Image and Z-Image on MPS.** FLUX.1, FLUX.2 Klein 9B, and
   runtime LoRA are now verified (see above).
4. **Release the AWQ checkpoint copy on MPS.** AWQ layers now run PyTorch's
   built-in 4-bit MPS matmul (`aten._weight_int4pack_mm`) instead of a dense
   bf16 cache. On FLUX.1's 76 modulation layers that cut resident memory from
   13.9 GB to 9.3 GB and the step from 8.9 s to 8.5 s; the op is about 3x
   faster than the dense matmul and slightly more accurate (no custom kernel
   was needed). The checkpoint `qweight` (1.6 GB) is still held next to the
   op's packed copy; releasing it, as SVDQ layers do, needs a way to rebuild
   it from that copy.
5. **MLX adapters for the remaining families,** using mflux's Qwen-Image and
   Z-Image models. FLUX.1 is done (`flux1`, lite checkpoints only). Manifest
   checkpoints need only a name map, which `PrefixMap.from_mflux` derives from
   mflux's weight mapping.

### 3. Distribution

6. **Hub build for Metal.** Add a `cpp-metal` section to `build.toml` and verify it
   with kernel-builder, which needs Nix on a Mac. Without it, Mac users must build
   the kernels locally while CUDA users get prebuilt ones.
7. **Release the torch/mlx extras split.** `nunchaku_lite.mlx` now lives in the
   main package. A bare `pip install nunchaku-lite` no longer installs torch;
   CUDA users need `nunchaku-lite[torch]`. Call this out in the release notes.

### 4. Performance

8. **Attention.** It is now about half of each FLUX.2 step on MPS and the largest
   remaining cost. Options: a Metal flash-attention kernel registered through
   Diffusers' `set_attention_backend`, or reusing MLX's own accelerator attention.
9. **Long-K matmuls** (for example `mlp_fc2`, K = 12288) reach about 20 of the
   30 available TOPS. Tile and pipeline tuning could add 10-20% on those layers.
10. **Quantize in the previous layer's epilogue.** Profiling estimated about 10%
    per layer from removing the quantize stage and the GEMM wait it causes.
11. **Load-time memory.** Weight conversion creates temporary float32 copies that
    inflate PyTorch's allocator heaps. Converting in a lighter form would lower
    peak memory.

### 5. Format and ecosystem

12. **NVFP4 checkpoints on Apple GPUs.** The accelerators have no fp4 input type.
    FP4 values times 2 convert exactly to int8, but there is a scale every 16
    values, so it needs a different kernel design.
13. **Upstream to Diffusers:** add `backend=` to Diffusers' own
    `NunchakuLiteQuantizationConfig`, following `AutoRoundConfig`. Best done once
    the MPS path is stable and on the Hub.

## Recommendation

Do 1-2 now (commit, PR, macOS CI). Then do 8 (attention) for the biggest remaining speedup.
