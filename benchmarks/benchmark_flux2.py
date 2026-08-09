#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from utils import (
    add_single_step_latency,
    cleanup,
    dtype_from_arg,
    import_diffusers_pipeline,
    pipeline_transformer_size_gb,
    run_generation_loop,
    timed_cuda_call,
    write_comparison_plot,
)

DEFAULT_MODEL_ID = "tonera/FLUX.2-klein-9B-Nunchaku"
DEFAULT_CHECKPOINT = "tonera/FLUX.2-klein-9B-Nunchaku/svdq-fp4_r32-FLUX.2-klein-9B-Nunchaku.safetensors"
DEFAULT_PROMPT = "A cat holding a sign that says hello world"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark original Flux2 Klein against nunchaku_lite Flux2.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--local-diffusers-src", default=None)
    parser.add_argument("--output-dir", default="outputs/benchmark_flux2")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--precision", choices=["auto", "fp4", "int4"], default="fp4")
    parser.add_argument("--skip-original", action="store_true")
    parser.add_argument("--skip-lite", action="store_true")
    parser.add_argument("--skip-plot", action="store_true", help="Do not write comparison.png after both runs finish.")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--image", default=None, help="Optional reference image path for Flux2 edit/KV paths.")
    return parser.parse_args()


def load_reference_image(path: str | None):
    if path is None:
        return None
    from PIL import Image

    return Image.open(path).convert("RGB")


def run_generation(pipe, args: argparse.Namespace, label: str, output_dir: Path) -> dict:
    reference_image = load_reference_image(args.image)

    def generate_image(generator: torch.Generator):
        call_kwargs = {
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "generator": generator,
        }
        if reference_image is not None:
            call_kwargs["image"] = reference_image
        return pipe(**call_kwargs).images[0]

    return run_generation_loop(pipe, args, label, output_dir, generate_image)


def run_original(args: argparse.Namespace, output_dir: Path, pipeline_cls, torch_dtype: torch.dtype) -> dict:
    cleanup()
    print("loading original Flux2 Klein pipeline", flush=True)
    pipe, load_seconds = timed_cuda_call(
        lambda: pipeline_cls.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    )
    pipe = pipe.to("cuda")
    result = run_generation(pipe, args, "original_diffusers", output_dir)
    result["load_seconds"] = load_seconds
    result["model_size_gb"] = pipeline_transformer_size_gb(pipe)
    del pipe
    cleanup()
    return result


def run_lite(args: argparse.Namespace, output_dir: Path, pipeline_cls, torch_dtype: torch.dtype) -> dict:
    from nunchaku_lite import load_nunchaku_pipeline

    cleanup()
    print("loading nunchaku_lite Flux2 Klein pipeline", flush=True)
    pipe, load_seconds = timed_cuda_call(
        lambda: load_nunchaku_pipeline(
            args.model_id,
            pipeline_cls=pipeline_cls,
            checkpoint=args.checkpoint,
            target="flux2",
            precision=args.precision,
            torch_dtype=torch_dtype,
            device="cuda",
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    )
    pipe = pipe.to("cuda")
    result = run_generation(pipe, args, "nunchaku_lite", output_dir)
    result["load_seconds"] = load_seconds
    result["model_size_gb"] = pipeline_transformer_size_gb(pipe)
    del pipe
    cleanup()
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_cls = import_diffusers_pipeline(args.local_diffusers_src, "Flux2KleinPipeline")
    torch_dtype = dtype_from_arg(args.dtype)
    results = {
        "metadata": {
            "model_id": args.model_id,
            "checkpoint": args.checkpoint,
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "dtype": args.dtype,
            "precision": args.precision,
            "image": args.image,
            "device": torch.cuda.get_device_name(0),
        }
    }

    if not args.skip_original:
        results["original_diffusers"] = run_original(args, output_dir, pipeline_cls, torch_dtype)
    if not args.skip_lite:
        results["nunchaku_lite"] = run_lite(args, output_dir, pipeline_cls, torch_dtype)
    for key in ("original_diffusers", "nunchaku_lite"):
        if key in results:
            add_single_step_latency(results[key], args.steps)
    if "original_diffusers" in results and "nunchaku_lite" in results:
        results["speedup"] = (
            results["original_diffusers"]["seconds"]["mean"] / results["nunchaku_lite"]["seconds"]["mean"]
        )
        print(f"speedup: {results['speedup']:.3f}x", flush=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)
    if not args.skip_plot:
        plot_path = write_comparison_plot(results, output_dir, "FLUX.2 Klein Benchmark: Original vs Nunchaku Lite")
        if plot_path is not None:
            print(f"wrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
