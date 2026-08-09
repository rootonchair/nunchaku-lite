#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from utils import (
    add_benchmark_speedups,
    add_single_step_latency,
    cleanup,
    dtype_from_arg,
    import_diffusers_pipeline,
    import_original_nunchaku_class,
    pipeline_transformer_size_gb,
    print_benchmark_speedups,
    run_generation_loop,
    timed_cuda_call,
    write_comparison_plot,
)

DEFAULT_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
DEFAULT_CHECKPOINT = "nunchaku-ai/nunchaku-z-image-turbo/svdq-fp4_r128-z-image-turbo.safetensors"
DEFAULT_PROMPT = (
    "a cinematic photo of a glass greenhouse full of tropical plants during golden hour, detailed, natural light"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark original Diffusers Z-Image against nunchaku_lite patched Z-Image."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--local-diffusers-src", default=None)
    parser.add_argument("--output-dir", default="outputs/benchmark_z_image")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--precision", choices=["auto", "fp4", "int4"], default="fp4")
    parser.add_argument("--skip-original", action="store_true")
    parser.add_argument("--skip-lite", action="store_true")
    parser.add_argument("--run-original-nunchaku", action="store_true")
    parser.add_argument("--skip-plot", action="store_true", help="Do not write comparison.png after both runs finish.")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    return parser.parse_args()


def run_generation(pipe, args: argparse.Namespace, label: str, output_dir: Path) -> dict:
    return run_generation_loop(
        pipe,
        args,
        label,
        output_dir,
        lambda generator: pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0],
    )


def run_original(args: argparse.Namespace, output_dir: Path, pipeline_cls, torch_dtype: torch.dtype) -> dict:
    cleanup()
    print("loading original diffusers pipeline", flush=True)
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


def run_original_nunchaku(args: argparse.Namespace, output_dir: Path, pipeline_cls, torch_dtype: torch.dtype) -> dict:
    transformer_cls = import_original_nunchaku_class("NunchakuZImageTransformer2DModel")

    def load_pipeline():
        transformer = transformer_cls.from_pretrained(args.checkpoint, torch_dtype=torch_dtype, device="cuda")
        return pipeline_cls.from_pretrained(
            args.model_id,
            transformer=transformer,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )

    cleanup()
    print("loading original Nunchaku Z-Image pipeline", flush=True)
    pipe, load_seconds = timed_cuda_call(load_pipeline)
    pipe = pipe.to("cuda")
    result = run_generation(pipe, args, "original_nunchaku", output_dir)
    result["load_seconds"] = load_seconds
    result["model_size_gb"] = pipeline_transformer_size_gb(pipe)
    del pipe
    cleanup()
    return result


def run_lite(args: argparse.Namespace, output_dir: Path, pipeline_cls, torch_dtype: torch.dtype) -> dict:
    from nunchaku_lite import load_nunchaku_pipeline

    cleanup()
    print("loading nunchaku_lite pipeline", flush=True)
    pipe, load_seconds = timed_cuda_call(
        lambda: load_nunchaku_pipeline(
            args.model_id,
            pipeline_cls=pipeline_cls,
            checkpoint=args.checkpoint,
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
    pipeline_cls = import_diffusers_pipeline(args.local_diffusers_src, "ZImagePipeline")
    torch_dtype = dtype_from_arg(args.dtype)

    metadata = {
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
        "device": torch.cuda.get_device_name(0),
    }
    results = {"metadata": metadata}

    if not args.skip_original:
        results["original_diffusers"] = run_original(args, output_dir, pipeline_cls, torch_dtype)
    if args.run_original_nunchaku:
        results["original_nunchaku"] = run_original_nunchaku(args, output_dir, pipeline_cls, torch_dtype)
    if not args.skip_lite:
        results["nunchaku_lite"] = run_lite(args, output_dir, pipeline_cls, torch_dtype)
    for key in ("original_diffusers", "original_nunchaku", "nunchaku_lite"):
        if key in results:
            add_single_step_latency(results[key], args.steps)

    add_benchmark_speedups(results)
    print_benchmark_speedups(results)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)
    if not args.skip_plot:
        plot_path = write_comparison_plot(results, output_dir, "Z-Image Turbo Benchmark: Original vs Nunchaku Lite")
        if plot_path is not None:
            print(f"wrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
