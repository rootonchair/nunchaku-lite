#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from utils import (
    add_single_step_latency,
    benchmark_device_label,
    cleanup,
    cuda_gb,
    dtype_from_arg,
    import_diffusers_pipeline,
    pipeline_transformer_size_gb,
    summarize,
    timed_cuda_call,
)

DEFAULT_PROMPT = "A flowing river in a forest at golden hour, gentle wind in the leaves."
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]
VARIANT_DEFAULTS = {
    "ltx2.3": {
        "model_id": "diffusers/LTX-2.3-Diffusers",
        "checkpoint": "outputs/checkpoints/svdq-nvfp4_r32-ltx2.3.safetensors",
        "output_dir": "outputs/benchmark_ltx2_3",
        "steps": 30,
        "guidance_scale": 3.0,
        "sigmas": None,
        "title": "LTX 2.3 Benchmark: BF16 Baseline vs Nunchaku Lite",
    },
    "ltx2.3-distilled": {
        "model_id": "dg845/LTX-2.3-Distilled-Diffusers",
        "checkpoint": "outputs/checkpoints/svdq-nvfp4_r32-ltx2.3.safetensors",
        "output_dir": "outputs/benchmark_ltx2_3_distilled",
        "steps": 8,
        "guidance_scale": 1.0,
        "sigmas": DISTILLED_SIGMA_VALUES,
        "title": "LTX 2.3 Distilled Benchmark: BF16 Baseline vs Nunchaku Lite",
    },
}


def parse_float_list(value: str) -> list[float]:
    try:
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float list: {value!r}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark original Diffusers LTX 2.3 against nunchaku_lite patched LTX 2.3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variant", choices=sorted(VARIANT_DEFAULTS), default="ltx2.3-distilled")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--local-diffusers-src", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--sigmas", type=parse_float_list, default=None, help="Comma-separated custom scheduler sigmas.")
    parser.add_argument("--distilled-sigmas", action="store_true", help="Use the distilled 8-step LTX sigma schedule.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--precision", choices=["auto", "fp4", "int4"], default="fp4")
    parser.add_argument("--skip-original", action="store_true")
    parser.add_argument("--skip-lite", action="store_true")
    parser.add_argument("--skip-manifest", action="store_true")
    parser.add_argument("--skip-plot", action="store_true", help="Do not write comparison.png after both runs finish.")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    args = parser.parse_args()

    defaults = VARIANT_DEFAULTS[args.variant]
    if args.model_id is None:
        args.model_id = defaults["model_id"]
    if args.checkpoint is None:
        args.checkpoint = defaults["checkpoint"]
    if args.output_dir is None:
        args.output_dir = defaults["output_dir"]
    if args.steps is None:
        args.steps = defaults["steps"]
    if args.guidance_scale is None:
        args.guidance_scale = defaults["guidance_scale"]
    if args.sigmas is None and defaults["sigmas"] is not None:
        args.sigmas = list(defaults["sigmas"])
    if args.distilled_sigmas:
        args.sigmas = list(DISTILLED_SIGMA_VALUES)
    return args


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class TimedProgressBar:
    def __init__(self, progress_bar: object, measurements: list[float]) -> None:
        self.progress_bar = progress_bar
        self.measurements = measurements
        self.started_at: float | None = None

    def __enter__(self) -> object:
        synchronize()
        self.started_at = time.perf_counter()
        return self.progress_bar.__enter__()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> object:
        result = self.progress_bar.__exit__(exc_type, exc_value, traceback)
        synchronize()
        if self.started_at is not None:
            self.measurements.append(time.perf_counter() - self.started_at)
        return result

    def __getattr__(self, name: str) -> object:
        return getattr(self.progress_bar, name)


def run_ltx_inference(pipe, args: argparse.Namespace) -> tuple[Any, Any, float, float]:
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    denoising_times: list[float] = []
    original_progress_bar = pipe.progress_bar

    def timed_progress_bar(*progress_args: object, **progress_kwargs: object) -> TimedProgressBar:
        return TimedProgressBar(original_progress_bar(*progress_args, **progress_kwargs), denoising_times)

    pipe.progress_bar = timed_progress_bar
    synchronize()
    started_at = time.perf_counter()
    try:
        video, audio = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            num_inference_steps=args.steps,
            sigmas=args.sigmas,
            guidance_scale=args.guidance_scale,
            output_type="np",
            return_dict=False,
            generator=generator,
        )
        synchronize()
        elapsed = time.perf_counter() - started_at
    finally:
        pipe.progress_bar = original_progress_bar
    if not denoising_times:
        raise RuntimeError("Failed to capture denoising-loop timing from the pipeline progress bar.")
    return video, audio, elapsed, denoising_times[-1]


def prepare_video_for_encode(video: object) -> object:
    if not isinstance(video, np.ndarray) or not np.issubdtype(video.dtype, np.floating):
        return video
    if not np.isfinite(video).all():
        raise ValueError("Video output contains NaN or infinite values; refusing to encode invalid frames.")
    min_value = float(np.nanmin(video))
    max_value = float(np.nanmax(video))
    if min_value >= -0.05 and max_value <= 1.05:
        return np.clip(video, 0.0, 1.0)
    return np.clip(video, 0.0, 255.0).round().astype(np.uint8)


def save_video(pipe, video: object, audio: object, output_path: Path, frame_rate: float) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    encode_video(
        prepare_video_for_encode(video[0]),
        fps=frame_rate,
        audio=audio[0].float().cpu(),
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
        output_path=str(output_path),
    )
    return time.perf_counter() - started_at


def run_generation(pipe, args: argparse.Namespace, label: str, output_dir: Path) -> dict:
    pipeline_timings = []
    denoising_timings = []
    peaks = []
    last_video = None
    last_audio = None
    total_runs = args.warmup_runs + args.runs

    for index in range(total_runs):
        measured = index >= args.warmup_runs
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        video, audio, elapsed, denoising_elapsed = run_ltx_inference(pipe, args)
        peak = cuda_gb() if torch.cuda.is_available() else 0.0
        print(
            f"{label} run {index + 1}/{total_runs}: "
            f"{elapsed:.3f}s, denoising {denoising_elapsed:.3f}s, peak {peak:.2f} GB",
            flush=True,
        )

        if measured:
            pipeline_timings.append(elapsed)
            denoising_timings.append(denoising_elapsed)
            peaks.append(peak)
            last_video = video
            last_audio = audio

    video_path = output_dir / f"{label}.mp4"
    encode_seconds = 0.0
    if last_video is not None and last_audio is not None:
        encode_seconds = save_video(pipe, last_video, last_audio, video_path, args.frame_rate)
        print(f"{label} encoded {video_path}: {encode_seconds:.3f}s", flush=True)

    return {
        "video": str(video_path),
        "seconds": summarize(denoising_timings),
        "latency_scope": "denoising_loop",
        "pipeline_call_seconds": summarize(pipeline_timings),
        "denoising_seconds": summarize(denoising_timings),
        "peak_cuda_gb": summarize(peaks),
        "encode_seconds": encode_seconds,
    }


def run_original(args: argparse.Namespace, output_dir: Path, pipeline_cls, torch_dtype: torch.dtype) -> dict:
    cleanup()
    print("loading original LTX 2.3 pipeline", flush=True)
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
    print("loading nunchaku_lite LTX 2.3 adapter pipeline", flush=True)
    pipe, load_seconds = timed_cuda_call(
        lambda: load_nunchaku_pipeline(
            args.model_id,
            pipeline_cls=pipeline_cls,
            checkpoint=args.checkpoint,
            target="ltx2",
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


def run_manifest(args: argparse.Namespace, output_dir: Path, pipeline_cls, torch_dtype: torch.dtype) -> dict:
    from nunchaku_lite import load_nunchaku_pipeline

    cleanup()
    print("loading nunchaku_lite LTX 2.3 manifest pipeline", flush=True)
    pipe, load_seconds = timed_cuda_call(
        lambda: load_nunchaku_pipeline(
            args.model_id,
            pipeline_cls=pipeline_cls,
            checkpoint=args.checkpoint,
            target="manifest",
            precision=args.precision,
            torch_dtype=torch_dtype,
            device="cuda",
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    )
    pipe = pipe.to("cuda")
    result = run_generation(pipe, args, "nunchaku_lite_manifest", output_dir)
    result["load_seconds"] = load_seconds
    result["model_size_gb"] = pipeline_transformer_size_gb(pipe)
    del pipe
    cleanup()
    return result


def run_case(name: str, runner, args: argparse.Namespace, output_dir: Path, pipeline_cls, torch_dtype: torch.dtype):
    try:
        return runner(args, output_dir, pipeline_cls, torch_dtype)
    except torch.cuda.OutOfMemoryError as exc:
        cleanup()
        message = f"{type(exc).__name__}: {exc}"
        print(f"{name} failed: {message}", flush=True)
        return {"error": message}


def result_succeeded(result: dict | None) -> bool:
    return bool(result) and "seconds" in result


def write_ltx_comparison_plot(results: dict, output_dir: Path, title: str) -> Path | None:
    if "original_diffusers" not in results or (
        "nunchaku_lite_manifest" not in results and "nunchaku_lite" not in results
    ):
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping comparison plot", flush=True)
        return None

    metadata = results["metadata"]
    series = [("original_diffusers", "BF16\nBaseline")]
    if result_succeeded(results.get("nunchaku_lite_manifest")):
        series.append(("nunchaku_lite_manifest", "Nunchaku\nLite"))
    if result_succeeded(results.get("nunchaku_lite")):
        series.append(("nunchaku_lite", "Nunchaku Lite\nAdapter"))
    labels = [label for _, label in series]
    denoising = [results[key]["seconds"]["mean"] for key, _ in series]
    denoising_err = [results[key]["seconds"]["stdev"] for key, _ in series]
    pipeline_call = [results[key]["pipeline_call_seconds"]["mean"] for key, _ in series]
    pipeline_call_err = [results[key]["pipeline_call_seconds"]["stdev"] for key, _ in series]
    memory = [results[key]["peak_cuda_gb"]["mean"] for key, _ in series]
    model_size = [results[key]["model_size_gb"] for key, _ in series]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), dpi=180)
    colors = ["#4C78A8", "#54A24B", "#F58518"][: len(series)]
    plots = [
        (axes[0], denoising, denoising_err, "Denoising Latency", "Seconds, lower is better", "{:.3f}s"),
        (axes[1], pipeline_call, pipeline_call_err, "Pipeline Call Latency", "Seconds, lower is better", "{:.3f}s"),
        (axes[2], memory, None, "Peak CUDA Memory", "GiB, lower is better", "{:.2f} GiB"),
        (axes[3], model_size, None, "Transformer Model Size", "GiB, lower is better", "{:.2f} GiB"),
    ]

    for ax, values, errors, subtitle, ylabel, value_format in plots:
        ax.bar(labels, values, yerr=errors, capsize=5 if errors is not None else 0, color=colors, width=0.62)
        ax.set_title(subtitle)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.35)
        for index, value in enumerate(values):
            ax.text(index, value + max(values) * 0.04, value_format.format(value), ha="center", va="bottom", fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        f"{benchmark_device_label(metadata['device'])} | {metadata['width']}x{metadata['height']} | "
        f"{metadata['steps']} steps | {metadata['runs']} measured runs",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    plot_path = output_dir / "comparison.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_cls = import_diffusers_pipeline(args.local_diffusers_src, "LTX2Pipeline")
    torch_dtype = dtype_from_arg(args.dtype)
    results = {
        "metadata": {
            "variant": args.variant,
            "model_id": args.model_id,
            "checkpoint": args.checkpoint,
            "nunchaku_lite_adapter_target": "ltx2",
            "nunchaku_lite_manifest_target": "manifest",
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "frame_rate": args.frame_rate,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "sigmas": args.sigmas,
            "seed": args.seed,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "dtype": args.dtype,
            "precision": args.precision,
            "device": torch.cuda.get_device_name(0),
        }
    }

    if not args.skip_original:
        results["original_diffusers"] = run_case(
            "original_diffusers", run_original, args, output_dir, pipeline_cls, torch_dtype
        )
    if not args.skip_manifest:
        results["nunchaku_lite_manifest"] = run_case(
            "nunchaku_lite_manifest", run_manifest, args, output_dir, pipeline_cls, torch_dtype
        )
    if not args.skip_lite:
        results["nunchaku_lite"] = run_case("nunchaku_lite", run_lite, args, output_dir, pipeline_cls, torch_dtype)
    for key in ("original_diffusers", "nunchaku_lite_manifest", "nunchaku_lite"):
        if result_succeeded(results.get(key)):
            add_single_step_latency(results[key], args.steps)

    if result_succeeded(results.get("original_diffusers")):
        original_seconds = results["original_diffusers"]["seconds"]["mean"]
        if result_succeeded(results.get("nunchaku_lite_manifest")):
            results["nunchaku_lite_manifest_vs_original_diffusers_speedup"] = (
                original_seconds / results["nunchaku_lite_manifest"]["seconds"]["mean"]
            )
        if result_succeeded(results.get("nunchaku_lite")):
            adapter_speedup = original_seconds / results["nunchaku_lite"]["seconds"]["mean"]
            results["speedup"] = adapter_speedup
            results["nunchaku_lite_vs_original_diffusers_speedup"] = adapter_speedup
    if result_succeeded(results.get("nunchaku_lite_manifest")) and result_succeeded(results.get("nunchaku_lite")):
        results["nunchaku_lite_adapter_vs_manifest_speedup"] = (
            results["nunchaku_lite_manifest"]["seconds"]["mean"] / results["nunchaku_lite"]["seconds"]["mean"]
        )
    for key, label in (
        ("nunchaku_lite_manifest_vs_original_diffusers_speedup", "nunchaku_lite manifest vs original Diffusers"),
        ("nunchaku_lite_vs_original_diffusers_speedup", "nunchaku_lite adapter vs original Diffusers"),
        ("nunchaku_lite_adapter_vs_manifest_speedup", "nunchaku_lite adapter vs manifest"),
    ):
        if key in results:
            print(f"{label}: {results[key]:.3f}x", flush=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)
    if not args.skip_plot and result_succeeded(results.get("original_diffusers")) and result_succeeded(
        results.get("nunchaku_lite")
    ):
        plot_path = write_ltx_comparison_plot(results, output_dir, VARIANT_DEFAULTS[args.variant]["title"])
        if plot_path is not None:
            print(f"wrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
