import gc
import importlib
import sys
import time
from collections.abc import Callable
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch


ORIGINAL_NUNCHAKU_SRC = Path("/mnt/disks/workspace/research/nunchaku")


def import_diffusers_pipeline(local_diffusers_src: str | None, pipeline_name: str):
    if local_diffusers_src:
        path = Path(local_diffusers_src)
        if path.exists():
            sys.path.insert(0, str(path))
    diffusers = importlib.import_module("diffusers")
    return getattr(diffusers, pipeline_name)


def import_original_nunchaku_class(class_name: str):
    if not ORIGINAL_NUNCHAKU_SRC.exists():
        raise FileNotFoundError(f"original Nunchaku checkout not found: {ORIGINAL_NUNCHAKU_SRC}")

    original_path = str(ORIGINAL_NUNCHAKU_SRC)
    if sys.path[0:1] != [original_path]:
        sys.path = [path for path in sys.path if path != original_path]
        sys.path.insert(0, original_path)

    loaded = sys.modules.get("nunchaku")
    if loaded is not None:
        loaded_file = Path(getattr(loaded, "__file__", "") or "").resolve()
        if ORIGINAL_NUNCHAKU_SRC.resolve() not in loaded_file.parents:
            for module_name in list(sys.modules):
                if module_name == "nunchaku" or module_name.startswith("nunchaku."):
                    del sys.modules[module_name]

    importlib.invalidate_caches()
    nunchaku = importlib.import_module("nunchaku")
    return getattr(nunchaku, class_name)


def original_nunchaku_source() -> str:
    return str(ORIGINAL_NUNCHAKU_SRC)


def dtype_from_arg(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def cuda_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def timed_cuda_call(fn: Callable[[], Any]):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def summarize(values: list[float]) -> dict[str, float | list[float]]:
    summary: dict[str, float | list[float]] = {"values": values, "mean": mean(values)}
    summary["stdev"] = stdev(values) if len(values) > 1 else 0.0
    return summary


def tensor_storage_gb(module: torch.nn.Module) -> float:
    """Return parameter and buffer storage for one model component in GiB."""

    tensors = list(module.parameters()) + list(module.buffers())
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors) / 1024**3


def pipeline_transformer_size_gb(pipe) -> float:
    """Return transformer component storage for benchmark size comparison."""

    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return 0.0
    return tensor_storage_gb(transformer)


def add_single_step_latency(result: dict, steps: int) -> None:
    """Add derived per-step latency statistics to one benchmark result."""

    if steps <= 0:
        return
    result["single_step_seconds"] = summarize([value / steps for value in result["seconds"]["values"]])


def add_benchmark_speedups(results: dict) -> None:
    if "original_diffusers" in results and "nunchaku_lite" in results:
        speedup = results["original_diffusers"]["seconds"]["mean"] / results["nunchaku_lite"]["seconds"]["mean"]
        results["speedup"] = speedup
        results["nunchaku_lite_vs_original_diffusers_speedup"] = speedup

    if "original_diffusers" in results and "original_nunchaku" in results:
        results["original_nunchaku_vs_original_diffusers_speedup"] = (
            results["original_diffusers"]["seconds"]["mean"] / results["original_nunchaku"]["seconds"]["mean"]
        )

    if "original_nunchaku" in results and "nunchaku_lite" in results:
        results["nunchaku_lite_vs_original_nunchaku_speedup"] = (
            results["original_nunchaku"]["seconds"]["mean"] / results["nunchaku_lite"]["seconds"]["mean"]
        )


def print_benchmark_speedups(results: dict) -> None:
    labels = {
        "nunchaku_lite_vs_original_diffusers_speedup": "nunchaku_lite vs original Diffusers",
        "original_nunchaku_vs_original_diffusers_speedup": "original Nunchaku vs original Diffusers",
        "nunchaku_lite_vs_original_nunchaku_speedup": "nunchaku_lite vs original Nunchaku",
    }
    for key, label in labels.items():
        if key in results:
            print(f"{label}: {results[key]:.3f}x", flush=True)


def run_generation_loop(
    pipe,
    args,
    label: str,
    output_dir: Path,
    generate_image: Callable[[torch.Generator], Any],
) -> dict:
    timings = []
    peaks = []
    last_image = None
    total_runs = args.warmup_runs + args.runs

    for index in range(total_runs):
        measured = index >= args.warmup_runs
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        image, elapsed = timed_cuda_call(lambda: generate_image(generator))
        peak = cuda_gb() if torch.cuda.is_available() else 0.0
        print(f"{label} run {index + 1}/{total_runs}: {elapsed:.3f}s, peak {peak:.2f} GB", flush=True)

        if measured:
            timings.append(elapsed)
            peaks.append(peak)
            last_image = image

    image_path = output_dir / f"{label}.png"
    if last_image is not None:
        last_image.save(image_path)
    return {"image": str(image_path), "seconds": summarize(timings), "peak_cuda_gb": summarize(peaks)}


def write_comparison_plot(results: dict, output_dir: Path, title: str) -> Path | None:
    """Write a latency, memory, and model-size comparison figure."""

    if "original_diffusers" not in results or "nunchaku_lite" not in results:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping comparison plot", flush=True)
        return None

    metadata = results["metadata"]
    series = [("original_diffusers", "Original\nDiffusers")]
    if "original_nunchaku" in results:
        series.append(("original_nunchaku", "Original\nNunchaku"))
    series.append(("nunchaku_lite", "Nunchaku\nLite"))

    labels = [label for _, label in series]
    latency = [results[key]["seconds"]["mean"] for key, _ in series]
    latency_err = [results[key]["seconds"]["stdev"] for key, _ in series]
    steps = max(int(metadata["steps"]), 1)
    per_step = [
        results[key].get("single_step_seconds", {"mean": results[key]["seconds"]["mean"] / steps})["mean"]
        for key, _ in series
    ]
    per_step_err = [
        results[key].get("single_step_seconds", {"stdev": results[key]["seconds"]["stdev"] / steps})["stdev"]
        for key, _ in series
    ]
    memory = [results[key]["peak_cuda_gb"]["mean"] for key, _ in series]
    model_size = [results[key].get("model_size_gb") for key, _ in series]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), dpi=180)
    axes = axes.flatten()
    colors = ["#4C78A8", "#54A24B", "#F58518"] if len(series) == 3 else ["#4C78A8", "#F58518"]

    axes[0].bar(labels, latency, yerr=latency_err, capsize=5, color=colors, width=0.62)
    axes[0].set_title("End-to-End Latency")
    axes[0].set_ylabel("Seconds, lower is better")
    axes[0].set_ylim(0, max(latency) * 1.35)
    for index, value in enumerate(latency):
        axes[0].text(index, value + max(latency) * 0.04, f"{value:.3f}s", ha="center", va="bottom", fontsize=10)

    axes[1].bar(labels, per_step, yerr=per_step_err, capsize=5, color=colors, width=0.62)
    axes[1].set_title("Single-Step Latency")
    axes[1].set_ylabel("Seconds / step, lower is better")
    axes[1].set_ylim(0, max(per_step) * 1.35)
    for index, value in enumerate(per_step):
        axes[1].text(index, value + max(per_step) * 0.04, f"{value:.3f}s", ha="center", va="bottom", fontsize=10)

    axes[2].bar(labels, memory, color=colors, width=0.62)
    axes[2].set_title("Peak CUDA Memory")
    axes[2].set_ylabel("GB, lower is better")
    axes[2].set_ylim(0, max(memory) * 1.25)
    for index, value in enumerate(memory):
        axes[2].text(index, value + max(memory) * 0.035, f"{value:.2f} GB", ha="center", va="bottom", fontsize=10)

    axes[3].set_title("Transformer Model Size")
    axes[3].set_ylabel("GiB, lower is better")
    if any(value is None for value in model_size):
        axes[3].axis("off")
        axes[3].text(0.5, 0.5, "Run benchmark again\nto collect model size", ha="center", va="center", fontsize=12)
    else:
        axes[3].bar(labels, model_size, color=colors, width=0.62)
        axes[3].set_ylim(0, max(model_size) * 1.25)
        for index, value in enumerate(model_size):
            axes[3].text(
                index,
                value + max(model_size) * 0.035,
                f"{value:.2f} GiB",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        f"{metadata['device']} | {metadata['width']}x{metadata['height']} | "
        f"{metadata['steps']} steps | {metadata['runs']} measured runs",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    plot_path = output_dir / "comparison.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path
