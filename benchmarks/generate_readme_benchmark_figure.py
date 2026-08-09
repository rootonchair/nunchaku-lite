#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from utils import benchmark_device_label

DEFAULT_ENTRIES = [
    ("FLUX.2\nKlein 4B", Path("outputs/benchmark_flux2_klein_4b/summary.json")),
    ("FLUX.2\nKlein 9B", Path("outputs/benchmark_flux2_klein_9b/summary.json")),
    ("Qwen\nImage", Path("outputs/benchmark_qwen_image/summary.json")),
    ("Z-Image\nTurbo", Path("outputs/benchmark_z_image_turbo/summary.json")),
]
DEFAULT_OUTPUT = Path("docs/assets/benchmarks/readme-benchmark-summary.png")


@dataclass(frozen=True)
class BenchmarkEntry:
    label: str
    summary_path: Path
    latency_original: float
    latency_lite: float
    memory_original: float
    memory_lite: float
    size_original: float
    size_lite: float
    metadata: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the README benchmark summary figure from benchmark summary.json files."
    )
    parser.add_argument(
        "--entry",
        action="append",
        default=[],
        metavar="LABEL=SUMMARY_JSON",
        help=(
            "Benchmark entry to plot. Repeat to control order. Use literal '\\n' in LABEL for line breaks. "
            "If omitted, the current documented benchmark set is used."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    return parser.parse_args()


def parse_entry_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"entry must use LABEL=SUMMARY_JSON format: {spec}")
    label, summary_path = spec.split("=", 1)
    if not label:
        raise ValueError(f"entry label must not be empty: {spec}")
    if not summary_path:
        raise ValueError(f"entry summary path must not be empty: {spec}")
    return label.replace("\\n", "\n"), Path(summary_path)


def load_entry(label: str, summary_path: Path) -> BenchmarkEntry:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    original = data["original_diffusers"]
    lite = data["nunchaku_lite"]
    return BenchmarkEntry(
        label=label,
        summary_path=summary_path,
        latency_original=original["seconds"]["mean"],
        latency_lite=lite["seconds"]["mean"],
        memory_original=original["peak_cuda_gb"]["mean"],
        memory_lite=lite["peak_cuda_gb"]["mean"],
        size_original=original["model_size_gb"],
        size_lite=lite["model_size_gb"],
        metadata=data["metadata"],
    )


def common_value(entries: list[BenchmarkEntry], key: str):
    values = {entry.metadata[key] for entry in entries}
    if len(values) == 1:
        return values.pop()
    return None


def footer_text(entries: list[BenchmarkEntry]) -> str:
    width = common_value(entries, "width")
    height = common_value(entries, "height")
    precision = common_value(entries, "precision")
    runs = common_value(entries, "runs")
    warmups = common_value(entries, "warmup_runs")
    devices = {benchmark_device_label(entry.metadata["device"]) for entry in entries}
    device = devices.pop() if len(devices) == 1 else "multiple GPUs"

    size = f"{width}x{height}" if width is not None and height is not None else "mixed resolutions"
    precision_text = f"{str(precision).upper()} " if precision is not None else ""
    run_text = (
        f"{runs} measured runs, {warmups} warmups"
        if runs is not None and warmups is not None
        else "see summary.json files for run counts"
    )
    return f"{size} {precision_text}benchmarks on {device} | {run_text}"


def draw_figure(entries: list[BenchmarkEntry], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [entry.label for entry in entries]
    latency_original = [entry.latency_original for entry in entries]
    latency_lite = [entry.latency_lite for entry in entries]
    memory_original = [entry.memory_original for entry in entries]
    memory_lite = [entry.memory_lite for entry in entries]
    size_original = [entry.size_original for entry in entries]
    size_lite = [entry.size_lite for entry in entries]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.9), dpi=180)
    fig.patch.set_facecolor("#FBFBFD")
    original_color = "#4C78A8"
    lite_color = "#F58518"
    x = np.arange(len(labels))
    width = 0.36

    plots = [
        (axes[0], latency_original, latency_lite, "Latency", "seconds, lower is better", "{:.3f}s"),
        (axes[1], memory_original, memory_lite, "Peak CUDA Memory", "GB, lower is better", "{:.1f}"),
        (axes[2], size_original, size_lite, "Transformer Storage", "GiB, lower is better", "{:.1f}"),
    ]

    for ax, original_values, lite_values, title, ylabel, value_format in plots:
        original_bars = ax.bar(x - width / 2, original_values, width, label="Original Diffusers", color=original_color)
        lite_bars = ax.bar(x + width / 2, lite_values, width, label="nunchaku_lite", color=lite_color)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        max_value = max(max(original_values), max(lite_values))
        ax.set_ylim(0, max_value * 1.24)
        ax.grid(axis="y", alpha=0.26)
        ax.grid(axis="x", visible=False)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        for bars in (original_bars, lite_bars):
            for bar in bars:
                value = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + max_value * 0.025,
                    value_format.format(value),
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#111827",
                )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.905), frameon=False, fontsize=11
    )
    fig.suptitle("nunchaku_lite vs. original Diffusers", fontsize=20, fontweight="bold", y=0.985)
    fig.text(0.5, 0.035, footer_text(entries), ha="center", fontsize=10, color="#4B5563")
    fig.tight_layout(rect=(0.02, 0.08, 0.995, 0.85))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    entry_specs = [parse_entry_spec(spec) for spec in args.entry] if args.entry else DEFAULT_ENTRIES
    entries = [load_entry(label, path) for label, path in entry_specs]
    draw_figure(entries, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
