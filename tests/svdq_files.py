"""Synthetic Nunchaku Lite checkpoint files for loader / integration tests (no kernels needed)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from nunchaku_lite.manifest import ROTATION_MARKER_KEY

WTSCALE = 0.0123


def svdq_tensors(
    in_features: int,
    out_features: int,
    *,
    precision: str = "int4",
    rank: int = 16,
    bias: bool = True,
    wtscale: bool = True,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    group = 64 if precision == "int4" else 16
    tensors = {
        "qweight": torch.randint(-128, 128, (out_features, in_features // 2), dtype=torch.int8, generator=g),
        "wscales": torch.rand(in_features // group, out_features, generator=g).to(
            torch.bfloat16 if precision == "int4" else torch.float8_e4m3fn
        ),
        "smooth_factor": torch.rand(in_features, generator=g).to(torch.bfloat16) + 0.5,
        "proj_down": torch.randn(in_features, rank, generator=g).to(torch.bfloat16),
        "proj_up": torch.randn(out_features, rank, generator=g).to(torch.bfloat16),
    }
    if bias:
        tensors["bias"] = torch.randn(out_features, generator=g).to(torch.bfloat16)
    if precision == "nvfp4":
        tensors["wcscales"] = torch.ones(out_features, dtype=torch.bfloat16)
        if wtscale:
            tensors["wtscale"] = torch.tensor([WTSCALE], dtype=torch.bfloat16)
    return tensors


def quant_config(
    targets: list[str],
    *,
    precision: str = "int4",
    rank: int = 16,
    rotated: list[str] | None = None,
) -> dict:
    config = {
        "quant_method": "nunchaku_lite",
        "compute_dtype": "bfloat16",
        "svdq_w4a4": {
            "precision": precision,
            "rank": rank,
            "group_size": 64 if precision == "int4" else 16,
            "targets": sorted(targets),
        },
    }
    if rotated:
        config["input_rotation"] = {"format_version": 2, "type": "hadamard", "targets": sorted(rotated)}
    return config


def write_checkpoint(
    path: Path,
    layers: dict[str, dict[str, torch.Tensor]],
    *,
    config: dict | None,
    prefix: str = "",
    extra: dict[str, torch.Tensor] | None = None,
    marker: bool = False,
    sidecar: bool = False,
    metadata: dict[str, str] | None = None,
) -> Path:
    tensors = {f"{prefix}{name}.{suffix}": t for name, ts in layers.items() for suffix, t in ts.items()}
    tensors.update({f"{prefix}{k}": v for k, v in (extra or {}).items()})
    if marker:
        tensors[ROTATION_MARKER_KEY] = torch.empty(0)
    meta = dict(metadata or {})
    if config is not None and not sidecar:
        meta["quantization_config"] = json.dumps(config)
    if config is not None and sidecar:
        (path.parent / "config.json").write_text(json.dumps({"quantization_config": config}))
    save_file(tensors, str(path), metadata=meta or None)
    return path
