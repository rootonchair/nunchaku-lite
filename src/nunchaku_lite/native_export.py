"""Write native-layout Nunchaku Lite checkpoints by splicing SVDQ layers into an official single-file transformer.

The published LTX checkpoint is Lightricks' own transformer file (``model.diffusion_model.*``, text connectors
included) in which every quantization target's ``weight`` / ``bias`` is replaced by our SVDQ tensors. Every other
tensor, and the file metadata (ltx-core reads the model ``config`` from it), is copied unchanged, so ltx-core and
ComfyUI see the official model except for the quantized layers. ``quantization_config`` goes into the metadata;
rotated checkpoints also get the root ``nunchaku_lite.input_rotation`` marker.

The SVDQ tensors come from any checkpoint :func:`nunchaku_lite.loader.read_checkpoint_spec` accepts (typically a
Diffusers-layout export). Quantized layer names are the same in both layouts, so no other key is renamed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from .loader import NATIVE_PREFIX, _quantization_config, read_checkpoint_spec
from .manifest import ROTATION_MARKER_KEY

__all__ = ["splice_native", "verify_native_splice"]


def splice_native(
    official: str | os.PathLike,
    quantized: str | os.PathLike,
    out: str | os.PathLike,
    *,
    allow_fp4_rotation: bool = False,
) -> dict[str, int]:
    """Write ``out``: ``official`` with the SVDQ layers of ``quantized`` spliced in. Returns a small report."""

    from safetensors import safe_open
    from safetensors.torch import save_file

    official, quantized, out = Path(official), Path(quantized), Path(out)
    if out.resolve() in (official.resolve(), quantized.resolve()):
        raise ValueError("Refusing to overwrite an input file.")
    spec = read_checkpoint_spec(quantized, allow_fp4_rotation=allow_fp4_rotation)
    raw_config = _quantization_config(_metadata(quantized), quantized, native=spec.layout == "native")

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(official), framework="pt") as f:
        metadata = dict(f.metadata() or {})
        keys = set(f.keys())
        if not any(key.startswith(NATIVE_PREFIX) for key in keys):
            raise ValueError(f"{official} is not a native-layout transformer (no {NATIVE_PREFIX!r} keys).")
        if "quantization_config" in metadata or ROTATION_MARKER_KEY in keys:
            raise ValueError(f"{official} is already quantized; splice into the official bf16 file.")
        replaced = set()
        for name, layer in spec.layers.items():
            weight_key = f"{NATIVE_PREFIX}{name}.weight"
            bias_key = f"{NATIVE_PREFIX}{name}.bias"
            if weight_key not in keys:
                raise ValueError(f"Official file has no {weight_key!r} for quantization target {name!r}.")
            shape = tuple(f.get_slice(weight_key).get_shape())
            if shape != (layer.out_features, layer.in_features):
                raise ValueError(f"{weight_key} is {shape}, but the quantized layer is "
                                 f"({layer.out_features}, {layer.in_features}); not the same model.")
            if (bias_key in keys) != layer.has_bias:
                raise ValueError(f"{name!r}: the official file {'has' if bias_key in keys else 'has no'} bias, "
                                 "the quantized layer disagrees.")
            replaced |= {weight_key, bias_key} & keys
        for key in sorted(keys - replaced):
            tensors[key] = f.get_tensor(key)

    with safe_open(str(quantized), framework="pt") as q:
        for name, layer in spec.layers.items():
            for suffix in layer.tensors:
                tensors[f"{NATIVE_PREFIX}{name}.{suffix}"] = q.get_tensor(spec.file_key(name, suffix))
    if spec.rotated:
        tensors[ROTATION_MARKER_KEY] = torch.empty(0)

    metadata["quantization_config"] = json.dumps(raw_config, sort_keys=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".partial")
    save_file(tensors, str(tmp), metadata=metadata)
    os.replace(tmp, out)
    return {"layers": len(spec.layers), "replaced_official_tensors": len(replaced), "tensors": len(tensors)}


def verify_native_splice(official: str | os.PathLike, spliced: str | os.PathLike, quantized: str | os.PathLike) -> None:
    """Check a spliced file: non-target tensors and metadata byte-identical to ``official``, SVDQ tensors to ``quantized``."""

    from safetensors import safe_open

    spec = read_checkpoint_spec(spliced)
    source = read_checkpoint_spec(quantized)
    if set(spec.layers) != set(source.layers):
        raise AssertionError("Spliced file has different quantization targets than its source.")
    for name, layer in spec.layers.items():
        if layer != _relocated(source.layers[name], layer):
            raise AssertionError(f"Layer {name!r} differs between the spliced file and its source.")
    with safe_open(str(official), framework="pt") as o, safe_open(str(spliced), framework="pt") as s, safe_open(
        str(quantized), framework="pt"
    ) as q:
        official_meta, spliced_meta = dict(o.metadata() or {}), dict(s.metadata() or {})
        if {k: v for k, v in spliced_meta.items() if k != "quantization_config"} != official_meta:
            raise AssertionError("Spliced metadata differs from the official file's (besides quantization_config).")
        replaced = {f"{NATIVE_PREFIX}{name}.{p}" for name in spec.layers for p in ("weight", "bias")}
        expected_official = set(o.keys()) - replaced
        quantized_keys = spec.quantized_file_keys()
        if set(s.keys()) != expected_official | quantized_keys:
            raise AssertionError("Spliced key set is not (official - replaced) + quantized layers.")
        for key in expected_official:
            if _digest(o.get_tensor(key)) != _digest(s.get_tensor(key)):
                raise AssertionError(f"{key} differs from the official tensor.")
        for name, layer in spec.layers.items():
            for suffix in layer.tensors:
                if _digest(q.get_tensor(source.file_key(name, suffix))) != _digest(s.get_tensor(spec.file_key(name, suffix))):
                    raise AssertionError(f"{name}.{suffix} differs from the quantized source.")


def _relocated(layer, like):
    from dataclasses import replace

    return replace(layer, dropped=like.dropped)


def _metadata(path: Path) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        return dict(f.metadata() or {})


def _digest(tensor: torch.Tensor) -> tuple:
    raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() if tensor.numel() else b""
    return tensor.dtype, tuple(tensor.shape), hashlib.sha256(raw).hexdigest()
