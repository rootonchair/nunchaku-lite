"""Runtime-manifest patching for MLX models (mirrors ``nunchaku_lite.adapters.manifest``).

A Nunchaku Lite runtime manifest (``quantization_config.runtime_manifest`` in the checkpoint's safetensors metadata)
lists every quantized layer with its op (``svdq_w4a4``, ``awq_w4a16`` or ``adanorm_awq_w4a16``), rank, group size and
bias, plus optional structural patches that split a fused ``nn.Linear`` before quantized children replace it. The
manifest is parsed by the same code as the PyTorch runtime (:mod:`nunchaku_lite.manifest`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import mlx.core as mx
from mlx import nn

from ..manifest import RuntimeManifest, RuntimeManifestTarget, parse_runtime_manifest
from .linear import AWQW4A16Linear, SVDQW4A4Linear

__all__ = [
    "RuntimeManifest",
    "RuntimeManifestTarget",
    "SplitLinearInput",
    "SplitLinearOutput",
    "apply_runtime_manifest",
    "parse_runtime_manifest",
]

SVDQ_PARAMS = ("qweight", "wscales", "smooth_factor", "proj_down", "proj_up", "bias")
AWQ_PARAMS = ("qweight", "wscales", "wzeros", "bias")


def _features(module) -> tuple[int, int] | None:
    in_features, out_features = getattr(module, "in_features", None), getattr(module, "out_features", None)
    if isinstance(in_features, int) and isinstance(out_features, int):
        return in_features, out_features
    if isinstance(module, nn.Linear):
        out_features, in_features = module.weight.shape
        return in_features, out_features
    return None


class SplitLinearOutput(nn.Module):
    """``nn.Linear`` replacement whose output features are split across ``linears`` (outputs concatenated)."""

    def __init__(self, linears: list[nn.Module]):
        super().__init__()
        self.linears = linears
        self.in_features = _features(linears[0])[0]
        self.out_features = sum(_features(linear)[1] for linear in linears)

    @classmethod
    def from_linear(cls, linear: nn.Linear, splits: list[int]) -> SplitLinearOutput:
        has_bias = "bias" in linear
        children, start = [], 0
        for split in splits:
            child = nn.Linear(linear.weight.shape[1], split, bias=has_bias)
            child.weight = linear.weight[start : start + split]
            if has_bias:
                child.bias = linear.bias[start : start + split]
            children.append(child)
            start += split
        return cls(children)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.concatenate([linear(x) for linear in self.linears], axis=-1)


class SplitLinearInput(nn.Module):
    """``nn.Linear`` replacement whose input features are split across ``linears`` (outputs summed)."""

    def __init__(self, linears: list[nn.Module]):
        super().__init__()
        self.linears = linears
        self.in_features_list = [_features(linear)[0] for linear in linears]
        self.in_features = sum(self.in_features_list)
        self.out_features = _features(linears[0])[1]

    @classmethod
    def from_linear(cls, linear: nn.Linear, splits: list[int]) -> SplitLinearInput:
        has_bias = "bias" in linear
        out_features = linear.weight.shape[0]
        children, start = [], 0
        for index, split in enumerate(splits):
            last = index == len(splits) - 1
            child = nn.Linear(split, out_features, bias=has_bias and last)
            child.weight = linear.weight[:, start : start + split]
            if has_bias and last:
                child.bias = linear.bias
            children.append(child)
            start += split
        return cls(children)

    def __call__(self, x: mx.array) -> mx.array:
        indices, total = [], 0
        for size in self.in_features_list[:-1]:
            total += size
            indices.append(total)
        chunks = mx.split(x, indices, axis=-1)
        out = self.linears[0](chunks[0])
        for linear, chunk in zip(self.linears[1:], chunks[1:]):
            out = out + linear(chunk)
        return out


def _resolve(model: nn.Module, path: str):
    parent, parts = model, path.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, list) else getattr(parent, part)
    last = parts[-1]
    child = parent[int(last)] if isinstance(parent, list) else getattr(parent, last, None)
    return parent, last, child


def _try_resolve(model: nn.Module, path: str):
    try:
        return _resolve(model, path)
    except (AttributeError, IndexError, ValueError, KeyError, TypeError):
        return None, None, None


def _assign(parent, name: str, module: nn.Module) -> None:
    if isinstance(parent, list):
        parent[int(name)] = module
    else:
        setattr(parent, name, module)


def _resolve_splits(total_in: int, total_out: int, splits: list, *, input_split: bool) -> list[int]:
    resolved = []
    for split in splits:
        if split == "out_features":
            resolved.append(total_out)
        elif split == "in_features":
            resolved.append(total_in)
        elif isinstance(split, int):
            resolved.append(split)
        else:
            raise ValueError(f"Unsupported runtime_manifest split value {split!r}.")
    total = total_in if input_split else total_out
    if total - sum(resolved) > 0:
        resolved.append(total - sum(resolved))
    resolved = [split for split in resolved if split > 0]
    if len(resolved) < 2 or sum(resolved) != total:
        raise ValueError(f"Structural split {splits!r} does not partition {total} features into at least two parts.")
    return resolved


def _path_matches(pattern: str, path: str) -> bool:
    pattern_parts, path_parts = pattern.split("."), path.split(".")
    return len(pattern_parts) == len(path_parts) and all(p == "*" or p == q for p, q in zip(pattern_parts, path_parts))


def _checkpoint_modules(state: Mapping[str, mx.array]) -> set[str]:
    """Every dotted ancestor of a checkpoint key: the checkpoint's module names (Diffusers namespace)."""

    names = set()
    for key in state:
        parts = key.split(".")
        names.update(".".join(parts[:i]) for i in range(1, len(parts)))
    return names


def _apply_structural_patches(model, state, manifest: RuntimeManifest, key_map) -> None:
    if not manifest.structural_patches:
        return
    modules = _checkpoint_modules(state)
    for patch in manifest.structural_patches:
        input_split = patch["type"] == "split_linear_input"
        matched = sorted(name for name in modules if _path_matches(patch["module"], name))
        if not matched:
            raise ValueError(f"runtime_manifest structural patch pattern {patch['module']!r} matched no modules.")
        for name in matched:
            parent, child_name, module = _try_resolve(model, key_map(name))
            if isinstance(module, (SplitLinearInput, SplitLinearOutput)):
                continue
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"{patch['type']} expected nn.Linear at {key_map(name)!r}, got {type(module).__name__}."
                )
            in_features, out_features = _features(module)
            splits = _resolve_splits(in_features, out_features, patch["args"]["splits"], input_split=input_split)
            cls = SplitLinearInput if input_split else SplitLinearOutput
            _assign(parent, child_name, cls.from_linear(module, splits))


def _build(target: RuntimeManifestTarget, in_features: int, out_features: int, dtype) -> nn.Module:
    if target.nunchaku_op == "svdq_w4a4":
        if target.group_size != 64:
            raise ValueError(f"svdq_w4a4 target {target.checkpoint_prefix!r} requires group_size=64.")
        return SVDQW4A4Linear(in_features, out_features, rank=target.rank, bias=target.has_bias, dtype=dtype)
    if target.nunchaku_op == "adanorm_awq_w4a16" and not target.checkpoint_prefix.endswith(".linear"):
        raise ValueError(f"adanorm_awq_w4a16 target {target.checkpoint_prefix!r} must reference a '.linear' child.")
    splits = target.op_options.get("adanorm_splits", 1) if target.nunchaku_op == "adanorm_awq_w4a16" else 1
    return AWQW4A16Linear(
        in_features,
        out_features,
        bias=target.has_bias,
        group_size=target.group_size,
        adanorm_splits=splits,
        dtype=dtype,
    )


def apply_runtime_manifest(
    model: nn.Module,
    state: Mapping[str, mx.array],
    manifest: RuntimeManifest,
    *,
    key_map: Callable[[str], str] | None = None,
    dtype=None,
) -> set[str]:
    """Replace every manifest target in ``model`` with a Nunchaku layer and load its checkpoint tensors.

    Args:
        model: MLX module to patch in place.
        state: Checkpoint tensors (Diffusers names).
        manifest: Parsed runtime manifest (see :func:`parse_runtime_manifest`).
        key_map: Maps checkpoint module names to ``model`` paths (default: identity).
        dtype: Floating-point dtype of the new layers (default: the checkpoint's ``wscales`` dtype).

    Returns:
        The checkpoint keys consumed by quantized layers; the caller loads the rest as dense weights.
    """

    key_map = key_map or (lambda name: name)
    if any(target.precision != "int4" for target in manifest.targets):
        raise NotImplementedError("NVFP4 (fp4) manifest targets are not supported on MLX yet; use an INT4 checkpoint.")
    _apply_structural_patches(model, state, manifest, key_map)

    consumed: set[str] = set()
    for target in manifest.targets:
        for source in target.source_modules:
            if _try_resolve(model, key_map(source))[2] is None:
                raise ValueError(
                    f"runtime_manifest target {target.checkpoint_prefix!r} source module {source!r} "
                    f"has no module at model path {key_map(source)!r}."
                )
        path = key_map(target.checkpoint_prefix)
        parent, name, module = _try_resolve(model, path)
        if module is None:
            raise ValueError(
                f"runtime_manifest target {target.checkpoint_prefix!r} has no module at model path {path!r}."
            )
        svdq = target.nunchaku_op == "svdq_w4a4"
        prefix = target.checkpoint_prefix
        tensors = [
            (n, state[f"{prefix}.{n}"]) for n in (SVDQ_PARAMS if svdq else AWQ_PARAMS) if f"{prefix}.{n}" in state
        ]
        found = dict(tensors)
        missing = {"qweight", "wscales"} - found.keys()
        if missing:
            raise ValueError(f"Checkpoint is missing {sorted(missing)} for manifest target {prefix!r}.")
        # Shapes come from the checkpoint, so pre-quantized (QuantizedLinear) base layers can be replaced too.
        rows, cols = found["qweight"].shape
        in_features, out_features = cols * 2, rows if svdq else rows * 4
        expected = _features(module)
        if expected is not None and expected != (in_features, out_features):
            raise ValueError(
                f"runtime_manifest target {prefix!r} is {in_features}->{out_features} in the checkpoint but "
                f"{expected[0]}->{expected[1]} at model path {path!r}."
            )
        layer = _build(target, in_features, out_features, dtype or found["wscales"].dtype)
        int_dtype = mx.int8 if svdq else mx.int32
        layer.load_weights([(n, v.astype(int_dtype) if n == "qweight" else v) for n, v in tensors])
        _assign(parent, name, layer)
        consumed.update(f"{prefix}.{n}" for n, _ in tensors)
    return consumed
