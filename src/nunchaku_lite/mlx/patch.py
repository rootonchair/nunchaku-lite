"""Patch an MLX model with Nunchaku SVDQ INT4 layers (mirrors ``nunchaku_lite.patch_transformer``)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import mlx.core as mx
from mlx import nn

from .linear import AWQW4A16Linear, SVDQW4A4Linear, resolve_backend
from .manifest import apply_runtime_manifest, parse_runtime_manifest

__all__ = ["configure_backend", "load_checkpoint", "load_dense_weights", "patch_transformer", "svdq_layer_from_state"]

# Original Nunchaku checkpoints use these names; nunchaku_lite checkpoints use the right-hand side.
_KEY_ALIASES = {"lora_down": "proj_down", "lora_up": "proj_up", "smooth": "smooth_factor"}
_IGNORED_SUFFIXES = ("smooth_orig", "smooth_factor_orig")
_SVDQ_PARAMS = ("qweight", "wscales", "smooth_factor", "proj_down", "proj_up", "bias")
_AWQ_PARAMS = ("qweight", "wscales", "wzeros", "bias")


def _prefix_rule(rule, default):
    if rule is None:
        return lambda prefix: default
    if callable(rule):
        return rule
    if isinstance(rule, Mapping):
        return lambda prefix: rule.get(prefix, default)
    names = set(rule)
    return lambda prefix: prefix in names


def _normalize_keys(state: Mapping[str, mx.array]) -> dict[str, mx.array]:
    out = {}
    for key, value in state.items():
        prefix, _, name = key.rpartition(".")
        if name in _IGNORED_SUFFIXES:
            continue
        out[f"{prefix}.{_KEY_ALIASES.get(name, name)}" if prefix else _KEY_ALIASES.get(name, name)] = value
    return out


def _checkpoint_files(checkpoint: str | Path) -> list[Path]:
    """Resolve a checkpoint reference to its safetensors files.

    Accepts a ``.safetensors`` file, a directory (a Diffusers repo root with a ``transformer/`` subfolder, or the
    transformer folder itself, possibly sharded), ``org/repo/path/file.safetensors`` on the Hub, ``org/repo``
    (a Diffusers repo on the Hub; only ``transformer/`` is downloaded) or ``org/repo/subfolder``.
    """

    path = Path(checkpoint)
    if not path.exists():
        parts = str(checkpoint).strip("/").split("/")
        if len(parts) < 2:
            raise FileNotFoundError(f"Checkpoint {checkpoint!s} not found.")
        repo = "/".join(parts[:2])
        if str(checkpoint).endswith(".safetensors"):
            from huggingface_hub import hf_hub_download

            return [Path(hf_hub_download(repo, "/".join(parts[2:])))]
        from huggingface_hub import snapshot_download

        subfolder = "/".join(parts[2:]) or "transformer"
        root = Path(snapshot_download(repo, allow_patterns=[f"{subfolder}/*.safetensors", f"{subfolder}/*.json"]))
        path = root / subfolder
    if path.is_file():
        return [path]
    if path.is_dir() and (path / "transformer").is_dir():
        path = path / "transformer"
    files = sorted(path.glob("*.safetensors")) if path.is_dir() else []
    if not files:
        raise FileNotFoundError(f"No .safetensors checkpoint found at {path!s}.")
    return files


def _quantization_config(metadata: Mapping[str, str], folder: Path | None) -> dict:
    raw = metadata.get("quantization_config")
    if raw:
        return json.loads(raw) if isinstance(raw, str) else dict(raw)
    config = folder / "config.json" if folder is not None else None
    if config is not None and config.is_file():
        value = json.loads(config.read_text()).get("quantization_config")
        if isinstance(value, dict):
            return value
    return {}


def load_checkpoint(
    checkpoint: str | Path | Mapping[str, mx.array], *, return_quantization_config: bool = False
) -> dict[str, mx.array] | tuple[dict[str, mx.array], dict]:
    """Load a Nunchaku checkpoint and normalize original Nunchaku key names to the ``nunchaku_lite`` ones.

    Args:
        checkpoint: A mapping of tensors, a ``.safetensors`` file, a (possibly sharded) checkpoint folder, a
            Diffusers repo folder with a ``transformer/`` subfolder, or a Hub reference (``org/repo``,
            ``org/repo/subfolder`` or ``org/repo/path/file.safetensors``).
        return_quantization_config: Also return the checkpoint's ``quantization_config`` (from the safetensors
            metadata, else the folder's ``config.json``; ``{}`` when absent), which carries the runtime manifest.
    """

    quantization_config: dict = {}
    if isinstance(checkpoint, Mapping):
        state = dict(checkpoint)
    else:
        state, metadata = {}, {}
        files = _checkpoint_files(checkpoint)
        for file in files:
            tensors, file_metadata = mx.load(str(file), return_metadata=True)
            state.update(tensors)
            if "quantization_config" in file_metadata and "quantization_config" not in metadata:
                metadata = file_metadata
        quantization_config = _quantization_config(metadata, files[0].parent)
    state = _normalize_keys(state)
    if any(k.endswith((".wcscales", ".wtscale")) for k in state):
        raise NotImplementedError("NVFP4 SVDQ checkpoints are not supported on MLX yet; use an INT4 checkpoint.")
    return (state, quantization_config) if return_quantization_config else state


def svdq_layer_from_state(
    state: Mapping[str, mx.array], prefix: str, *, act_unsigned: bool = False, dtype=mx.bfloat16, backend: str = "auto"
) -> SVDQW4A4Linear:
    """Build an :class:`SVDQW4A4Linear` from one checkpoint prefix (shapes are read from the tensors)."""

    qweight = state[f"{prefix}.qweight"]
    out_features, in_features = qweight.shape[0], qweight.shape[1] * 2
    layer = SVDQW4A4Linear(
        in_features,
        out_features,
        rank=state[f"{prefix}.proj_down"].shape[1],
        bias=f"{prefix}.bias" in state,
        act_unsigned=act_unsigned,
        dtype=dtype,
        backend=backend,
    )
    tensors = [(n, state[f"{prefix}.{n}"]) for n in _SVDQ_PARAMS if f"{prefix}.{n}" in state]
    layer.load_weights([(n, v.astype(mx.int8) if n == "qweight" else v) for n, v in tensors])
    return layer


def configure_backend(model: nn.Module, backend: str = "auto") -> str:
    """Resolve ``backend`` once for the whole model and store it on every SVDQ layer."""

    resolved = resolve_backend(backend, _int8_resident_bytes(model) if backend == "auto" else None)
    for _, module in model.named_modules():
        if isinstance(module, SVDQW4A4Linear):
            module.backend = resolved
    return resolved


def _resolve(model: nn.Module, path: str):
    parent, parts = model, path.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, list) else getattr(parent, part)
    last = parts[-1]
    child = parent[int(last)] if isinstance(parent, list) else getattr(parent, last, None)
    return parent, last, child


def _assign(parent, name: str, module: nn.Module) -> None:
    if isinstance(parent, list):
        parent[int(name)] = module
    else:
        setattr(parent, name, module)


def patch_transformer(
    model: nn.Module,
    checkpoint: str | Path | Mapping[str, mx.array],
    *,
    act_unsigned: Callable[[str], bool] | Iterable[str] | None = None,
    adanorm_splits: Callable[[str], int] | Mapping[str, int] | None = None,
    key_map: Callable[[str], str] | None = None,
    dtype=None,
    strict: bool = True,
    backend: str = "auto",
) -> nn.Module:
    """Replace ``nn.Linear`` layers with Nunchaku quantized layers and load an INT4 checkpoint.

    Every checkpoint prefix with SVDQ tensors (``<prefix>.qweight`` plus
    ``wscales`` / ``proj_down`` / ``proj_up``) becomes an :class:`SVDQW4A4Linear`,
    and every AWQ W4A16 prefix (``qweight`` / ``wscales`` / ``wzeros``) an
    :class:`AWQW4A16Linear`, replacing the ``nn.Linear`` at ``key_map(prefix)``.
    All remaining checkpoint tensors (norms, embeddings, unquantized layers) are
    loaded into the model as regular weights.

    When the checkpoint carries a Nunchaku Lite runtime manifest (``quantization_config.runtime_manifest``, or
    the ``svdq_w4a4`` / ``awq_w4a16`` target lists of older lite checkpoints), the manifest decides which layers
    are quantized and how (as in ``nunchaku_lite``); ``act_unsigned`` and ``adanorm_splits`` are then ignored.
    Lite checkpoints run with signed activations and plain AWQ layers: their converter folds the unsigned shift
    and AdaNorm layout into the stored tensors.

    Args:
        model: MLX model to patch in place.
        checkpoint: Checkpoint reference accepted by :func:`load_checkpoint` or an already-loaded mapping.
        act_unsigned: Checkpoint prefixes (or a predicate on them) whose inputs are
            post-GELU activations quantized with Nunchaku's unsigned INT4 shift,
            e.g. FLUX.1 / Qwen-Image MLP ``fc2`` layers.
        adanorm_splits: AdaNorm chunk count per AWQ prefix (mapping or callable) for layers
            stored in Nunchaku's AdaNorm layout, e.g. 6 for FLUX.1 ``norm1.linear`` and 3 for
            single-block ``norm.linear``. The layers then load in the original ``nn.Linear`` order.
        key_map: Maps checkpoint key prefixes to model module paths when names differ.
        dtype: Floating-point dtype for SVDQ parameters (default: the replaced layer's dtype).
        strict: Raise when a checkpoint tensor has no matching model module or parameter.
        backend: ``"auto"``, ``"int8"``, ``"int4"``, or ``"dense"``, resolved once for the
            whole model. ``"auto"`` picks ``"int8"`` when the model fits in half of the GPU
            working set with int8 weights, else ``"int4"`` (``"dense"`` without Neural
            Accelerators).

    Returns:
        ``model``, patched in place.
    """

    resolve_backend(backend)  # validate before loading anything
    state, quantization_config = load_checkpoint(checkpoint, return_quantization_config=True)
    key_map = key_map or (lambda key: key)
    manifest = parse_runtime_manifest(quantization_config, checkpoint_keys=state)
    if manifest is not None:
        consumed = apply_runtime_manifest(model, state, manifest, key_map=key_map, dtype=dtype)
        load_dense_weights(model, [(key_map(k), v) for k, v in state.items() if k not in consumed], strict)
        configure_backend(model, backend)
        return model
    is_unsigned = _prefix_rule(act_unsigned, False)
    splits_for = _prefix_rule(adanorm_splits, 1)

    prefixes = sorted(
        k[: -len(".qweight")] for k in state if k.endswith(".qweight") and k[: -len(".qweight")] + ".proj_down" in state
    )
    consumed: set[str] = set()
    for prefix in prefixes:
        path = key_map(prefix)
        try:
            parent, name, linear = _resolve(model, path)
        except (AttributeError, IndexError, ValueError, KeyError):
            linear = None
        if not isinstance(linear, (nn.Linear, SVDQW4A4Linear)):
            if strict:
                raise ValueError(f"Checkpoint layer {prefix!r} has no nn.Linear at model path {path!r}.")
            continue
        rank = state[f"{prefix}.proj_down"].shape[1]
        module = (
            SVDQW4A4Linear.from_linear(linear, rank=rank, act_unsigned=is_unsigned(prefix), dtype=dtype)
            if isinstance(linear, nn.Linear)
            else linear
        )
        tensors = [(n, state[f"{prefix}.{n}"]) for n in _SVDQ_PARAMS if f"{prefix}.{n}" in state]
        module.load_weights([(n, v.astype(mx.int8) if n == "qweight" else v) for n, v in tensors])
        _assign(parent, name, module)
        consumed.update(f"{prefix}.{n}" for n, _ in tensors)

    awq_prefixes = sorted(
        k[: -len(".wzeros")] for k in state if k.endswith(".wzeros") and k[: -len(".wzeros")] + ".qweight" in state
    )
    for prefix in awq_prefixes:
        path = key_map(prefix)
        try:
            parent, name, linear = _resolve(model, path)
        except (AttributeError, IndexError, ValueError, KeyError):
            linear = None
        if not isinstance(linear, (nn.Linear, AWQW4A16Linear)):
            if strict:
                raise ValueError(f"Checkpoint layer {prefix!r} has no nn.Linear at model path {path!r}.")
            continue
        wscales = state[f"{prefix}.wscales"]
        in_features = state[f"{prefix}.qweight"].shape[1] * 2
        group_size = in_features // wscales.shape[0]
        module = (
            AWQW4A16Linear.from_linear(linear, group_size=group_size, adanorm_splits=splits_for(prefix), dtype=dtype)
            if isinstance(linear, nn.Linear)
            else linear
        )
        tensors = [(n, state[f"{prefix}.{n}"]) for n in _AWQ_PARAMS if f"{prefix}.{n}" in state]
        module.load_weights([(n, v.astype(mx.int32) if n == "qweight" else v) for n, v in tensors])
        _assign(parent, name, module)
        consumed.update(f"{prefix}.{n}" for n, _ in tensors)

    load_dense_weights(model, [(key_map(k), v) for k, v in state.items() if k not in consumed], strict)

    configure_backend(model, backend)
    return model


def load_dense_weights(model: nn.Module, items: list[tuple[str, mx.array]], strict: bool = True) -> set[str]:
    """Load unquantized checkpoint tensors (already mapped to model paths) into ``model``.

    Raises when ``strict`` and a tensor has no matching model parameter. Returns the loaded parameter paths.
    """

    if not items:
        return set()
    model_keys = {k for k, _ in nn.utils.tree_flatten(model.parameters())}
    unknown = [k for k, _ in items if k not in model_keys]
    if unknown and strict:
        raise ValueError(f"{len(unknown)} checkpoint tensor(s) do not match model parameters, e.g. {unknown[0]!r}.")
    known = [(k, v) for k, v in items if k in model_keys]
    model.load_weights(known, strict=False)
    return {k for k, _ in known}


def _int8_resident_bytes(model: nn.Module) -> int:
    """Approximate the model footprint with int8 SVDQ weights (int8 copy plus the packed checkpoint)."""

    total = sum(v.nbytes for _, v in nn.utils.tree_flatten(model.parameters()))
    for _, m in model.named_modules():
        if isinstance(m, SVDQW4A4Linear):
            total += m.in_features * m.out_features
    return total
