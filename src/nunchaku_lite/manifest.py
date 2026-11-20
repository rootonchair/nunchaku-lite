"""Runtime manifest parsing for Nunchaku Lite checkpoint metadata.

Shared with the MLX runtime (:mod:`nunchaku_lite.mlx`), which must import without torch: keep this module free
of torch and diffusers imports (``tests/test_torch_free_imports.py``).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

SCHEMA = "nunchaku_lite.runtime_manifest"
SUPPORTED_VERSION = 1
SUPPORTED_FORMAT_VERSIONS = (1, 2)
# Format version 2 adds online activation rotation: per target ``op_options.rotation``, or for
# checkpoints loaded through a model-specific adapter a model-level
# ``quantization_config.input_rotation`` declaration listing the rotated targets. Version 1
# checkpoints must not use it, so loaders that predate rotation refuse rotated checkpoints
# instead of silently producing wrong outputs.
ROTATION_FORMAT_VERSION = 2
# ``hadamard``: per contiguous 64-channel block (logical channel order) of the smoothed
# activations, multiply by the orthonormal Sylvester Hadamard matrix ``H64 / 8``; the low-rank
# branch sees the unrotated input and the packed residual weight is ``Q(R @ blockdiag(H64 / 8))``.
# The name is fixed to that definition: a different transform needs a new name.
SUPPORTED_ROTATION_TYPES = {"hadamard"}
ROTATION_DECLARATION_KEY = "input_rotation"
# Tensors of the abandoned pre-release ``block_matrix`` format, refused with a conversion hint.
LEGACY_INPUT_TRANSFORM_SUFFIX = ".input_transform"
SUPPORTED_OPS = {"svdq_w4a4", "awq_w4a16", "adanorm_awq_w4a16"}
SUPPORTED_PRECISIONS = {"int4", "fp4"}
SUPPORTED_KINDS = {"linear"}


@dataclass(frozen=True)
class RuntimeManifestTarget:
    """One runtime target declared by a Nunchaku Lite manifest."""

    name: str | None
    checkpoint_prefix: str
    source_modules: tuple[str, ...]
    roles: tuple[str, ...]
    kind: str
    nunchaku_op: str
    precision: str
    group_size: int
    rank: int
    has_bias: bool
    op_options: dict[str, Any] = field(default_factory=dict)
    activation: dict[str, Any] = field(default_factory=dict)

    @property
    def rotation(self) -> dict[str, Any] | None:
        """Return the validated ``op_options.rotation`` metadata, or ``None``."""

        rotation = self.op_options.get("rotation")
        return dict(rotation) if isinstance(rotation, dict) else None

    @property
    def runtime_precision(self) -> str:
        """Return the native precision name used by runtime modules."""

        return "nvfp4" if self.precision == "fp4" else self.precision


@dataclass(frozen=True)
class RuntimeManifest:
    """Validated v1 Nunchaku Lite runtime manifest."""

    schema: str
    version: int
    component: str
    nunchaku_format_version: int
    producer: dict[str, Any]
    requirements: dict[str, Any]
    structural_patches: tuple[dict[str, Any], ...]
    targets: tuple[RuntimeManifestTarget, ...]

    @property
    def precision(self) -> str | None:
        """Return the common manifest precision, or ``None`` for mixed."""

        precision = self.requirements.get("precision")
        if precision in SUPPORTED_PRECISIONS:
            return str(precision)
        return None

    @property
    def runtime_precision(self) -> str | None:
        """Return the common native precision, or ``None`` for mixed."""

        precision = self.precision
        if precision is None:
            return None
        return "nvfp4" if precision == "fp4" else precision

    @property
    def rank(self) -> int:
        """Return the common rank declared in requirements."""

        return int(self.requirements.get("rank", 32))

    @property
    def rotated_targets(self) -> tuple[RuntimeManifestTarget, ...]:
        """Return the targets that declare an online activation rotation."""

        return tuple(target for target in self.targets if target.rotation is not None)


def parse_runtime_manifest(
    quantization_config: dict[str, Any], checkpoint_keys: Iterable[str] | None = None
) -> RuntimeManifest | None:
    """Parse and validate ``quantization_config.runtime_manifest`` if present.

    Some Nunchaku Lite checkpoints (e.g. ``lite-infer/flux.1-dev-nunchaku-lite-int4-r32``) predate the runtime
    manifest. Their ``config.json`` lists the quantized modules instead (``svdq_w4a4.targets`` /
    ``awq_w4a16.targets``, one per Diffusers linear, no structural patches). When ``checkpoint_keys`` is given, a
    manifest is built from those lists, with ``has_bias`` taken from the presence of ``<target>.bias``.
    """

    raw = quantization_config.get("runtime_manifest")
    if raw is None and checkpoint_keys is not None:
        raw = _manifest_from_target_lists(quantization_config, set(checkpoint_keys))
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("quantization_config.runtime_manifest must be a JSON object.")

    schema = _required(raw, "schema", str)
    if schema != SCHEMA:
        raise ValueError(f"Unsupported runtime_manifest schema {schema!r}; expected {SCHEMA!r}.")

    version = _required(raw, "version", int)
    if version != SUPPORTED_VERSION:
        raise ValueError(f"Unsupported runtime_manifest version {version}; expected {SUPPORTED_VERSION}.")

    nunchaku_format_version = _required(raw, "nunchaku_format_version", int)
    if nunchaku_format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            "Unsupported runtime_manifest nunchaku_format_version "
            f"{nunchaku_format_version}; expected one of {SUPPORTED_FORMAT_VERSIONS}."
        )

    component = _required(raw, "component", str)
    producer = _required(raw, "producer", dict)
    requirements = _required(raw, "requirements", dict)
    _validate_producer(producer)
    _validate_requirements(requirements)
    structural_patches = _required(raw, "structural_patches", list)
    targets_raw = _required(raw, "targets", list)
    if not targets_raw:
        raise ValueError("runtime_manifest.targets must contain at least one target.")

    precision = requirements.get("precision")
    if precision not in (*SUPPORTED_PRECISIONS, "mixed"):
        raise ValueError(f"Unsupported runtime_manifest requirements.precision {precision!r}.")

    targets = tuple(_parse_target(index, target) for index, target in enumerate(targets_raw))
    rotated = [target.checkpoint_prefix for target in targets if target.rotation is not None]
    if rotated and nunchaku_format_version < ROTATION_FORMAT_VERSION:
        raise ValueError(
            f"runtime_manifest targets {rotated[:3]} declare op_options.rotation, which requires "
            f"nunchaku_format_version >= {ROTATION_FORMAT_VERSION}; got {nunchaku_format_version}."
        )

    return RuntimeManifest(
        schema=schema,
        version=version,
        component=component,
        nunchaku_format_version=nunchaku_format_version,
        producer=dict(producer),
        requirements=dict(requirements),
        structural_patches=tuple(_validate_structural_patch(patch) for patch in structural_patches),
        targets=targets,
    )


_TARGET_LIST_OPS = ("svdq_w4a4", "awq_w4a16")


def with_target_list_manifest(quantization_config: dict[str, Any], checkpoint_keys: Iterable[str]) -> dict[str, Any]:
    """``quantization_config`` with a ``runtime_manifest`` built from its ``svdq_w4a4`` / ``awq_w4a16`` target lists
    (checkpoints that predate the manifest); unchanged when it already has one or has no target lists."""

    if "runtime_manifest" in quantization_config:
        return quantization_config
    raw = _manifest_from_target_lists(quantization_config, set(checkpoint_keys))
    return quantization_config if raw is None else {**quantization_config, "runtime_manifest": raw}


def _manifest_from_target_lists(quantization_config: dict[str, Any], keys: set[str]) -> dict[str, Any] | None:
    if quantization_config.get("quant_method") != "nunchaku_lite":
        return None
    targets: list[dict[str, Any]] = []
    precisions: set[str] = set()
    rank = 0
    for op in _TARGET_LIST_OPS:
        section = quantization_config.get(op)
        if not isinstance(section, dict):
            continue
        names = section.get("targets")
        if not isinstance(names, list):
            raise TypeError(f"quantization_config.{op}.targets must be a list of module names.")
        precision = section.get("precision", "int4")
        precision = "fp4" if precision == "nvfp4" else precision
        group_size = section.get("group_size", 64)
        op_rank = section.get("rank", 0) if op == "svdq_w4a4" else 0
        if op == "svdq_w4a4":
            rank = op_rank
        precisions.add(precision)
        targets.extend(
            {
                "name": name,
                "checkpoint_prefix": name,
                "source_modules": [name],
                "roles": [],
                "kind": "linear",
                "nunchaku_op": op,
                "precision": precision,
                "group_size": group_size,
                "rank": op_rank,
                "has_bias": f"{name}.bias" in keys,
                "op_options": {},
                "activation": {},
            }
            for name in names
        )
    if not targets:
        return None
    rotated = _apply_target_list_rotation(quantization_config, targets)
    precision = precisions.pop() if len(precisions) == 1 else "mixed"
    return {
        "schema": SCHEMA,
        "version": SUPPORTED_VERSION,
        "component": "transformer",
        "nunchaku_format_version": ROTATION_FORMAT_VERSION if rotated else 1,
        "producer": {"name": "nunchaku_lite.config_target_lists", "version": "1"},
        "requirements": {
            "method": "svdquant",
            "precision": precision,
            "rank": rank,
            "weight_dtype": precision,
            "activation_dtype": precision,
        },
        "structural_patches": [],
        "targets": targets,
    }


def _validate_producer(producer: dict[str, Any]) -> None:
    _required(producer, "name", str)
    _required(producer, "version", str)


def _validate_requirements(requirements: dict[str, Any]) -> None:
    _required(requirements, "method", str)
    _required(requirements, "precision", str)
    _required(requirements, "rank", int)
    _required(requirements, "weight_dtype", str)
    _required(requirements, "activation_dtype", str)


def _parse_target(index: int, raw: Any) -> RuntimeManifestTarget:
    if not isinstance(raw, dict):
        raise TypeError(f"runtime_manifest.targets[{index}] must be a JSON object.")

    checkpoint_prefix = _required(raw, "checkpoint_prefix", str)
    source_modules = _required(raw, "source_modules", list)
    roles = _required(raw, "roles", list)
    kind = _required(raw, "kind", str)
    nunchaku_op = _required(raw, "nunchaku_op", str)
    precision = _required(raw, "precision", str)
    group_size = _required(raw, "group_size", int)
    rank = _required(raw, "rank", int)
    has_bias = _required(raw, "has_bias", bool)
    op_options = _required(raw, "op_options", dict)
    activation = _required(raw, "activation", dict)

    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"Unsupported runtime_manifest target kind {kind!r} at {checkpoint_prefix!r}.")
    if nunchaku_op not in SUPPORTED_OPS:
        raise ValueError(f"Unsupported runtime_manifest nunchaku_op {nunchaku_op!r} at {checkpoint_prefix!r}.")
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"Unsupported runtime_manifest target precision {precision!r} at {checkpoint_prefix!r}.")
    if group_size <= 0:
        raise ValueError(f"runtime_manifest target {checkpoint_prefix!r} must have positive group_size.")
    if rank < 0:
        raise ValueError(f"runtime_manifest target {checkpoint_prefix!r} must have non-negative rank.")
    if not all(isinstance(item, str) for item in source_modules):
        raise ValueError(f"runtime_manifest target {checkpoint_prefix!r} source_modules must be strings.")
    if not all(isinstance(item, str) for item in roles):
        raise ValueError(f"runtime_manifest target {checkpoint_prefix!r} roles must be strings.")

    if "rotation" in op_options:
        _validate_rotation(f"runtime_manifest target {checkpoint_prefix!r}", nunchaku_op, op_options["rotation"])

    if nunchaku_op == "adanorm_awq_w4a16":
        splits = op_options.get("adanorm_splits")
        if splits not in {3, 6}:
            raise ValueError(
                f"runtime_manifest target {checkpoint_prefix!r} requires op_options.adanorm_splits to be 3 or 6."
            )

    return RuntimeManifestTarget(
        name=raw.get("name") if isinstance(raw.get("name"), str) else None,
        checkpoint_prefix=checkpoint_prefix,
        source_modules=tuple(source_modules),
        roles=tuple(roles),
        kind=kind,
        nunchaku_op=nunchaku_op,
        precision=precision,
        group_size=group_size,
        rank=rank,
        has_bias=has_bias,
        op_options=dict(op_options),
        activation=dict(activation),
    )


def _validate_rotation(where: str, nunchaku_op: str, rotation: Any) -> None:
    """Validate one rotation object, ``{"type": "hadamard"}``; ``where`` names it in errors."""

    if not isinstance(rotation, dict):
        raise TypeError(f"{where}: rotation must be a JSON object like {{'type': 'hadamard'}}.")
    if nunchaku_op != "svdq_w4a4":
        raise ValueError(f"{where}: rotation is only supported for svdq_w4a4.")
    rotation_type = rotation.get("type")
    if rotation_type == "block_matrix":
        raise ValueError(
            f"{where} uses the pre-release {{'type': 'block_matrix'}} rotation format; replace it with "
            "{'type': 'hadamard'} and drop the .input_transform tensors (the packed weights are unchanged)."
        )
    if rotation_type not in SUPPORTED_ROTATION_TYPES:
        raise ValueError(f"{where} has unsupported rotation type {rotation_type!r}; expected 'hadamard'.")
    unknown = sorted(set(rotation) - {"type"})
    if unknown:
        raise ValueError(f"{where}: unknown rotation fields {unknown}; 'hadamard' takes none.")


def parse_rotation_declaration(quantization_config: dict[str, Any]) -> tuple[str, ...] | None:
    """Parse the model-level ``quantization_config["input_rotation"]`` declaration.

    ``{"format_version": 2, "type": "hadamard", "targets": [<checkpoint prefix>, ...]}`` declares
    which ``svdq_w4a4`` layers were exported rotated, for checkpoints without per-target manifest
    metadata (model-specific adapters, Diffusers-style target lists).

    Returns:
        The declared checkpoint prefixes, or ``None`` when there is no declaration.

    Raises:
        TypeError, ValueError: If the declaration is malformed or has an unsupported version.
    """

    raw = quantization_config.get(ROTATION_DECLARATION_KEY)
    if raw is None:
        return None
    where = f"quantization_config.{ROTATION_DECLARATION_KEY}"
    if not isinstance(raw, dict):
        raise TypeError(f"{where} must be a JSON object.")
    version = raw.get("format_version")
    if version != ROTATION_FORMAT_VERSION:
        raise ValueError(f"Unsupported {where}.format_version {version!r}; expected {ROTATION_FORMAT_VERSION}.")
    targets = raw.get("targets")
    if not isinstance(targets, list) or not targets or not all(isinstance(t, str) and t for t in targets):
        raise TypeError(f"{where}.targets must be a non-empty list of checkpoint prefixes.")
    if len(set(targets)) != len(targets):
        raise ValueError(f"{where}.targets lists a target more than once.")
    _validate_rotation(where, "svdq_w4a4", {k: v for k, v in raw.items() if k not in ("format_version", "targets")})
    return tuple(targets)


def _apply_target_list_rotation(quantization_config: dict[str, Any], targets: list[dict[str, Any]]) -> bool:
    """Carry a model-level rotation declaration onto synthesized target-list manifest targets."""

    declared = parse_rotation_declaration(quantization_config)
    if declared is None:
        return False
    by_name = {target["checkpoint_prefix"]: target for target in targets}
    unknown = [name for name in declared if by_name.get(name, {}).get("nunchaku_op") != "svdq_w4a4"]
    if unknown:
        raise ValueError(
            f"quantization_config.{ROTATION_DECLARATION_KEY}.targets {unknown[:3]} are not svdq_w4a4 targets "
            "of this checkpoint."
        )
    for name in declared:
        by_name[name]["op_options"] = {"rotation": {"type": "hadamard"}}
    return True


def _validate_structural_patch(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("runtime_manifest structural patches must be JSON objects.")
    patch_type = _required(raw, "type", str)
    if patch_type not in {"split_linear_output", "split_linear_input"}:
        raise ValueError(f"Unsupported runtime_manifest structural patch type {patch_type!r}.")
    module = _required(raw, "module", str)
    args = _required(raw, "args", dict)
    splits = args.get("splits")
    if not isinstance(splits, list):
        raise TypeError(f"runtime_manifest structural patch {module!r} requires args.splits list.")
    return {"type": patch_type, "module": module, "args": dict(args)}


def _required(raw: dict[str, Any], key: str, expected_type: type) -> Any:
    if key not in raw:
        raise ValueError(f"runtime_manifest is missing required field {key!r}.")
    value = raw[key]
    if not isinstance(value, expected_type):
        raise TypeError(f"runtime_manifest field {key!r} must be {expected_type.__name__}.")
    return value
