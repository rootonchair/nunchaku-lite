"""Generic runtime LoRA conversion for manifest-patched transformers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from ..linear import AWQW4A16Linear, SVDQW4A4Linear
from ..manifest import RuntimeManifest, RuntimeManifestTarget
from .core.convert import (
    LORA_ERROR_LABEL,
    is_nunchaku_lite_lora_state_dict,
    normalize_nunchaku_lora_state_dict,
    set_standard_converted_lora_pair,
)
from .core.layout import lora_modules
from .core.peft import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    apply_network_alphas,
    extract_network_alphas,
    normalize_float_tensor,
    peft_lora_pairs,
)
from .core.runtime import NunchakuLoraMixin, load_lora_state_dict, raise_if_text_encoder_lora

KOHYA_DOWN_SUFFIX = ".lora_down.weight"
KOHYA_UP_SUFFIX = ".lora_up.weight"
ALPHA_SUFFIX = ".alpha"
NUNCHAKU_FORMAT = "nunchaku"
PEFT_FORMAT = "peft"
KOHYA_FORMAT = "kohya"
COMFYUI_FORMAT = "comfyui"


@dataclass(frozen=True)
class _ManifestLoraMatch:
    target_name: str
    source_name: str | None


class NunchakuManifestLoraMixin(NunchakuLoraMixin):
    """Mixin-style LoRA runtime for manifest-declared quantized targets."""

    def _convert_lora_to_nunchaku(
        self,
        path_or_state_dict: str | Path | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert supported generic LoRA inputs into Nunchaku Lite tensors."""

        state_dict = load_lora_state_dict(path_or_state_dict)
        return convert_manifest_lora_state_dict(state_dict, self)


def convert_manifest_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    """Detect a manifest LoRA format, then dispatch to its converter."""

    lora_format = detect_manifest_lora_format(state_dict)
    if lora_format == NUNCHAKU_FORMAT:
        return normalize_nunchaku_lora_state_dict(state_dict, transformer)
    if lora_format == PEFT_FORMAT:
        return convert_manifest_peft_lora_state_dict(state_dict, transformer)
    if lora_format == KOHYA_FORMAT:
        return convert_manifest_kohya_lora_state_dict(state_dict, transformer)
    if lora_format == COMFYUI_FORMAT:
        return convert_manifest_comfyui_lora_state_dict(state_dict, transformer)
    raise ValueError(f"Unsupported manifest LoRA format {lora_format!r}.")


def detect_manifest_lora_format(state_dict: dict[str, torch.Tensor]) -> str:
    """Return the supported manifest LoRA format for a state dict."""

    if is_nunchaku_lite_lora_state_dict(state_dict):
        return NUNCHAKU_FORMAT

    tensor_keys = [key for key, value in state_dict.items() if torch.is_tensor(value)]
    has_kohya = any(key.endswith((KOHYA_DOWN_SUFFIX, KOHYA_UP_SUFFIX)) for key in tensor_keys)
    has_comfyui = any(_is_comfyui_lora_key(key) for key in tensor_keys)
    has_peft = any(key.endswith((LORA_A_SUFFIX, LORA_B_SUFFIX)) for key in tensor_keys)

    if has_kohya:
        return KOHYA_FORMAT
    if has_comfyui:
        return COMFYUI_FORMAT
    if has_peft:
        return PEFT_FORMAT
    raise ValueError(f"LoRA state dict did not contain any supported {LORA_ERROR_LABEL} projection tensors.")


def convert_manifest_peft_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    """Convert PEFT/Diffusers-style LoRA keys through manifest targets."""

    normalized = normalize_manifest_peft_lora_state_dict(state_dict)
    return _convert_manifest_peft_pairs(normalized, transformer)


def convert_manifest_kohya_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    """Convert Kohya-style LoRA suffixes through manifest targets."""

    normalized = normalize_manifest_kohya_lora_state_dict(state_dict)
    return _convert_manifest_peft_pairs(normalized, transformer)


def convert_manifest_comfyui_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    """Convert ComfyUI-style component-prefixed LoRA keys through manifest targets."""

    normalized = normalize_manifest_comfyui_lora_state_dict(state_dict)
    return _convert_manifest_peft_pairs(normalized, transformer)


def _convert_manifest_peft_pairs(
    normalized_state_dict: dict[str, torch.Tensor],
    transformer: nn.Module,
) -> dict[str, torch.Tensor]:
    """Map normalized PEFT-style pairs to manifest runtime targets."""

    manifest = getattr(transformer, "_nunchaku_lite_runtime_manifest", None)
    if manifest is None:
        raise RuntimeError("Manifest LoRA conversion requires _nunchaku_lite_runtime_manifest on the transformer.")
    raise_if_text_encoder_lora(normalized_state_dict)

    alphas = extract_network_alphas(normalized_state_dict)
    pairs = peft_lora_pairs(apply_network_alphas(normalized_state_dict, alphas))
    if not pairs:
        raise ValueError(f"LoRA state dict did not contain any supported {LORA_ERROR_LABEL} projection tensors.")

    modules = lora_modules(transformer)
    target_index = {target.checkpoint_prefix: target for target in manifest.targets}
    alias_index = _build_manifest_lora_alias_index(manifest)
    grouped: dict[str, list[tuple[str | None, str, torch.Tensor, torch.Tensor]]] = defaultdict(list)
    unsupported = []

    for base_name, (lora_a, lora_b) in pairs.items():
        match = _resolve_manifest_lora_base(base_name, alias_index)
        if match is None:
            unsupported.append(base_name)
            continue
        grouped[match.target_name].append((match.source_name, base_name, lora_a, lora_b))

    if unsupported:
        sample = ", ".join(unsupported[:5])
        raise ValueError(f"Unsupported {LORA_ERROR_LABEL} target(s) for manifest runtime: {sample}")

    converted: dict[str, torch.Tensor] = {}
    for target_name, entries in grouped.items():
        if target_name not in modules:
            raise ValueError(
                f"Manifest LoRA target {target_name!r} does not exist on this patched {LORA_ERROR_LABEL} transformer."
            )
        target = target_index[target_name]
        module = modules[target_name]
        down, up = _convert_manifest_target_entries(target, module, entries)
        set_standard_converted_lora_pair(converted, target_name, down, up, module)
    return converted


def normalize_manifest_peft_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize PEFT/Diffusers LoRA keys while preserving target bases."""

    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if key.endswith(LORA_A_SUFFIX):
            base = key[: -len(LORA_A_SUFFIX)]
            normalized[f"{_strip_common_lora_prefixes(base)}{LORA_A_SUFFIX}"] = normalize_float_tensor(value)
        elif key.endswith(LORA_B_SUFFIX):
            base = key[: -len(LORA_B_SUFFIX)]
            normalized[f"{_strip_common_lora_prefixes(base)}{LORA_B_SUFFIX}"] = normalize_float_tensor(value)
        elif key.endswith(ALPHA_SUFFIX):
            base = key[: -len(ALPHA_SUFFIX)]
            normalized[f"{_strip_common_lora_prefixes(base)}{ALPHA_SUFFIX}"] = value
    return normalized


def normalize_manifest_kohya_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize Kohya ``lora_down/lora_up`` keys to PEFT pair suffixes."""

    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if key.endswith(KOHYA_DOWN_SUFFIX):
            base = key[: -len(KOHYA_DOWN_SUFFIX)]
            normalized[f"{_strip_common_lora_prefixes(base)}{LORA_A_SUFFIX}"] = normalize_float_tensor(value)
        elif key.endswith(KOHYA_UP_SUFFIX):
            base = key[: -len(KOHYA_UP_SUFFIX)]
            normalized[f"{_strip_common_lora_prefixes(base)}{LORA_B_SUFFIX}"] = normalize_float_tensor(value)
        elif key.endswith(ALPHA_SUFFIX):
            base = key[: -len(ALPHA_SUFFIX)]
            normalized[f"{_strip_common_lora_prefixes(base)}{ALPHA_SUFFIX}"] = value
    return normalized


def normalize_manifest_comfyui_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize ComfyUI component-prefixed LoRA keys to PEFT pair suffixes."""

    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if key.endswith(LORA_A_SUFFIX):
            base = key[: -len(LORA_A_SUFFIX)]
            normalized[f"{_strip_common_lora_prefixes(base)}{LORA_A_SUFFIX}"] = normalize_float_tensor(value)
        elif key.endswith(LORA_B_SUFFIX):
            base = key[: -len(LORA_B_SUFFIX)]
            normalized[f"{_strip_common_lora_prefixes(base)}{LORA_B_SUFFIX}"] = normalize_float_tensor(value)
        elif key.endswith(KOHYA_DOWN_SUFFIX):
            base = key[: -len(KOHYA_DOWN_SUFFIX)]
            normalized[f"{_strip_common_lora_prefixes(base)}{LORA_A_SUFFIX}"] = normalize_float_tensor(value)
        elif key.endswith(KOHYA_UP_SUFFIX):
            base = key[: -len(KOHYA_UP_SUFFIX)]
            normalized[f"{_strip_common_lora_prefixes(base)}{LORA_B_SUFFIX}"] = normalize_float_tensor(value)
        elif key.endswith(ALPHA_SUFFIX):
            base = key[: -len(ALPHA_SUFFIX)]
            normalized[f"{_strip_common_lora_prefixes(base)}{ALPHA_SUFFIX}"] = value
    return normalized


def _convert_manifest_target_entries(
    target: RuntimeManifestTarget,
    module: SVDQW4A4Linear | AWQW4A16Linear,
    entries: list[tuple[str | None, str, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    direct_entries = [entry for entry in entries if entry[0] is None or entry[0] == target.checkpoint_prefix]
    if direct_entries:
        if len(direct_entries) > 1:
            bases = ", ".join(entry[1] for entry in direct_entries[:5])
            raise ValueError(f"Ambiguous manifest LoRA entries for {target.checkpoint_prefix!r}: {bases}")
        _source, _base, lora_a, lora_b = direct_entries[0]
        return lora_a.contiguous(), lora_b.contiguous()

    source_order = list(target.source_modules)
    if len(source_order) <= 1:
        _source, _base, lora_a, lora_b = entries[0]
        return lora_a.contiguous(), lora_b.contiguous()

    by_source: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for source_name, base_name, lora_a, lora_b in entries:
        if source_name is None:
            continue
        if source_name in by_source:
            raise ValueError(f"Multiple LoRA pairs map to manifest source module {source_name!r}: {base_name}")
        by_source[source_name] = (lora_a, lora_b)
    return _fuse_manifest_source_branches(target, module, by_source)


def _fuse_manifest_source_branches(
    target: RuntimeManifestTarget,
    module: SVDQW4A4Linear | AWQW4A16Linear,
    by_source: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    source_order = list(target.source_modules)
    if not source_order:
        raise ValueError(f"Manifest target {target.checkpoint_prefix!r} has no source_modules for LoRA fusion.")
    if module.out_features % len(source_order) != 0:
        raise ValueError(
            f"Cannot infer equal branch sizes for manifest LoRA target {target.checkpoint_prefix!r} "
            f"with out_features={module.out_features} and {len(source_order)} source modules."
        )

    observed = [(source_name, by_source[source_name]) for source_name in source_order if source_name in by_source]
    if not observed:
        raise ValueError(f"No LoRA branches were provided for manifest target {target.checkpoint_prefix!r}.")

    branch_out = module.out_features // len(source_order)
    first_a, first_b = observed[0][1]
    for source_name, (lora_a, lora_b) in observed:
        if lora_a.shape[1] != module.in_features:
            raise ValueError(
                f"LoRA A tensor for {source_name!r} has input size {lora_a.shape[1]}, "
                f"expected {module.in_features}."
            )
        if lora_b.shape[0] > branch_out:
            raise ValueError(
                f"LoRA B tensor for {source_name!r} has output size {lora_b.shape[0]}, "
                f"but inferred branch size is {branch_out}."
            )

    if all(lora_a.equal(first_a) for _source_name, (lora_a, _lora_b) in observed):
        up_branches = []
        for source_name in source_order:
            pair = by_source.get(source_name)
            if pair is None:
                up_branches.append(
                    torch.zeros(branch_out, first_a.shape[0], dtype=first_b.dtype, device=first_b.device)
                )
                continue
            _lora_a, lora_b = pair
            if lora_b.shape[0] < branch_out:
                padded = torch.zeros(branch_out, lora_b.shape[1], dtype=lora_b.dtype, device=lora_b.device)
                padded[: lora_b.shape[0]] = lora_b
                lora_b = padded
            up_branches.append(lora_b)
        return first_a.contiguous(), torch.cat(up_branches, dim=0).contiguous()

    total_rank = sum(lora_a.shape[0] for _source_name, (lora_a, _lora_b) in observed)
    down = torch.zeros(total_rank, module.in_features, dtype=first_a.dtype, device=first_a.device)
    up = torch.zeros(module.out_features, total_rank, dtype=first_b.dtype, device=first_b.device)
    col = 0
    for source_index, source_name in enumerate(source_order):
        pair = by_source.get(source_name)
        if pair is None:
            continue
        lora_a, lora_b = pair
        rank = lora_a.shape[0]
        down[col : col + rank] = lora_a
        row = source_index * branch_out
        up[row : row + lora_b.shape[0], col : col + rank] = lora_b
        col += rank
    return down.contiguous(), up.contiguous()


def _build_manifest_lora_alias_index(manifest: RuntimeManifest) -> dict[str, set[_ManifestLoraMatch]]:
    index: dict[str, set[_ManifestLoraMatch]] = defaultdict(set)
    for target in manifest.targets:
        for path in (target.checkpoint_prefix, target.name):
            if path:
                _add_manifest_lora_aliases(index, path, _ManifestLoraMatch(target.checkpoint_prefix, None))
        for source_name in target.source_modules:
            source = None if source_name == target.checkpoint_prefix else source_name
            _add_manifest_lora_aliases(index, source_name, _ManifestLoraMatch(target.checkpoint_prefix, source))
    return index


def _add_manifest_lora_aliases(
    index: dict[str, set[_ManifestLoraMatch]],
    path: str,
    match: _ManifestLoraMatch,
) -> None:
    for alias in _manifest_lora_aliases(path):
        index[alias].add(match)


def _manifest_lora_aliases(path: str) -> set[str]:
    stripped = _strip_common_lora_prefixes(path)
    aliases = {path, stripped}
    aliases.update(
        {
            f"transformer.{stripped}",
            f"base_model.model.transformer.{stripped}",
            f"diffusion_model.{stripped}",
        }
    )
    compressed = stripped.replace(".", "_")
    aliases.update(
        {
            compressed,
            f"lora_transformer_{compressed}",
            f"lora_diffusion_model_{compressed}",
        }
    )
    return aliases


def _resolve_manifest_lora_base(
    base_name: str,
    alias_index: dict[str, set[_ManifestLoraMatch]],
) -> _ManifestLoraMatch | None:
    candidates: set[_ManifestLoraMatch] = set()
    for alias in _base_lookup_aliases(base_name):
        candidates.update(alias_index.get(alias, set()))
    if not candidates:
        return None

    direct = {candidate for candidate in candidates if candidate.source_name is None}
    if direct:
        candidates = direct
    if len(candidates) == 1:
        return next(iter(candidates))
    sample = ", ".join(
        f"{candidate.target_name}:{candidate.source_name or '<direct>'}"
        for candidate in sorted(candidates, key=lambda item: (item.target_name, item.source_name or ""))
    )
    raise ValueError(f"Ambiguous manifest LoRA target {base_name!r}; matched {sample}.")


def _base_lookup_aliases(base_name: str) -> set[str]:
    stripped = _strip_common_lora_prefixes(base_name)
    aliases = {base_name, stripped}
    if stripped.startswith("lora_transformer_"):
        aliases.add(stripped[len("lora_transformer_") :])
    if stripped.startswith("lora_diffusion_model_"):
        aliases.add(stripped[len("lora_diffusion_model_") :])
    return aliases


def _is_comfyui_lora_key(key: str) -> bool:
    return key.startswith(
        (
            "diffusion_model.",
            "model.diffusion_model.",
            "base_model.model.diffusion_model.",
            "lora_diffusion_model_",
        )
    )


def _strip_common_lora_prefixes(base_name: str) -> str:
    prefixes = (
        "base_model.model.transformer.",
        "base_model.model.diffusion_model.",
        "base_model.model.",
        "model.diffusion_model.",
        "diffusion_model.",
        "transformer.",
    )
    for prefix in prefixes:
        if base_name.startswith(prefix):
            return base_name[len(prefix) :]
    return base_name
