"""Load MLX diffusion pipelines with a Nunchaku INT4 transformer (mirrors ``nunchaku_lite.load_nunchaku_pipeline``).

MLX has no Diffusers; pipelines come from `mflux <https://github.com/filipstrand/mflux>`_. The pipeline is built
from the base model (text encoder, VAE, tokenizer, scheduler), then a registered adapter swaps the transformer's
projections for Nunchaku layers and loads the checkpoint's remaining (unquantized) transformer weights.

Checkpoints with a Nunchaku Lite runtime manifest (e.g. the ``lite-infer`` Diffusers repos) are patched from the
manifest, like ``nunchaku_lite``'s ``ManifestAdapter``. Manifest module names are Diffusers names; a model-family
adapter supplies the Diffusers -> mflux name map (derived from mflux's own weight mapping), and the generic
``"manifest"`` adapter uses names as they are.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import mlx.core as mx
from mlx import nn

from ..checkpoint_keys import convert_flux2_state_dict
from .keymap import PrefixMap
from .linear import AWQW4A16Linear, SVDQW4A4Linear, resolve_backend
from .manifest import RuntimeManifest, apply_runtime_manifest, parse_runtime_manifest
from .patch import configure_backend, load_checkpoint, load_dense_weights, svdq_layer_from_state

__all__ = [
    "Flux1Adapter",
    "Flux2Adapter",
    "ManifestAdapter",
    "PipelineAdapter",
    "list_adapters",
    "load_nunchaku_pipeline",
    "register_adapter",
]


class PipelineAdapter(Protocol):
    """Model-family adapter: installs Nunchaku layers into an MLX pipeline's transformer.

    ``manifest`` is the checkpoint's parsed runtime manifest, or ``None`` for checkpoints without one.
    """

    target: str

    def matches(self, pipeline: Any) -> bool: ...

    def patch(
        self,
        pipeline: Any,
        state: dict[str, mx.array],
        *,
        manifest: RuntimeManifest | None,
        backend: str,
        strict: bool,
    ) -> None: ...


_ADAPTERS: dict[str, PipelineAdapter] = {}


def register_adapter(adapter: PipelineAdapter) -> None:
    """Register an adapter under ``adapter.target``."""

    _ADAPTERS[adapter.target] = adapter


def list_adapters() -> list[str]:
    """Return the registered adapter targets."""

    return sorted(_ADAPTERS)


def load_nunchaku_pipeline(
    pretrained_model_name_or_path: str | Path,
    *,
    pipeline_cls: type,
    checkpoint: str | Path | Mapping[str, mx.array],
    target: str = "auto",
    precision: str = "int4",
    backend: str = "auto",
    strict: bool = True,
    **pipeline_kwargs,
):
    """Build an MLX pipeline and install a Nunchaku INT4 transformer.

    Args:
        pretrained_model_name_or_path: Base model (Hub id or local path) providing the text encoder, VAE,
            tokenizer and scheduler; passed to ``pipeline_cls(model_path=...)``.
        pipeline_cls: MLX pipeline class, e.g. ``mflux.models.flux2.variants.Flux2Klein``.
        checkpoint: Nunchaku checkpoint: a ``.safetensors`` file, a checkpoint or Diffusers repo folder, a Hub
            reference (``org/repo`` for a Diffusers repo such as ``lite-infer/flux.2-klein-4b-nunchaku-lite-int4-r32``,
            ``org/repo/subfolder`` or ``org/repo/path/file.safetensors``), or a mapping of tensors.
        target: Adapter target (see :func:`list_adapters`), ``"manifest"`` (generic runtime-manifest patching
            with Diffusers names used as model paths) or ``"auto"``.
        precision: ``"int4"`` (``"auto"`` is accepted); NVFP4 is not supported on MLX yet.
        backend: Quantized-kernel backend, ``"auto"``, ``"int8"``, ``"int4"``, or ``"dense"`` (same as
            ``nunchaku_lite``); resolved once for the whole transformer.
        strict: Raise when checkpoint tensors do not match the transformer.
        **pipeline_kwargs: Forwarded to ``pipeline_cls``.

    Returns:
        The pipeline, with ``pipeline.transformer`` patched in place.
    """

    if precision not in ("auto", "int4"):
        raise NotImplementedError("Only INT4 SVDQ checkpoints are supported on MLX (no NVFP4 yet).")
    resolve_backend(backend)  # validate before loading anything
    state, quantization_config = load_checkpoint(checkpoint, return_quantization_config=True)
    manifest = parse_runtime_manifest(quantization_config, checkpoint_keys=state)
    if manifest is not None and manifest.precision == "fp4":
        raise NotImplementedError("NVFP4 (fp4) checkpoints are not supported on MLX yet; use an INT4 checkpoint.")
    pipeline = pipeline_cls(model_path=str(pretrained_model_name_or_path), **pipeline_kwargs)
    adapter = _select_adapter(pipeline, target, manifest)
    adapter.patch(pipeline, state, manifest=manifest, backend=backend, strict=strict)
    pipeline._nunchaku_lite_target = adapter.target
    pipeline._nunchaku_lite_runtime_manifest = manifest
    return pipeline


def _select_adapter(pipeline: Any, target: str, manifest: RuntimeManifest | None = None) -> PipelineAdapter:
    if target == "manifest" and manifest is None:
        raise ValueError("target='manifest' requires a checkpoint with quantization_config.runtime_manifest metadata.")
    if target != "auto":
        if target not in _ADAPTERS:
            raise ValueError(f"Unknown adapter target {target!r}; available: {', '.join(list_adapters())}.")
        return _ADAPTERS[target]
    matches = [a for a in _ADAPTERS.values() if a.matches(pipeline)]
    if len(matches) == 1:
        return matches[0]
    if not matches and manifest is not None:
        return _ADAPTERS["manifest"]
    raise ValueError(
        f"Could not select an adapter for {type(pipeline).__name__}; pass target= (one of {list_adapters()})."
    )


def patch_from_manifest(
    transformer: nn.Module,
    state: dict[str, mx.array],
    manifest: RuntimeManifest,
    *,
    key_map=None,
    backend: str = "auto",
    strict: bool = True,
) -> set[str]:
    """Install the manifest's Nunchaku layers, load the remaining tensors as dense weights and resolve the backend.

    ``key_map`` maps checkpoint (Diffusers) names to ``transformer`` paths; default identity. Returns the dense
    parameter paths that were loaded.
    """

    key_map = key_map or (lambda name: name)
    consumed = apply_runtime_manifest(transformer, state, manifest, key_map=key_map)
    loaded = load_dense_weights(transformer, [(key_map(k), v) for k, v in state.items() if k not in consumed], strict)
    configure_backend(transformer, backend)
    return loaded


def _is_quantized_base(transformer: nn.Module) -> bool:
    return any(isinstance(m, nn.QuantizedLinear) for _, m in transformer.named_modules())


def _fresh_transformer(pipeline: Any) -> nn.Module:
    """Rebuild a pre-quantized base transformer (mflux ``-q4`` / ``-q8`` repo or ``quantize=``) as an unloaded
    skeleton, so the Nunchaku checkpoint provides every weight. The skeleton's initial weights are lazy and never
    materialized; :func:`_require_complete` checks that the checkpoint replaced all of them."""

    config = getattr(getattr(pipeline, "model_config", None), "transformer_overrides", None)
    if config is None:
        raise ValueError(
            "The base pipeline's transformer is quantized (QuantizedLinear layers) and has no "
            "model_config.transformer_overrides to rebuild it from; load the base pipeline without quantization."
        )
    return type(pipeline.transformer)(**config)


def _require_complete(transformer: nn.Module, dense_loaded: set[str]) -> None:
    """Raise if a rebuilt transformer still has parameters the checkpoint did not provide."""

    quantized = [
        path + "." for path, m in transformer.named_modules() if isinstance(m, (SVDQW4A4Linear, AWQW4A16Linear))
    ]
    missing = [
        k
        for k, _ in nn.utils.tree_flatten(transformer.parameters())
        if k not in dense_loaded and not any(k.startswith(p) for p in quantized)
    ]
    if missing:
        raise ValueError(
            f"The base transformer is quantized, so every weight must come from the Nunchaku checkpoint, but "
            f"{len(missing)} parameter(s) are missing from it, e.g. {missing[0]!r}. Use a non-quantized base pipeline."
        )


class ManifestAdapter:
    """Generic runtime-manifest adapter for transformers whose module names match the checkpoint's."""

    target = "manifest"

    def matches(self, pipeline: Any) -> bool:
        return False  # selected from checkpoint metadata, not the model type

    def patch(self, pipeline: Any, state, *, manifest: RuntimeManifest | None, backend: str, strict: bool) -> None:
        if manifest is None:
            raise ValueError("target='manifest' requires quantization_config.runtime_manifest metadata.")
        patch_from_manifest(pipeline.transformer, state, manifest, backend=backend, strict=strict)


# ---------------------------------------------------------------------------
# Composite projections for fused / split checkpoint layers
# ---------------------------------------------------------------------------


class SharedProjection(nn.Module):
    """A fused projection read by several :class:`ProjectionSlice` s: computed once per input, freed after the
    last slice reads it (e.g. a checkpoint ``to_qkv`` serving a model's separate ``to_q`` / ``to_k`` / ``to_v``)."""

    def __init__(self, layer: nn.Module, consumers: int):
        super().__init__()
        self.layer = layer
        self._consumers = consumers
        self._x = None
        self._out = None
        self._left = 0

    def output(self, x: mx.array) -> mx.array:
        if self._x is not x or self._left == 0:
            self._out, self._x, self._left = self.layer(x), x, self._consumers
        out = self._out
        self._left -= 1
        if self._left == 0:
            self._x = self._out = None
        return out


class ProjectionSlice(nn.Module):
    """Output columns ``[start, stop)`` of a :class:`SharedProjection`. Only the owning slice registers the shared
    projection as a child, so its parameters appear once in ``parameters()``."""

    def __init__(self, shared: SharedProjection, start: int, stop: int, owner: bool):
        super().__init__()
        if owner:
            self.shared = shared
        else:
            self._shared = shared
        self._start, self._stop = start, stop

    def __call__(self, x: mx.array) -> mx.array:
        shared = self.get("shared")  # owner slices hold the child; the others keep a non-child reference
        if shared is None:
            shared = self._shared
        return shared.output(x)[..., self._start : self._stop]


class ConcatProjection(nn.Module):
    """Concatenate the outputs of projections that read the same input (checkpoint split of a fused layer)."""

    def __init__(self, parts: list[nn.Module]):
        super().__init__()
        self.parts = parts

    def __call__(self, x: mx.array) -> mx.array:
        return mx.concatenate([part(x) for part in self.parts], axis=-1)


class SplitSumProjection(nn.Module):
    """Split the input along features and sum the parts' projections (checkpoint split of a layer whose input is
    a concatenation, e.g. FLUX.2 single-block ``to_out`` over ``[attention, mlp]``)."""

    def __init__(self, parts: list[nn.Module], split_indices: list[int]):
        super().__init__()
        self.parts = parts
        self._split_indices = list(split_indices)

    def __call__(self, x: mx.array) -> mx.array:
        chunks = mx.split(x, self._split_indices, axis=-1)
        out = self.parts[0](chunks[0])
        for part, chunk in zip(self.parts[1:], chunks[1:]):
            out = out + part(chunk)
        return out


# ---------------------------------------------------------------------------
# FLUX.2 (mflux Flux2Klein / Flux2KleinEdit)
# ---------------------------------------------------------------------------


# Diffusers -> mflux names used when mflux is not importable; with mflux the map comes from Flux2WeightMapping.
_FLUX2_RENAMES = (
    ("time_guidance_embed.timestep_embedder.linear_{i}", "time_guidance_embed.linear_{i}"),
    ("transformer_blocks.{block}.attn.to_out.0", "transformer_blocks.{block}.attn.to_out"),
)


def flux2_key_map() -> PrefixMap:
    """Diffusers -> mflux FLUX.2 transformer name map, derived from mflux's ``Flux2WeightMapping``."""

    try:
        from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping
    except ImportError:
        return PrefixMap(_FLUX2_RENAMES)
    return PrefixMap.from_mflux(Flux2WeightMapping.get_transformer_mapping())


class Flux2Adapter:
    """Nunchaku FLUX.2 Klein checkpoints on mflux's FLUX.2 transformer.

    Two checkpoint layouts are supported:

    * Nunchaku Lite checkpoints with a runtime manifest (``lite-infer`` repos): Diffusers module names, separate
      ``to_q`` / ``to_k`` / ``to_v``, fused single-block ``to_qkv_mlp_proj``, AWQ W4A16 modulation layers. These
      match mflux's modules one to one, so the manifest is applied through :func:`flux2_key_map`.
    * Original Nunchaku checkpoints (e.g. ``tonera/FLUX.2-klein-4B-Nunchaku``): keys are normalized with the
      same ``convert_flux2_state_dict`` as ``nunchaku_lite``, giving fused ``to_qkv`` / ``to_added_qkv`` and split
      single-block ``qkv_proj`` / ``mlp_fc1`` / ``out_proj`` / ``mlp_fc2`` layers, which are wrapped to serve
      mflux's separate projections.

    The base pipeline may be pre-quantized (e.g. ``mflux-community/flux2-klein-9b-mflux-q4``, used only for its
    text encoder and VAE): its transformer is then rebuilt from ``model_config`` and filled entirely from the
    Nunchaku checkpoint.
    """

    target = "flux2"

    def matches(self, pipeline: Any) -> bool:
        transformer = getattr(pipeline, "transformer", None)
        return type(transformer).__name__ == "Flux2Transformer"

    def patch(self, pipeline: Any, state, *, manifest: RuntimeManifest | None, backend: str, strict: bool) -> None:
        key_map = flux2_key_map()
        rebuilt = _is_quantized_base(pipeline.transformer)
        if rebuilt:
            pipeline.transformer = _fresh_transformer(pipeline)
        if manifest is not None:
            loaded = patch_from_manifest(
                pipeline.transformer, state, manifest, key_map=key_map, backend=backend, strict=strict
            )
        else:
            loaded = self._patch_nunchaku(
                pipeline.transformer, convert_flux2_state_dict(state), key_map, backend, strict
            )
        if rebuilt:
            _require_complete(pipeline.transformer, loaded)

    def _patch_nunchaku(self, transformer, state, key_map, backend: str, strict: bool) -> set[str]:
        used: set[str] = set()

        def svdq(prefix: str):
            used.update(k for k in state if k.startswith(prefix + "."))
            # FLUX.2 uses SwiGLU: no act_unsigned layers
            return svdq_layer_from_state(state, prefix, dtype=state[f"{prefix}.wscales"].dtype)

        for i, block in enumerate(transformer.transformer_blocks):
            p, attn = f"transformer_blocks.{i}", block.attn
            for fused, names in (
                (f"{p}.attn.to_qkv", ("to_q", "to_k", "to_v")),
                (f"{p}.attn.to_added_qkv", ("add_q_proj", "add_k_proj", "add_v_proj")),
            ):
                shared = SharedProjection(svdq(fused), consumers=3)
                inner = attn.inner_dim
                for j, name in enumerate(names):
                    setattr(attn, name, ProjectionSlice(shared, j * inner, (j + 1) * inner, owner=j == 0))
            attn.to_out = svdq(f"{p}.attn.to_out.0")
            attn.to_add_out = svdq(f"{p}.attn.to_add_out")
            for ff in ("ff", "ff_context"):
                getattr(block, ff).linear_in = svdq(f"{p}.{ff}.linear_in")
                getattr(block, ff).linear_out = svdq(f"{p}.{ff}.linear_out")

        for i, block in enumerate(transformer.single_transformer_blocks):
            p, attn = f"single_transformer_blocks.{i}", block.attn
            attn.to_qkv_mlp_proj = ConcatProjection([svdq(f"{p}.attn.qkv_proj"), svdq(f"{p}.attn.mlp_fc1")])
            attn.to_out = SplitSumProjection(
                [svdq(f"{p}.attn.out_proj"), svdq(f"{p}.attn.mlp_fc2")], split_indices=[attn.inner_dim]
            )

        loaded = load_dense_weights(transformer, [(key_map(k), v) for k, v in state.items() if k not in used], strict)
        configure_backend(transformer, backend)
        return loaded


# ---------------------------------------------------------------------------
# FLUX.1 (mflux Flux1)
# ---------------------------------------------------------------------------


# Diffusers -> mflux names used when mflux is not importable; with mflux the map comes from FluxWeightMapping.
_FLUX1_RENAMES = tuple(
    (f"transformer_blocks.{{block}}.{ff}.{src}", f"transformer_blocks.{{block}}.{ff}.{dst}")
    for ff in ("ff", "ff_context")
    for src, dst in (("net.0.proj", "linear1"), ("net.2", "linear2"))
)


def flux1_key_map() -> PrefixMap:
    """Diffusers -> mflux FLUX.1 transformer name map, derived from mflux's ``FluxWeightMapping``."""

    try:
        from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping
    except ImportError:
        return PrefixMap(_FLUX1_RENAMES)
    return PrefixMap.from_mflux(FluxWeightMapping.get_transformer_mapping())


class Flux1Adapter:
    """Nunchaku Lite FLUX.1 checkpoints on mflux's FLUX.1 transformer (``mflux.models.flux.variants.Flux1``).

    Supports Nunchaku Lite checkpoints (e.g. ``lite-infer/flux.1-dev-nunchaku-lite-int4-r32``): Diffusers module
    names with one quantized layer per Diffusers linear (separate ``to_q`` / ``to_k`` / ``to_v``, single-block
    ``proj_out`` over ``[attention, mlp]``), which match mflux's modules one to one. The targets come from the
    runtime manifest, or from the ``config.json`` target lists of checkpoints that predate it. Original Nunchaku
    FLUX.1 checkpoints (fused ``qkv_proj``, split single-block projections) are not supported.

    The base pipeline may be pre-quantized (e.g. ``mflux-community/flux-1-dev-mflux-q4``): its transformer is then
    rebuilt and filled entirely from the Nunchaku checkpoint.
    """

    target = "flux1"

    def matches(self, pipeline: Any) -> bool:
        transformer = getattr(pipeline, "transformer", None)
        cls = type(transformer)
        return cls.__name__ == "Transformer" and cls.__module__.startswith("mflux.models.flux.model.")

    def patch(self, pipeline: Any, state, *, manifest: RuntimeManifest | None, backend: str, strict: bool) -> None:
        if manifest is None:
            raise ValueError(
                "The FLUX.1 MLX adapter needs a Nunchaku Lite checkpoint (runtime manifest or config.json target "
                "lists, e.g. lite-infer/flux.1-dev-nunchaku-lite-int4-r32); original Nunchaku FLUX.1 checkpoints "
                "are not supported on MLX."
            )
        rebuilt = _is_quantized_base(pipeline.transformer)
        if rebuilt:
            old = pipeline.transformer
            pipeline.transformer = type(old)(
                model_config=pipeline.model_config,
                num_transformer_blocks=len(old.transformer_blocks),
                num_single_transformer_blocks=len(old.single_transformer_blocks),
            )
        loaded = patch_from_manifest(
            pipeline.transformer, state, manifest, key_map=flux1_key_map(), backend=backend, strict=strict
        )
        if rebuilt:
            _require_complete(pipeline.transformer, loaded)


register_adapter(ManifestAdapter())
register_adapter(Flux1Adapter())
register_adapter(Flux2Adapter())
