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

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

import mlx.core as mx
from mlx import nn

from ..checkpoint_keys import convert_flux2_state_dict
from .fused_attention import install_fused_attention
from .keymap import PrefixMap
from .linear import AWQW4A16Linear, SVDQW4A4Linear, resolve_backend
from .manifest import RuntimeManifest, apply_runtime_manifest, parse_runtime_manifest
from .patch import configure_backend, load_checkpoint, load_dense_weights, svdq_layer_from_state

__all__ = [
    "Flux1Adapter",
    "Flux2Adapter",
    "ManifestAdapter",
    "PipelineAdapter",
    "QwenImageAdapter",
    "ZImageAdapter",
    "list_adapters",
    "load_nunchaku_pipeline",
    "register_adapter",
]


class PipelineAdapter(Protocol):
    """Model-family adapter: installs Nunchaku layers into an MLX pipeline's transformer.

    ``manifest`` is the checkpoint's parsed runtime manifest, or ``None`` for checkpoints without one. ``patch``
    returns the transformer parameter paths it loaded as dense weights, which :func:`load_nunchaku_pipeline` uses
    to check that a transformer built without base weights was filled completely.
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
    ) -> set[str]: ...


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
    load_base_transformer: bool = False,
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
        load_base_transformer: Also download and load the base model's transformer weights. Off by default:
            Nunchaku checkpoints hold the whole transformer, so the pipeline is built with an empty (lazy, never
            materialized) transformer that the checkpoint fills; a checkpoint missing any weight raises. This
            skips e.g. the 23 GB Z-Image-Turbo transformer download.
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
    if load_base_transformer:
        pipeline = pipeline_cls(model_path=str(pretrained_model_name_or_path), **pipeline_kwargs)
    else:
        with _without_base_weights("transformer"):
            pipeline = pipeline_cls(model_path=str(pretrained_model_name_or_path), **pipeline_kwargs)
    adapter = _select_adapter(pipeline, target, manifest)
    loaded = adapter.patch(pipeline, state, manifest=manifest, backend=backend, strict=strict)
    if not load_base_transformer:
        if loaded is None:
            raise TypeError(
                f"Adapter {adapter.target!r} did not return the dense parameter paths it loaded, so a transformer "
                "built without base weights cannot be checked for completeness. Return them from patch() (see "
                "PipelineAdapter), or pass load_base_transformer=True."
            )
        _require_complete(pipeline.transformer, loaded, reason="the base transformer weights were not loaded")
    if mx.metal.is_available() and isinstance(pipeline.transformer, nn.Module):
        install_fused_attention(pipeline.transformer)  # q/k norm + RoPE in one kernel (Metal GPUs)
    pipeline._nunchaku_lite_target = adapter.target
    pipeline._nunchaku_lite_runtime_manifest = manifest
    return pipeline


_WEIGHT_LOADER_LOCK = threading.Lock()


@contextmanager
def _without_base_weights(component: str) -> Iterator[None]:
    """Build an mflux pipeline without downloading or loading one component's weights.

    mflux reads every component through ``WeightLoader``; this drops the component's weight-file download patterns
    (its configs are still fetched) and returns no weights for it, so the module keeps its lazy initial parameters
    (never evaluated, so never allocated). The patch is process-wide while active, so concurrent pipeline builds
    are serialized; it is a no-op when mflux is not installed.
    """

    try:
        from mflux.models.common.weights.loading.weight_loader import WeightLoader
    except ImportError:  # not an mflux pipeline (or mflux missing): nothing to skip
        yield
        return

    def is_weight_pattern(pattern: str) -> bool:
        return pattern.startswith(f"{component}/") and pattern.endswith((".safetensors", ".bin", ".pt", ".pth"))

    with _WEIGHT_LOADER_LOCK:
        saved = {name: WeightLoader.__dict__[name] for name in ("load", "_load_component")}
        original_load, original_component = WeightLoader.load, WeightLoader._load_component

        def load(weight_definition, *args, **kwargs):
            positional = len(args) >= 2  # (model_path, download_patterns, ...)
            patterns = args[1] if positional else kwargs.get("download_patterns")
            if patterns is None:
                patterns = weight_definition.get_download_patterns()
            patterns = [p for p in patterns if not is_weight_pattern(p)]
            if positional:
                args = (args[0], patterns, *args[2:])
            else:
                kwargs["download_patterns"] = patterns
            return original_load(weight_definition, *args, **kwargs)

        def load_component(root_path, definition, *args, **kwargs):
            if definition.name == component:
                return {}, None, None
            return original_component(root_path, definition, *args, **kwargs)

        WeightLoader.load = staticmethod(load)
        WeightLoader._load_component = staticmethod(load_component)
        try:
            yield
        finally:
            for name, value in saved.items():
                setattr(WeightLoader, name, value)


def _require_dense_base(pipeline: Any) -> None:
    """Raise early for a quantized base transformer (``quantize=`` or a pre-quantized repo) in adapters that do not
    rebuild one: dense checkpoint weights cannot be loaded into ``QuantizedLinear`` layers."""

    if _is_quantized_base(pipeline.transformer):
        raise ValueError(
            f"The base pipeline's transformer ({type(pipeline.transformer).__name__}) is quantized (quantize= or a "
            "pre-quantized repo); the Nunchaku checkpoint replaces it entirely, so build the base pipeline without "
            "quantization."
        )


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


def _require_complete(
    transformer: nn.Module, dense_loaded: set[str], reason: str = "the base transformer is quantized"
) -> None:
    """Raise if a rebuilt (or never loaded) transformer still has parameters the checkpoint did not provide."""

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
            f"Every transformer weight must come from the Nunchaku checkpoint ({reason}), but {len(missing)} "
            f"parameter(s) are missing from it, e.g. {missing[0]!r}. Pass load_base_transformer=True with a "
            "non-quantized base model."
        )


class ManifestAdapter:
    """Generic runtime-manifest adapter for transformers whose module names match the checkpoint's."""

    target = "manifest"

    def matches(self, pipeline: Any) -> bool:
        return False  # selected from checkpoint metadata, not the model type

    def patch(self, pipeline: Any, state, *, manifest: RuntimeManifest | None, backend: str, strict: bool) -> set[str]:
        if manifest is None:
            raise ValueError("target='manifest' requires quantization_config.runtime_manifest metadata.")
        return patch_from_manifest(pipeline.transformer, state, manifest, backend=backend, strict=strict)


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

    def patch(self, pipeline: Any, state, *, manifest: RuntimeManifest | None, backend: str, strict: bool) -> set[str]:
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
        return loaded

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


def _block_count(state: Mapping[str, Any], group: str) -> int:
    """Number of ``<group>.<i>.`` blocks in a checkpoint (highest index + 1)."""

    prefix = group + "."
    ids = {int(k[len(prefix) :].split(".", 1)[0]) for k in state if k.startswith(prefix)}
    return max(ids) + 1 if ids else 0


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

    def patch(self, pipeline: Any, state, *, manifest: RuntimeManifest | None, backend: str, strict: bool) -> set[str]:
        if manifest is None:
            raise ValueError(
                "The FLUX.1 MLX adapter needs a Nunchaku Lite checkpoint (runtime manifest or config.json target "
                "lists, e.g. lite-infer/flux.1-dev-nunchaku-lite-int4-r32); original Nunchaku FLUX.1 checkpoints "
                "are not supported on MLX."
            )
        # mflux sizes the FLUX.1 transformer from the weights it loads, so without base weights it has no blocks;
        # rebuild it from the checkpoint's block counts (also replaces a quantized base).
        old = pipeline.transformer
        blocks = (_block_count(state, "transformer_blocks"), _block_count(state, "single_transformer_blocks"))
        rebuilt = _is_quantized_base(old) or (len(old.transformer_blocks), len(old.single_transformer_blocks)) != blocks
        if rebuilt:
            pipeline.transformer = type(old)(
                model_config=pipeline.model_config,
                num_transformer_blocks=blocks[0],
                num_single_transformer_blocks=blocks[1],
            )
        loaded = patch_from_manifest(
            pipeline.transformer, state, manifest, key_map=flux1_key_map(), backend=backend, strict=strict
        )
        if rebuilt:
            _require_complete(pipeline.transformer, loaded)
        return loaded


# ---------------------------------------------------------------------------
# Z-Image (mflux ZImage)
# ---------------------------------------------------------------------------


# Diffusers -> mflux names used when mflux is not importable; with mflux the map comes from ZImageWeightMapping.
_Z_IMAGE_RENAMES = (
    ("t_embedder.mlp.0", "t_embedder.linear1"),
    ("t_embedder.mlp.2", "t_embedder.linear2"),
    ("all_final_layer.2-1.adaLN_modulation.1", "all_final_layer.2-1.adaLN_modulation.0"),
)
_Z_IMAGE_BLOCK_GROUPS = ("noise_refiner", "context_refiner", "layers")


def z_image_key_map() -> PrefixMap:
    """Diffusers -> mflux Z-Image transformer name map, derived from mflux's ``ZImageWeightMapping``."""

    try:
        from mflux.models.z_image.weights.z_image_weight_mapping import ZImageWeightMapping
    except ImportError:
        return PrefixMap(_Z_IMAGE_RENAMES)
    return PrefixMap.from_mflux(ZImageWeightMapping.get_transformer_mapping())


def _split_rows(state: dict[str, mx.array], key: str, names: list[str], used: set[str]) -> None:
    """Replace a fused dense ``key`` (``<prefix>.weight``) by equal row blocks named ``names`` (in order)."""

    fused = state[key]
    used.add(key)
    for name, part in zip(names, mx.split(fused, len(names), axis=0)):
        state[name] = part


class ZImageAdapter:
    """Nunchaku Z-Image checkpoints on mflux's Z-Image transformer (``mflux.models.z_image.ZImage``).

    Original Nunchaku checkpoints (e.g. ``nunchaku-tech/nunchaku-z-image-turbo``) store Diffusers names with a
    fused ``attention.to_qkv`` and the SwiGLU feed-forward as ``feed_forward.net.0.proj`` (``[w3; w1]``: Diffusers'
    SwiGLU multiplies the first half by ``silu`` of the second) and ``feed_forward.net.2`` (``w2``). They are
    wrapped to serve mflux's separate ``to_q`` / ``to_k`` / ``to_v`` and ``w1`` / ``w3``. Refiners stored dense
    (``skip_refiners``) are split the same way. Checkpoints with a runtime manifest use Diffusers' module names,
    which match mflux's one to one.
    """

    target = "z_image"

    def matches(self, pipeline: Any) -> bool:
        return type(getattr(pipeline, "transformer", None)).__name__ == "ZImageTransformer"

    def patch(self, pipeline: Any, state, *, manifest: RuntimeManifest | None, backend: str, strict: bool) -> set[str]:
        key_map = z_image_key_map()
        _require_dense_base(pipeline)
        if manifest is not None:
            return patch_from_manifest(
                pipeline.transformer, state, manifest, key_map=key_map, backend=backend, strict=strict
            )
        transformer, state, used = pipeline.transformer, dict(state), set()

        def svdq(prefix: str):
            used.update(k for k in state if k.startswith(prefix + "."))
            return svdq_layer_from_state(state, prefix, dtype=state[f"{prefix}.wscales"].dtype)

        for group in _Z_IMAGE_BLOCK_GROUPS:
            for i, block in enumerate(getattr(transformer, group)):
                p = f"{group}.{i}"
                attn, ff = block.attention, block.feed_forward
                if f"{p}.attention.to_qkv.qweight" in state:
                    shared = SharedProjection(svdq(f"{p}.attention.to_qkv"), consumers=3)
                    for j, name in enumerate(("to_q", "to_k", "to_v")):
                        setattr(attn, name, ProjectionSlice(shared, j * attn.dim, (j + 1) * attn.dim, owner=j == 0))
                    attn.to_out = [svdq(f"{p}.attention.to_out.0")]
                elif f"{p}.attention.to_qkv.weight" in state:
                    names = [f"{p}.attention.{n}.weight" for n in ("to_q", "to_k", "to_v")]
                    _split_rows(state, f"{p}.attention.to_qkv.weight", names, used)
                if f"{p}.feed_forward.net.0.proj.qweight" in state:
                    gate_up = svdq(f"{p}.feed_forward.net.0.proj")
                    hidden = gate_up.out_features // 2
                    shared = SharedProjection(gate_up, consumers=2)
                    ff.w3 = ProjectionSlice(shared, 0, hidden, owner=True)
                    ff.w1 = ProjectionSlice(shared, hidden, 2 * hidden, owner=False)
                    ff.w2 = svdq(f"{p}.feed_forward.net.2")
                elif f"{p}.feed_forward.net.0.proj.weight" in state:
                    names = [f"{p}.feed_forward.w3.weight", f"{p}.feed_forward.w1.weight"]
                    _split_rows(state, f"{p}.feed_forward.net.0.proj.weight", names, used)
                    state[f"{p}.feed_forward.w2.weight"] = state[f"{p}.feed_forward.net.2.weight"]
                    used.add(f"{p}.feed_forward.net.2.weight")

        loaded = load_dense_weights(transformer, [(key_map(k), v) for k, v in state.items() if k not in used], strict)
        configure_backend(transformer, backend)
        return loaded


# ---------------------------------------------------------------------------
# Qwen-Image (mflux QwenImage)
# ---------------------------------------------------------------------------


def qwen_image_key_map() -> PrefixMap:
    """Diffusers -> mflux Qwen-Image transformer name map, derived from mflux's ``QwenWeightMapping``."""

    from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping

    return PrefixMap.from_mflux(QwenWeightMapping.get_transformer_mapping())


def _awq_layer_from_state(state: Mapping[str, mx.array], prefix: str, **kwargs) -> AWQW4A16Linear:
    """An :class:`AWQW4A16Linear` from one checkpoint prefix (shapes read from ``qweight``)."""

    qweight = state[f"{prefix}.qweight"]
    layer = AWQW4A16Linear(
        qweight.shape[1] * 2,
        qweight.shape[0] * 4,
        bias=f"{prefix}.bias" in state,
        dtype=state[f"{prefix}.wscales"].dtype,
        **kwargs,
    )
    layer.load_weights(
        [(n, state[f"{prefix}.{n}"]) for n in ("qweight", "wscales", "wzeros", "bias") if f"{prefix}.{n}" in state]
    )
    return layer


class QwenImageAdapter:
    """Original Nunchaku Qwen-Image checkpoints on mflux's Qwen-Image transformer (``mflux.models.qwen``).

    Nunchaku stores Diffusers names with fused ``attn.to_qkv`` / ``attn.add_qkv_proj`` (served to mflux's separate
    projections from one matmul), GELU MLPs whose second layer takes the shifted, unsigned GELU output, and AWQ
    W4A16 modulation layers (``img_mod.1`` / ``txt_mod.1``) in Nunchaku's AdaNorm layout: six interleaved output
    chunks with ``+1`` folded into the scale chunks' bias, as for FLUX.1 (the PyTorch adapter runs them with
    ``scale_shift=0``; mflux adds the ``1`` itself, so the layer undoes the fold). Checkpoints with a runtime
    manifest are patched from it with the same name map.
    """

    target = "qwen_image"

    def matches(self, pipeline: Any) -> bool:
        return type(getattr(pipeline, "transformer", None)).__name__ == "QwenTransformer"

    def patch(self, pipeline: Any, state, *, manifest: RuntimeManifest | None, backend: str, strict: bool) -> set[str]:
        key_map = qwen_image_key_map()
        _require_dense_base(pipeline)
        if manifest is not None:
            return patch_from_manifest(
                pipeline.transformer, state, manifest, key_map=key_map, backend=backend, strict=strict
            )
        transformer, used = pipeline.transformer, set()

        def take(prefix: str) -> str:
            used.update(k for k in state if k.startswith(prefix + "."))
            return prefix

        def svdq(prefix: str, act_unsigned: bool = False):
            dtype = state[f"{prefix}.wscales"].dtype
            return svdq_layer_from_state(state, take(prefix), act_unsigned=act_unsigned, dtype=dtype)

        for i, block in enumerate(transformer.transformer_blocks):
            p, attn = f"transformer_blocks.{i}", block.attn
            for fused, names in (
                (f"{p}.attn.to_qkv", ("to_q", "to_k", "to_v")),
                (f"{p}.attn.add_qkv_proj", ("add_q_proj", "add_k_proj", "add_v_proj")),
            ):
                shared = SharedProjection(svdq(fused), consumers=3)
                for j, name in enumerate(names):
                    setattr(attn, name, ProjectionSlice(shared, j * attn.dim, (j + 1) * attn.dim, owner=j == 0))
            attn.attn_to_out = [svdq(f"{p}.attn.to_out.0")]
            attn.to_add_out = svdq(f"{p}.attn.to_add_out")
            for stream in ("img", "txt"):
                ff = getattr(block, f"{stream}_ff")
                ff.mlp_in = svdq(f"{p}.{stream}_mlp.net.0.proj")
                ff.mlp_out = svdq(f"{p}.{stream}_mlp.net.2", act_unsigned=True)  # shifted GELU output
                setattr(
                    block,
                    f"{stream}_mod_linear",
                    _awq_layer_from_state(state, take(f"{p}.{stream}_mod.1"), adanorm_splits=6),
                )

        loaded = load_dense_weights(transformer, [(key_map(k), v) for k, v in state.items() if k not in used], strict)
        configure_backend(transformer, backend)
        return loaded


register_adapter(ManifestAdapter())
register_adapter(Flux1Adapter())
register_adapter(Flux2Adapter())
register_adapter(QwenImageAdapter())
register_adapter(ZImageAdapter())
