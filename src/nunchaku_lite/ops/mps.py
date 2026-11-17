"""Apple-silicon (MPS) execution paths for Nunchaku Lite quantized modules.

Nunchaku SVDQ checkpoints store CUDA-MMA-packed tensors. On MPS we keep those
parameters untouched (so state dicts and runtime LoRA keep working) and derive a
per-module cache in the layout of the ``nunchaku_lite_kernels`` Metal ops:

* ``svdq_quantize_act_int8``: per-token int8 activations after SVDQ smoothing.
* ``svdq_gemm_w8a8`` / ``svdq_gemm_w4a8``: int8 x int8 Neural Accelerator matmul
  with per-channel scales, bias and the SVD low-rank branch fused in the epilogue.

Per-64-channel weight scales are folded into a per-channel scale plus integer
multipliers (``w8 = rint(q * wr)``, ``|w8| <= 127``) so the int32 accumulator
runs over the whole K dimension.

GPUs without Neural Accelerators (pre-M5) use a dense fallback that dequantizes
the INT4 weights to the activation dtype once and runs ``F.linear``.

To avoid holding the weights twice (Apple GPUs share system memory), the packed
``qweight`` is released once the kernel copy is built, whenever the 4-bit values
can be recovered exactly from it (always for ``int4``; per layer for ``int8``,
where ``w8 = rint(q * wr)`` is invertible when every ``wr >= 1``). ``state_dict``
re-packs it on demand and ``load_state_dict`` restores it.

Backend selection (``backend=`` on ``load_nunchaku_pipeline`` / ``patch_transformer``,
stored per layer as ``SVDQW4A4Linear.backend``):

``auto`` (default)
    ``int8`` when the transformer fits comfortably in GPU memory with int8
    weights, else ``int4``; ``dense`` on GPUs without Neural Accelerators.
``int8``
    W8A8 kernel, weights expanded to int8. Fastest.
``int4``
    W4A8 kernel, weights stay 4-bit (about half the weight memory of ``int8``,
    a few percent slower, identical output).
``dense``
    Dequantized fallback (exact W4A16 math); any Apple GPU.

Two attention kernels come from the same ``nunchaku_lite_kernels`` build. When the installed build predates them,
they are compiled at runtime with ``torch.mps.compile_shader`` from the same sources:

* q/k RMSNorm + RoPE (:func:`qk_norm_rope_mps` for Diffusers' FLUX.2 processors, and in place on the dense qkv in
  :func:`qkv_norm_rotary_mps` for the Nunchaku processors); op ``qk_norm_rope_mps_``. Its body is shared with the
  MLX runtime (:data:`nunchaku_lite.shaders.QK_NORM_ROPE`).
* :func:`attention_mps`: MLX's Neural Accelerator flash attention (vendored in :mod:`._mlx_steel_attention`),
  about 3.7x faster than PyTorch's MPS attention on M5-class GPUs, with lower error; op ``attention_mps``.
"""

from __future__ import annotations

import functools
import re
import struct
import subprocess
import warnings
import weakref

import torch
import torch.nn.functional as F

from ..shaders import ATTENTION_HEAD_DIMS, ATTENTION_TILE

GELU_SHIFT = 0.171875
"""Shift Nunchaku INT4 applies to post-GELU activations of ``act_unsigned`` projections."""

_TILE_M = 128
_W_PERM_INV = (0, 3, 6, 4, 8, 1, 2, 7, 5, 9)  # argsort of the packer permutation (0, 5, 6, 1, 3, 8, 2, 7, 4, 9)
_S_PERM_INV = (0, 2, 3, 5, 4, 6, 1)  # argsort of the scale packer permutation (0, 6, 1, 2, 4, 3, 5)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


@functools.cache
def has_neural_accelerators() -> bool:
    """Return whether this Mac's GPU has M5-class Neural Accelerators (int8 matmul)."""

    try:
        brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return False
    parts = brand.split()
    if len(parts) < 2 or parts[0] != "Apple" or not parts[1].startswith("M"):
        return False
    try:
        return int(parts[1][1:]) >= 5
    except ValueError:
        return False


BACKENDS = ("auto", "int8", "int4", "dense")
_AUTO_MEMORY_FRACTION = 0.5
"""``auto`` picks ``int8`` when the estimated int8 transformer fits in this fraction of the GPU working set."""


def validate_backend(choice: str) -> str:
    choice = str(choice).lower()
    if choice not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}; got {choice!r}.")
    return choice


def resolve_backend(choice: str = "auto", int8_resident_bytes: int | None = None) -> str:
    """Resolve ``"auto"`` to a concrete backend.

    Args:
        choice: ``"auto"``, ``"int8"``, ``"int4"``, or ``"dense"``.
        int8_resident_bytes: Estimated model footprint with int8 weights. When given
            on an MPS machine, ``"auto"`` falls back to ``"int4"`` if it exceeds half
            of ``torch.mps.recommended_max_memory()``.
    """

    choice = validate_backend(choice)
    if choice != "auto":
        return choice
    if not has_neural_accelerators():
        return "dense"
    if not mps_kernels_available():
        return "dense"
    if int8_resident_bytes is not None and torch.backends.mps.is_available():
        budget = _AUTO_MEMORY_FRACTION * torch.mps.recommended_max_memory()
        if int8_resident_bytes > budget:
            return "int4"
    return "int8"


def estimate_int8_resident_bytes(model: torch.nn.Module) -> int:
    """Approximate a patched model's footprint on MPS with the ``int8`` backend."""

    from ..linear import SVDQW4A4Linear

    total = 0
    for module in model.modules():
        if isinstance(module, SVDQW4A4Linear):
            total += module.in_features * module.out_features  # int8 weight copy (packed qweight is released)
            params = [p for name, p in module.named_parameters(recurse=False) if name != "qweight"]
        else:
            params = list(module.parameters(recurse=False))
        total += sum(p.numel() * p.element_size() for p in params)
    return total


def configure_backend(model: torch.nn.Module, choice: str = "auto") -> str:
    """Resolve the backend once for a whole model and store it on every SVDQ layer."""

    from ..linear import SVDQW4A4Linear

    choice = validate_backend(choice)
    layers = [m for m in model.modules() if isinstance(m, SVDQW4A4Linear)]
    if choice == "auto" and torch.backends.mps.is_available():
        resolved = resolve_backend("auto", estimate_int8_resident_bytes(model))
    else:
        resolved = choice  # "auto" left for per-layer resolution off MPS
    for layer in layers:
        layer.backend = resolved
    model._nunchaku_lite_backend = resolved
    return resolved


@functools.cache
def mps_kernels_available() -> bool:
    """Whether the ``nunchaku_lite_kernels`` MPS ops load; ``"auto"`` falls back to ``"dense"`` (with a warning)
    when they don't, e.g. a PyTorch version without a prebuilt Hub build and no local kernels."""

    try:
        _kernel_ops()
    except Exception as exc:  # noqa: BLE001 - any load failure means the kernels are unusable
        warnings.warn(
            f'backend="auto" uses the dense MPS fallback: the Metal kernels are unavailable ({exc}). The Hub build '
            'covers PyTorch 2.10-2.15; otherwise build them with "pip install --no-build-isolation '
            './nunchaku-lite-kernels" for the faster int8/int4 backends.',
            stacklevel=3,
        )
        return False
    return True


def _kernel_ops():
    from .backend import get_ops

    ops = get_ops()
    if not hasattr(ops, "svdq_gemm_w8a8"):
        raise RuntimeError(
            "The loaded nunchaku_lite_kernels ops have no MPS ops (a CUDA-only local build?). Rebuild it on "
            'macOS with "pip install --no-build-isolation ./nunchaku-lite-kernels", or pass backend="dense".'
        )
    return ops


# ---------------------------------------------------------------------------
# Nunchaku packed layout -> logical tensors (device-side)
# ---------------------------------------------------------------------------


def unpack_svdq_weight(qweight: torch.Tensor) -> torch.Tensor:
    """Unpack Nunchaku INT4 ``qweight`` ``(N, K // 2)`` int8 into signed values ``(N, K)`` int8."""

    n, half_k = qweight.shape
    k = half_k * 2
    if n % 128 or k % 128:
        raise ValueError(f"qweight shape {tuple(qweight.shape)} is not a valid Nunchaku INT4 layout.")
    # Each little-endian int32 word holds 8 nibbles at shifts 0, 4, ..., 28: byte i holds nibbles 2i (low) and
    # 2i + 1 (high). Unpacking bytes keeps every temporary int8 (an int32 per weight would be 4x the memory).
    b = qweight.contiguous().view(torch.uint8).reshape(n // 128, k // 64, 1, 8, 8, 4, 2, 2, 1, 4)
    nibbles = torch.stack([b & 0xF, b >> 4], dim=-1).view(torch.int8).flatten(-2)
    signed = (nibbles ^ 8) - 8
    return signed.permute(_W_PERM_INV).contiguous().reshape(n, k)


_W_PERM = (0, 5, 6, 1, 3, 8, 2, 7, 4, 9)


def pack_svdq_weight(q: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 values ``(N, K)`` into Nunchaku ``qweight`` ``(N, K // 2)`` int8 (inverse of unpack)."""

    n, k = q.shape
    w = q.to(torch.int32).reshape(n // 128, 8, 2, 8, 1, k // 64, 1, 2, 4, 8).permute(_W_PERM) & 0xF
    shifts = torch.arange(0, 32, 4, device=q.device, dtype=torch.int32)
    words = (w << shifts).sum(dim=-1, dtype=torch.int32)  # int32 wraparound gives the packed bit pattern
    return words.contiguous().view(torch.int8).reshape(n, k // 2)


def unpack_svdq_scale(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Unpack Nunchaku-ordered scales (``wscales``, ``bias``, ``smooth_factor``) into ``(N, G)``."""

    if n % 128:
        raise ValueError(f"scale length {n} must be divisible by 128.")
    groups = packed.numel() // n
    s = packed.reshape(n // 128, groups, 1, 8, 4, 2, 2).permute(_S_PERM_INV)
    return s.contiguous().reshape(n, groups)


def _unpack_lowrank(tensor: torch.Tensor, features: int, down: bool) -> torch.Tensor:
    from ..lora.core.layout import unpack_lowrank_weight

    if tensor.shape[1] == 0:
        return tensor.new_zeros((0, features) if down else (features, 0))
    logical = unpack_lowrank_weight(tensor, down=down)
    return logical[:, :features] if down else logical[:features]


# ---------------------------------------------------------------------------
# SVDQ linear
# ---------------------------------------------------------------------------


def _param_state(t: torch.Tensor | None) -> tuple | None:
    if t is None:
        return None
    return (id(t), t._version, tuple(t.shape), t.dtype, t.device, t.data_ptr())


def _states_match(saved: tuple | None, current: tuple | None) -> bool:
    if saved is None or current is None:
        return saved is current
    if len(saved) != 6 or len(current) != 6:  # sentinels such as ("released",)
        return saved == current
    # Same parameter object/version/shape/dtype. A changed data pointer on the *same* device means
    # the storage was swapped (``param.data = ...``); a device move (``module.to``) is not a change.
    return saved[:4] == current[:4] and (saved[4] != current[4] or saved[5] == current[5])


_SVDQ_KEY_TENSORS = ("qweight", "wscales", "smooth_factor", "proj_down", "proj_up", "bias")
_AWQ_KEY_TENSORS = ("qweight", "wscales", "wzeros")


def _tensor_states(module, names: tuple[str, ...]) -> tuple:
    released = getattr(module, "_nunchaku_lite_qweight_released", False)
    return tuple(
        ("released",) if name == "qweight" and released else _param_state(getattr(module, name)) for name in names
    )


def preserve_cache_across_move(module, apply):
    """Run ``apply`` (``nn.Module._apply``) and keep a valid MPS cache valid across the move.

    A move changes every data pointer; without this, an ``mps -> cpu -> mps`` round trip (CPU offload)
    would look like swapped weights and rebuild the cache.

    The cache keeps its own dtypes (float32 scales, float16 group multipliers) through a dtype cast such
    as ``module.to(torch.bfloat16)``: cast buffers are replaced by their originals on the new device, since
    a released ``qweight`` is recovered from them. A cast that changes parameter dtypes invalidates the
    cache, which is then rebuilt from the cast parameters on the next call.
    """

    key = getattr(module, "_nunchaku_lite_mps_key", None)
    names = _AWQ_KEY_TENSORS if hasattr(module, "wzeros") else _SVDQ_KEY_TENSORS
    before = _tensor_states(module, names)
    valid = key is not None and _cache_valid(module, (*key[:2], *before))
    originals = {name: _cached(module, name) for name in getattr(module, "_nunchaku_lite_mps_buffers", ())}
    result = apply()
    for name, original in originals.items():
        moved = _cached(module, name)
        if moved is not None and original is not None and moved.dtype != original.dtype:
            module.register_buffer(f"_nl_mps_{name}", original.to(moved.device), persistent=False)
    after = _tensor_states(module, names)
    if valid:
        dtypes_kept = all(_state_dtype(a) == _state_dtype(b) for a, b in zip(before, after))
        module._nunchaku_lite_mps_key = (*key[:2], *after) if dtypes_kept else None
    return result


def _state_dtype(state: tuple | None):
    return state[3] if state is not None and len(state) == 6 else state


def _cache_valid(module, key: tuple) -> bool:
    saved = getattr(module, "_nunchaku_lite_mps_key", None)
    if saved is None or saved[:2] != key[:2] or len(saved) != len(key):
        return False
    return all(_states_match(a, b) for a, b in zip(saved[2:], key[2:]))


def _compact(tensors: dict) -> dict:
    """Re-allocate long-lived cache tensors from an empty MPS buffer pool.

    The MPS allocator serves a new tensor from any cached free buffer that is large enough and keeps the
    whole buffer allocated. A cache built from float32 temporaries would otherwise pin buffers several
    times its size (measured: ~10 GB wasted on FLUX.1-dev).
    """

    if not any(t.device.type == "mps" for t in tensors.values()):
        return tensors
    torch.mps.empty_cache()
    return {name: t.clone() for name, t in tensors.items()}


def _store_cache(module, key: tuple, tensors: dict, meta: dict) -> None:
    """Keep cached tensors as non-persistent buffers so they follow ``module.to`` (e.g. CPU offload)."""

    tensors = _compact(tensors)
    for name in getattr(module, "_nunchaku_lite_mps_buffers", ()):
        if name not in tensors:
            delattr(module, f"_nl_mps_{name}")
    for name, tensor in tensors.items():
        module.register_buffer(f"_nl_mps_{name}", tensor, persistent=False)
    module._nunchaku_lite_mps_buffers = tuple(tensors)
    module._nunchaku_lite_mps_meta = meta
    module._nunchaku_lite_mps_key = key


def _cached(module, name: str) -> torch.Tensor | None:
    return getattr(module, f"_nl_mps_{name}", None)


@torch.no_grad()
def _build_svdq_cache(module, backend: str, dtype: torch.dtype) -> tuple[dict, dict]:
    if module.precision != "int4":
        raise NotImplementedError("NVFP4 SVDQ checkpoints are not supported on MPS yet; use an INT4 checkpoint.")
    n, k = module.out_features, module.in_features
    q = _source_q(module)
    wscale = unpack_svdq_scale(module.wscales.float(), n)  # (N, K / 64)
    smooth = unpack_svdq_scale(module.smooth_factor.float(), k).reshape(k)
    bias = None if module.bias is None else unpack_svdq_scale(module.bias.float(), n).reshape(n)
    down = _unpack_lowrank(module.proj_down, k, down=True).float()  # (R, K)
    up = _unpack_lowrank(module.proj_up, n, down=False).float()  # (N, R)

    tensors: dict = {"inv_smooth": (1.0 / smooth).contiguous()}
    if bias is not None:
        tensors["bias"] = bias.contiguous()
    if backend == "dense":
        weight = torch.empty(n, k, dtype=dtype, device=q.device)
        for rows in _row_chunks(n, k):
            w = (q[rows].float().view(-1, k // 64, 64) * wscale[rows, :, None]).view(-1, k)
            weight[rows] = (w / smooth).to(dtype)  # fold 1/smooth into the dense weight
        tensors["weight"] = weight
        if down.shape[0]:  # keep the SVD branch low-rank: two thin matmuls, not a dense (N, K) matrix
            tensors["down"] = down.to(dtype).contiguous()
            tensors["up"] = up.to(dtype).contiguous()
        return tensors, {"backend": backend, "q_invertible": False}

    lr_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float16
    rank = down.shape[0]
    rank_pad = -rank % 32 if rank else 32
    down = F.pad(down, (0, 0, 0, rank_pad))
    up = F.pad(up, (0, rank_pad))
    tensors["down_t"] = down.t().contiguous().to(lr_dtype)  # (K, R)
    tensors["lup"] = up.contiguous().to(lr_dtype)  # (N, R)

    wsc = wscale.amax(dim=1) * (8.0 / 127.0)
    wsc = torch.where(wsc > 0, wsc, torch.ones_like(wsc))
    wr = (wscale / wsc[:, None]).half()
    tensors["wsc"] = wsc.float().contiguous()
    tensors["wr"] = wr.t().contiguous()  # (K / 64, N); also lets int8 recover q exactly
    if backend == "int8":
        w8 = torch.empty(n, k, dtype=torch.int8, device=q.device)
        for rows in _row_chunks(n, k):
            chunk = torch.round(q[rows].float().view(-1, k // 64, 64) * wr[rows].float()[..., None]).clamp_(-127, 127)
            w8[rows] = chunk.to(torch.int8).view(-1, k)
        tensors["w8"] = w8
        # q = round(w8 / wr) recovers q exactly when every multiplier wr >= 1 (the rounding error divided by wr
        # stays below 0.5); only layers with smaller multipliers need the full check.
        invertible = bool((wr >= 1).all()) or all(
            torch.equal(_q_from_w8(w8[rows], tensors["wr"][:, rows]), q[rows]) for rows in _row_chunks(n, k)
        )
    else:
        nib = (q & 0xF).to(torch.uint8).view(n, k // 2, 2)
        tensors["w4"] = (nib[..., 0] | (nib[..., 1] << 4)).contiguous()
        invertible = True
    return tensors, {"backend": backend, "lr_dtype": lr_dtype, "q_invertible": invertible}


def _row_chunks(n: int, k: int, budget: int = 1 << 24):
    """Row slices covering ``n`` rows of width ``k``, each at most ``budget`` elements (64 MB of float32), so
    weight conversion never holds a full-size float32 copy of a layer."""

    step = max(128, budget // k // 128 * 128)
    return [slice(start, min(start + step, n)) for start in range(0, n, step)]


def _q_from_w8(w8: torch.Tensor, wr_t: torch.Tensor) -> torch.Tensor:
    n, k = w8.shape
    q = torch.round(w8.float().view(n, k // 64, 64) / wr_t.t().float()[..., None])
    return q.clamp_(-8, 7).to(torch.int8).view(n, k)


def _q_from_w4(w4: torch.Tensor) -> torch.Tensor:
    lo = ((w4 & 0xF).to(torch.int8) ^ 8) - 8
    hi = ((w4 >> 4).to(torch.int8) ^ 8) - 8
    return torch.stack([lo, hi], dim=-1).reshape(w4.shape[0], -1)


def _source_q(module) -> torch.Tensor:
    """Signed int4 weights ``(N, K)``: from ``qweight``, or recovered from the kernel copy once released."""

    if not getattr(module, "_nunchaku_lite_qweight_released", False):
        return unpack_svdq_weight(module.qweight)
    w4 = _cached(module, "w4")
    if w4 is not None:
        return _q_from_w4(w4)
    return _q_from_w8(_cached(module, "w8"), _cached(module, "wr"))


def _state_dict_hook(module, state_dict, prefix, local_metadata):
    if getattr(module, "_nunchaku_lite_qweight_released", False):
        state_dict[prefix + "qweight"] = pack_svdq_weight(_source_q(module))


def _load_state_dict_pre_hook(module, state_dict, prefix, *args):
    if getattr(module, "_nunchaku_lite_qweight_released", False) and prefix + "qweight" in state_dict:
        dtype = getattr(module, "_nunchaku_lite_qweight_dtype", torch.int8)
        module.qweight = torch.nn.Parameter(
            torch.empty(module._nunchaku_lite_qweight_shape, dtype=dtype, device=module.wscales.device),
            requires_grad=False,
        )
        module._nunchaku_lite_qweight_released = False


def _release_qweight(module) -> None:
    """Free the packed ``qweight``; the kernel copy can reproduce it exactly."""

    if not getattr(module, "_nunchaku_lite_mps_hooks", False):
        module._register_state_dict_hook(_state_dict_hook)
        module._register_load_state_dict_pre_hook(_load_state_dict_pre_hook, with_module=True)
        module._nunchaku_lite_mps_hooks = True
    module._nunchaku_lite_qweight_shape = tuple(module.qweight.shape)
    module.qweight = torch.nn.Parameter(
        torch.empty(0, dtype=torch.int8, device=module.qweight.device), requires_grad=False
    )
    module._nunchaku_lite_qweight_released = True


def svdq_mps_cache(module, dtype: torch.dtype) -> dict:
    """Ensure the MPS-layout cache for an SVDQ module is current and return its metadata."""

    backend = resolve_backend(getattr(module, "backend", "auto"))
    released = getattr(module, "_nunchaku_lite_qweight_released", False)
    key = (backend, dtype, *_tensor_states(module, _SVDQ_KEY_TENSORS))
    if not _cache_valid(module, key):
        tensors, meta = _build_svdq_cache(module, backend, dtype)
        if released and not meta["q_invertible"]:
            # e.g. switched to the dense backend after release: keep the 4-bit values packed again
            module.qweight = torch.nn.Parameter(pack_svdq_weight(_source_q(module)), requires_grad=False)
            module._nunchaku_lite_qweight_released = released = False
        _store_cache(module, key, tensors, meta)
        if meta["q_invertible"] and not released:
            _release_qweight(module)
            module._nunchaku_lite_mps_key = (*key[:2], ("released",), *key[3:])
    return module._nunchaku_lite_mps_meta


def svdq_linear_mps(module, x: torch.Tensor, act_shift: float | None = None) -> torch.Tensor:
    """Apply an SVDQ projection on MPS.

    Args:
        module: :class:`~nunchaku_lite.linear.SVDQW4A4Linear` with loaded parameters.
        x: Activations ``(..., in_features)``.
        act_shift: Shift added to the quantized path's input. Defaults to
            :data:`GELU_SHIFT` for ``act_unsigned`` projections, else ``0``.
    """

    if act_shift is None:
        act_shift = GELU_SHIFT if module.act_unsigned else 0.0
    lead = x.shape[:-1]
    x2 = x.reshape(-1, module.in_features)
    dtype = x2.dtype
    meta = svdq_mps_cache(module, dtype)
    bias = _cached(module, "bias")
    bias = None if bias is None else bias.float()  # module.to(dtype) may have cast float32 buffers

    if meta["backend"] == "dense":
        weight = _cached(module, "weight")
        out = F.linear((x2 + act_shift).to(weight.dtype), weight)
        down, up = _cached(module, "down"), _cached(module, "up")
        if down is not None:
            out = out + F.linear(F.linear(x2.to(down.dtype), down), up)
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out.to(dtype).reshape(*lead, module.out_features)

    ops = _kernel_ops()
    lr_dtype = meta["lr_dtype"]
    m = x2.shape[0]
    pad = -m % _TILE_M
    xp = F.pad(x2, (0, 0, 0, pad)) if pad else x2
    xp = xp.contiguous()
    a, ascale = ops.svdq_quantize_act_int8(xp, _cached(module, "inv_smooth").float(), float(act_shift))
    lact = (xp.to(lr_dtype) @ _cached(module, "down_t").to(lr_dtype)).contiguous()
    lup = _cached(module, "lup").to(lr_dtype)
    wsc = _cached(module, "wsc").float()
    if meta["backend"] == "int8":
        out = ops.svdq_gemm_w8a8(a, _cached(module, "w8"), ascale, wsc, lact, lup, bias)
    else:
        out = ops.svdq_gemm_w4a8(a, _cached(module, "w4"), _cached(module, "wr").half(), ascale, wsc, lact, lup, bias)
    if pad:
        out = out[:m]
    return out.to(dtype).reshape(*lead, module.out_features)


# ---------------------------------------------------------------------------
# Fused-op equivalents
# ---------------------------------------------------------------------------


def gelu_mlp_mps(x: torch.Tensor, fc1, fc2) -> torch.Tensor:
    """Equivalent of the CUDA ``fused_gelu_mlp``: fc1 -> tanh-GELU -> fc2 (with the INT4 GELU shift)."""

    hidden = F.gelu(svdq_linear_mps(fc1, x, act_shift=0.0), approximate="tanh")
    return svdq_linear_mps(fc2, hidden)


def unpack_rotemb(packed: torch.Tensor) -> torch.Tensor:
    """Invert :func:`nunchaku_lite.adapters.common.pack_rotemb`: ``(B, S, D)`` -> ``(B, S, D // 2, 2)`` (sin, cos)."""

    batch, seq_len, dim = packed.shape
    r = packed.view(batch, seq_len // 16, dim // 8, 8, 4, 2, 2)
    r = r.permute(0, 1, 2, 5, 3, 4, 6).reshape(batch, seq_len // 16, dim // 8, 16, 8)
    r = r.permute(0, 1, 3, 2, 4).reshape(batch, seq_len, dim // 2, 2)
    return r


def _packed_rope_tables(rotary_emb: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)`` float32 ``(seq_len, D // 2)`` tables (one value per pair) from a packed rotary embedding."""

    table = unpack_rotemb(rotary_emb)[0, :seq_len].float()
    return table[..., 1].contiguous(), table[..., 0].contiguous()


def _rms_norm_heads(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        xf = xf * weight.float()
    return xf


def _apply_packed_rope(x: torch.Tensor, sincos: torch.Tensor) -> torch.Tensor:
    # x: (B, S, H, D) float32; sincos: (B or 1, S, D // 2, 2) with (sin, cos) per interleaved pair.
    sin = sincos[..., 0].unsqueeze(2)
    cos = sincos[..., 1].unsqueeze(2)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out


def qkv_norm_rotary_mps(
    x: torch.Tensor,
    proj,
    norm_q_weight: torch.Tensor | None,
    norm_k_weight: torch.Tensor | None,
    rotary_emb: torch.Tensor | None,
    eps_q: float = 1e-6,
    eps_k: float = 1e-6,
    head_dim: int = 128,
) -> torch.Tensor:
    """Equivalent of the CUDA ``fused_qkv_norm_rotary`` dense-output path.

    Returns ``(batch, seq, 3 * heads * head_dim)`` with per-head RMSNorm and packed RoPE applied to Q and K.
    """

    batch, seq_len, _ = x.shape
    qkv = svdq_linear_mps(proj, x)
    if norm_q_weight is None and norm_k_weight is None and rotary_emb is None:
        return qkv
    heads = qkv.shape[-1] // (3 * head_dim)
    if (
        head_dim == 128
        and qkv.dtype in (torch.float16, torch.bfloat16)
        and norm_q_weight is not None
        and norm_k_weight is not None
        and rotary_emb is not None
        and rotary_emb.shape[0] == 1
        and qk_norm_rope_mps_available()
    ):
        # in place on the dense qkv rows; packed tables hold (sin, cos) once per pair
        qkv2 = qkv.reshape(batch * seq_len, -1).contiguous()
        width = heads * head_dim
        parts = (qkv2[:, :width], qkv2[:, width : 2 * width], qkv2[:, 2 * width :])
        cos, sin = _derived(rotary_emb, f"packed_rope:{seq_len}", lambda r: _packed_rope_tables(r, seq_len))
        device = qkv.device
        _launch_qk_norm_rope(
            parts, None, parts, (_f32(norm_q_weight, device), _f32(norm_k_weight, device)) * 2,
            cos, sin, (eps_q, eps_k), heads,
            rot_rows=seq_len, cs_pair=1, out_strides=(head_dim, qkv2.stride(0)),
        )  # fmt: skip
        return qkv2.view(batch, seq_len, -1)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.reshape(batch, seq_len, -1, head_dim)
    k = k.reshape(batch, seq_len, -1, head_dim)
    if norm_q_weight is not None:
        q = _rms_norm_heads(q, norm_q_weight, eps_q)
    if norm_k_weight is not None:
        k = _rms_norm_heads(k, norm_k_weight, eps_k)
    if rotary_emb is not None:
        sincos = unpack_rotemb(rotary_emb)[:, :seq_len]
        q = _apply_packed_rope(q.float(), sincos)
        k = _apply_packed_rope(k.float(), sincos)
    q = q.to(qkv.dtype).reshape(batch, seq_len, -1)
    k = k.to(qkv.dtype).reshape(batch, seq_len, -1)
    return torch.cat([q, k, v], dim=-1)


# ---------------------------------------------------------------------------
# AWQ W4A16 (modulation projections)
# ---------------------------------------------------------------------------


def _awq_g64_channel_maps(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    channel = torch.arange(64, device=device)
    packed_index = (channel // 32) * 4 + (channel % 8) // 2
    nibble = ((channel % 32) // 8) + 4 * (channel % 2)
    return packed_index, nibble


@torch.no_grad()
def _awq_g64_codes(qweight: torch.Tensor, out_features: int) -> torch.Tensor:
    """Unsigned 4-bit codes ``(N, K)`` int32 from Nunchaku AWQ W4A16 group-64 int32 ``qweight``."""

    rows = qweight.shape[0]
    groups = qweight.shape[1] * 2 // 64
    packed = qweight.to(torch.int32).view(rows, groups, 4, 8)
    packed_index, nibble = _awq_g64_channel_maps(qweight.device)
    # Advanced indexing, not index_select: MPS index_select corrupts large-magnitude int32 values.
    selected = packed[..., packed_index]  # (rows, groups, 4, 64)
    codes = (selected >> (4 * nibble).to(torch.int32)) & 0xF
    return codes.permute(0, 2, 1, 3).reshape(out_features, groups * 64)


@torch.no_grad()
def dequantize_awq_g64_int32(qweight: torch.Tensor, wscales: torch.Tensor, wzeros: torch.Tensor) -> torch.Tensor:
    """Dequantize Nunchaku AWQ W4A16 group-64 int32 weights to a dense ``(N, K)`` float32 tensor."""

    groups, out_features = wscales.shape
    codes = _awq_g64_codes(qweight, out_features).reshape(out_features, groups, 64).float()
    scale = wscales.float().t().reshape(out_features, groups, 1)
    zeros = wzeros.float().t().reshape(out_features, groups, 1)
    return (codes * scale + zeros).reshape(out_features, groups * 64)


@torch.no_grad()
def _awq_g64_pack(codes: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_awq_g64_codes`: unsigned codes ``(N, K)`` -> Nunchaku AWQ group-64 int32 ``qweight``."""

    n, k = codes.shape
    groups = k // 64
    c = codes.to(device="cpu", dtype=torch.int64).view(n // 4, 4, groups, 64).permute(0, 2, 1, 3)
    packed_index, nibble = _awq_g64_channel_maps(torch.device("cpu"))
    # the 8 channels sharing a packed word occupy disjoint nibbles, so adding them is an OR
    words = torch.zeros(n // 4, groups, 4, 8, dtype=torch.int64).index_add_(-1, packed_index, c << (4 * nibble))
    words = torch.where(words >= 2**31, words - 2**32, words)  # reinterpret as signed int32
    return words.to(torch.int32).view(n // 4, groups * 32)


_INT4PACK_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _awq_int4pack_codes(module) -> torch.Tensor:
    """Unsigned codes ``(N, K)`` recovered exactly from the 4-bit matmul copy: ``I @ W`` with scale 1 and zero 8."""

    # The packed layout is the MPS one even after the module moves to the CPU (offload), so decode on MPS.
    w4 = _cached(module, "w4").to("mps")
    k, n = module.in_features, module.out_features
    ones = torch.ones(k // 64, n, device=w4.device)
    unit = torch.stack([ones, 8 * ones], dim=-1).to(torch.bfloat16)
    codes = torch.empty(n, k, dtype=torch.int32, device=w4.device)
    for start in range(0, k, 2048):  # bounded temporaries for wide layers
        rows = min(2048, k - start)
        eye = torch.zeros(rows, k, dtype=torch.bfloat16, device=w4.device)
        eye[:, start : start + rows].fill_diagonal_(1)
        codes[:, start : start + rows] = torch.ops.aten._weight_int4pack_mm(eye, w4, 64, unit).t().round().int()
    return codes.to(module.wscales.device)


def _awq_codes(module) -> torch.Tensor:
    if getattr(module, "_nunchaku_lite_qweight_released", False):
        return _awq_int4pack_codes(module)
    return _awq_g64_codes(module.qweight, module.out_features)


@torch.no_grad()
def _build_awq_int4pack(module, dtype: torch.dtype) -> dict | None:
    """Convert to PyTorch's MPS 4-bit matmul format (``aten._weight_int4pack_mm``), or ``None`` if unavailable.

    That op computes ``w = (q - 8) * scale + zero`` per group of 64 input channels, from codes packed two per
    byte with the even input channel in the high nibble. AWQ stores ``w = q * scale + wzero``, so
    ``zero = wzero + 8 * scale``. Same codes, a quarter of the dense bf16 weight, and about 3x faster for the
    1-2 row modulation matmuls.
    """

    if dtype not in _INT4PACK_DTYPES or not hasattr(torch.ops.aten, "_weight_int4pack_mm"):
        return None
    n = module.out_features
    codes = _awq_codes(module).view(n, -1, 2)
    try:
        packed = torch.ops.aten._convert_weight_to_int4pack(((codes[..., 0] << 4) | codes[..., 1]).to(torch.uint8), 2)
    except RuntimeError:
        return None
    scale, wzero = module.wscales.float(), module.wzeros.float()  # (K / 64, N)
    scales_and_zeros = torch.stack([scale, wzero + 8 * scale], dim=-1).to(dtype).contiguous()  # (K / 64, N, 2)
    return {"w4": packed, "scales_and_zeros": scales_and_zeros}


def _awq_state_dict_hook(module, state_dict, prefix, local_metadata):
    if getattr(module, "_nunchaku_lite_qweight_released", False):
        state_dict[prefix + "qweight"] = _awq_g64_pack(_awq_int4pack_codes(module)).to(module.wscales.device)


def _release_awq_qweight(module) -> None:
    """Free the checkpoint ``qweight`` once the 4-bit matmul copy holds the same codes (1.6 GB on FLUX.1)."""

    if not getattr(module, "_nunchaku_lite_mps_hooks", False):
        module._register_state_dict_hook(_awq_state_dict_hook)
        module._register_load_state_dict_pre_hook(_load_state_dict_pre_hook, with_module=True)
        module._nunchaku_lite_mps_hooks = True
    module._nunchaku_lite_qweight_shape = tuple(module.qweight.shape)
    module._nunchaku_lite_qweight_dtype = module.qweight.dtype
    module.qweight = torch.nn.Parameter(
        torch.empty(0, dtype=module.qweight.dtype, device=module.qweight.device), requires_grad=False
    )
    module._nunchaku_lite_qweight_released = True


def awq_linear_mps(module, x: torch.Tensor) -> torch.Tensor:
    """Apply an AWQ W4A16 projection on MPS.

    Uses PyTorch's 4-bit MPS matmul when available (weights stay 4-bit; the checkpoint ``qweight`` is then
    released and re-packed on ``state_dict()``), else a cached dequantized weight.
    """

    if module.group_size != 64:
        raise NotImplementedError("Only group-size-64 AWQ W4A16 weights are supported on MPS.")

    key = ("awq", x.dtype, *_tensor_states(module, _AWQ_KEY_TENSORS))
    if not _cache_valid(module, key):
        released = getattr(module, "_nunchaku_lite_qweight_released", False)
        int4pack = _build_awq_int4pack(module, x.dtype)
        if int4pack is not None:
            _store_cache(module, key, int4pack, {"backend": "awq_int4pack"})
            if not released:
                _release_awq_qweight(module)
                module._nunchaku_lite_mps_key = (*key[:2], ("released",), *key[3:])
        else:
            if released:  # e.g. float64 activations: bring the packed weights back first
                module.qweight = torch.nn.Parameter(
                    _awq_g64_pack(_awq_int4pack_codes(module)).to(module.wscales.device), requires_grad=False
                )
                module._nunchaku_lite_qweight_released = False
                key = ("awq", x.dtype, *_tensor_states(module, _AWQ_KEY_TENSORS))
            dense = dequantize_awq_g64_int32(module.qweight, module.wscales, module.wzeros).to(x.dtype)
            _store_cache(module, key, {"weight": dense}, {"backend": "awq_dense"})
    if module._nunchaku_lite_mps_meta["backend"] == "awq_int4pack":
        return torch.ops.aten._weight_int4pack_mm(x, _cached(module, "w4"), 64, _cached(module, "scales_and_zeros"))
    weight = _cached(module, "weight")
    return F.linear(x, weight.to(x.dtype))


# ---------------------------------------------------------------------------
# Attention kernels: from nunchaku_lite_kernels, or compiled at runtime when the installed build predates them
# ---------------------------------------------------------------------------


@functools.cache
def _lite_op(name: str):
    """The ``nunchaku_lite_kernels`` op ``name``, or ``None`` when no build loads or the build predates it."""

    try:
        return getattr(_kernel_ops(), name, None)
    except Exception:  # noqa: BLE001 - any load failure: fall back to the runtime-compiled kernel
        return None


def _can_compile_shaders() -> bool:
    return torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")


# FLUX.2 q/k norm + RoPE. The kernel body is shared with the MLX runtime (:data:`nunchaku_lite.shaders.QK_NORM_ROPE`).
_QK_TYPES = {torch.float16: "half", torch.bfloat16: "bfloat", torch.float32: "float"}


@functools.cache
def _qk_norm_rope_library():
    from ..shaders import qk_norm_rope_library_source

    return torch.mps.compile_shader(qk_norm_rope_library_source())


@functools.cache
def qk_norm_rope_mps_available() -> bool:
    """Whether the q/k norm + RoPE kernel runs here (from ``nunchaku_lite_kernels``, or compiled at runtime)."""

    if not torch.backends.mps.is_available():
        return False
    if _lite_op("qk_norm_rope_mps_") is not None:
        return True
    if not _can_compile_shaders():
        return False
    try:
        _qk_norm_rope_library()
    except Exception:  # noqa: BLE001 - any compile failure means: use the unfused path
        return False
    return True


def _rows(t: torch.Tensor) -> torch.Tensor:
    """A ``(..., L, C)`` projection output (or a column slice of one) as a 2-D view with a contiguous last dim."""

    t2 = t.reshape(-1, t.shape[-1]) if t.is_contiguous() else t.flatten(0, -2)
    return t2 if t2.stride(-1) == 1 else t2.contiguous()


@functools.lru_cache(maxsize=256)
def _device_constant(values: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
    """A small read-only kernel argument (dims, eps, unit norm weight), uploaded once per distinct value."""

    return torch.tensor(values, dtype=dtype, device=device)


_DERIVED: dict[tuple[int, str], tuple[weakref.ref, int, object]] = {}


def _derived(t: torch.Tensor, key: str, build):
    """``build(t)``, cached while ``t`` is alive and unmodified (weights, per-step rotary tables)."""

    entry = _DERIVED.get((id(t), key))
    if entry is not None and entry[0]() is t and entry[1] == t._version:
        return entry[2]
    value = build(t)
    cache_key = (id(t), key)
    _DERIVED[cache_key] = (weakref.ref(t, lambda _ref: _DERIVED.pop(cache_key, None)), t._version, value)
    return value


def _f32(t: torch.Tensor | None, device, n: int = 128) -> torch.Tensor:
    if t is None:
        return _device_constant((1.0,) * n, torch.float32, str(device))
    if t.dtype == torch.float32 and t.is_contiguous() and t.device == device:
        return t
    return _derived(t, f"f32:{device}", lambda w: w.to(device=device, dtype=torch.float32).contiguous())


def _launch_qk_norm_rope(
    image: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    text: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: tuple[float, float],
    heads: int,
    *,
    rot_rows: int,
    cs_pair: int,
    out_strides: tuple[int, int],
) -> None:
    """Run the shared kernel. Sources and outputs are 2-D row views (outputs may alias sources)."""

    op = _lite_op("qk_norm_rope_mps_")
    if op is not None:
        op(
            *image, *(text if text is not None else (None, None, None)), weights[2], weights[3], weights[0],
            weights[1], cos, sin, *outputs, float(eps[0]), float(eps[1]), heads, rot_rows, cs_pair, *out_strides,
        )  # fmt: skip
        return
    device = image[0].device
    txt = text if text is not None else image
    lt = text[0].shape[0] if text is not None else 0
    total = lt + image[0].shape[0]
    copy_v = int(outputs[2].data_ptr() != image[2].data_ptr())
    dims = (lt, txt[0].stride(0), image[0].stride(0), 0, 0, 0, rot_rows, cos.stride(0), cs_pair, *out_strides, copy_v)
    group = next(g for g in (8, 4, 2, 1) if heads % g == 0)
    kernel = getattr(_qk_norm_rope_library(), f"qk_norm_rope_{_QK_TYPES[image[0].dtype]}")
    kernel(
        *txt, *image, *weights, cos, sin,
        _device_constant(tuple(map(float, eps)), torch.float32, str(device)),
        _device_constant(dims, torch.int32, str(device)),
        *outputs, threads=(heads * 32, total, 1), group_size=(32 * group, 1, 1),
    )  # fmt: skip


def qk_norm_rope_mps(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    eps_q: float,
    eps_k: float,
    text: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    text_q_weight: torch.Tensor | None = None,
    text_k_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """q/k RMSNorm + RoPE for FLUX.2-style attention (head dim 128, batch 1) on MPS.

    Args:
        q, k, v: Image-token projections ``(1, Li, heads * 128)``. They may be column slices of one fused
            projection output; they must share the row stride.
        q_weight, k_weight: RMSNorm weights ``(128,)`` (``None``: no affine weight).
        cos, sin: Diffusers rotary tables ``(Lt + Li, 128)`` (each pair's value repeated), text rows first.
        heads: Number of attention heads.
        eps_q, eps_k: RMSNorm epsilons.
        text: Optional text-token ``(q, k, v)`` placed before the image rows (joint attention).
        text_q_weight, text_k_weight: RMSNorm weights for the text rows.

    Returns:
        ``(q, k, v)`` of shape ``(1, Lt + Li, heads, 128)`` in the projections' dtype.
    """

    dtype, width = q.dtype, heads * 128
    groups = [tuple(_rows(t) for t in g) for g in ((q, k, v), text) if g is not None]
    for group in groups:
        if len({t.stride(0) for t in group}) != 1 or any(t.shape[-1] != width or t.dtype != dtype for t in group):
            raise ValueError("q, k and v must share dtype, width heads * 128 and row stride.")
    image, txt = groups[0], (groups[1] if text is not None else None)
    total = image[0].shape[0] + (txt[0].shape[0] if txt is not None else 0)
    if cos.shape != (total, 128) or sin.shape != (total, 128):
        raise ValueError(f"Rotary tables must be ({total}, 128), got {tuple(cos.shape)} and {tuple(sin.shape)}.")
    device = q.device
    wq, wk = _f32(q_weight, device), _f32(k_weight, device)
    wqt, wkt = (_f32(text_q_weight, device), _f32(text_k_weight, device)) if text is not None else (wq, wk)
    out = tuple(torch.empty((1, total, heads, 128), dtype=dtype, device=device) for _ in range(3))
    _launch_qk_norm_rope(
        image, txt, tuple(o.view(total, width) for o in out), (wqt, wkt, wq, wk),
        _f32(cos, device, total), _f32(sin, device, total), (eps_q, eps_k), heads,
        rot_rows=total, cs_pair=2, out_strides=(128, width),
    )  # fmt: skip
    return out


# Flash attention on the Neural Accelerators (MLX's steel NAX kernel)
_ATTN_TYPES = {torch.float16: "half", torch.bfloat16: "bfloat"}


@functools.cache
def _attention_library(align_q: bool, align_k: bool, has_mask: bool):
    """Compile the kernel for one (query alignment, key alignment, mask) combination. MLX selects these features
    with Metal function constants, which ``compile_shader`` cannot set, so they become compile-time constants
    (the ``nunchaku_lite_kernels`` build specializes the function constants instead)."""

    from ..shaders import attention_instantiations
    from ._mlx_steel_attention import SOURCE

    consts = {
        "align_Q": align_q,
        "align_K": align_k,
        "has_mask": has_mask,
        "do_causal": False,
        "has_sinks": False,
        "save_lse": False,
    }
    source = SOURCE
    for name, value in consts.items():
        source, n = re.subn(
            rf"constant bool {name} \[\[function_constant\(\d+\)\]\];",
            f"constant constexpr bool {name} = {str(value).lower()};",
            source,
        )
        if n != 1:
            raise RuntimeError(f"Vendored MLX attention source has no function constant {name!r}.")
    source = re.sub(r", function_constant\(\w+\)\]\]", "]]", source)
    return torch.mps.compile_shader(source + attention_instantiations())


@functools.cache
def attention_mps_available() -> bool:
    """Whether :func:`attention_mps` can run: an M5-class GPU, and the ``nunchaku_lite_kernels`` op or a PyTorch
    with ``torch.mps.compile_shader``."""

    if not (torch.backends.mps.is_available() and has_neural_accelerators()):
        return False
    if _lite_op("attention_mps") is not None:
        return True
    if not _can_compile_shaders():
        return False
    try:
        _attention_library(True, True, False)
    except Exception:  # noqa: BLE001 - any compile failure means: use PyTorch's attention
        return False
    return True


def _mask_4d(attn_mask: torch.Tensor, query: torch.Tensor, k_len: int) -> torch.Tensor | None:
    """The mask as a ``(B, H, Lq, Lk)`` view with a contiguous key dim, or ``None`` if it cannot broadcast."""

    if attn_mask.ndim == 2 and attn_mask.shape == (query.shape[0], k_len):  # Diffusers' (B, Lk) key padding mask
        attn_mask = attn_mask[:, None, None, :]
    while attn_mask.ndim < 4:
        attn_mask = attn_mask.unsqueeze(0)
    try:
        attn_mask = attn_mask.expand(query.shape[0], query.shape[2], query.shape[1], k_len)
    except RuntimeError:
        return None
    return attn_mask if attn_mask.stride(-1) == 1 else attn_mask.contiguous()


def attention_mps_supported(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: torch.Tensor | None = None
) -> bool:
    """Whether :func:`attention_mps` handles these ``(B, L, H, D)`` tensors and optional mask."""

    if not (
        query.device.type == "mps"
        and query.dtype in _ATTN_TYPES
        and query.dtype == key.dtype == value.dtype
        and query.ndim == key.ndim == value.ndim == 4
        and query.shape[-1] in ATTENTION_HEAD_DIMS
        and key.shape[-1] == value.shape[-1] == query.shape[-1]
        and key.shape[:3] == value.shape[:3]
        and query.shape[0] == key.shape[0]
        and query.shape[2] % key.shape[2] == 0
        and query.stride(-1) == key.stride(-1) == value.stride(-1) == 1
    ):
        return False
    if attn_mask is not None and (
        attn_mask.device != query.device
        or not (attn_mask.dtype == torch.bool or attn_mask.is_floating_point())
        or _mask_4d(attn_mask, query, key.shape[1]) is None
    ):
        return False
    return attention_mps_available()


def attention_mps(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Softmax attention over ``(B, L, H, D)`` tensors (Diffusers' layout), returning ``(B, Lq, H, D)``.

    Any strides with a contiguous head dim are read in place; grouped-query attention (fewer key/value heads) is
    supported. ``attn_mask`` follows ``F.scaled_dot_product_attention``: broadcastable to ``(B, H, Lq, Lk)`` (or
    Diffusers' ``(B, Lk)``), bool (``True`` = attend) or additive float. Call :func:`attention_mps_supported` first.
    """

    batch, q_len, heads, dim = query.shape
    k_len, kv_heads = key.shape[1], key.shape[2]
    scale = dim**-0.5 if scale is None else float(scale)
    mask = None
    if attn_mask is not None:
        mask = _mask_4d(attn_mask, query, k_len)
        if mask.dtype != torch.bool and mask.dtype != query.dtype:
            mask = mask.to(query.dtype)
    op = _lite_op("attention_mps")
    if op is not None:
        return op(query, key, value, mask, scale)
    out = torch.empty_like(query, memory_format=torch.contiguous_format)
    bq, bk, wm = ATTENTION_TILE
    nq, nk, nq_full, nk_full = -(-q_len // bq), -(-k_len // bk), q_len // bq, k_len // bk

    def strides(t):  # (B, L, H, D) -> the kernel's (B, H, L) strides
        return t.stride(0), t.stride(2), t.stride(1)

    params = struct.pack(
        "<6if7i12q",
        batch, heads, dim, q_len, k_len, heads // kv_heads, scale, nq, nk, nq_full, nk_full,
        q_len - nq_full * bq, k_len - nk_full * bk, k_len - q_len,
        *strides(query), *strides(key), *strides(value), *strides(out),
    )  # fmt: skip
    device = query.device
    params = torch.frombuffer(bytearray(params), dtype=torch.uint8).to(device)
    unused = torch.empty(16, dtype=torch.uint8, device=device)  # sinks / lse slots (disabled)
    mask_kind = "bool" if mask is not None and mask.dtype == torch.bool else "add"
    if mask is None:
        mask, mask_params = unused, unused
    else:
        mask_params = torch.tensor(mask.stride()[:3], dtype=torch.int64).view(torch.uint8).to(device)
    library = _attention_library(q_len % bq == 0, k_len % bk == 0, attn_mask is not None)
    kernel = getattr(library, f"attention_{_ATTN_TYPES[query.dtype]}_{dim}_{mask_kind}")
    kernel(
        query, key, value, out, params, mask_params, mask, unused, unused,
        threads=(nq * 32, heads * wm, batch), group_size=(32, wm, 1),
    )  # fmt: skip
    return out
