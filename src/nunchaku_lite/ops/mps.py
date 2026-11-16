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
"""

from __future__ import annotations

import functools
import subprocess

import torch
import torch.nn.functional as F

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


def _kernel_ops():
    from .backend import get_ops

    ops = get_ops()
    if not hasattr(ops, "svdq_gemm_w8a8"):
        raise RuntimeError(
            "The installed nunchaku_lite_kernels package has no MPS ops. Build it on macOS with "
            '"pip install --no-build-isolation ./nunchaku-lite-kernels", or pass backend="dense".'
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
    words = qweight.contiguous().view(torch.int32).reshape(n // 128, k // 64, 1, 8, 8, 4, 2, 2, 1)
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    nibbles = (words.unsqueeze(-1) >> shifts) & 0xF
    signed = ((nibbles ^ 8) - 8).to(torch.int8)
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
    """

    key = getattr(module, "_nunchaku_lite_mps_key", None)
    names = _AWQ_KEY_TENSORS if key is not None and key[0] == "awq" else _SVDQ_KEY_TENSORS
    valid = key is not None and _cache_valid(module, (*key[:2], *_tensor_states(module, names)))
    result = apply()
    if valid:
        module._nunchaku_lite_mps_key = (*key[:2], *_tensor_states(module, names))
    return result


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
        w = (q.float().view(n, k // 64, 64) * wscale[..., None]).view(n, k)
        tensors["weight"] = (w / smooth).to(dtype)  # fold 1/smooth into the dense weight
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
        w8 = torch.round(q.float().view(n, k // 64, 64) * wr.float()[..., None]).clamp_(-127, 127)
        tensors["w8"] = w8.to(torch.int8).view(n, k).contiguous()
        invertible = bool(torch.equal(_q_from_w8(tensors["w8"], tensors["wr"]), q))
    else:
        nib = (q & 0xF).to(torch.uint8).view(n, k // 2, 2)
        tensors["w4"] = (nib[..., 0] | (nib[..., 1] << 4)).contiguous()
        invertible = True
    return tensors, {"backend": backend, "lr_dtype": lr_dtype, "q_invertible": invertible}


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
        module.qweight = torch.nn.Parameter(
            torch.empty(module._nunchaku_lite_qweight_shape, dtype=torch.int8, device=module.wscales.device),
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


def _fused_qk_norm_rope_op():
    """Return the in-place Metal QK-norm/RoPE op, or ``None`` when the kernels build predates it."""

    try:
        return getattr(_kernel_ops(), "svdq_qk_norm_rope_", None)
    except (ImportError, RuntimeError, AttributeError):
        return None


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
    fused = _fused_qk_norm_rope_op() if head_dim == 128 and qkv.dtype in (torch.float16, torch.bfloat16) else None
    if fused is not None:
        rope = None if rotary_emb is None else unpack_rotemb(rotary_emb).float().contiguous()
        qkv2 = qkv.reshape(batch * seq_len, -1).contiguous()
        fused(
            qkv2,
            None if norm_q_weight is None else norm_q_weight.to(qkv.dtype).contiguous(),
            None if norm_k_weight is None else norm_k_weight.to(qkv.dtype).contiguous(),
            rope,
            float(eps_q),
            float(eps_k),
            heads,
            seq_len,
        )
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
def dequantize_awq_g64_int32(qweight: torch.Tensor, wscales: torch.Tensor, wzeros: torch.Tensor) -> torch.Tensor:
    """Dequantize Nunchaku AWQ W4A16 group-64 int32 weights to a dense ``(N, K)`` float32 tensor."""

    groups, out_features = wscales.shape
    rows = qweight.shape[0]
    packed = qweight.to(torch.int32).view(rows, groups, 4, 8)
    packed_index, nibble = _awq_g64_channel_maps(qweight.device)
    # Advanced indexing, not index_select: MPS index_select corrupts large-magnitude int32 values.
    selected = packed[..., packed_index]  # (rows, groups, 4, 64)
    codes = (selected >> (4 * nibble).to(torch.int32)) & 0xF
    codes = codes.permute(0, 2, 1, 3).reshape(out_features, groups, 64).float()
    scale = wscales.float().t().reshape(out_features, groups, 1)
    zeros = wzeros.float().t().reshape(out_features, groups, 1)
    return (codes * scale + zeros).reshape(out_features, groups * 64)


def awq_linear_mps(module, x: torch.Tensor) -> torch.Tensor:
    """Apply an AWQ W4A16 projection on MPS via a cached dequantized weight (small-M modulation layers)."""

    if module.group_size != 64:
        raise NotImplementedError("Only group-size-64 AWQ W4A16 weights are supported on MPS.")

    key = ("awq", x.dtype, *_tensor_states(module, _AWQ_KEY_TENSORS))
    if not _cache_valid(module, key):
        dense = dequantize_awq_g64_int32(module.qweight, module.wscales, module.wzeros).to(x.dtype)
        _store_cache(module, key, {"weight": dense}, {"backend": "awq"})
    weight = _cached(module, "weight")
    return F.linear(x, weight.to(x.dtype))
