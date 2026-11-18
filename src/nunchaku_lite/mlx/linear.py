"""MLX SVDQ linear layer mirroring ``nunchaku_lite.linear.SVDQW4A4Linear``.

The module stores the Nunchaku checkpoint tensors under the same parameter names
as the PyTorch runtime (``qweight``, ``wscales``, ``smooth_factor``,
``proj_down``, ``proj_up``, ``bias``), so checkpoints load with
``module.load_weights``. On the first call (and whenever those parameters are
replaced) the weights are converted to the layout of the selected backend.

Backends (``backend=`` on the layer or :func:`~nunchaku_lite.mlx.patch_transformer`,
same values as the PyTorch MPS runtime's ``mps_backend``):

``auto`` (default)
    ``int8`` on GPUs with Neural Accelerators (M5 class), ``dense`` elsewhere.
    ``patch_transformer`` additionally falls back to ``int4`` when the model would
    not fit comfortably in GPU memory with int8 weights.
``int8``
    W8A8 Neural Accelerator kernel; weights expanded to int8. Fastest.
``int4``
    W4A8 kernel; weights stay 4-bit and are expanded in-kernel. Same output as
    ``int8``, about half the weight memory, slightly slower.
``dense``
    Exact W4A16 fallback for any Apple GPU (``mx.quantized_matmul`` with a
    lossless mapping of the INT4 weights; activations stay in floating point).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn

from . import mx_layout as mxl
from .kernels import TILE_M, has_neural_accelerators, svdq_gemm_w4a8, svdq_gemm_w8a8, svdq_quantize_act_int8
from .packing import _LR_PACK, _LR_PERM

__all__ = ["BACKENDS", "GELU_SHIFT", "AWQW4A16Linear", "SVDQW4A4Linear", "resolve_backend"]

GELU_SHIFT = 0.171875
"""Shift Nunchaku INT4 applies to post-GELU activations of ``act_unsigned`` projections."""

_PARAM_NAMES = ("qweight", "wscales", "smooth_factor", "proj_down", "proj_up", "bias")


BACKENDS = ("auto", "int8", "int4", "dense")
_AUTO_MEMORY_FRACTION = 0.5


def resolve_backend(choice: str = "auto", int8_resident_bytes: int | None = None) -> str:
    """Resolve ``"auto"`` to ``"int8"``, ``"int4"``, or ``"dense"``.

    Args:
        choice: One of :data:`BACKENDS`.
        int8_resident_bytes: Estimated model footprint with int8 weights; when given,
            ``"auto"`` picks ``"int4"`` if it exceeds half of the GPU's recommended
            working set.
    """

    choice = str(choice).lower()
    if choice not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}; got {choice!r}.")
    if choice != "auto":
        return choice
    if not has_neural_accelerators():
        return "dense"
    if int8_resident_bytes is not None:
        budget = _AUTO_MEMORY_FRACTION * mx.device_info().get("max_recommended_working_set_size", float("inf"))
        if int8_resident_bytes > budget:
            return "int4"
    return "int8"


def _to_numpy(value: mx.array) -> np.ndarray:
    if value.dtype == mx.bfloat16:
        value = value.astype(mx.float32)
    return np.array(value)


_SHIFTS4 = None


def _nibble_shifts() -> mx.array:
    global _SHIFTS4
    if _SHIFTS4 is None:
        _SHIFTS4 = mx.arange(0, 32, 4, dtype=mx.uint32)
    return _SHIFTS4


def _pack_nibbles_u32(codes: mx.array) -> mx.array:
    """Pack unsigned 4-bit codes ``(N, K)`` into uint32 words, low nibble first (MLX affine layout)."""

    n, k = codes.shape
    return (codes.astype(mx.uint32).reshape(n, k // 8, 8) << _nibble_shifts()).sum(axis=-1).astype(mx.uint32)


def _unpack_nibbles_u32(words: mx.array) -> mx.array:
    n = words.shape[0]
    return ((words[..., None] >> _nibble_shifts()) & 0xF).reshape(n, -1)


def _pack_svdq_qweight(q: mx.array) -> mx.array:
    """Nunchaku INT4 ``qweight`` ``(N, K // 2)`` int8 from signed values ``(N, K)`` (lazy MLX graph)."""

    n, k = q.shape
    w = (q.astype(mx.int32) & 0xF).astype(mx.uint32).reshape(n // 128, 8, 2, 8, 1, k // 64, 1, 2, 4, 8)
    w = w.transpose(0, 5, 6, 1, 3, 8, 2, 7, 4, 9)
    words = (w << _nibble_shifts()).sum(axis=-1).astype(mx.uint32)
    return words.reshape(n, k // 8).view(mx.int8).reshape(n, k // 2)


def _pack_lowrank(w: mx.array, down: bool) -> mx.array:
    """Lazy MLX twin of :func:`packing.pack_lowrank`: ``down`` ``(R, K)`` -> ``(K, R)``; up ``(N, R)`` -> ``(N, R)``."""

    p = _LR_PACK
    if down:
        r, c = w.shape
        w = w.reshape(r // p, p, c // p, p).transpose(2, 0, 1, 3)
    else:
        c, r = w.shape
        w = w.reshape(c // p, p, r // p, p).transpose(0, 2, 1, 3)
    return w.reshape(c // p, r // p, 2, 8, 1, 2, 4, 2).transpose(_LR_PERM).reshape(c, r)


def _redo_adanorm_layout(t: mx.array, splits: int) -> mx.array:
    """Inverse of the channel un-permutation in :func:`packing.undo_adanorm_layout` (lazy)."""

    if splits <= 1:
        return t
    n = t.shape[0]
    return t.reshape(splits, n // splits, *t.shape[1:]).swapaxes(0, 1).reshape(t.shape)


def _pack_awq_qweight(codes: mx.array) -> mx.array:
    """Lazy MLX twin of :func:`packing.pack_awq_w4`: unsigned codes ``(N, K)`` -> int32 ``(N / 4, K / 2)``."""

    oc, ic = codes.shape
    w = codes.astype(mx.uint32).reshape(-1, 4, 8)
    w = w[:, 0] | (w[:, 1] << 4) | (w[:, 2] << 8) | (w[:, 3] << 12)
    w = w.reshape(oc // 4, 4, ic // 64, 16).transpose(0, 2, 1, 3).reshape(oc // 4, ic)
    return w.astype(mx.uint16).view(mx.int32).reshape(oc // 4, ic // 2)


class SVDQW4A4Linear(nn.Module):
    """SVDQ INT4 projection ``y = W4((x + shift) / smooth) + up(down(x)) + bias``.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        rank: Low-rank correction rank of the checkpoint.
        bias: Whether the projection has a bias.
        precision: Checkpoint precision; only ``"int4"`` is supported.
        act_unsigned: Nunchaku ``act_unsigned`` flag. INT4 checkpoints quantize these
            projections' inputs after a GELU with a +0.171875 shift (FLUX.1 and
            Qwen-Image MLP ``fc2``); the shift is applied automatically.
        dtype: Dtype of the floating-point parameters.
        backend: ``"auto"``, ``"int8"``, ``"int4"``, or ``"dense"`` (see module docstring).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        bias: bool = True,
        precision: str = "int4",
        act_unsigned: bool = False,
        dtype=mx.bfloat16,
        backend: str = "auto",
    ):
        super().__init__()
        resolve_backend(backend)  # validate early
        if precision != "int4":
            raise NotImplementedError("Only INT4 SVDQ checkpoints are supported on MLX (no NVFP4 yet).")
        if in_features % 128 or out_features % 128:
            raise ValueError("SVDQ INT4 layers require in_features and out_features divisible by 128.")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.precision = precision
        self.act_unsigned = act_unsigned
        self.backend = backend
        self.qweight = mx.zeros((out_features, in_features // 2), dtype=mx.int8)
        self.wscales = mx.zeros((in_features // 64, out_features), dtype=dtype)
        self.smooth_factor = mx.ones((in_features,), dtype=dtype)
        self.proj_down = mx.zeros((in_features, rank), dtype=dtype)
        self.proj_up = mx.zeros((out_features, rank), dtype=dtype)
        self.bias = mx.zeros((out_features,), dtype=dtype) if bias else None
        self.freeze()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 32,
        precision: str = "int4",
        act_unsigned: bool = False,
        dtype=None,
        backend: str = "auto",
    ) -> SVDQW4A4Linear:
        """Create an empty SVDQ module with the shape (and bias presence) of ``linear``."""

        out_features, in_features = linear.weight.shape
        return cls(
            in_features,
            out_features,
            rank=rank,
            bias="bias" in linear,
            precision=precision,
            act_unsigned=act_unsigned,
            dtype=dtype or linear.weight.dtype,
            backend=backend,
        )

    # ------------------------------------------------------------------ cache

    def _params(self) -> tuple:
        return tuple(getattr(self, name) for name in _PARAM_NAMES)

    def _cache_for(self, backend: str, dtype) -> dict:
        params = self._params()
        cached = getattr(self, "_cache", None)
        if (
            cached is not None
            and cached["backend"] == backend
            and cached["dtype"] == dtype
            and all(a is b for a, b in zip(cached["params"], params))
        ):
            return cached
        cached = self._build_cache(backend, dtype)
        # Release checkpoint copies the backend copy duplicates: they become lazy re-packs of the cache, costing no
        # memory until evaluated (e.g. when saving weights). Evaluating them reproduces the exact checkpoint bytes.
        if cached.pop("q_invertible"):
            self.qweight = _pack_svdq_qweight(self._q_from_cache(cached))
        rank = self.proj_down.shape[1]
        if cached["down_t"].dtype == self.proj_down.dtype:
            self.proj_down = _pack_lowrank(cached["down_t"][:, :rank].T, down=True)
        if backend == "dense":
            self.proj_up = _pack_lowrank(cached["up_t"].T[:, :rank], down=False)  # stored in the checkpoint dtype
        elif cached.pop("up_invertible"):
            self.proj_up = _pack_lowrank(cached["lup"][:, :rank], down=False).astype(self.proj_up.dtype)
        params = self._params()
        cached["params"] = params  # identity check: load_weights / update replace the arrays
        self._cache = cached
        return cached

    def _q_from_cache(self, c: dict) -> mx.array:
        """Signed int4 weights ``(N, K)`` recovered exactly from the backend copy."""

        n, k = self.out_features, self.in_features
        if c["backend"] == "dense":
            return _unpack_nibbles_u32(c["qw"]).astype(mx.int32) - 8
        if c["backend"] == "int4":
            w4 = c["w4"]
            nib = mx.stack([w4 & 0xF, w4 >> 4], axis=-1).reshape(n, k).astype(mx.int32)
            return (nib ^ 8) - 8
        q = mx.round(c["w8"].astype(mx.float32).reshape(n, k // 64, 64) / c["wr"].T.astype(mx.float32)[..., None])
        return q.astype(mx.int32).reshape(n, k)

    def _build_cache(self, backend: str, dtype) -> dict:
        """Convert the checkpoint tensors to the backend's layout on the GPU (bit-identical to the NumPy
        references in :mod:`.packing`) and evaluate the result, so the cache holds no graph back to them."""

        n, k = self.out_features, self.in_features
        q = mxl.unpack_weight(self.qweight)  # (N, K) int32
        wscale = mxl.unpack_scale(self.wscales, n)  # (N, K / 64) float32
        down = mxl.unpack_lowrank(self.proj_down, down=True)  # (R, K) float32
        up = mxl.unpack_lowrank(self.proj_up, down=False)  # (N, R) float32
        rank = down.shape[0]
        rank_pad = -rank % 32 if rank else 32  # kernels need a non-empty rank that is a multiple of 32
        down = mx.pad(down, [(0, rank_pad), (0, 0)])
        up = mx.pad(up, [(0, 0), (0, rank_pad)])
        cache = {
            "backend": backend,
            "dtype": dtype,
            "shift": mx.array([GELU_SHIFT if self.act_unsigned else 0.0], dtype=mx.float32),
            "inv_smooth": 1.0 / mxl.unpack_scale(self.smooth_factor, k).reshape(k),
            # Low-rank weights stay in the checkpoint dtype whatever the activation dtype (mflux FLUX.1 runs float32
            # activations on bf16 checkpoints): same values, half the memory, and the checkpoint copy can be released.
            "down_t": down.T.astype(self.proj_down.dtype),  # (K, R)
            "bias": None if self.bias is None else mxl.unpack_scale(self.bias, n).reshape(n),
        }
        if backend == "dense":
            # MLX affine 4-bit: w = scale * u + bias, with u = q + 8 in [0, 15] (lossless).
            cache["qw"] = _pack_nibbles_u32(q + 8)
            cache["qscales"] = wscale.astype(dtype)
            cache["qbiases"] = (-8.0 * cache["qscales"].astype(mx.float32)).astype(dtype)
            cache["up_t"] = up.T.astype(self.proj_up.dtype)  # (R, N)
            cache["q_invertible"] = True  # the affine packing stores q + 8 exactly
            mx.eval([v for v in cache.values() if isinstance(v, mx.array)])
            return cache

        lr_dtype = mx.bfloat16 if dtype == mx.bfloat16 else mx.float16  # low-rank runs on the accelerators
        if lr_dtype == mx.float16 and mx.abs(up).max().item() > 65504:
            raise ValueError("proj_up values exceed float16 range.")
        wsc, wr = mxl.kernel_scales(wscale)
        cache["lr_dtype"] = lr_dtype
        cache["lup"] = up.astype(lr_dtype)  # (N, R)
        cache["wsc"] = wsc
        cache["wr"] = wr.T  # (K / 64, N)
        checks = {"up_invertible": mx.array_equal(cache["lup"].astype(mx.float32), up)}
        if backend == "int8":
            cache["w8"] = mxl.kernel_w8(q, wr)
            recovered = mx.round(
                cache["w8"].astype(mx.float32).reshape(n, k // 64, 64) / wr.astype(mx.float32)[..., None]
            )
            checks["q_invertible"] = mx.array_equal(recovered.reshape(n, k), q)  # every group multiplier >= 1
        else:
            cache["w4"] = mxl.kernel_w4(q)
            cache["q_invertible"] = True
        # proj_up can be released when lup holds its values exactly (e.g. bf16 checkpoint values in fp16)
        mx.eval([v for v in (*cache.values(), *checks.values()) if isinstance(v, mx.array)])
        cache.update({name: bool(flag.item()) for name, flag in checks.items()})
        return cache

    # ---------------------------------------------------------------- forward

    def __call__(self, x: mx.array) -> mx.array:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        cache = self._cache_for(resolve_backend(self.backend), x.dtype)
        out = self._forward_dense(x2, cache) if cache["backend"] == "dense" else self._forward_kernel(x2, cache)
        return out.reshape(*lead, self.out_features)

    @staticmethod
    def _forward_dense(x: mx.array, c: dict) -> mx.array:
        dtype = x.dtype
        xs = ((x.astype(mx.float32) + c["shift"]) * c["inv_smooth"]).astype(dtype)
        out = mx.quantized_matmul(xs, c["qw"], c["qscales"], c["qbiases"], transpose=True, group_size=64, bits=4)
        out = out + ((x @ c["down_t"]) @ c["up_t"]).astype(dtype)
        if c["bias"] is not None:
            out = out + c["bias"].astype(dtype)
        return out

    @staticmethod
    def _forward_kernel(x: mx.array, c: dict) -> mx.array:
        m = x.shape[0]
        pad = -m % TILE_M
        if pad:
            x = mx.pad(x, [(0, pad), (0, 0)])
        a, ascale = svdq_quantize_act_int8(x, c["inv_smooth"], c["shift"])
        lact = (x @ c["down_t"]).astype(c["lr_dtype"])
        if c["backend"] == "int8":
            out = svdq_gemm_w8a8(a, c["w8"], ascale, c["wsc"], lact, c["lup"], c["bias"], out_dtype=x.dtype)
        else:
            out = svdq_gemm_w4a8(a, c["w4"], c["wr"], ascale, c["wsc"], lact, c["lup"], c["bias"], out_dtype=x.dtype)
        return out[:m] if pad else out

    def _extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, "
            f"precision={self.precision}, act_unsigned={self.act_unsigned}, backend={self.backend}"
        )


class AWQW4A16Linear(nn.Module):
    """AWQ W4A16 projection (Nunchaku AdaNorm / modulation layers), mirroring ``nunchaku_lite.linear.AWQW4A16Linear``.

    Parameters use the checkpoint names (``qweight`` int32 ``(N/4, K/2)``, ``wscales`` and ``wzeros``
    ``(K/64, N)``, ``bias``). Nunchaku's AWQ format (``w = code * scale + zero``, group size 64) maps
    losslessly onto MLX's affine 4-bit format, so the layer runs ``mx.quantized_matmul`` at 4 bits.

    Args:
        adanorm_splits: Number of AdaNorm chunks when the checkpoint stores the layer in Nunchaku's
            AdaNorm layout (output channels interleaved across chunks, ``+1`` folded into two chunks'
            bias), e.g. 6 for FLUX.1 ``norm1.linear`` and 3 for single-block ``norm.linear``. The layer
            then produces outputs in the original ``nn.Linear`` order.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 64,
        adanorm_splits: int = 1,
        dtype=mx.bfloat16,
    ):
        super().__init__()
        if group_size != 64:
            raise NotImplementedError("Only group-size-64 AWQ W4A16 weights are supported on MLX.")
        if in_features % 64 or out_features % 4:
            raise ValueError("AWQ W4A16 layers require in_features % 64 == 0 and out_features % 4 == 0.")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.adanorm_splits = adanorm_splits
        self.qweight = mx.zeros((out_features // 4, in_features // 2), dtype=mx.int32)
        self.wscales = mx.zeros((in_features // group_size, out_features), dtype=dtype)
        self.wzeros = mx.zeros((in_features // group_size, out_features), dtype=dtype)
        self.bias = mx.zeros((out_features,), dtype=dtype) if bias else None
        self.freeze()

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 64, adanorm_splits: int = 1, dtype=None):
        out_features, in_features = linear.weight.shape
        return cls(
            in_features,
            out_features,
            bias="bias" in linear,
            group_size=group_size,
            adanorm_splits=adanorm_splits,
            dtype=dtype or linear.weight.dtype,
        )

    def _cache_for(self, dtype) -> dict:
        params = (self.qweight, self.wscales, self.wzeros, self.bias)
        cached = getattr(self, "_cache", None)
        if cached is not None and cached["dtype"] == dtype and all(a is b for a, b in zip(cached["params"], params)):
            return cached
        splits = self.adanorm_splits
        codes = mxl.undo_adanorm_channels(mxl.unpack_awq_codes(self.qweight, self.out_features), splits)  # (N, K)
        scale = mxl.undo_adanorm_channels(self.wscales.T.astype(mx.float32), splits)  # (N, G)
        zero = mxl.undo_adanorm_channels(self.wzeros.T.astype(mx.float32), splits)
        bias = None if self.bias is None else mxl.undo_adanorm_bias(self.bias.astype(mx.float32), splits)
        cached = {
            "dtype": dtype,
            "wq": _pack_nibbles_u32(codes),
            "scales": scale.astype(dtype),
            "biases": zero.astype(dtype),
            "bias": bias,
        }
        mx.eval([v for v in cached.values() if isinstance(v, mx.array)])  # drop the graph back to the checkpoint
        # Release the checkpoint copies (as SVDQW4A4Linear does): lazy re-packs of the cache, exact when evaluated.
        self.qweight = _pack_awq_qweight(_redo_adanorm_layout(_unpack_nibbles_u32(cached["wq"]), splits))
        if dtype == self.wscales.dtype:
            self.wscales = _redo_adanorm_layout(cached["scales"], splits).T
        if dtype == self.wzeros.dtype:
            self.wzeros = _redo_adanorm_layout(cached["biases"], splits).T
        cached["params"] = (self.qweight, self.wscales, self.wzeros, self.bias)
        self._cache = cached
        return cached

    def __call__(self, x: mx.array) -> mx.array:
        c = self._cache_for(x.dtype)
        out = mx.quantized_matmul(x, c["wq"], c["scales"], c["biases"], transpose=True, group_size=64, bits=4)
        if c["bias"] is not None:
            out = out + c["bias"].astype(out.dtype)
        return out

    def _extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, adanorm_splits={self.adanorm_splits}"
        )
