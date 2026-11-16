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

from .kernels import TILE_M, has_neural_accelerators, svdq_gemm_w4a8, svdq_gemm_w8a8, svdq_quantize_act_int8
from .packing import to_kernel_layout, undo_adanorm_layout, unpack_awq_w4, unpack_svdq_int4

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
        if cached.pop("q_invertible"):
            # Release the packed checkpoint copy: qweight becomes a lazy re-pack of the backend copy, costing no
            # memory until evaluated (e.g. when saving weights). Evaluating it reproduces the exact packed bytes.
            self.qweight = _pack_svdq_qweight(self._q_from_cache(cached))
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
        weights = unpack_svdq_int4(
            qweight=np.array(self.qweight).astype(np.int8),
            wscales=_to_numpy(self.wscales),
            smooth_factor=_to_numpy(self.smooth_factor),
            proj_down=_to_numpy(self.proj_down),
            proj_up=_to_numpy(self.proj_up),
            bias=None if self.bias is None else _to_numpy(self.bias),
        )
        rank = weights.down.shape[0]
        rank_pad = -rank % 32 if rank else 32  # kernels need a non-empty rank that is a multiple of 32
        down = np.pad(weights.down, ((0, rank_pad), (0, 0)))
        up = np.pad(weights.up, ((0, 0), (0, rank_pad)))
        cache = {
            "backend": backend,
            "dtype": dtype,
            "shift": mx.array([GELU_SHIFT if self.act_unsigned else 0.0], dtype=mx.float32),
            "inv_smooth": mx.array(1.0 / weights.smooth.astype(np.float32)),
            "down_t": mx.array(np.ascontiguousarray(down.T)).astype(dtype),  # (K, R)
            "bias": None if weights.bias is None else mx.array(weights.bias.astype(np.float32)),
        }
        if backend == "dense":
            n, k = weights.q.shape
            # MLX affine 4-bit: w = scale * u + bias, with u = q + 8 in [0, 15] (lossless).
            u = (weights.q.astype(np.int16) + 8).astype(np.uint32).reshape(n, k // 8, 8)
            packed = np.zeros((n, k // 8), dtype=np.uint32)
            for i in range(8):
                packed |= u[..., i] << (4 * i)
            cache["qw"] = mx.array(packed)
            cache["qscales"] = mx.array(weights.wscale.astype(np.float32)).astype(dtype)
            cache["qbiases"] = (-8.0 * cache["qscales"].astype(mx.float32)).astype(dtype)
            cache["up_t"] = mx.array(np.ascontiguousarray(up.T)).astype(dtype)  # (R, N)
            cache["q_invertible"] = True  # the affine packing stores q + 8 exactly
            return cache

        lr_dtype = mx.bfloat16 if dtype == mx.bfloat16 else mx.float16  # low-rank runs on the accelerators
        if lr_dtype == mx.float16 and np.abs(up).max() > 65504:
            raise ValueError("proj_up values exceed float16 range.")
        layout = to_kernel_layout(weights)
        cache["lr_dtype"] = lr_dtype
        cache["lup"] = mx.array(up.astype(np.float32)).astype(lr_dtype)  # (N, R)
        cache["wsc"] = mx.array(layout["wsc"])
        cache["wr"] = mx.array(layout["wr"])
        if backend == "int8":
            cache["w8"] = mx.array(layout["w8"])
            n, k = weights.q.shape
            recovered = np.rint(
                layout["w8"].astype(np.float32).reshape(n, k // 64, 64) / layout["wr"].T.astype(np.float32)[..., None]
            ).reshape(n, k)
            cache["q_invertible"] = bool(np.array_equal(recovered, weights.q))  # every group multiplier >= 1
        else:
            cache["w4"] = mx.array(layout["w4"])
            cache["q_invertible"] = True
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
        out = out + (x @ c["down_t"]) @ c["up_t"]
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
        codes = unpack_awq_w4(np.array(self.qweight).astype(np.int32), self.out_features)
        scale = _to_numpy(self.wscales).T.astype(np.float32)  # (N, G)
        zero = _to_numpy(self.wzeros).T.astype(np.float32)
        bias = None if self.bias is None else _to_numpy(self.bias).astype(np.float32)
        codes, scale, zero, bias = undo_adanorm_layout(codes, scale, zero, bias, self.adanorm_splits)
        cached = {
            "dtype": dtype,
            "wq": _pack_nibbles_u32(mx.array(codes)),
            "scales": mx.array(scale).astype(dtype),
            "biases": mx.array(zero).astype(dtype),
            "bias": None if bias is None else mx.array(bias),
        }
        cached["params"] = params
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
