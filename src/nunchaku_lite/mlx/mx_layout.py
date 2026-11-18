"""MLX (GPU) versions of the checkpoint -> kernel layout conversions in :mod:`.packing`.

:mod:`.packing` holds the NumPy reference implementations. The functions here produce bit-identical results with
lazy MLX ops, so converting a layer runs on the GPU instead of round-tripping through NumPy (on FLUX.1-dev: 53 s
of first-step conversion in NumPy). ``tests/mlx_runtime/test_mlx_linear.py`` checks them against the references.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .packing import _LR_PACK, _LR_PERM, _S_PERM, _W_PERM, _scale_dims, adanorm_scale_chunks

_W_PERM_INV = tuple(int(i) for i in np.argsort(_W_PERM))
_S_PERM_INV = tuple(int(i) for i in np.argsort(_S_PERM))
_LR_PERM_INV = tuple(int(i) for i in np.argsort(_LR_PERM))

_AWQ_CHANNEL = np.arange(64)
_AWQ_PACKED_INDEX = mx.array((_AWQ_CHANNEL // 32) * 4 + (_AWQ_CHANNEL % 8) // 2, dtype=mx.int32)
_AWQ_SHIFT = mx.array(4 * (((_AWQ_CHANNEL % 32) // 8) + 4 * (_AWQ_CHANNEL % 2)), dtype=mx.uint32)
_SHIFTS4 = mx.arange(0, 32, 4, dtype=mx.uint32)


def unpack_weight(qweight: mx.array) -> mx.array:
    """Nunchaku SVDQ ``qweight`` ``(N, K // 2)`` int8 -> signed int4 values ``(N, K)`` int32."""

    n, half_k = qweight.shape
    k = half_k * 2
    words = qweight.view(mx.uint32).reshape(n // 128, k // 64, 1, 8, 8, 4, 2, 2, 1)
    nib = ((words[..., None] >> _SHIFTS4) & 0xF).astype(mx.int32)
    return ((nib ^ 8) - 8).transpose(_W_PERM_INV).reshape(n, k)


def unpack_scale(packed: mx.array, n: int) -> mx.array:
    """Nunchaku-ordered scales (``wscales``, ``bias``, ``smooth_factor``) -> float32 ``(N, G)``."""

    groups = packed.size // n
    packs, lanes4, half_pack, blocks = _scale_dims(n)
    s = packed.astype(mx.float32).reshape(blocks, groups, packs, lanes4, 4, half_pack, 2)
    return s.transpose(_S_PERM_INV).reshape(n, groups)


def unpack_lowrank(packed: mx.array, down: bool) -> mx.array:
    """Low-rank weight in Nunchaku order -> float32; ``down``: ``(K, R)`` -> ``(R, K)``, up: ``(N, R)`` -> ``(N, R)``."""

    c, r = packed.shape
    p = _LR_PACK
    w = packed.astype(mx.float32).reshape(c // p, r // p, 8, 4, 2, 2, 1, 2).transpose(_LR_PERM_INV)
    w = w.reshape(c // p, r // p, p, p)
    return w.transpose(1, 2, 0, 3).reshape(r, c) if down else w.transpose(0, 2, 1, 3).reshape(c, r)


def kernel_scales(wscale: mx.array) -> tuple[mx.array, mx.array]:
    """Per-channel scale ``wsc`` float32 ``(N,)`` and group multipliers ``wr`` float16 ``(N, G)``
    (as :func:`.packing.to_kernel_layout`)."""

    wsc = wscale.max(axis=1) * (8.0 / 127.0)
    wsc = mx.where(wsc > 0, wsc, mx.array(1.0, dtype=mx.float32))
    return wsc, (wscale / wsc[:, None]).astype(mx.float16)


def kernel_w8(q: mx.array, wr: mx.array) -> mx.array:
    """``rint(q * wr)`` int8 ``(N, K)`` (as :func:`.packing.to_kernel_layout`)."""

    n, k = q.shape
    w8 = mx.round(q.astype(mx.float32).reshape(n, k // 64, 64) * wr.astype(mx.float32)[..., None])
    return mx.clip(w8, -127, 127).astype(mx.int8).reshape(n, k)


def kernel_w4(q: mx.array) -> mx.array:
    """Signed int4 ``(N, K)`` -> uint8 ``(N, K // 2)``, low nibble = even input channel."""

    n, k = q.shape
    nib = (q & 0xF).astype(mx.uint8).reshape(n, k // 2, 2)
    return nib[..., 0] | (nib[..., 1] << 4)


def unpack_awq_codes(qweight: mx.array, out_features: int) -> mx.array:
    """Nunchaku AWQ int32 ``qweight`` ``(N / 4, K / 2)`` -> unsigned codes ``(N, K)`` uint32."""

    rows = qweight.shape[0]
    groups = qweight.shape[1] * 2 // 64
    packed = qweight.view(mx.uint32).reshape(rows, groups, 4, 8)
    codes = (mx.take(packed, _AWQ_PACKED_INDEX, axis=-1) >> _AWQ_SHIFT) & 0xF  # (rows, groups, 4, 64)
    return codes.transpose(0, 2, 1, 3).reshape(out_features, groups * 64)


def undo_adanorm_channels(t: mx.array, splits: int) -> mx.array:
    """Restore ``nn.Linear`` output order from Nunchaku's AdaNorm interleave (as :func:`.packing.undo_adanorm_layout`)."""

    if splits <= 1:
        return t
    n = t.shape[0]
    return t.reshape(n // splits, splits, *t.shape[1:]).swapaxes(0, 1).reshape(t.shape)


def undo_adanorm_bias(bias: mx.array, splits: int) -> mx.array:
    """AdaNorm bias: drop the ``+1`` folded into the scale chunks, then restore the channel order."""

    if splits <= 1:
        return bias
    offset = np.zeros(splits, dtype=np.float32)
    offset[adanorm_scale_chunks(splits)] = 1
    n = bias.shape[0]
    return (bias.reshape(n // splits, splits) - mx.array(offset)).T.reshape(n)
