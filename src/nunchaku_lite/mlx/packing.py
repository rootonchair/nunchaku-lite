"""Convert Nunchaku SVDQ INT4 checkpoint tensors to logical and kernel layouts.

Nunchaku stores SVDQ W4A4 tensors in CUDA MMA fragment order. The layouts below
mirror the reference packer, ``NunchakuWeightPacker`` with ``bits=4`` and
``warp_n=128``, from DeepCompressor's ``backend/nunchaku/utils.py``. The
``pack_*`` functions reproduce that packer and exist for round-trip tests.
The ``unpack_*`` functions are their exact inverses.

All functions operate on NumPy arrays so the layout logic is device-independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SVDQLogicalWeights",
    "pack_awq_w4",
    "pack_lowrank",
    "pack_scale",
    "pack_svdq_int4",
    "pack_weight",
    "to_kernel_layout",
    "undo_adanorm_layout",
    "unpack_awq_w4",
    "unpack_lowrank",
    "unpack_scale",
    "unpack_svdq_int4",
    "unpack_weight",
]

_WARP_N = 128  # mem_n: output channels per weight tile
_MEM_K = 64  # mem_k: input channels per weight tile
_W_SHAPE_TAIL = (8, 2, 8, 1)  # num_n_packs, n_pack_size, num_n_lanes, reg_n
_W_K_TAIL = (1, 2, 4, 8)  # num_k_packs, k_pack_size, num_k_lanes, reg_k
_W_PERM = (0, 5, 6, 1, 3, 8, 2, 7, 4, 9)
_S_PERM = (0, 6, 1, 2, 4, 3, 5)
_LR_PERM = (0, 1, 3, 6, 2, 5, 4, 7)
_LR_PACK = 16


def pack_weight(q: np.ndarray) -> np.ndarray:
    """Pack signed int4 weights ``(N, K)`` in ``[-8, 7]`` into Nunchaku ``qweight`` ``(N, K // 2)`` int8."""

    n, k = q.shape
    if n % _WARP_N or k % (2 * _MEM_K):
        raise ValueError(f"weight shape {q.shape} must be divisible by ({_WARP_N}, {2 * _MEM_K}).")
    w = q.astype(np.int64).reshape(n // _WARP_N, *_W_SHAPE_TAIL, k // _MEM_K, *_W_K_TAIL)
    w = w.transpose(_W_PERM) & 0xF
    words = (w << (4 * np.arange(8, dtype=np.int64))).sum(axis=-1).astype(np.uint32)
    return np.ascontiguousarray(words).view(np.int8).reshape(n, k // 2)


def unpack_weight(qweight: np.ndarray) -> np.ndarray:
    """Unpack Nunchaku ``qweight`` ``(N, K // 2)`` into signed int4 values ``(N, K)`` as int8."""

    n, half_k = qweight.shape
    k = half_k * 2
    if n % _WARP_N or k % (2 * _MEM_K):
        raise ValueError(f"qweight shape {qweight.shape} is not a valid Nunchaku INT4 layout.")
    words = np.ascontiguousarray(qweight).view(np.uint32)
    words = words.reshape(n // _WARP_N, k // _MEM_K, 1, 8, 8, 4, 2, 2, 1)
    nib = (words[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 0xF
    signed = ((nib.astype(np.int8) ^ 8) - 8).astype(np.int8)
    return np.ascontiguousarray(signed.transpose(np.argsort(_W_PERM))).reshape(n, k)


def _scale_dims(n: int) -> tuple[int, int, int, int]:
    s_pack_size = min(max(_WARP_N // 32, 2), 8)
    num_s_lanes = min(32, _WARP_N // s_pack_size)
    num_s_packs = _WARP_N // (s_pack_size * num_s_lanes)
    if n % _WARP_N:
        raise ValueError(f"scale length {n} must be divisible by {_WARP_N}.")
    return num_s_packs, num_s_lanes // 4, s_pack_size // 2, n // _WARP_N


def pack_scale(scale: np.ndarray) -> np.ndarray:
    """Pack logical scales ``(N, G)`` into Nunchaku order: ``(G, N)`` when G > 1, ``(N,)`` when G == 1."""

    n, groups = scale.shape
    packs, lanes4, half_pack, blocks = _scale_dims(n)
    s = scale.reshape(blocks, packs, lanes4, half_pack, 4, 2, groups).transpose(_S_PERM)
    s = np.ascontiguousarray(s)
    return s.reshape(-1) if groups == 1 else s.reshape(groups, n)


def unpack_scale(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack Nunchaku-ordered scales (``wscales``, ``bias``, ``smooth_factor``) into ``(N, G)``."""

    groups = packed.size // n
    packs, lanes4, half_pack, blocks = _scale_dims(n)
    s = packed.reshape(blocks, groups, packs, lanes4, 4, half_pack, 2).transpose(np.argsort(_S_PERM))
    return np.ascontiguousarray(s).reshape(n, groups)


def pack_lowrank(weight: np.ndarray, down: bool) -> np.ndarray:
    """Pack a low-rank weight. ``down``: ``(R, K)`` -> ``(K, R)``. Up: ``(N, R)`` -> ``(N, R)``."""

    if down:
        r, c = weight.shape
        w = weight.reshape(r // _LR_PACK, _LR_PACK, c // _LR_PACK, _LR_PACK).transpose(2, 0, 1, 3)
    else:
        c, r = weight.shape
        w = weight.reshape(c // _LR_PACK, _LR_PACK, r // _LR_PACK, _LR_PACK).transpose(0, 2, 1, 3)
    w = w.reshape(c // _LR_PACK, r // _LR_PACK, 2, 8, 1, 2, 4, 2).transpose(_LR_PERM)
    return np.ascontiguousarray(w).reshape(c, r)


def unpack_lowrank(packed: np.ndarray, down: bool) -> np.ndarray:
    """Unpack a low-rank weight. ``down``: ``(K, R)`` -> ``(R, K)``. Up: ``(N, R)`` -> ``(N, R)``."""

    c, r = packed.shape
    w = packed.reshape(c // _LR_PACK, r // _LR_PACK, 8, 4, 2, 2, 1, 2).transpose(np.argsort(_LR_PERM))
    w = np.ascontiguousarray(w).reshape(c // _LR_PACK, r // _LR_PACK, _LR_PACK, _LR_PACK)
    if down:
        return np.ascontiguousarray(w.transpose(1, 2, 0, 3)).reshape(r, c)
    return np.ascontiguousarray(w.transpose(0, 2, 1, 3)).reshape(c, r)


@dataclass
class SVDQLogicalWeights:
    """SVDQ INT4 linear weights in plain row-major layout.

    The projection is ``y = (q * wscale) @ ((x + shift) / smooth) + up @ (down @ x) + bias``.
    """

    q: np.ndarray  # int8 (N, K) in [-8, 7]
    wscale: np.ndarray  # float32 (N, K // 64)
    smooth: np.ndarray  # float32 (K,)
    down: np.ndarray  # float32 (R, K)
    up: np.ndarray  # float32 (N, R)
    bias: np.ndarray | None  # float32 (N,)

    @property
    def out_features(self) -> int:
        return self.q.shape[0]

    @property
    def in_features(self) -> int:
        return self.q.shape[1]

    def dequantize(self) -> np.ndarray:
        """Return the dense smoothed 4-bit weight ``(N, K)`` in float32."""

        n, k = self.q.shape
        w = self.q.astype(np.float32).reshape(n, k // 64, 64) * self.wscale[..., None]
        return w.reshape(n, k)


def unpack_svdq_int4(
    qweight: np.ndarray,
    wscales: np.ndarray,
    smooth_factor: np.ndarray,
    proj_down: np.ndarray,
    proj_up: np.ndarray,
    bias: np.ndarray | None = None,
) -> SVDQLogicalWeights:
    """Convert Nunchaku INT4 SVDQ checkpoint tensors to :class:`SVDQLogicalWeights`.

    Args:
        qweight: int8 ``(N, K // 2)``.
        wscales: ``(K // 64, N)`` in Nunchaku scale order (any float dtype).
        smooth_factor: ``(K,)`` in Nunchaku scale order.
        proj_down: ``(K, R)`` in Nunchaku low-rank order.
        proj_up: ``(N, R)`` in Nunchaku low-rank order.
        bias: Optional ``(N,)`` in Nunchaku scale order.
    """

    n = qweight.shape[0]
    k = qweight.shape[1] * 2
    if wscales.size != n * (k // 64):
        raise ValueError(f"wscales size {wscales.size} does not match INT4 group-64 weights of shape ({n}, {k}).")
    return SVDQLogicalWeights(
        q=unpack_weight(qweight),
        wscale=unpack_scale(np.asarray(wscales, dtype=np.float32), n),
        smooth=unpack_scale(np.asarray(smooth_factor, dtype=np.float32).reshape(-1), k).reshape(k),
        down=unpack_lowrank(np.asarray(proj_down, dtype=np.float32), down=True),
        up=unpack_lowrank(np.asarray(proj_up, dtype=np.float32), down=False),
        bias=None if bias is None else unpack_scale(np.asarray(bias, dtype=np.float32).reshape(-1), n).reshape(n),
    )


def to_kernel_layout(w: SVDQLogicalWeights) -> dict[str, np.ndarray]:
    """Fold group scales into per-channel scales plus integer-domain group multipliers.

    Returns arrays used by :mod:`nunchaku_lite.mlx.kernels`:

    * ``wsc`` float32 ``(N,)``: per-channel scale, ``max_g wscale * 8 / 127``.
    * ``wr`` float16 ``(K // 64, N)``: group multiplier ``wscale / wsc``, at most 15.875.
    * ``w8`` int8 ``(N, K)``: ``rint(q * wr)``, bit-identical to the in-kernel int4 expansion.
    * ``w4`` uint8 ``(N, K // 2)``: signed int4, low nibble = even input channel.
    """

    n, k = w.q.shape
    wscale = w.wscale.astype(np.float32)
    wsc = wscale.max(axis=1) * (8.0 / 127.0)
    wsc = np.where(wsc > 0, wsc, 1.0).astype(np.float32)
    wr = (wscale / wsc[:, None]).astype(np.float16)
    w8 = np.rint(w.q.astype(np.float32).reshape(n, k // 64, 64) * wr.astype(np.float32)[..., None])
    w8 = np.clip(w8, -127, 127).astype(np.int8).reshape(n, k)
    nib = (w.q.astype(np.int16) & 0xF).astype(np.uint8).reshape(n, k // 2, 2)
    w4 = (nib[..., 0] | (nib[..., 1] << 4)).astype(np.uint8)
    return {"wsc": wsc, "wr": np.ascontiguousarray(wr.T), "w8": w8, "w4": w4}


def pack_svdq_int4(w: SVDQLogicalWeights) -> dict[str, np.ndarray]:
    """Inverse of :func:`unpack_svdq_int4`: logical weights -> Nunchaku checkpoint tensors (float32 scales)."""

    out = {
        "qweight": pack_weight(w.q),
        "wscales": pack_scale(w.wscale.astype(np.float32)),
        "smooth_factor": pack_scale(w.smooth.astype(np.float32)[:, None]),
        "proj_down": pack_lowrank(w.down.astype(np.float32), down=True),
        "proj_up": pack_lowrank(w.up.astype(np.float32), down=False),
    }
    if w.bias is not None:
        out["bias"] = pack_scale(w.bias.astype(np.float32)[:, None])
    return out


# ---------------------------------------------------------------------------
# AWQ W4A16 (TinyChat layout, group size 64): Nunchaku AdaNorm / modulation layers
# ---------------------------------------------------------------------------

_AWQ_CHANNEL = np.arange(64)
_AWQ_PACKED_INDEX = (_AWQ_CHANNEL // 32) * 4 + (_AWQ_CHANNEL % 8) // 2
_AWQ_NIBBLE = ((_AWQ_CHANNEL % 32) // 8) + 4 * (_AWQ_CHANNEL % 2)


def pack_awq_w4(codes: np.ndarray) -> np.ndarray:
    """Pack unsigned 4-bit codes ``(N, K)`` like DeepCompressor's TinyChat ``pack_w4``; returns int32 ``(N/4, K/2)``."""

    oc, ic = codes.shape
    w = codes.astype(np.int32).reshape(-1, 4, 8)
    w = w[:, 0] | (w[:, 1] << 4) | (w[:, 2] << 8) | (w[:, 3] << 12)
    w = w.reshape(oc // 4, 4, ic // 64, 16).transpose(0, 2, 1, 3).reshape(oc // 4, ic)
    return np.ascontiguousarray(w.astype(np.int16)).view(np.int32)


def unpack_awq_w4(qweight: np.ndarray, out_features: int) -> np.ndarray:
    """Unpack Nunchaku AWQ int32 ``qweight`` ``(N/4, K/2)`` into unsigned codes ``(N, K)`` uint8."""

    rows = qweight.shape[0]
    groups = qweight.shape[1] * 2 // 64
    packed = np.ascontiguousarray(qweight).view(np.int32).reshape(rows, groups, 4, 8)
    selected = packed[..., _AWQ_PACKED_INDEX]  # (rows, groups, 4, 64)
    codes = (selected >> (4 * _AWQ_NIBBLE)) & 0xF
    return codes.transpose(0, 2, 1, 3).reshape(out_features, groups * 64).astype(np.uint8)


def adanorm_scale_chunks(splits: int) -> list[int]:
    """AdaNorm chunks holding a scale with ``+1`` folded in: ``scale_msa`` (1) and, for 6 splits, ``scale_mlp`` (4).
    With 3 splits (``shift, scale, gate``) chunk ``-2`` *is* chunk 1, so the ``+1`` is folded in once."""

    return sorted({1, splits - 2})


def undo_adanorm_layout(
    codes: np.ndarray, scale: np.ndarray, zero: np.ndarray, bias: np.ndarray | None, splits: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Invert Nunchaku's AdaNorm AWQ layout (output channels interleaved across ``splits`` chunks,
    ``+1`` folded into the bias of each scale chunk, see :func:`adanorm_scale_chunks`), restoring the original ``nn.Linear`` order."""

    if splits <= 1:
        return codes, scale, zero, bias
    n = codes.shape[0]

    def unperm(t):
        return np.ascontiguousarray(t.reshape(n // splits, splits, *t.shape[1:]).swapaxes(0, 1).reshape(t.shape))

    if bias is not None:
        b = bias.reshape(n // splits, splits).copy()
        b[:, adanorm_scale_chunks(splits)] -= 1
        bias = np.ascontiguousarray(b.T).reshape(n)
    return unperm(codes), unperm(scale), unperm(zero), bias
