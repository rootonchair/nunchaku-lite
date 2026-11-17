"""Fused attention prologue for PyTorch MPS: per-head q/k RMSNorm + RoPE, compiled at runtime.

Diffusers runs the FLUX.2 q/k norm and rotary embedding as a chain of float32 elementwise ops plus concatenations;
on MPS that costs about a sixth of a FLUX.2-klein denoising step. :func:`qk_norm_rope_mps` does it in one Metal
pass. The kernel is compiled with ``torch.mps.compile_shader``, so it needs no ``nunchaku_lite_kernels`` build.
"""

from __future__ import annotations

import functools

import torch

__all__ = ["fused_qk_norm_rope_available", "qk_norm_rope_mps"]

# One simdgroup per (token, head): 32 lanes x 4 channels of a 128-wide head. Text rows (if any) come first, then
# image rows. q, k and v each have their own source pointer (they may alias one fused projection buffer), so
# separate to_q / to_k / to_v projections need no concatenation. Output is (tokens, heads, 128), contiguous.
_SRC = r"""
#include <metal_stdlib>
using namespace metal;

kernel void qk_norm_rope_T(
    device const T* qt [[buffer(0)]], device const T* kt [[buffer(1)]], device const T* vt [[buffer(2)]],
    device const T* qi [[buffer(3)]], device const T* ki [[buffer(4)]], device const T* vi [[buffer(5)]],
    device const float* wqt [[buffer(6)]], device const float* wkt [[buffer(7)]],
    device const float* wqi [[buffer(8)]], device const float* wki [[buffer(9)]],
    device const float* cosb [[buffer(10)]], device const float* sinb [[buffer(11)]],
    device T* q_out [[buffer(12)]], device T* k_out [[buffer(13)]], device T* v_out [[buffer(14)]],
    device const int* dims [[buffer(15)]], device const float* epsb [[buffer(16)]],
    uint3 tg [[threadgroup_position_in_grid]], uint3 tpt [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint head = tg.x * (tpt.x / 32) + sg;
    const uint row = tg.y;
    const int Lt = dims[0], H = dims[1], st_t = dims[2], st_i = dims[3], cs_stride = dims[4];
    const bool txt = int(row) < Lt;
    const size_t base = txt ? (size_t)row * st_t : (size_t)(row - Lt) * st_i;
    const uint c = head * 128 + lane * 4;
    const device float* wq = txt ? wqt : wqi;
    const device float* wk = txt ? wkt : wki;
    float4 q = float4(*(const device vec<T, 4>*)((txt ? qt : qi) + base + c));
    float4 k = float4(*(const device vec<T, 4>*)((txt ? kt : ki) + base + c));
    const vec<T, 4> v = *(const device vec<T, 4>*)((txt ? vt : vi) + base + c);
    // diffusers tables hold each pair's angle twice: (cos[2i], cos[2i + 1]) are equal
    const float4 cs4 = *(const device float4*)(cosb + (size_t)row * cs_stride + lane * 4);
    const float4 sn4 = *(const device float4*)(sinb + (size_t)row * cs_stride + lane * 4);
    q *= rsqrt(simd_sum(dot(q, q)) / 128.f + epsb[0]) * *(const device float4*)(wq + lane * 4);
    k *= rsqrt(simd_sum(dot(k, k)) / 128.f + epsb[1]) * *(const device float4*)(wk + lane * 4);
    const float4 qr = float4(q.x * cs4.x - q.y * sn4.x, q.y * cs4.y + q.x * sn4.y,
                             q.z * cs4.z - q.w * sn4.z, q.w * cs4.w + q.z * sn4.w);
    const float4 kr = float4(k.x * cs4.x - k.y * sn4.x, k.y * cs4.y + k.x * sn4.y,
                             k.z * cs4.z - k.w * sn4.z, k.w * cs4.w + k.z * sn4.w);
    const size_t o = ((size_t)row * H + head) * 128 + lane * 4;
    *(device vec<T, 4>*)(q_out + o) = vec<T, 4>(qr);
    *(device vec<T, 4>*)(k_out + o) = vec<T, 4>(kr);
    *(device vec<T, 4>*)(v_out + o) = v;
}
"""

_TYPES = {torch.float16: "half", torch.bfloat16: "bfloat", torch.float32: "float"}


@functools.cache
def _library():
    header, body = _SRC.split("kernel void", 1)
    body = "kernel void" + body
    return torch.mps.compile_shader(
        header
        + "".join(
            body.replace("_T(", f"_{name}(").replace("T*", f"{name}*").replace("<T,", f"<{name},")
            for name in _TYPES.values()
        )
    )


@functools.cache
def fused_qk_norm_rope_available() -> bool:
    """Whether the runtime-compiled MPS kernel builds on this machine."""

    if not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")):
        return False
    try:
        _library()
    except Exception:  # noqa: BLE001 - any compile failure means: use the unfused path
        return False
    return True


def _rows(t: torch.Tensor) -> tuple[torch.Tensor, int]:
    """A ``(..., L, C)`` projection output (or a column slice of one) as its base tensor and row stride."""

    t2 = t.reshape(-1, t.shape[-1]) if t.is_contiguous() else t.flatten(0, -2)
    if t2.stride(-1) != 1:
        t2 = t2.contiguous()
    return t2, t2.stride(0)


def qk_norm_rope_mps(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    *,
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

    dtype = q.dtype
    width = heads * 128
    srcs = []
    for group in (text, (q, k, v)):
        if group is None:
            continue
        rows = [_rows(t) for t in group]
        if len({s for _, s in rows}) != 1 or any(t.shape[-1] != width or t.dtype != dtype for t in group):
            raise ValueError("q, k and v must share dtype, width heads * 128 and row stride.")
        srcs.append(rows)
    img = srcs[-1]
    txt = srcs[0] if text is not None else img
    lt = txt[0][0].shape[0] if text is not None else 0
    total = lt + img[0][0].shape[0]
    if cos.shape != (total, 128) or sin.shape != (total, 128):
        raise ValueError(f"Rotary tables must be ({total}, 128), got {tuple(cos.shape)} and {tuple(sin.shape)}.")

    device = q.device
    ones = torch.ones(128, device=device, dtype=torch.float32)

    def weight(w):
        return ones if w is None else w.to(device=device, dtype=torch.float32).contiguous()

    wq, wk = weight(q_weight), weight(k_weight)
    wqt, wkt = (weight(text_q_weight), weight(text_k_weight)) if text is not None else (wq, wk)
    cos32 = cos.to(device=device, dtype=torch.float32).contiguous()
    sin32 = sin.to(device=device, dtype=torch.float32).contiguous()
    dims = torch.tensor([lt, heads, txt[0][1], img[0][1], 128], dtype=torch.int32, device=device)
    eps = torch.tensor([eps_q, eps_k], dtype=torch.float32, device=device)
    out = [torch.empty((1, total, heads, 128), dtype=dtype, device=device) for _ in range(3)]
    group = next(g for g in (8, 4, 2, 1) if heads % g == 0)
    kernel = getattr(_library(), f"qk_norm_rope_{_TYPES[dtype]}")
    kernel(
        txt[0][0],
        txt[1][0],
        txt[2][0],
        img[0][0],
        img[1][0],
        img[2][0],
        wqt,
        wkt,
        wq,
        wk,
        cos32,
        sin32,
        *out,
        dims,
        eps,
        threads=(heads * 32, total, 1),
        group_size=(32 * group, 1, 1),
    )
    return tuple(out)
