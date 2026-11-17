"""Metal kernel sources shared by the MLX runtime and the PyTorch MPS runtime.

This module imports neither framework, so ``nunchaku_lite.mlx`` stays torch-free and ``nunchaku_lite.ops.mps``
stays mlx-free. Each source is a kernel *body*. MLX wraps it with ``mx.fast.metal_kernel`` (which generates the
signature from input and output names); PyTorch compiles :func:`mps_library_source` with
``torch.mps.compile_shader``. Both see the same buffer names, the same ``T`` element type and the same
thread-position names.
"""

from __future__ import annotations

__all__ = ["QK_NORM_ROPE", "QK_NORM_ROPE_INPUTS", "QK_NORM_ROPE_OUTPUTS", "mps_library_source"]

QK_NORM_ROPE_INPUTS = ("qt", "kt", "vt", "qi", "ki", "vi", "wqt", "wkt", "wqi", "wki", "cosb", "sinb", "epsb", "dims")
QK_NORM_ROPE_OUTPUTS = ("q_out", "k_out", "v_out")

QK_NORM_ROPE = r"""
    // Attention prologue for 128-wide heads: per-head RMSNorm of q and k, interleaved-pair RoPE (pair i is
    // channels 2i, 2i + 1), and a layout change, in one pass. One simdgroup per (row, head): 32 lanes x 4 channels.
    // Rows below Lt are text rows (sources qt/kt/vt, norm weights wqt/wkt); the rest are image rows (qi/ki/vi,
    // wqi/wki). q, k and v may alias one projection buffer (column offsets qo/ko/vo) and the outputs may alias the
    // inputs (in place): every lane reads its channels before writing them.
    //
    // dims: [Lt, st_t, st_i, qo, ko, vo, rot_rows, cs_row, cs_pair, os_h, os_l, copy_v]
    //   st_t / st_i       row strides of the text / image sources
    //   rot_rows          rotary table rows; row r uses table row r % rot_rows (flattened batches)
    //   cs_row / cs_pair  rotary table strides between rows and between pairs (1: one value per pair,
    //                     2: Diffusers' tables, which store each pair's value twice)
    //   os_h / os_l       output strides of a head and of a row (head-major or row-major outputs)
    //   copy_v            0 when v_out is v itself (in place): v is left untouched
    // epsb: [eps_q, eps_k]
    const uint lane = thread_index_in_simdgroup;
    const uint head = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.y;
    const int Lt = dims[0], st_t = dims[1], st_i = dims[2], qo = dims[3], ko = dims[4], vo = dims[5];
    const int rot_rows = dims[6], cs_row = dims[7], cs_pair = dims[8], os_h = dims[9], os_l = dims[10];
    const bool copy_v = dims[11] != 0;
    const bool txt = int(row) < Lt;
    const size_t base = txt ? (size_t)row * st_t : (size_t)(int(row) - Lt) * st_i;
    const uint c = head * 128 + lane * 4;
    const device float* wq = txt ? wqt : wqi;
    const device float* wk = txt ? wkt : wki;
    float4 q = float4(*(const device vec<T, 4>*)((txt ? qt : qi) + base + qo + c));
    float4 k = float4(*(const device vec<T, 4>*)((txt ? kt : ki) + base + ko + c));
    const size_t r = (size_t)(int(row) % rot_rows) * cs_row + (size_t)lane * 2 * cs_pair;
    const float c0 = cosb[r], c1 = cosb[r + cs_pair], s0 = sinb[r], s1 = sinb[r + cs_pair];
    q *= rsqrt(simd_sum(dot(q, q)) / 128.f + epsb[0]) * *(const device float4*)(wq + lane * 4);
    k *= rsqrt(simd_sum(dot(k, k)) / 128.f + epsb[1]) * *(const device float4*)(wk + lane * 4);
    const float4 qr = float4(q.x * c0 - q.y * s0, q.y * c0 + q.x * s0, q.z * c1 - q.w * s1, q.w * c1 + q.z * s1);
    const float4 kr = float4(k.x * c0 - k.y * s0, k.y * c0 + k.x * s0, k.z * c1 - k.w * s1, k.w * c1 + k.z * s1);
    const size_t o = (size_t)head * os_h + (size_t)row * os_l + lane * 4;
    *(device vec<T, 4>*)(q_out + o) = vec<T, 4>(qr);
    *(device vec<T, 4>*)(k_out + o) = vec<T, 4>(kr);
    if (copy_v) {
        *(device vec<T, 4>*)(v_out + o) = *(const device vec<T, 4>*)((txt ? vt : vi) + base + vo + c);
    }
"""

_THREAD_ARGS = (
    "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
    "uint3 threads_per_threadgroup [[threads_per_threadgroup]]",
    "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]",
    "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]",
)


def mps_library_source(
    name: str,
    body: str,
    inputs: dict[str, str],
    outputs: tuple[str, ...],
    types: dict[str, str],
) -> str:
    """A ``torch.mps.compile_shader`` library: ``body`` as kernel ``{name}_{type}`` for each Metal type in ``types``.

    Args:
        name: Kernel base name.
        body: Kernel body (as passed to ``mx.fast.metal_kernel``).
        inputs: Input buffer name -> element type (``"T"`` for the templated type).
        outputs: Output buffer names (element type ``T``).
        types: Host name suffix -> Metal type, e.g. ``{"bfloat16": "bfloat"}``.

    Buffers are bound in order: inputs, then outputs.
    """

    params = [f"const device {t}* {n} [[buffer({i})]]" for i, (n, t) in enumerate(inputs.items())]
    params += [f"device T* {n} [[buffer({len(inputs) + i})]]" for i, n in enumerate(outputs)]
    signature = ",\n    ".join([*params, *_THREAD_ARGS])
    source = (
        "#include <metal_stdlib>\nusing namespace metal;\n\n"
        f"template <typename T>\n[[kernel]] void {name}(\n    {signature}) {{{body}}}\n\n"
    )
    for suffix, metal_type in types.items():
        source += (
            f'template [[host_name("{name}_{suffix}")]] [[kernel]] decltype({name}<{metal_type}>) '
            f"{name}<{metal_type}>;\n"
        )
    return source
