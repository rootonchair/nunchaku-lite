"""Activation quantization wrappers exposed by the active backend."""

import torch

from ..utils import ceil_divide
from .backend import get_ops

HADAMARD_BLOCK_SIZE = 64


def _alloc_quantize_buffers(
    input: torch.Tensor,
    output: torch.Tensor | None,
    oscales: torch.Tensor | None,
    lora_down: torch.Tensor,
    lora_act_out: torch.Tensor | None,
    fp4: bool,
    pad_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, channels = input.shape
    rank = lora_down.shape[1]
    batch_size_pad = ceil_divide(batch_size, pad_size) * pad_size
    if output is None:
        output = torch.empty(batch_size_pad, channels // 2, dtype=torch.uint8, device=input.device)
    if oscales is None:
        if fp4:
            if channels % 16 != 0:
                raise ValueError("NVFP4 activation channels must be divisible by 16")
            oscales = torch.empty(channels // 16, batch_size_pad, dtype=torch.float8_e4m3fn, device=input.device)
        else:
            if channels % 64 != 0:
                raise ValueError("INT4 activation channels must be divisible by 64")
            oscales = torch.empty(channels // 64, batch_size_pad, dtype=input.dtype, device=input.device)
    if lora_act_out is None:
        lora_act_out = torch.empty(batch_size_pad, rank, dtype=torch.float32, device=input.device)
    return output, oscales, lora_act_out


def svdq_quantize_w4a4_act_fuse_lora_cuda(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    fuse_glu: bool = False,
    fp4: bool = False,
    pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize W4A4 activations and optionally fuse smooth and LoRA-down paths.

    Args:
        input: Flattened activation tensor with shape ``(tokens, channels)``.
        output: Optional preallocated quantized activation buffer.
        oscales: Optional preallocated activation-scale buffer.
        lora_down: Low-rank down-projection used to compute side activations.
        lora_act_out: Optional preallocated low-rank activation output.
        smooth: Optional smooth factor applied during quantization.
        fuse_glu: Whether the native quantizer should fuse GLU handling.
        fp4: Whether to use the NVFP4 activation scale layout.
        pad_size: Token padding multiple required by the kernel.

    Returns:
        Tuple of ``(quantized_output, output_scales, lora_activation_output)``.

    Raises:
        ValueError: If channel dimensions are incompatible with the selected
            precision layout.
    """

    output, oscales, lora_act_out = _alloc_quantize_buffers(
        input, output, oscales, lora_down, lora_act_out, fp4, pad_size
    )
    get_ops().quantize_w4a4_act_fuse_lora(input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4)
    return output, oscales, lora_act_out


def svdq_quantize_w4a4_act_hadamard_fuse_lora_cuda(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    lora_down: torch.Tensor | None = None,
    lora_act_out: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    fuse_glu: bool = False,
    fp4: bool = False,
    pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize W4A4 activations with an online blockwise Hadamard rotation.

    Same as :func:`svdq_quantize_w4a4_act_fuse_lora_cuda`, except that each
    64-channel block of the smoothed activations is multiplied by the orthonormal
    Sylvester Hadamard matrix ``H64 / 8`` (see :func:`hadamard_matrix`) before
    quantization. The low-rank branch still consumes the unrotated input. The
    matching weight residual must be packed as ``Q(R @ blockdiag(H64 / 8))``.

    The op itself accepts ``fp4=True``; refusing NVFP4 + rotation by default is
    the loader's job, since rotation measured slightly worse than none for NVFP4.

    Raises:
        ValueError: If the channel count is not a multiple of 64.
        NotImplementedError: If the active kernel backend predates this op.
    """

    if input.shape[-1] % HADAMARD_BLOCK_SIZE != 0:
        raise ValueError("Hadamard activation quantization needs channels divisible by 64")
    ops = get_ops()
    op = getattr(ops, "quantize_w4a4_act_hadamard_fuse_lora", None)
    if op is None:
        raise NotImplementedError(
            "The active Nunchaku Lite kernel backend has no quantize_w4a4_act_hadamard_fuse_lora op. "
            'Rotated checkpoints need a kernel build that includes it: run "pip install ./nunchaku-lite-kernels".'
        )
    output, oscales, lora_act_out = _alloc_quantize_buffers(
        input, output, oscales, lora_down, lora_act_out, fp4, pad_size
    )
    op(input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4)
    return output, oscales, lora_act_out


def hadamard_matrix(
    size: int = HADAMARD_BLOCK_SIZE, dtype: torch.dtype = torch.float32, device=None
) -> torch.Tensor:
    """Return the orthonormal Sylvester Hadamard matrix ``H_size / sqrt(size)``.

    This is the exact matrix the Hadamard quantize op applies per 64-channel block
    (Sylvester order: ``H_2n = [[H_n, H_n], [H_n, -H_n]]``). It is symmetric and
    orthonormal, so ``H @ H == I``.
    """

    if size < 1 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {size}")
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < size:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return (h / size**0.5).to(dtype=dtype, device=device)


def hadamard_rotate(x: torch.Tensor, block_size: int = HADAMARD_BLOCK_SIZE) -> torch.Tensor:
    """Apply ``blockdiag(H_block / sqrt(block))`` along the last dimension of ``x``.

    Reference for the kernel's activation rotation, and the transform an exporter
    applies to the weight residual: ``qweight = Q(hadamard_rotate(R))``.
    """

    channels = x.shape[-1]
    if channels % block_size != 0:
        raise ValueError(f"Last dimension {channels} is not a multiple of block size {block_size}")
    h = hadamard_matrix(block_size, dtype=x.dtype, device=x.device)
    return (x.reshape(*x.shape[:-1], channels // block_size, block_size) @ h).reshape(x.shape)
