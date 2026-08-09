"""Quantized linear modules backed by Nunchaku Lite native kernels."""

import torch
import torch.nn.functional as F
from torch import nn

from .ops.gemm import awq_gemm_w4a16_g128_int16, awq_gemm_w4a16_g64_int32, svdq_gemm_w4a4_cuda
from .ops.gemv import awq_gemv_w4a16_cuda
from .ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda


class DenseRuntimeLoraLinear(nn.Linear):
    """Dense linear layer with runtime LoRA branches managed by Nunchaku Lite."""

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "DenseRuntimeLoraLinear":
        """Wrap an existing dense linear while preserving its state-dict keys.

        Args:
            linear: Source dense linear module.
        """

        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        module.weight = linear.weight
        module.bias = linear.bias
        return module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the dense projection plus any active runtime LoRA branch.

        Args:
            input: Activation tensor whose last dimension is ``in_features``.
        """

        output = F.linear(input, self.weight, self.bias)
        lora_down = getattr(self, "_nunchaku_lite_lora_down", None)
        lora_up = getattr(self, "_nunchaku_lite_lora_up", None)
        if lora_down is None or lora_up is None or lora_down.shape[1] == 0:
            return output
        if lora_down.device != input.device:
            lora_down = lora_down.to(input.device)
            self._nunchaku_lite_lora_down = lora_down
        if lora_up.device != input.device:
            lora_up = lora_up.to(input.device)
            self._nunchaku_lite_lora_up = lora_up
        lora = torch.matmul(input.to(lora_down.dtype), lora_down)
        lora = torch.matmul(lora, lora_up.transpose(0, 1))
        return output + lora.to(output.dtype)


class SVDQW4A4Linear(nn.Module):
    """SVDQ W4A4 linear projection with low-rank correction parameters.

    The module owns the parameter buffers expected by Nunchaku SVDQ
    checkpoints. Parameters are allocated empty and are populated later through
    ``load_state_dict``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        bias: bool = True,
        precision: str = "int4",
        act_unsigned: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ):
        """Allocate SVDQ parameter buffers for a quantized linear projection.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            rank: Low-rank correction rank.
            bias: Whether to allocate a bias parameter.
            precision: Native weight precision, one of ``"int4"``,
                ``"nvfp4"``, or ``"int8"``.
            act_unsigned: Whether the activation quantization path should use
                unsigned activations.
            torch_dtype: Runtime dtype for floating-point buffers.
            device: Device for parameter allocation.

        Raises:
            ValueError: If ``precision`` is unsupported.

        Returns:
            None.
        """

        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.precision = precision
        self.torch_dtype = torch_dtype

        if precision == "nvfp4":
            self.group_size = 16
        elif precision == "int4":
            self.group_size = 64
        elif precision == "int8":
            self.group_size = 32
        else:
            raise ValueError(f"Invalid precision: {precision}")

        qweight_in_features = in_features if precision == "int8" else in_features // 2
        self.qweight = nn.Parameter(
            torch.empty(out_features, qweight_in_features, dtype=torch.int8, device=device), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=torch_dtype, device=device), requires_grad=True)
            if bias
            else None
        )
        self.wscales = nn.Parameter(
            torch.empty(
                in_features // self.group_size,
                out_features,
                dtype=torch_dtype if precision in ("int4", "int8") else torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.smooth_factor = nn.Parameter(
            torch.empty(in_features, dtype=torch_dtype, device=device), requires_grad=False
        )
        self.proj_down = nn.Parameter(torch.empty(in_features, rank, dtype=torch_dtype, device=device))
        self.proj_up = nn.Parameter(torch.empty(out_features, rank, dtype=torch_dtype, device=device))

        if precision == "nvfp4":
            self.wcscales = nn.Parameter(
                torch.ones(out_features, dtype=torch_dtype, device=device), requires_grad=False
            )
            self.wtscale = 1.0
        else:
            self.wtscale = None
            self.wcscales = None

        self.act_unsigned = act_unsigned

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs):
        """Create an empty SVDQ module with metadata copied from ``linear``.

        Args:
            linear: Source dense linear layer.
            **kwargs: Constructor overrides such as ``rank``, ``precision``,
                ``torch_dtype``, ``device``, or ``in_features``.

        Returns:
            New :class:`SVDQW4A4Linear` with matching output shape and bias
            presence.
        """

        in_features = kwargs.pop("in_features", linear.in_features)
        torch_dtype = kwargs.pop("torch_dtype", linear.weight.dtype)
        device = kwargs.pop("device", linear.weight.device)
        return cls(
            in_features=in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            torch_dtype=torch_dtype,
            device=device,
            **kwargs,
        )

    def quantize(self, x: torch.Tensor, pad_size: int = 256) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize activations and compute the low-rank activation side output.

        Args:
            x: Flattened activation tensor with shape ``(tokens, channels)``.
            pad_size: Token padding multiple required by the native quantizer.

        Returns:
            Tuple of quantized activations, activation scales, and low-rank
            activation output.
        """

        return svdq_quantize_w4a4_act_fuse_lora_cuda(
            x,
            lora_down=self.proj_down,
            smooth=self.smooth_factor,
            fp4=self.precision == "nvfp4",
            w8a8=self.precision == "int8",
            pad_size=pad_size,
        )

    def forward_quant(
        self,
        quantized_x: torch.Tensor,
        ascales: torch.Tensor,
        lora_act: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run native W4A4 GEMM on already-quantized activations.

        Args:
            quantized_x: Quantized activation tensor returned by
                :meth:`quantize`.
            ascales: Activation scales returned by :meth:`quantize`.
            lora_act: Low-rank activation tensor returned by :meth:`quantize`.
            output: Optional preallocated output tensor.

        Returns:
            Projection output tensor.
        """

        if output is None:
            output = torch.empty(
                quantized_x.shape[0], self.out_features, dtype=self.proj_up.dtype, device=quantized_x.device
            )
        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=self.qweight,
            out=output,
            ascales=ascales,
            wscales=self.wscales,
            lora_act_in=lora_act,
            lora_up=self.proj_up,
            bias=self.bias,
            fp4=self.precision == "nvfp4",
            alpha=self.wtscale,
            wcscales=self.wcscales,
            act_unsigned=self.act_unsigned,
            w8a8=self.precision == "int8",
        )
        return output

    def forward(self, x: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the quantized projection to a batched sequence tensor.

        Args:
            x: Input tensor with shape ``(batch, sequence, channels)``.
            output: Optional flattened preallocated output buffer.

        Returns:
            Output tensor with shape ``(batch, sequence, out_features)``.
        """

        batch_size, seq_len, channels = x.shape
        x = x.reshape(batch_size * seq_len, channels)
        if output is None:
            output = torch.empty(batch_size * seq_len, self.out_features, dtype=x.dtype, device=x.device)
        quantized_x, ascales, lora_act_out = self.quantize(x)
        output = self.forward_quant(quantized_x, ascales, lora_act_out, output)
        return output.reshape(batch_size, seq_len, -1)

    def __repr__(self):
        """Return a compact representation with quantization metadata.

        Args:
            None.

        Returns:
            Debug string containing feature sizes, rank, precision, and
            activation signedness.
        """

        return (
            f"SVDQW4A4Linear(in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, precision={self.precision}, act_unsigned={self.act_unsigned})"
        )


class AWQW4A16Linear(nn.Module):
    """AWQ W4A16 linear projection used by selected Flux adapter paths.

    This module stores packed AWQ checkpoint buffers and dispatches to native
    GEMM or chunked GEMV kernels at runtime.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 64,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ):
        """Create an empty AWQ linear module with packed weight buffers.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            bias: Whether to allocate a bias parameter.
            group_size: AWQ scale/zero group size.
            torch_dtype: Runtime dtype for floating-point buffers.
            device: Device for parameter allocation.

        Returns:
            None.
        """

        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        self.qweight = nn.Parameter(
            torch.empty(out_features // 4, in_features // 2, dtype=torch.int32, device=device), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=torch_dtype, device=device), requires_grad=True)
            if bias
            else None
        )
        self.wscales = nn.Parameter(
            torch.empty(in_features // group_size, out_features, dtype=torch_dtype, device=device),
            requires_grad=False,
        )
        self.wzeros = nn.Parameter(
            torch.empty(in_features // group_size, out_features, dtype=torch_dtype, device=device),
            requires_grad=False,
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 64,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
        **kwargs,
    ):
        """Create an empty AWQ module with metadata copied from ``linear``.

        Args:
            linear: Source dense linear layer.
            group_size: AWQ group size.
            torch_dtype: Runtime dtype for floating-point buffers.
            device: Optional allocation device. Defaults to the source
                linear's device.
            **kwargs: Ignored compatibility kwargs accepted by adapter helpers.

        Returns:
            New :class:`AWQW4A16Linear`.
        """

        if device is None:
            device = linear.weight.device
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
            torch_dtype=torch_dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the AWQ projection using native AWQ kernels.

        Args:
            x: Input activation tensor whose last dimension is
                ``in_features``.

        Returns:
            Output tensor with last dimension ``out_features``.
        """

        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"AWQW4A16Linear expected input last dimension {self.in_features}, got shape {tuple(x.shape)}."
            )
        output_shape = (*x.shape[:-1], self.out_features)
        x_flat = x.reshape(-1, self.in_features).contiguous()
        if x_flat.shape[0] == 0:
            output = x.new_empty(output_shape)
        else:
            if self._use_gemm(x_flat.shape[0]):
                output = awq_gemm_w4a16_g64_int32(
                    in_feats=x_flat,
                    kernel=self.qweight,
                    scaling_factors=self.wscales,
                    zeros=self.wzeros,
                ).reshape(output_shape)
            else:
                output = self._forward_gemv_chunks(x_flat).reshape(output_shape)
        if self.bias is not None:
            output.add_(self.bias.view([1] * (output.ndim - 1) + [-1]))
        lora_down = getattr(self, "_nunchaku_lite_lora_down", None)
        lora_up = getattr(self, "_nunchaku_lite_lora_up", None)
        if lora_down is not None and lora_up is not None and lora_down.shape[1] > 0:
            if lora_down.device != x.device:
                lora_down = lora_down.to(x.device)
                self._nunchaku_lite_lora_down = lora_down
            if lora_up.device != x.device:
                lora_up = lora_up.to(x.device)
                self._nunchaku_lite_lora_up = lora_up
            lora = torch.matmul(x.to(lora_down.dtype), lora_down)
            lora = torch.matmul(lora, lora_up.transpose(0, 1))
            output.add_(lora.to(output.dtype))
        return output

    def _use_gemm(self, rows: int) -> bool:
        return (
            rows >= 16
            and self.group_size == 64
            and self.in_features % 64 == 0
            and self.out_features % 128 == 0
        )

    def _forward_gemv_chunks(self, x_flat: torch.Tensor) -> torch.Tensor:
        outputs = []
        for start in range(0, x_flat.shape[0], 8):
            chunk = x_flat[start : start + 8]
            outputs.append(
                awq_gemv_w4a16_cuda(
                    in_feats=chunk,
                    kernel=self.qweight,
                    scaling_factors=self.wscales,
                    zeros=self.wzeros,
                    m=chunk.shape[0],
                    n=self.out_features,
                    k=self.in_features,
                    group_size=self.group_size,
                )
            )
        return torch.cat(outputs, dim=0)

    def __repr__(self):
        """Return a compact representation with AWQ metadata.

        Args:
            None.

        Returns:
            Debug string containing feature sizes and group size.
        """

        return (
            f"AWQW4A16Linear(in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size})"
        )


class TinyChatAWQW4A16Linear(nn.Module):
    """TinyChat-layout AWQ W4A16 linear projection.

    This module matches original Nunchaku text encoder checkpoints that store
    packed TinyChat AWQ tensors as ``qweight``, ``scales``, and
    ``scaled_zeros``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 128,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        if group_size != 128:
            raise ValueError("TinyChatAWQW4A16Linear currently supports group_size=128 only.")
        if in_features % group_size != 0:
            raise ValueError("in_features must be divisible by group_size.")
        if out_features % 4 != 0:
            raise ValueError("out_features must be divisible by 4.")
        if torch_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("torch_dtype must be torch.float16 or torch.bfloat16.")
        if device is None:
            device = torch.device("cpu")

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        self.register_buffer(
            "qweight",
            torch.empty(out_features // 4, in_features, dtype=torch.int16, device=device),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                _tinychat_ceil_num_groups(in_features, group_size), out_features, dtype=torch_dtype, device=device
            ),
        )
        self.register_buffer(
            "scaled_zeros",
            torch.empty(
                _tinychat_ceil_num_groups(in_features, group_size), out_features, dtype=torch_dtype, device=device
            ),
        )
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch_dtype, device=device))
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        group_size: int = 128,
        torch_dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
    ) -> "TinyChatAWQW4A16Linear":
        """Allocate an empty quantized module with metadata copied from ``linear``."""

        if torch_dtype is None:
            torch_dtype = linear.weight.dtype
        if device is None:
            device = linear.weight.device
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
            torch_dtype=torch_dtype,
            device=device,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the packed AWQ projection using the TinyChat GEMM path."""

        output = awq_gemm_w4a16_g128_int16(x, self.qweight, self.scales, self.scaled_zeros)
        if self.bias is not None:
            output = output + self.bias.view([1] * (output.ndim - 1) + [-1])
        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, weight_bits=4, group_size={self.group_size}"
        )


def _tinychat_ceil_num_groups(in_features: int, group_size: int, weight_bits: int = 4) -> int:
    if in_features % group_size != 0:
        raise ValueError("in_features must be divisible by group_size.")
    if weight_bits != 4:
        raise ValueError("Only 4-bit weights are supported.")
    num_groups = in_features // group_size
    pack_size = 32 // weight_bits
    num_packs = (num_groups + pack_size - 1) // pack_size
    return num_packs * pack_size
