"""Quantized linear modules backed by Nunchaku Lite native kernels."""

import torch
import torch.nn.functional as F
from torch import nn

from .ops.gemm import awq_gemm_w4a16_g64_int32, awq_gemm_w4a16_g128_int16, svdq_gemm_w4a4_cuda
from .ops.gemv import awq_gemv_w4a16_cuda
from .ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda, svdq_quantize_w4a4_act_hadamard_fuse_lora_cuda

INPUT_ROTATIONS = (None, "hadamard")
# The SVDQ checkpoint format, defined here once and used by every loading path:
SVDQ_GROUP_SIZE = {"int4": 64, "nvfp4": 16}  # input channels per weight scale
SVDQ_RANK_TENSOR = "proj_down"  # (in_features, rank): the tensor that fixes the low-rank branch's rank
SVDQ_CHECKPOINT_SCALARS = ("wtscale",)  # optional one-element per-tensor scale (NVFP4); absent means 1.0
SVDQ_DEFAULTED_CHECKPOINT_TENSORS = ("wcscales",)  # parameters a checkpoint may omit (NVFP4 channel scales: ones)
SVDQ_LEGACY_CHECKPOINT_SUFFIXES = ("smooth_factor_orig",)  # leftover of older exports; every loader drops it
_ALPHA_CACHE_ATTR = "_nunchaku_lite_alpha_cache"
# Tensors of SVDQW4A4Linear whose dtype a module-wide cast must not change.
_CAST_PROTECTED_DTYPES = {"wscales": (torch.float8_e4m3fn,), "wtscale": (torch.float32,)}


def wtscale_alpha(module: nn.Module) -> float | torch.Tensor | None:
    """The NVFP4 per-tensor weight scale to pass to the GEMM as ``alpha``.

    ``wtscale`` is a float, a CPU tensor (set by the Diffusers loader), or, for layers built by
    :mod:`nunchaku_lite.loader`, a persistent buffer that moves with the module. The native op
    reads ``alpha`` with ``item()``, so a device-resident buffer would force a host sync on every
    GEMM (and break CUDA-graph capture): it is read once per (re)load and handed over as a CPU
    float32 scalar, exactly the value ``item()`` would have read.
    """

    wtscale = getattr(module, "wtscale", None)
    if not isinstance(wtscale, torch.Tensor) or wtscale.device.type == "cpu":
        return wtscale
    # Inference tensors (models built under torch.inference_mode, as ltx-core does) have no version counter; a
    # reload still replaces the tensor, so its data pointer alone keys the cache.
    version = None if wtscale.is_inference() else wtscale._version
    key = (wtscale.data_ptr(), version, wtscale.dtype)
    cached = module.__dict__.get(_ALPHA_CACHE_ATTR)
    if cached is None or cached[0] != key:
        cached = (key, torch.tensor(float(wtscale.reshape(-1)[0]), dtype=torch.float32))
        module.__dict__[_ALPHA_CACHE_ATTR] = cached
    return cached[1]


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
        input_rotation: str | None = None,
    ):
        """Allocate SVDQ parameter buffers for a quantized linear projection.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            rank: Low-rank correction rank.
            bias: Whether to allocate a bias parameter.
            precision: Native weight precision, either ``"int4"`` or
                ``"nvfp4"``.
            act_unsigned: Whether the activation quantization path should use
                unsigned activations.
            torch_dtype: Runtime dtype for floating-point buffers.
            device: Device for parameter allocation.
            input_rotation: Online activation rotation the packed weight was
                exported for: ``None`` or ``"hadamard"`` (blockwise H64/8,
                applied after smoothing; the low-rank branch stays unrotated).
                Rotation needs a signed activation path, so it cannot be
                combined with ``act_unsigned``.

        Raises:
            ValueError: If ``precision`` or ``input_rotation`` is unsupported,
                or rotation is combined with unsigned activations.

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

        if precision not in SVDQ_GROUP_SIZE:
            raise ValueError(f"Invalid precision: {precision}")
        self.group_size = SVDQ_GROUP_SIZE[precision]

        self.qweight = nn.Parameter(
            torch.empty(out_features, in_features // 2, dtype=torch.int8, device=device), requires_grad=False
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
                dtype=torch_dtype if precision == "int4" else torch.float8_e4m3fn,
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

        if input_rotation not in INPUT_ROTATIONS:
            raise ValueError(f"Invalid input_rotation: {input_rotation!r}; expected one of {INPUT_ROTATIONS}")
        if input_rotation is not None and act_unsigned:
            raise ValueError("input_rotation requires signed activations; it cannot be combined with act_unsigned")
        self.act_unsigned = act_unsigned
        self.input_rotation = input_rotation
        # Quantized-kernel backend on Apple silicon ("auto", "int8", "int4", "dense"); set by patch_transformer.
        self.backend = "auto"

    @staticmethod
    def checkpoint_dims(shapes: dict[str, tuple[int, ...]]) -> tuple[int, int, int]:
        """``(in_features, out_features, rank)`` of a layer from its checkpoint tensor shapes (``suffix -> shape``).

        Raises:
            KeyError: If ``qweight`` or ``proj_down`` is missing.
            ValueError: If either is not 2-D.
        """

        qweight, proj_down = shapes["qweight"], shapes[SVDQ_RANK_TENSOR]
        if len(qweight) != 2 or len(proj_down) != 2:
            raise ValueError(f"qweight {tuple(qweight)} and {SVDQ_RANK_TENSOR} {tuple(proj_down)} must be 2-D")
        out_features, packed_in = qweight
        return 2 * packed_in, out_features, SVDQW4A4Linear.checkpoint_rank(proj_down)

    @staticmethod
    def checkpoint_rank(proj_down_shape: tuple[int, ...]) -> int:
        """The low-rank branch rank from the shape of the checkpoint's ``proj_down`` tensor ``(in_features, rank)``."""

        return int(proj_down_shape[1])

    def use_checkpoint_scalar(self, name: str, shape: tuple[int, ...]) -> None:
        """Hold the checkpoint's per-tensor scale ``name`` (e.g. ``wtscale``) as a persistent float32 buffer, so any
        state-dict loader fills it and moves keep it on the module's device; without one it stays a plain ``1.0``."""

        if name not in SVDQ_CHECKPOINT_SCALARS or getattr(self, name, None) is None:
            raise ValueError(f"{self.precision} SVDQ layers have no checkpoint scalar {name!r}.")
        delattr(self, name)
        self.register_buffer(name, torch.ones(shape, dtype=torch.float32, device=self.qweight.device), persistent=True)

    def default_missing_tensor(self, name: str) -> None:
        """Turn parameter ``name``, which the checkpoint omits, into a non-persistent buffer of its default (ones), so a
        strict loader does not expect it. Built on meta, the buffer is created on the CPU for the caller to move."""

        if name not in SVDQ_DEFAULTED_CHECKPOINT_TENSORS or getattr(self, name, None) is None:
            raise ValueError(f"{self.precision} SVDQ layers cannot default {name!r}.")
        old = getattr(self, name)
        device = torch.device("cpu") if old.device.type == "meta" else old.device
        delattr(self, name)
        self.register_buffer(name, torch.ones(old.shape, dtype=old.dtype, device=device), persistent=False)

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

        if self.input_rotation is not None and self.act_unsigned:
            # act_unsigned is often set by adapters after construction, so re-check here.
            raise ValueError("input_rotation requires signed activations; it cannot be combined with act_unsigned")
        quantize_fn = (
            svdq_quantize_w4a4_act_hadamard_fuse_lora_cuda
            if self.input_rotation == "hadamard"
            else svdq_quantize_w4a4_act_fuse_lora_cuda
        )
        return quantize_fn(
            x,
            lora_down=self.proj_down,
            smooth=self.smooth_factor,
            fp4=self.precision == "nvfp4",
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
            alpha=wtscale_alpha(self),
            wcscales=self.wcscales,
            act_unsigned=self.act_unsigned,
        )
        return output

    def _apply(self, fn, recurse=True):
        from .ops.mps import preserve_cache_across_move

        # A dtype cast (module.to(torch.float16), as framework model managers do) must not touch the packed
        # NVFP4 scales (float8) or a wtscale buffer (float32): they keep their dtype and only move.
        # detach(): nn.Module._apply rebinds ``param.data`` in place, which would also change a plain reference.
        kept = {
            name: tensor.detach()
            for name, tensor in (("wscales", self.wscales), ("wtscale", getattr(self, "wtscale", None)))
            if isinstance(tensor, torch.Tensor) and tensor.dtype in _CAST_PROTECTED_DTYPES[name]
        }
        result = preserve_cache_across_move(self, lambda: super(SVDQW4A4Linear, self)._apply(fn, recurse))
        for name, original in kept.items():
            moved = getattr(self, name)
            if moved.dtype != original.dtype:
                restored = original.to(moved.device)
                if name in self._parameters:
                    self._parameters[name] = nn.Parameter(restored, requires_grad=False)
                else:
                    self._buffers[name] = restored
        return result

    def forward(self, x: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the quantized projection to a batched sequence tensor.

        Args:
            x: Input tensor with shape ``(batch, sequence, channels)``.
            output: Optional flattened preallocated output buffer.

        Returns:
            Output tensor with shape ``(batch, sequence, out_features)``.
        """

        if x.device.type == "mps":
            from .ops.mps import svdq_linear_mps

            return svdq_linear_mps(self, x)
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
            f"rank={self.rank}, precision={self.precision}, act_unsigned={self.act_unsigned}, "
            f"input_rotation={self.input_rotation})"
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

    def _apply(self, fn, recurse=True):
        from .ops.mps import preserve_cache_across_move

        return preserve_cache_across_move(self, lambda: super(AWQW4A16Linear, self)._apply(fn, recurse))

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
        elif x.device.type == "mps":
            from .ops.mps import awq_linear_mps

            output = awq_linear_mps(self, x_flat).reshape(output_shape)
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
