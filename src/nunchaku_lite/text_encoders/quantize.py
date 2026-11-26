"""Load-time weight-only quantization of Hugging Face text encoders (Qwen3, Qwen2.5-VL, T5, ...).

No quantized checkpoint is needed: :func:`load_quantized_text_encoder` loads the bf16 encoder onto the CPU (memory
mapped by Transformers when the checkpoint dtype matches, so it costs almost no RAM) and :func:`quantize_text_encoder`
then quantizes one ``nn.Linear`` at a time onto the GPU. The dense encoder is never resident on the GPU.

Weights are quantized round-to-nearest; activations stay in bf16/fp16 (LLM activations have outliers, and the encoder
runs once per prompt, so speed matters little).

:func:`load_quantized_text_encoder` (and ``load_nunchaku_pipeline(text_encoder_quantization=...)``) picks a backend per
device. On CUDA with ``bitsandbytes`` installed it loads the encoder with bitsandbytes (NF4 for ``"int4"``, LLM.int8
for ``"int8"``) through :func:`~nunchaku_lite.text_encoders.load_text_encoder_bnb`: on LTX-2.5's Gemma 4 12B, NF4 was
smaller than group-64 round-to-nearest int4 (7.0 vs 7.7 GiB) and closer to the bf16 encoder (2.9% vs 3.2% hidden-state
error). Everywhere else (Apple silicon, CPU, CUDA without bitsandbytes) it uses this module's own formats:

- ``"int4"``: asymmetric, groups of 64 input channels. On CUDA as :class:`~nunchaku_lite.linear.AWQW4A16Linear` (the
  AWQ W4A16 kernels); elsewhere (MPS, CPU) as :class:`~nunchaku_lite.linear.Int4WeightOnlyLinear`, which dequantizes
  one layer at a time per call.
- ``"int8"``: symmetric per output channel, as :class:`~nunchaku_lite.linear.Int8WeightOnlyLinear`.
"""

from __future__ import annotations

import gc
import importlib.util
import os
from collections import Counter
from collections.abc import Iterable

import torch
from torch import nn

from ..linear import AWQW4A16Linear, Int4WeightOnlyLinear, Int8WeightOnlyLinear

TEXT_ENCODER_QUANTIZATIONS = ("int4", "int8")
TEXT_ENCODER_BACKENDS = ("auto", "bitsandbytes", "nunchaku-lite")
_BNB_MODE = {"int4": "nf4", "int8": "int8"}

# Vision towers of VLM text encoders (Qwen2.5-VL's ``visual``) are kept dense by default.
DEFAULT_EXCLUDE = ("visual", "vision_tower", "vision_model")

_INT4_GROUP_SIZE = 64
_ROW_CHUNK = 2048  # rows quantized at once: bounds the float32 temporaries (a multiple of 4 for int4 packing)


def validate_text_encoder_quantization(weights: str) -> str:
    if weights not in TEXT_ENCODER_QUANTIZATIONS:
        raise ValueError(
            f"Unsupported text encoder quantization {weights!r}; expected one of {TEXT_ENCODER_QUANTIZATIONS}."
        )
    return weights


def resolve_text_encoder_backend(backend: str = "auto", device: str | torch.device | None = None) -> str:
    """``"bitsandbytes"`` or ``"nunchaku-lite"`` for loading a quantized text encoder on ``device``.

    ``"auto"`` picks bitsandbytes on CUDA when it is installed, else this module's own int4/int8 formats.
    """

    if backend not in TEXT_ENCODER_BACKENDS:
        raise ValueError(f"Unsupported text encoder backend {backend!r}; expected one of {TEXT_ENCODER_BACKENDS}.")
    device = torch.device(device) if device is not None else default_quantization_device()
    if backend == "auto":
        on_cuda = device.type == "cuda"
        return "bitsandbytes" if on_cuda and importlib.util.find_spec("bitsandbytes") is not None else "nunchaku-lite"
    if backend == "bitsandbytes" and device.type != "cuda":
        raise ValueError(f"The bitsandbytes text encoder backend needs a CUDA device, got {device}.")
    return backend


def default_quantization_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def quantize_text_encoder(
    model: nn.Module,
    weights: str = "int4",
    *,
    device: str | torch.device | None = None,
    exclude: Iterable[str] | None = DEFAULT_EXCLUDE,
    torch_dtype: torch.dtype | None = None,
) -> nn.Module:
    """Quantize the ``nn.Linear`` layers of ``model`` in place and move the model to ``device``.

    Each layer is quantized on ``device`` and its dense weight released before the next one, so the model can come
    from a CPU (memory-mapped) load. Layers are kept dense when their weight is tied to another parameter (``lm_head``
    tied to the input embeddings), when a dotted component of their name is in ``exclude``, or, for ``"int4"``, when
    their shape does not fit groups of 64 input channels.

    Args:
        model: Text encoder, typically a Transformers ``PreTrainedModel``.
        weights: ``"int4"`` or ``"int8"``.
        device: Device to quantize on and move the model to. Defaults to CUDA, then MPS, then CPU.
        exclude: Module-name components to keep dense. Defaults to vision towers.
        torch_dtype: Dtype of scales and biases. Defaults to each layer's weight dtype (bf16 when it is not 16-bit).

    Returns:
        ``model``.
    """

    validate_text_encoder_quantization(weights)
    device = torch.device(device) if device is not None else default_quantization_device()
    excluded = set(exclude or ())
    shared = Counter(id(p) for _, p in model.named_parameters(remove_duplicate=False))

    targets = []
    for name, module in model.named_modules():
        if type(module) is not nn.Linear:
            continue
        if shared[id(module.weight)] > 1 or excluded.intersection(name.split(".")):
            continue
        if weights == "int4" and (module.in_features % _INT4_GROUP_SIZE or module.out_features % 4):
            continue
        targets.append(name)

    # Allocate every quantized layer and move the rest of the model before quantizing anything: the persistent
    # buffers then sit together, and the float32 temporaries allocated afterwards are released whole by
    # ``empty_cache``. Interleaving them leaves about 1.4 GB of MPS heap reserved for Qwen3-4B.
    pending = []
    for name in targets:
        linear = model.get_submodule(name)
        dtype = torch_dtype or _runtime_dtype(linear.weight.dtype)
        quantized = _allocate(linear, weights, device, dtype)
        parent_name, _, child_name = name.rpartition(".")
        setattr(model.get_submodule(parent_name) if parent_name else model, child_name, quantized)
        pending.append((quantized, linear))
        del linear, quantized
    model.to(device)
    # A single-device ``device_map="cpu"`` load leaves no hooks, only this attribute; it is stale after the move and
    # would make Diffusers treat the encoder as device-mapped.
    if getattr(model, "hf_device_map", None) is not None:
        del model.hf_device_map

    pending.reverse()
    while pending:
        _fill(*pending.pop())  # each dense (memory-mapped) layer is dropped once it is quantized
    gc.collect()  # the replaced layers can sit in reference cycles from loading
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    model._nunchaku_lite_text_encoder_quantization = weights
    return model


def load_quantized_text_encoder(
    pretrained_model_name_or_path: str | os.PathLike,
    weights: str = "int4",
    *,
    model_cls: type | None = None,
    subfolder: str | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    exclude: Iterable[str] | None = DEFAULT_EXCLUDE,
    backend: str = "auto",
    **from_pretrained_kwargs,
) -> nn.Module:
    """Load a Transformers text encoder and quantize its weights at load time.

    Args:
        pretrained_model_name_or_path: Model id or local path (for Diffusers repos, pass ``subfolder``).
        weights: ``"int4"`` or ``"int8"``.
        model_cls: Transformers model class. Defaults to the first entry of the config's ``architectures``.
        subfolder: Subfolder holding the encoder, for example ``"text_encoder"``.
        torch_dtype: Dtype the dense weights load in, and of scales, biases and the unquantized layers.
        device: Device to quantize on and run on. Defaults to CUDA, then MPS, then CPU.
        exclude: Module-name components to keep dense (see :func:`quantize_text_encoder`).
        backend: ``"auto"`` (bitsandbytes on CUDA when installed, else this module's formats), ``"bitsandbytes"``
            or ``"nunchaku-lite"``. A bitsandbytes encoder is placed on ``device`` at load time; don't move it after.
        **from_pretrained_kwargs: Forwarded to ``model_cls.from_pretrained`` (``revision``, ``cache_dir``, ...).

    Returns:
        The quantized encoder in eval mode on ``device``.
    """

    import transformers

    validate_text_encoder_quantization(weights)
    hub_kwargs = dict(from_pretrained_kwargs)
    if subfolder is not None:
        hub_kwargs["subfolder"] = subfolder
    if model_cls is None:
        config = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path, **hub_kwargs)
        architectures = getattr(config, "architectures", None) or []
        if not architectures or not hasattr(transformers, architectures[0]):
            raise ValueError(
                f"Cannot infer the text encoder class of {pretrained_model_name_or_path!s}; pass model_cls."
            )
        model_cls = getattr(transformers, architectures[0])

    resolved = resolve_text_encoder_backend(backend, device)
    if resolved == "bitsandbytes":
        from .bnb import DEFAULT_SKIP_MODULES, load_text_encoder_bnb

        skip = tuple(dict.fromkeys((*DEFAULT_SKIP_MODULES, *(exclude or ()))))
        model = load_text_encoder_bnb(
            model_cls,
            pretrained_model_name_or_path,
            quantization=_BNB_MODE[weights],
            device=device or "cuda",
            torch_dtype=torch_dtype,
            skip_modules=skip,
            **hub_kwargs,
        )
        model.eval()
        model._nunchaku_lite_text_encoder_quantization = weights
        model._nunchaku_lite_text_encoder_backend = resolved
        return model

    dtype_key = "dtype" if _transformers_accepts_dtype() else "torch_dtype"
    model = model_cls.from_pretrained(
        pretrained_model_name_or_path,
        device_map="cpu",
        **{dtype_key: torch_dtype},
        **hub_kwargs,
    )
    model.eval()
    model = quantize_text_encoder(model, weights, device=device, exclude=exclude, torch_dtype=torch_dtype)
    model._nunchaku_lite_text_encoder_backend = resolved
    return model


def _transformers_accepts_dtype() -> bool:
    import transformers
    from packaging.version import Version

    return Version(transformers.__version__).release >= (4, 56)


def _runtime_dtype(dtype: torch.dtype) -> torch.dtype:
    return dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16


def _row_chunks(rows: int):
    for start in range(0, rows, _ROW_CHUNK):
        yield start, min(start + _ROW_CHUNK, rows)


def _allocate(linear: nn.Linear, weights: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    kwargs = {"bias": linear.bias is not None, "torch_dtype": dtype, "device": device}
    if weights == "int8":
        return Int8WeightOnlyLinear(linear.in_features, linear.out_features, **kwargs)
    if device.type == "cuda":
        module = AWQW4A16Linear(linear.in_features, linear.out_features, group_size=_INT4_GROUP_SIZE, **kwargs)
        module.requires_grad_(False)
        # Hugging Face T5 checks ``wo.weight.dtype`` before its feed-forward output projection.
        module.weight = torch.empty(0, dtype=dtype, device=device)
        return module
    return Int4WeightOnlyLinear(linear.in_features, linear.out_features, group_size=_INT4_GROUP_SIZE, **kwargs)


def _int4_groups(linear: nn.Linear, start: int, end: int, device: torch.device, dtype: torch.dtype):
    """Codes ``(rows, K)`` (uint8, 0..15) and per-group scale and zero ``(rows, K / 64)`` in ``dtype`` for a row range."""

    w = linear.weight[start:end].to(device=device, dtype=torch.float32)
    w = w.view(end - start, -1, _INT4_GROUP_SIZE)
    w_min, w_max = w.amin(dim=-1), w.amax(dim=-1)
    # Quantize against the scale and zero as stored (rounded to ``dtype``), so the matmul sees the same values.
    scale = ((w_max - w_min) / 15).clamp_min(1e-8).to(dtype)
    zero = w_min.to(dtype)
    codes = ((w - zero.float().unsqueeze(-1)) / scale.float().unsqueeze(-1)).round_().clamp_(0, 15)
    return codes.view(end - start, -1).to(torch.uint8), scale, zero


@torch.no_grad()
def _fill(module: nn.Module, linear: nn.Linear) -> None:
    device, dtype = module.wscales.device, module.wscales.dtype
    for start, end in _row_chunks(linear.out_features):
        if isinstance(module, Int8WeightOnlyLinear):
            w = linear.weight[start:end].to(device=device, dtype=torch.float32)
            scale = (w.abs().amax(dim=-1) / 127).clamp_min(1e-8).to(dtype)
            module.weight[start:end] = (w / scale.float().unsqueeze(-1)).round_().clamp_(-127, 127).to(torch.int8)
            module.wscales[start:end] = scale
        elif isinstance(module, Int4WeightOnlyLinear):
            codes, scale, zero = _int4_groups(linear, start, end, device, dtype)
            module.weight_packed[start:end] = (codes[:, 0::2] << 4) | codes[:, 1::2]
            module.wscales[start:end] = scale
            module.wzeros[start:end] = zero
        else:  # AWQW4A16Linear: (K / 64, N) scales and the Nunchaku AWQ group-64 int32 packing
            from ..ops.mps import _awq_g64_pack

            codes, scale, zero = _int4_groups(linear, start, end, device, dtype)
            module.wscales[:, start:end] = scale.t()
            module.wzeros[:, start:end] = zero.t()
            module.qweight[start // 4 : end // 4] = _awq_g64_pack(codes).to(device)
    if linear.bias is not None:
        module.bias.copy_(linear.bias)


__all__ = [
    "DEFAULT_EXCLUDE",
    "TEXT_ENCODER_BACKENDS",
    "TEXT_ENCODER_QUANTIZATIONS",
    "load_quantized_text_encoder",
    "quantize_text_encoder",
    "resolve_text_encoder_backend",
]
