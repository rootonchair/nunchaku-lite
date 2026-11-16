"""MLX / Apple-silicon runtime for Nunchaku SVDQ checkpoints (mirrors the ``nunchaku_lite`` API)."""

from .kernels import has_neural_accelerators, svdq_gemm_w4a8, svdq_gemm_w8a8, svdq_quantize_act_int8
from .keymap import PrefixMap
from .linear import BACKENDS, GELU_SHIFT, AWQW4A16Linear, SVDQW4A4Linear, resolve_backend
from .manifest import apply_runtime_manifest, parse_runtime_manifest
from .packing import SVDQLogicalWeights, pack_svdq_int4, to_kernel_layout, unpack_svdq_int4
from .patch import load_checkpoint, patch_transformer
from .pipeline import list_adapters, load_nunchaku_pipeline, register_adapter

__all__ = [
    "BACKENDS",
    "GELU_SHIFT",
    "AWQW4A16Linear",
    "PrefixMap",
    "SVDQLogicalWeights",
    "SVDQW4A4Linear",
    "apply_runtime_manifest",
    "has_neural_accelerators",
    "list_adapters",
    "load_checkpoint",
    "load_nunchaku_pipeline",
    "pack_svdq_int4",
    "parse_runtime_manifest",
    "patch_transformer",
    "register_adapter",
    "resolve_backend",
    "svdq_gemm_w4a8",
    "svdq_gemm_w8a8",
    "svdq_quantize_act_int8",
    "to_kernel_layout",
    "unpack_svdq_int4",
]
