"""Run Nunchaku Lite LTX-2.x transformers in Lightricks' reference code (``ltx-core`` / ``ltx-pipelines``).

A native-layout checkpoint (Lightricks' transformer file with our SVDQ layers in place of the quantized linears, as
published in ``rootonchair/nunchaku-lite-ltx-2.5-distilled-native``) plugs into ltx-core as a
:class:`~ltx_core.quantization.QuantizationPolicy`, the same hook as Lightricks' own ``nvfp4-prequant``::

    from nunchaku_lite.integrations.ltx_core import build_nunchaku_policy

    policy = build_nunchaku_policy("ltx-2.5-22b-distilled-transformer-nunchaku-nvfp4-r32.safetensors")
    pipeline = DistilledPipeline(..., quantization=policy)   # any ltx-pipelines pipeline

The policy

* keeps every checkpoint key (ltx-core's own ``model.diffusion_model.`` renaming map strips the prefix)
  and drops legacy ``smooth_factor_orig`` tensors;
* swaps each target ``nn.Linear`` of the meta shell for an SVDQ W4A4 layer (:func:`nunchaku_lite.loader.replace_linears`),
  so ltx-core's loader binds our tensors like any other weight;
* runs the feed-forward through the fused NVFP4 GELU-MLP when :func:`~nunchaku_lite.loader.gelu_mlp_plan` allows it;
* optionally replaces the attention Q/K RMSNorm + RoPE by one fused CUDA kernel (``fused_attention``), keeping the
  stock path for everything that kernel does not cover.

LoRAs cannot be fused into W4A4 layers; this version refuses them on quantized layers (see ``_REFUSE_LORA_RULE``).
"""

from __future__ import annotations

import hashlib
import inspect
import types
import warnings
from pathlib import Path

import torch
from torch import nn

from ..linear import SVDQW4A4Linear
from ..loader import CheckpointSpec, check_runtime_support, gelu_mlp_plan, read_checkpoint_spec, replace_linears
from ..ops.fused import FUSED_MLP_FC2_ATTR

__all__ = ["build_nunchaku_policy", "main", "patch_ltx_model"]

# Source hashes of the ltx-core forwards our replacements mirror (LTX-2 c3a7efb, 2026-11-24). If upstream changes
# one, the replacement is skipped with a warning and the stock forward runs: slower, never wrong.
_ATTENTION_FORWARD_SHA = "623d8b92f654e7a9"
_GELU_APPROX_FORWARD_SHA = "13e614a43f5caf75"
_FEED_FORWARD_FORWARD_SHA = "89515e721627c43d"


def build_nunchaku_policy(
    checkpoint_path: str | Path,
    *,
    fused_attention: bool = True,
    allow_fp4_rotation: bool = False,
    device: torch.device | str | None = None,
):
    """Build the ltx-core ``QuantizationPolicy`` that loads a native-layout Nunchaku Lite transformer.

    Pass the same file as the pipeline's checkpoint / ``--transformer-path``. ``device`` (default: the current
    CUDA device when there is one) is checked up front for precision and rotation support.
    """

    from ltx_core.loader.lora import FLOAT_FUSE_RULE  # noqa: F401  (import check: ltx-core present)
    from ltx_core.loader.module_ops import ModuleOps
    from ltx_core.loader.sd_ops import SDOps
    from ltx_core.quantization.policy import QuantizationPolicy

    spec = read_checkpoint_spec(checkpoint_path, allow_fp4_rotation=allow_fp4_rotation)
    if spec.layout != "native":
        raise ValueError(
            f"{checkpoint_path} is a Diffusers-layout checkpoint; ltx-core reads Lightricks' native layout (their "
            "transformer file with our quantized layers in place), e.g. rootonchair/nunchaku-lite-ltx-2.5-distilled-native."
        )
    if device is None and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    if device is not None:
        check_runtime_support(spec, device)

    name = f"nunchaku-lite:{_spec_digest(spec)}:attn={'fused' if fused_attention else 'stock'}"
    sd_ops = SDOps(f"{name}:load").with_kv_operation(_drop, key_suffix=".smooth_factor_orig")
    swap = ModuleOps(
        f"{name}:swap",
        matcher=lambda model: hasattr(model, "transformer_blocks"),
        mutator=lambda model: patch_ltx_model(model, spec, fused_attention=fused_attention),
    )
    return QuantizationPolicy(sd_ops=(sd_ops,), module_ops=(swap,), fuse_rule=_refuse_lora_rule())


def patch_ltx_model(model: nn.Module, spec: CheckpointSpec, *, fused_attention: bool = True) -> nn.Module:
    """Swap ``spec``'s targets in an ltx-core ``LTXModel`` (meta shell or built) and install the fused forwards."""

    from ltx_core.model.transformer.attention import Attention
    from ltx_core.model.transformer.feed_forward import FeedForward
    from ltx_core.model.transformer.gelu_approx import GELUApprox

    replace_linears(model, spec)

    ff_ok = _source_matches(FeedForward.forward, _FEED_FORWARD_FORWARD_SHA) and _source_matches(
        GELUApprox.forward, _GELU_APPROX_FORWARD_SHA
    )
    attn_ok = fused_attention and _source_matches(Attention.forward, _ATTENTION_FORWARD_SHA)
    for module in model.modules():
        if isinstance(module, FeedForward) and ff_ok:
            _patch_feed_forward(module, GELUApprox)
        elif isinstance(module, Attention) and attn_ok and isinstance(module.to_q, SVDQW4A4Linear):
            module.forward = types.MethodType(_attention_forward, module)
            module._nunchaku_lite_fused_qk_norm_rope = True
    model._nunchaku_lite_spec = spec
    return model


# --- feed-forward --------------------------------------------------------------------------------


def _patch_feed_forward(ff: nn.Module, gelu_approx_cls: type) -> None:
    fc1_owner, fc2 = ff.net[0], ff.net[2]
    fc1 = getattr(fc1_owner, "proj", None)
    if not (isinstance(fc1_owner, gelu_approx_cls) and isinstance(fc1, SVDQW4A4Linear) and isinstance(fc2, SVDQW4A4Linear)):
        return
    plan = gelu_mlp_plan(fc1, fc2)
    fc2.act_unsigned = plan.fc2_act_unsigned
    if plan.fc2_fused_flag:
        setattr(fc2, FUSED_MLP_FC2_ATTR, True)
    if plan.fused:
        ff.forward = types.MethodType(_fused_feed_forward, ff)


def _fused_feed_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if x.is_cuda and not torch.is_grad_enabled() and x.ndim == 3:
        from ..ops.fused import fused_gelu_mlp

        return fused_gelu_mlp(x, self.net[0].proj, self.net[2])
    return self.net(x)


# --- attention -----------------------------------------------------------------------------------


def _fused_qk(self: nn.Module, q: torch.Tensor, k: torch.Tensor, pe, k_pe):
    """Q/K RMSNorm + RoPE in one launch when the inputs fit the kernel, else ``None``."""

    if pe is None or not q.is_cuda or q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype:
        return None
    rope_type = getattr(self.rope_type, "value", self.rope_type)
    if rope_type != "split":  # the fused kernel's interleaved layout differs from ltx-core's legacy one
        return None
    key_pe = pe if k_pe is None else k_pe
    batch, heads, head_dim = q.shape[0], self.heads, self.dim_head
    expected_q = (batch, heads, q.shape[1], head_dim // 2)
    expected_k = (batch, heads, k.shape[1], head_dim // 2)
    if tuple(pe[0].shape) != expected_q or tuple(key_pe[0].shape) != expected_k:
        return None
    if self.q_norm.eps is None or self.k_norm.eps is None or self.q_norm.eps != self.k_norm.eps:
        return None
    from ..ops.fused import fused_cross_head_qk_norm_rope

    return fused_cross_head_qk_norm_rope(
        q,
        k,
        self.q_norm,
        self.k_norm,
        pe,
        key_pe,
        q_heads=heads,
        k_heads=k.shape[-1] // head_dim,
        head_dim=head_dim,
        rope_type="split",
    )


def _attention_forward(
    self: nn.Module,
    x: torch.Tensor,
    context: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    pe: torch.Tensor | None = None,
    k_pe: torch.Tensor | None = None,
    perturbation_mask: torch.Tensor | None = None,
    all_perturbed: bool = False,
) -> torch.Tensor:
    """ltx-core ``Attention.forward`` (c3a7efb) with Q/K norm + RoPE fused when possible."""

    from ltx_core.model.transformer.rope import apply_rotary_emb

    context = x if context is None else context
    use_attention = not all_perturbed

    v = self.to_v(context)

    if not use_attention:
        out = v
    else:
        q = self.to_q(x)
        k = self.to_k(context)
        fused = _fused_qk(self, q, k, pe, k_pe)
        if fused is not None:
            q, k = fused
        else:
            q = self.q_norm(q)
            k = self.k_norm(k)
            if pe is not None:
                q = apply_rotary_emb(q, pe, self.rope_type)
                k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)
        if mask is None:
            out = self.attention_function(q, k, v, self.heads)  # (B, T, H*D)
        else:
            out = self.masked_attention_function(q, k, v, self.heads, mask)

    if self.to_gate_logits is not None and use_attention:
        gate_logits = self.to_gate_logits(x)
        b, t, _ = out.shape
        out = out.view(b, t, self.heads, self.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)
        out = out * gates.unsqueeze(-1)
        out = out.view(b, t, self.heads * self.dim_head)

    if use_attention and perturbation_mask is not None:
        out = out * perturbation_mask + v * (1 - perturbation_mask)

    return self.to_out(out)


# --- helpers -------------------------------------------------------------------------------------


def _drop(key: str, inputs):
    del key, inputs
    return {}


def _refuse_lora_rule():
    """LoRA fusion: plain bf16 add for unquantized weights; a LoRA on an SVDQ layer is refused.

    An SVDQ layer has no ``.weight`` key, so ltx-core would otherwise have no weight to fuse into; this rule
    makes the failure explicit if a delta ever reaches a quantized tensor.
    """

    from ltx_core.loader.lora import FLOAT_FUSE_RULE
    from ltx_core.loader.units import FuseRule

    def patch(key, weight, delta, clean):
        if not weight.is_floating_point() or key.endswith((".qweight", ".wscales")):
            raise NotImplementedError(f"LoRA on the Nunchaku Lite W4A4 layer {key!r} is not supported yet.")
        return FLOAT_FUSE_RULE.patch(key, weight, delta, clean)

    return FuseRule("nunchaku-lite-bf16-add", torch.bfloat16, patch)


def _spec_digest(spec: CheckpointSpec) -> str:
    rows = sorted(
        f"{ly.name}:{ly.precision}:{ly.rank}:{ly.in_features}x{ly.out_features}:{ly.has_bias}:{ly.has_wtscale}:{ly.input_rotation}"
        for ly in spec.layers.values()
    )
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:12]


def _source_matches(fn, expected: str) -> bool:
    try:
        actual = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        actual = None
    if actual != expected:
        warnings.warn(
            f"ltx-core {fn.__qualname__} changed upstream (source {actual}, expected {expected}); "
            "using the stock forward there (unfused, still correct).",
            stacklevel=3,
        )
        return False
    return True


# --- command line --------------------------------------------------------------------------------

_CLI_FLAGS = {"--nunchaku": None, "--nunchaku-stock-attention": None, "--nunchaku-allow-fp4-rotation": None}


def main(argv: list[str] | None = None) -> None:
    """Run an ltx-pipelines entry point with a Nunchaku Lite transformer.

    ``python -m nunchaku_lite.integrations.ltx_core distilled --transformer-path <nunchaku file> ...`` runs
    ``ltx_pipelines.distilled`` exactly as its own CLI, with our policy in place of ``--quantization``. The
    transformer file is taken from ``--transformer-path`` (or ``--distilled-checkpoint-path`` /
    ``--checkpoint-path``, the order ltx-pipelines uses). Extra flags: ``--nunchaku-stock-attention``,
    ``--nunchaku-allow-fp4-rotation``.
    """

    import importlib
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        raise SystemExit("usage: python -m nunchaku_lite.integrations.ltx_core <pipeline module> [pipeline args]")
    pipeline = argv.pop(0)
    module_name = pipeline if "." in pipeline else f"ltx_pipelines.{pipeline}"
    stock_attention = "--nunchaku-stock-attention" in argv
    allow_fp4_rotation = "--nunchaku-allow-fp4-rotation" in argv
    rest = [a for a in argv if a not in _CLI_FLAGS]
    if "--quantization" in rest:
        raise SystemExit("--quantization selects Lightricks' own formats; leave it out for a Nunchaku Lite checkpoint.")

    args_module = importlib.import_module("ltx_pipelines.utils.args")
    original = args_module._resolve_quantization

    def resolve(namespace):
        original(namespace)
        path = (
            getattr(namespace, "transformer_path", None)
            or getattr(namespace, "distilled_checkpoint_path", None)
            or getattr(namespace, "checkpoint_path", None)
        )
        if path is None:
            raise SystemExit("Pass the Nunchaku Lite transformer with --transformer-path (or --checkpoint-path).")
        namespace.quantization = build_nunchaku_policy(
            path, fused_attention=not stock_attention, allow_fp4_rotation=allow_fp4_rotation
        )

    args_module._resolve_quantization = resolve
    sys.argv = [f"{module_name} (nunchaku-lite)", *rest]
    try:
        importlib.import_module(module_name).main()
    finally:
        args_module._resolve_quantization = original


if __name__ == "__main__":
    main()
