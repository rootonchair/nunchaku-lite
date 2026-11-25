"""ComfyUI loader decision table and invariants (no ComfyUI and no kernels needed)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nunchaku_lite.comfyui.core import (
    build_comfy_context,
    load_state_exactly,
    resolve_checkpoint_precision,
    select_comfy_family,
)


def _svdq(prefix: str, scale_dtype=torch.bfloat16, rank: int = 4, name: str = "proj_down") -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.qweight": torch.zeros(8, 4, dtype=torch.int8),
        f"{prefix}.wscales": torch.zeros(1, 8, dtype=scale_dtype),
        f"{prefix}.{name}": torch.zeros(8, rank),
    }


# --- precision -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scale_dtype", "declared", "expected"),
    [
        (torch.bfloat16, {}, "int4"),
        (torch.float8_e4m3fn, {}, "nvfp4"),
        (torch.bfloat16, {"weight": {"dtype": "int4"}}, "int4"),
        (torch.float8_e4m3fn, {"weight": {"dtype": "fp4_e2m1_all"}}, "nvfp4"),
    ],
)
def test_precision_follows_weight_scales(scale_dtype, declared, expected):
    assert resolve_checkpoint_precision(_svdq("a", scale_dtype), declared) == expected


@pytest.mark.parametrize(
    ("scale_dtype", "declared"),
    [(torch.bfloat16, {"weight": {"dtype": "fp4_e2m1_all"}}), (torch.float8_e4m3fn, {"weight": {"dtype": "int4"}})],
)
def test_precision_declaration_must_match_tensors(scale_dtype, declared):
    with pytest.raises(ValueError, match="declares"):
        resolve_checkpoint_precision(_svdq("a", scale_dtype), declared)


def test_precision_refuses_mixed_layers_and_awq_only_checkpoints():
    with pytest.raises(ValueError, match="mixes"):
        resolve_checkpoint_precision({**_svdq("a"), **_svdq("b", torch.float8_e4m3fn)}, {})
    awq = {"m.qweight": torch.zeros(1, dtype=torch.int32), "m.wscales": torch.zeros(1), "m.wzeros": torch.zeros(1)}
    with pytest.raises(ValueError, match="no SVDQ W4A4 layers"):
        resolve_checkpoint_precision(awq, {})


# --- rank and dtype ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["proj_down", "lora_down"])
def test_rank_is_inferred_from_either_low_rank_naming(name):
    context = build_comfy_context(_svdq("a", rank=16, name=name), {}, torch.bfloat16)
    assert context.rank == 16


def test_declared_rank_wins_and_missing_rank_is_an_error():
    assert build_comfy_context(_svdq("a", rank=16), {"rank": 32}, torch.bfloat16).rank == 32
    state = {k: v for k, v in _svdq("a").items() if not k.endswith("proj_down")}
    with pytest.raises(ValueError, match="rank"):
        build_comfy_context(state, {}, torch.bfloat16)


def test_compute_dtype_must_be_half_precision():
    with pytest.raises(ValueError, match="bfloat16 or float16"):
        build_comfy_context(_svdq("a"), {}, torch.float32)


# --- family selection -------------------------------------------------------------------------------------------


def _flux1_keys():
    return {
        "x_embedder.weight": torch.zeros(1),
        "time_text_embed.timestep_embedder.linear_1.weight": torch.zeros(1),
        "single_transformer_blocks.0.qkv_proj.qweight": torch.zeros(1),
    }


@pytest.mark.parametrize(
    ("state", "family"),
    [
        (_flux1_keys(), "flux"),
        (
            {
                "double_stream_modulation_img.linear.weight": torch.zeros(1),
                "transformer_blocks.0.attn.to_qkv.qweight": torch.zeros(1),
            },
            "flux2",
        ),
        (
            {"txt_norm.weight": torch.zeros(1), "transformer_blocks.0.attn.to_qkv.qweight": torch.zeros(1)},
            "qwen_image",
        ),
        (
            {
                "cap_pad_token": torch.zeros(1),
                "x_pad_token": torch.zeros(1),
                "layers.0.attention.to_qkv.qweight": torch.zeros(1),
            },
            "z_image",
        ),
        (
            {
                "conv_in.weight": torch.zeros(1),
                "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_qkv.qweight": torch.zeros(1),
            },
            "sdxl",
        ),
    ],
)
def test_exactly_one_family_matches(state, family):
    assert select_comfy_family(state, {}).name == family


def test_unknown_checkpoint_is_refused():
    with pytest.raises(ValueError, match="No Nunchaku Lite ComfyUI family"):
        select_comfy_family({"foo.weight": torch.zeros(1)}, {})


def test_flux1_and_flux2_never_both_match():
    from nunchaku_lite.comfyui.flux import Flux2ComfyFamily, FluxComfyFamily

    state = {
        **_flux1_keys(),
        "double_stream_modulation_img.linear.weight": torch.zeros(1),
        "transformer_blocks.0.attn.to_qkv.qweight": torch.zeros(1),
    }
    assert not FluxComfyFamily().matches(state, {})
    assert Flux2ComfyFamily().matches(state, {})


# --- exact load ---------------------------------------------------------------------------------------------------


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.device("meta"):
            self.linear = nn.Linear(4, 2)
        self.register_buffer("index", torch.empty(2, dtype=torch.int64, device="meta"))


def _tiny_state():
    return {"linear.weight": torch.ones(2, 4), "linear.bias": torch.ones(2), "index": torch.arange(2)}


def test_exact_load_assigns_everything():
    module = _Tiny()
    load_state_exactly(module, _tiny_state())
    assert not any(t.is_meta for t in [*module.parameters(), *module.buffers()])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda s: s.pop("linear.bias"), "missing"),
        (lambda s: s.update(extra=torch.ones(1)), "unexpected"),
        (lambda s: s.update({"linear.weight": torch.ones(4, 2)}), "shape"),
        (lambda s: s.update({"index": torch.ones(2)}), "cannot load"),
    ],
    ids=["missing", "unexpected", "shape", "float-into-int"],
)
def test_exact_load_refuses_mismatches(mutate, message):
    state = _tiny_state()
    mutate(state)
    with pytest.raises(ValueError, match=message):
        load_state_exactly(_Tiny(), state)


def test_exact_load_casts_floating_tensors_to_the_model_dtype():
    module = _Tiny().to(torch.bfloat16)
    load_state_exactly(module, _tiny_state())
    assert module.linear.weight.dtype == torch.bfloat16
    assert module.index.dtype == torch.int64


def test_exact_load_refuses_tensors_left_on_meta():
    module = _Tiny()
    module.extra = nn.Module()
    module.extra.register_buffer("cache", torch.empty(1, device="meta"), persistent=False)
    with pytest.raises(ValueError, match="meta"):
        load_state_exactly(module, _tiny_state())
