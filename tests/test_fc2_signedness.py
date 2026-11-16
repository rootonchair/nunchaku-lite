import pytest
import torch
from diffusers.models.attention import FeedForward

from nunchaku_lite.adapters.common import SVDQPatchContext
from nunchaku_lite.adapters.ltx2 import _patch_ltx2_feed_forward
from nunchaku_lite.linear import SVDQW4A4Linear
from nunchaku_lite.ops.fused import fused_gelu_mlp


@pytest.mark.parametrize("precision,expected_unsigned", [("int4", True), ("nvfp4", False)])
def test_ltx2_fc2_follows_fused_epilogue_signedness(precision, expected_unsigned):
    # fused_gelu_mlp writes unsigned GELU-shifted codes for INT4 and signed codes for NVFP4,
    # so LTX-2's fc2 must use the same convention as Flux and Qwen-Image.
    ff = FeedForward(64, activation_fn="gelu-approximate")
    context = SVDQPatchContext(precision=precision, rank=16, torch_dtype=torch.bfloat16)
    ff = _patch_ltx2_feed_forward(ff, context)
    assert isinstance(ff.net[2], SVDQW4A4Linear)
    assert ff.net[2].act_unsigned is expected_unsigned


@pytest.mark.parametrize("precision,bad_unsigned", [("int4", False), ("nvfp4", True)])
def test_fused_gelu_mlp_refuses_mismatched_fc2_signedness(precision, bad_unsigned):
    fc1 = SVDQW4A4Linear(128, 256, rank=16, precision=precision)
    fc2 = SVDQW4A4Linear(256, 128, rank=16, precision=precision, act_unsigned=bad_unsigned)
    with pytest.raises(ValueError, match="fc2.act_unsigned"):
        fused_gelu_mlp(torch.zeros(1, 4, 128), fc1, fc2)


def test_unsigned_linear_refuses_standalone_forward():
    layer = SVDQW4A4Linear(128, 128, rank=16, act_unsigned=True)
    with pytest.raises(ValueError, match="fused_gelu_mlp"):
        layer(torch.zeros(1, 4, 128))
