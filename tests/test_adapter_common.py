import pytest
import torch
from torch import nn

from nunchaku_lite.adapters.common import linear_class_from_checkpoint, linear_from_class
from nunchaku_lite.linear import AWQW4A16Linear, DenseRuntimeLoraLinear, SVDQW4A4Linear


def test_linear_class_from_checkpoint_detects_dense_awq_and_svdq():
    assert linear_class_from_checkpoint({"proj.weight": torch.empty(1)}, "proj") is DenseRuntimeLoraLinear
    assert linear_class_from_checkpoint({"proj.qweight": torch.empty(1, dtype=torch.int32)}, "proj") is AWQW4A16Linear
    assert linear_class_from_checkpoint({"proj.wzeros": torch.empty(1)}, "proj") is AWQW4A16Linear
    assert linear_class_from_checkpoint({"proj.qweight": torch.empty(1, dtype=torch.int8)}, "proj") is SVDQW4A4Linear
    assert linear_class_from_checkpoint({"proj.proj_down": torch.empty(1)}, "proj") is SVDQW4A4Linear


def test_linear_class_from_checkpoint_defaults_to_dense():
    assert linear_class_from_checkpoint({}, "proj") is DenseRuntimeLoraLinear
    assert linear_class_from_checkpoint({}, "proj", default=AWQW4A16Linear) is AWQW4A16Linear


def test_linear_class_from_checkpoint_rejects_ambiguous_keys():
    with pytest.raises(ValueError, match="Ambiguous linear checkpoint format"):
        linear_class_from_checkpoint(
            {
                "proj.weight": torch.empty(1),
                "proj.qweight": torch.empty(1, dtype=torch.int32),
            },
            "proj",
        )


def test_linear_class_from_checkpoint_requires_default_when_disabled():
    with pytest.raises(ValueError, match="Could not infer linear checkpoint format"):
        linear_class_from_checkpoint({}, "proj", default=None)


def test_linear_from_class_instantiates_selected_replacement():
    linear = nn.Linear(16, 32)

    dense = linear_from_class(linear, DenseRuntimeLoraLinear)
    awq = linear_from_class(linear, AWQW4A16Linear, torch_dtype=torch.bfloat16)
    svdq = linear_from_class(linear, SVDQW4A4Linear, torch_dtype=torch.bfloat16, rank=4, precision="int4")

    assert isinstance(dense, DenseRuntimeLoraLinear)
    assert isinstance(awq, AWQW4A16Linear)
    assert isinstance(svdq, SVDQW4A4Linear)
    assert svdq.rank == 4
