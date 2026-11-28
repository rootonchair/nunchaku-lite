"""Meta-device loading keeps non-persistent buffers; quantized layers take the model's Linear dtype.

Wan is the real-model case for both: its rotary tables are non-persistent buffers, and Diffusers keeps its first
parameter (``scale_shift_table``) in float32.
"""

import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from nunchaku_lite import patch_transformer
from nunchaku_lite.adapters.common import infer_transformer_dtype
from nunchaku_lite.adapters.manifest import ManifestAdapter
from nunchaku_lite.core import _load_nunchaku_component_from_config, _raise_on_meta_tensors
from nunchaku_lite.linear import SVDQW4A4Linear
from nunchaku_lite.manifest import parse_runtime_manifest

TARGET = "blocks.0.attn1.to_q"


def _wan_cls():
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "WanTransformer3DModel"):
        pytest.skip("diffusers has no WanTransformer3DModel")
    return diffusers.WanTransformer3DModel


def _tiny_wan():
    return _wan_cls()(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=64,
        in_channels=16,
        out_channels=16,
        text_dim=128,
        freq_dim=256,
        ffn_dim=256,
        num_layers=1,
        rope_max_seq_len=32,
    )


def _manifest():
    return {
        "schema": "nunchaku_lite.runtime_manifest",
        "version": 1,
        "component": "transformer",
        "nunchaku_format_version": 1,
        "producer": {"name": "test", "version": "0"},
        "requirements": {
            "method": "svdquant",
            "precision": "int4",
            "rank": 16,
            "weight_dtype": "int4",
            "activation_dtype": "int4",
            "torch_dtype": None,
        },
        "structural_patches": [],
        "targets": [
            {
                "name": TARGET,
                "checkpoint_prefix": TARGET,
                "source_modules": [TARGET],
                "roles": [],
                "kind": "linear",
                "nunchaku_op": "svdq_w4a4",
                "precision": "int4",
                "group_size": 64,
                "rank": 16,
                "has_bias": True,
                "op_options": {},
                "activation": {},
            }
        ],
    }


def _write_wan_repo(tmp_path):
    """A local Diffusers repo (``transformer/``) plus a manifest checkpoint with one SVDQ target."""

    root = tmp_path / "repo"
    _tiny_wan().save_pretrained(root / "transformer")
    model = _wan_cls().from_pretrained(root / "transformer", torch_dtype=torch.bfloat16)
    config = {"runtime_manifest": _manifest()}
    options = type(
        "Options",
        (),
        {
            "precision": parse_runtime_manifest(config).runtime_precision,
            "torch_dtype": torch.bfloat16,
            "device": None,
            "strict": True,
            "adapter_options": {},
        },
    )()
    ManifestAdapter().patch(model, {}, config, options)
    checkpoint = tmp_path / "wan.safetensors"
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()},
        checkpoint,
        metadata={"quantization_config": json.dumps(config)},
    )
    return root, checkpoint


def test_meta_load_builds_non_persistent_buffers(tmp_path):
    cls = _wan_cls()
    root, checkpoint = _write_wan_repo(tmp_path)

    transformer = _load_nunchaku_component_from_config(
        root,
        component_name="transformer",
        component_cls=cls,
        checkpoint=checkpoint,
        target="manifest",
        precision="int4",
        torch_dtype=torch.bfloat16,
        device="cpu",
        strict=True,
        adapter_options=None,
        pipeline_kwargs={},
    )

    assert isinstance(transformer.get_submodule(TARGET), SVDQW4A4Linear)
    assert not any(t.is_meta for t in transformer.parameters())
    assert not any(t.is_meta for t in transformer.buffers())
    reference = _tiny_wan()
    for name in ("freqs_cos", "freqs_sin"):
        assert name not in transformer.state_dict().get(f"rope.{name}", {})  # non-persistent: not in checkpoint
        expected = getattr(reference.rope, name).to(torch.bfloat16)
        torch.testing.assert_close(getattr(transformer.rope, name), expected)


def test_meta_tensors_left_after_loading_are_named():
    module = nn.Module()
    module.child = nn.Module()
    module.child.register_buffer("table", torch.empty(4, device="meta"), persistent=False)
    module.child.cache = torch.empty(2, device="meta")

    with pytest.raises(RuntimeError, match=r"child\.table, child\.cache"):
        _raise_on_meta_tensors(module)


def test_infer_transformer_dtype_ignores_float32_first_parameter():
    module = nn.Module()
    module.scale_shift_table = nn.Parameter(torch.zeros(1, 6, 8))  # float32, registered first
    module.time_embedder = nn.Linear(8, 8)  # float32 (kept in fp32 by Diffusers)
    module.a = nn.Linear(8, 8).to(torch.bfloat16)
    module.b = nn.Linear(8, 8).to(torch.bfloat16)

    assert next(module.parameters()).dtype == torch.float32
    assert infer_transformer_dtype(module) == torch.bfloat16
    assert infer_transformer_dtype(nn.Module()) == torch.bfloat16
    assert infer_transformer_dtype(nn.LayerNorm(4)) == torch.float32


def test_patch_transformer_without_torch_dtype_builds_bfloat16_layers_for_wan(tmp_path):
    root, checkpoint = _write_wan_repo(tmp_path)
    transformer = _wan_cls().from_pretrained(root / "transformer", torch_dtype=torch.bfloat16)
    assert next(transformer.parameters()).dtype == torch.float32  # scale_shift_table is kept in fp32

    patch_transformer(transformer, checkpoint, target="manifest", precision="int4")

    layer = transformer.get_submodule(TARGET)
    float_dtypes = {t.dtype for t in [*layer.parameters(), *layer.buffers()] if t.is_floating_point()}
    assert float_dtypes == {torch.bfloat16}
    assert transformer.scale_shift_table.dtype == torch.float32  # nothing else was cast
