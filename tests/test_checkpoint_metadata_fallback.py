"""Checkpoints without safetensors metadata: config.json fallback, target-list manifests and rank inference."""

import json

import pytest
import torch
from safetensors.torch import save_file

from nunchaku_lite.core import _infer_svdq_rank, _sibling_quantization_config
from nunchaku_lite.manifest import parse_runtime_manifest, with_target_list_manifest

_CONFIG = {
    "quant_method": "nunchaku_lite",
    "svdq_w4a4": {"precision": "int4", "group_size": 64, "rank": 64, "targets": ["blocks.0.attn.to_q"]},
    "awq_w4a16": {"precision": "int4", "group_size": 64, "targets": ["blocks.0.norm.linear"]},
}


def test_sibling_config_json_is_read_for_local_checkpoints(tmp_path):
    folder = tmp_path / "transformer"
    folder.mkdir()
    ckpt = folder / "diffusion_pytorch_model.safetensors"
    save_file({"x": torch.zeros(1)}, ckpt)
    assert _sibling_quantization_config(ckpt) == {}  # no config.json: no fallback, and no Hub lookup
    (folder / "config.json").write_text(json.dumps({"_class_name": "X", "quantization_config": _CONFIG}))
    assert _sibling_quantization_config(ckpt) == _CONFIG


def test_target_lists_become_a_runtime_manifest():
    keys = {"blocks.0.attn.to_q.qweight", "blocks.0.attn.to_q.bias", "blocks.0.norm.linear.qweight"}
    config = with_target_list_manifest(_CONFIG, keys)
    assert "runtime_manifest" in config and "runtime_manifest" not in _CONFIG
    manifest = parse_runtime_manifest(config)
    assert manifest.requirements["rank"] == 64
    by_name = {t.name: t for t in manifest.targets}
    assert by_name["blocks.0.attn.to_q"].has_bias and not by_name["blocks.0.norm.linear"].has_bias
    assert with_target_list_manifest({"rank": 32}, keys) == {"rank": 32}  # no target lists: unchanged


def test_rank_is_inferred_from_proj_down():
    state = {"a.qweight": torch.zeros(8, 4, dtype=torch.int8), "a.proj_down": torch.zeros(3072, 64)}
    assert _infer_svdq_rank(state) == 64
    assert _infer_svdq_rank({"a.qweight": torch.zeros(8, 4, dtype=torch.int8)}) is None
    with pytest.warns(UserWarning, match="mixed ranks"):
        assert _infer_svdq_rank({**state, "b.proj_down": torch.zeros(3072, 32)}) == 64


def test_malformed_sibling_config_warns(tmp_path):
    ckpt = tmp_path / "model.safetensors"
    save_file({"x": torch.zeros(1)}, ckpt)
    (tmp_path / "config.json").write_text("{not json")
    with pytest.warns(UserWarning, match="malformed"):
        assert _sibling_quantization_config(ckpt) == {}
