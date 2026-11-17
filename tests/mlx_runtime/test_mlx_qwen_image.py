"""Original Nunchaku Qwen-Image checkpoints on mflux's Qwen-Image transformer, built without base weights."""

import mlx.core as mx
import numpy as np
import pytest

from nunchaku_lite.mlx import SVDQLogicalWeights, load_nunchaku_pipeline, pack_svdq_int4
from nunchaku_lite.mlx.packing import pack_awq_w4

pytest.importorskip("mflux")
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping
from mlx.utils import tree_flatten

D, HEADS, JOINT = 256, 2, 128  # head dim 128; MLP hidden 4 * D


def _transformer():
    return QwenTransformer(num_layers=1, num_attention_heads=HEADS, joint_attention_dim=JOINT)


class FakeQwenPipeline:
    def __init__(self, model_path: str, **kwargs):
        self.transformer = _transformer()


def _svdq(n, k, seed):
    rng = np.random.default_rng(seed)
    return SVDQLogicalWeights(
        q=rng.integers(-8, 8, size=(n, k), dtype=np.int8),
        wscale=rng.uniform(0.001, 0.02, size=(n, k // 64)).astype(np.float32),
        smooth=rng.uniform(0.5, 2.0, size=k).astype(np.float32),
        down=(rng.standard_normal((32, k)) * 0.05).astype(np.float32),
        up=(rng.standard_normal((n, 32)) * 0.05).astype(np.float32),
        bias=rng.standard_normal(n).astype(np.float32),
    )


def _ref(w, x):
    x = x.astype(np.float64)
    return (x / w.smooth) @ w.dequantize().astype(np.float64).T + (x @ w.down.T) @ w.up.T + w.bias


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))


def _run(layer, x):
    return np.array(layer(mx.array(x).astype(mx.bfloat16)).astype(mx.float32))


def _interleave(t, splits=6):
    """Nunchaku's AdaNorm layout: logical row ``j * D + c`` stored at ``c * splits + j``."""
    n = t.shape[0]
    return t.reshape(splits, n // splits, *t.shape[1:]).swapaxes(0, 1).reshape(t.shape)


def _checkpoint():
    to_diffusers = {}
    for target in QwenWeightMapping.get_transformer_mapping():
        to_diffusers[target.to_pattern] = target.from_pattern[0]
    ckpt, ws = {}, {}
    replaced = (".attn.to_", ".attn.add_", ".attn.attn_to_out", "_ff.", "_mod_linear.")
    for name, value in tree_flatten(_transformer().parameters()):
        if any(r in name for r in replaced):
            continue
        pattern = name.replace("transformer_blocks.0.", "transformer_blocks.{block}.")
        diffusers = to_diffusers.get(pattern, pattern).replace("{block}", "0")
        ckpt[diffusers] = mx.random.normal(value.shape).astype(mx.bfloat16) * 0.02
    p = "transformer_blocks.0"
    layers = {
        "attn.to_qkv": (3 * D, D),
        "attn.add_qkv_proj": (3 * D, D),
        "attn.to_out.0": (D, D),
        "attn.to_add_out": (D, D),
        "img_mlp.net.0.proj": (4 * D, D),
        "img_mlp.net.2": (D, 4 * D),
        "txt_mlp.net.0.proj": (4 * D, D),
        "txt_mlp.net.2": (D, 4 * D),
    }
    for j, (name, (n, k)) in enumerate(layers.items()):
        ws[f"{p}.{name}"] = w = _svdq(n, k, j)
        for t, v in pack_svdq_int4(w).items():
            ckpt[f"{p}.{name}.{t}"] = mx.array(v)
    rng = np.random.default_rng(9)
    for stream in ("img", "txt"):
        n, k = 6 * D, D
        codes = rng.integers(0, 16, size=(n, k)).astype(np.int32)
        scale = rng.uniform(0.001, 0.02, size=(n, k // 64)).astype(np.float32)
        zero = (-7 * scale).astype(np.float32)
        bias = rng.standard_normal(n).astype(np.float32)
        ws[f"{p}.{stream}_mod.1"] = (
            (codes.reshape(n, k // 64, 64) * scale[..., None] + zero[..., None]).reshape(n, k),
            bias,
        )
        ckpt[f"{p}.{stream}_mod.1.qweight"] = mx.array(pack_awq_w4(_interleave(codes)))
        ckpt[f"{p}.{stream}_mod.1.wscales"] = mx.array(_interleave(scale).T.copy()).astype(mx.bfloat16)
        ckpt[f"{p}.{stream}_mod.1.wzeros"] = mx.array(_interleave(zero).T.copy()).astype(mx.bfloat16)
        ckpt[f"{p}.{stream}_mod.1.bias"] = mx.array(_interleave(bias)).astype(mx.bfloat16)
    return ckpt, ws


def test_qwen_image_checkpoint_fills_a_transformer_built_without_weights():
    ckpt, ws = _checkpoint()
    pipe = load_nunchaku_pipeline("p", pipeline_cls=FakeQwenPipeline, checkpoint=ckpt, backend="dense")
    assert pipe._nunchaku_lite_target == "qwen_image"
    block = pipe.transformer.transformer_blocks[0]
    x = np.random.default_rng(3).standard_normal((4, D)).astype(np.float32)

    attn = block.attn
    qkv = np.concatenate([_run(getattr(attn, n), x) for n in ("add_q_proj", "add_k_proj", "add_v_proj")], -1)
    assert _rel(qkv, _ref(ws["transformer_blocks.0.attn.add_qkv_proj"], x)) < 2e-2
    assert _rel(_run(attn.attn_to_out[0], x), _ref(ws["transformer_blocks.0.attn.to_out.0"], x)) < 2e-2
    assert block.img_ff.mlp_out.act_unsigned and not block.img_ff.mlp_in.act_unsigned

    # modulation: logical nn.Linear order, bias without a folded +1 (mflux adds it in _modulate)
    dense, bias = ws["transformer_blocks.0.txt_mod.1"]
    expected = x.astype(np.float64) @ dense.T + bias
    assert _rel(_run(block.txt_mod_linear, x), expected) < 2e-2
