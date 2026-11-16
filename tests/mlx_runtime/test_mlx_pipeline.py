import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from nunchaku_lite.mlx import SVDQLogicalWeights, SVDQW4A4Linear, list_adapters, load_nunchaku_pipeline, pack_svdq_int4
from nunchaku_lite.mlx.pipeline import ConcatProjection, ProjectionSlice, SharedProjection, SplitSumProjection

D, HEADS, HEAD_DIM = 256, 2, 128  # inner_dim = 256
MLP = 128  # single-block mlp hidden (SwiGLU: fc1 outputs 2 * MLP)


def _weights(n, k, seed):
    rng = np.random.default_rng(seed)
    return SVDQLogicalWeights(
        q=rng.integers(-8, 8, size=(n, k), dtype=np.int8),
        wscale=rng.uniform(0.001, 0.02, size=(n, k // 64)).astype(np.float32),
        smooth=rng.uniform(0.5, 2.0, size=k).astype(np.float32),
        down=(rng.standard_normal((32, k)) * 0.05).astype(np.float32),
        up=(rng.standard_normal((n, 32)) * 0.05).astype(np.float32),
        bias=None,
    )


def _ref(w, x):
    x = x.astype(np.float64)
    return (x / w.smooth) @ w.dequantize().astype(np.float64).T + (x @ w.down.T) @ w.up.T


def _ckpt(prefix, w):
    return {f"{prefix}.{k}": mx.array(v) for k, v in pack_svdq_int4(w).items()}


def _layer(w):
    layer = SVDQW4A4Linear(w.in_features, w.out_features, rank=32, bias=False, dtype=mx.float32, backend="dense")
    layer.load_weights([(k.split(".", 1)[1], v) for k, v in _ckpt("x", w).items()])
    return layer


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))


class _Counting(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self._calls = 0

    def __call__(self, x):
        self._calls += 1
        return self.layer(x)


def test_shared_projection_computes_once_and_frees():
    w = _weights(3 * D, D, 0)
    counting = _Counting(_layer(w))
    shared = SharedProjection(counting, consumers=3)
    slices = [ProjectionSlice(shared, j * D, (j + 1) * D, owner=j == 0) for j in range(3)]
    x = mx.array(np.random.default_rng(1).standard_normal((4, D)).astype(np.float32))
    outs = [np.array(s(x)) for s in slices]
    assert counting._calls == 1 and shared._out is None  # one fused matmul, released after the last read
    np.testing.assert_allclose(np.concatenate(outs, -1), _ref(w, np.array(x)), rtol=2e-3, atol=2e-3)
    slices[0](mx.array(np.ones((4, D), np.float32)))  # new input -> recomputed
    assert counting._calls == 2
    # the shared layer's parameters appear once (only under the owning slice)
    keys = [k for k, _ in nn.utils.tree_flatten(nn.Module.parameters(_Holder(slices)))]
    assert sum(k.endswith("qweight") for k in keys) == 1


class _Holder(nn.Module):
    def __init__(self, parts):
        super().__init__()
        self.parts = parts


def test_concat_and_split_sum():
    w1, w2 = _weights(256, D, 2), _weights(128, D, 3)
    x = np.random.default_rng(4).standard_normal((3, D)).astype(np.float32)
    out = np.array(ConcatProjection([_layer(w1), _layer(w2)])(mx.array(x)))
    np.testing.assert_allclose(out, np.concatenate([_ref(w1, x), _ref(w2, x)], -1), rtol=2e-3, atol=2e-3)

    wa, wb = _weights(D, 256, 5), _weights(D, 128, 6)
    xy = np.random.default_rng(7).standard_normal((3, 384)).astype(np.float32)
    out = np.array(SplitSumProjection([_layer(wa), _layer(wb)], [256])(mx.array(xy)))
    assert _rel(out, _ref(wa, xy[:, :256]) + _ref(wb, xy[:, 256:])) < 2e-3


# ---------------------------------------------------------------------------
# FLUX.2 adapter on a miniature mflux-shaped transformer
# ---------------------------------------------------------------------------


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_dim = D
        for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_out", "to_add_out"):
            setattr(self, name, nn.Linear(D, D, bias=False))
        self.norm_q = nn.RMSNorm(HEAD_DIM)


class _FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_in = nn.Linear(D, 2 * 128, bias=False)
        self.linear_out = nn.Linear(128, D, bias=False)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn, self.ff, self.ff_context = _Attn(), _FF(), _FF()


class _ParallelAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_dim = D
        self.to_qkv_mlp_proj = nn.Linear(D, 3 * D + 2 * MLP, bias=False)
        self.to_out = nn.Linear(D + MLP, D, bias=False)


class _Single(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _ParallelAttn()


class _TimeGuidance(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(256, D, bias=False)


class Flux2Transformer(nn.Module):  # name matches mflux for adapter auto-selection
    def __init__(self):
        super().__init__()
        self.x_embedder = nn.Linear(128, D, bias=False)
        self.time_guidance_embed = _TimeGuidance()
        self.transformer_blocks = [_Block()]
        self.single_transformer_blocks = [_Single()]


class FakeFlux2Pipeline:
    def __init__(self, model_path: str, **kwargs):
        self.model_path, self.kwargs = model_path, kwargs
        self.transformer = Flux2Transformer()


def _flux2_checkpoint():
    ws, ckpt = {}, {}
    specs = {
        "transformer_blocks.0.attn.to_qkv": (3 * D, D),
        "transformer_blocks.0.attn.to_added_qkv": (3 * D, D),
        "transformer_blocks.0.attn.to_out.0": (D, D),
        "transformer_blocks.0.attn.to_add_out": (D, D),
        "transformer_blocks.0.ff.linear_in": (256, D),
        "transformer_blocks.0.ff.linear_out": (D, 128),
        "transformer_blocks.0.ff_context.linear_in": (256, D),
        "transformer_blocks.0.ff_context.linear_out": (D, 128),
        "single_transformer_blocks.0.attn.qkv_proj": (3 * D, D),
        "single_transformer_blocks.0.attn.mlp_fc1": (2 * MLP, D),
        "single_transformer_blocks.0.attn.out_proj": (D, D),
        "single_transformer_blocks.0.attn.mlp_fc2": (D, MLP),
    }
    for i, (prefix, (n, k)) in enumerate(specs.items()):
        ws[prefix] = _weights(n, k, 100 + i)
        ckpt.update(_ckpt(prefix, ws[prefix]))
        ckpt[f"{prefix}.smooth_factor_orig"] = mx.ones((k,))  # ignored
    ckpt["x_embedder.weight"] = mx.full((D, 128), 0.5)
    ckpt["time_guidance_embed.timestep_embedder.linear_1.weight"] = mx.full((D, 256), 0.25)
    ckpt["transformer_blocks.0.attn.norm_q.weight"] = mx.full((HEAD_DIM,), 3.0)
    return ws, ckpt


def test_flux2_adapter_end_to_end():
    assert "flux2" in list_adapters()
    ws, ckpt = _flux2_checkpoint()
    pipe = load_nunchaku_pipeline("base/path", pipeline_cls=FakeFlux2Pipeline, checkpoint=ckpt, backend="dense", flag=1)
    assert pipe.model_path == "base/path" and pipe.kwargs == {"flag": 1} and pipe._nunchaku_lite_target == "flux2"
    tr = pipe.transformer
    # dense weights from the checkpoint, including the time_guidance_embed rename
    np.testing.assert_allclose(np.array(tr.x_embedder.weight), 0.5)
    np.testing.assert_allclose(np.array(tr.time_guidance_embed.linear_1.weight), 0.25)
    np.testing.assert_allclose(np.array(tr.transformer_blocks[0].attn.norm_q.weight), 3.0)

    rng = np.random.default_rng(9)
    x = rng.standard_normal((5, D)).astype(np.float32)
    attn = tr.transformer_blocks[0].attn
    q, k, v = (
        np.array(getattr(attn, n)(mx.array(x).astype(mx.bfloat16)).astype(mx.float32)) for n in ("to_q", "to_k", "to_v")
    )
    assert _rel(np.concatenate([q, k, v], -1), _ref(ws["transformer_blocks.0.attn.to_qkv"], x)) < 2e-2
    out = np.array(attn.to_out(mx.array(x).astype(mx.bfloat16)).astype(mx.float32))
    assert _rel(out, _ref(ws["transformer_blocks.0.attn.to_out.0"], x)) < 2e-2

    single = tr.single_transformer_blocks[0].attn
    proj = np.array(single.to_qkv_mlp_proj(mx.array(x).astype(mx.bfloat16)).astype(mx.float32))
    ref = np.concatenate(
        [
            _ref(ws["single_transformer_blocks.0.attn.qkv_proj"], x),
            _ref(ws["single_transformer_blocks.0.attn.mlp_fc1"], x),
        ],
        -1,
    )
    assert proj.shape == (5, 3 * D + 2 * MLP) and _rel(proj, ref) < 2e-2
    xy = rng.standard_normal((5, D + MLP)).astype(np.float32)
    out = np.array(single.to_out(mx.array(xy).astype(mx.bfloat16)).astype(mx.float32))
    ref = _ref(ws["single_transformer_blocks.0.attn.out_proj"], xy[:, :D]) + _ref(
        ws["single_transformer_blocks.0.attn.mlp_fc2"], xy[:, D:]
    )
    assert _rel(out, ref) < 2e-2
    layers = {id(m): m for _, m in tr.named_modules() if isinstance(m, SVDQW4A4Linear)}
    assert len(layers) == 12 and all(m.backend == "dense" for m in layers.values())


def test_flux2_adapter_strict_and_selection_errors():
    _, ckpt = _flux2_checkpoint()
    ckpt["unknown.weight"] = mx.zeros((2,))
    with pytest.raises(ValueError, match="do not match model parameters"):
        load_nunchaku_pipeline("p", pipeline_cls=FakeFlux2Pipeline, checkpoint=ckpt, backend="dense")
    load_nunchaku_pipeline("p", pipeline_cls=FakeFlux2Pipeline, checkpoint=ckpt, backend="dense", strict=False)
    with pytest.raises(ValueError, match="Unknown adapter target"):
        load_nunchaku_pipeline("p", pipeline_cls=FakeFlux2Pipeline, checkpoint=ckpt, target="qwen_image")
    with pytest.raises(NotImplementedError):
        load_nunchaku_pipeline("p", pipeline_cls=FakeFlux2Pipeline, checkpoint=ckpt, precision="fp4")
    with pytest.raises(ValueError, match="backend must be one of"):
        load_nunchaku_pipeline("p", pipeline_cls=FakeFlux2Pipeline, checkpoint=ckpt, backend="w8a8")
