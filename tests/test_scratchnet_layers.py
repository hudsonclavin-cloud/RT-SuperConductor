"""Every hand-written backward pass is checked against finite differences.

The checks run in float64. For a layer with output y, the test loss is
sum(y * R) for a fixed random R, so the gradient handed to ``backward`` is R
and every output entry is exercised.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rt_superconductor.scratchnet import layers as L  # noqa: E402
from rt_superconductor.scratchnet.optim import (  # noqa: E402
    AdamW, clip_by_global_norm, numerical_gradient, relative_error, warmup_cosine,
)

F64 = np.float64
TOLERANCE = 1e-6


def sample_entries(array, rng, count=12):
    flat = rng.choice(array.size, size=min(count, array.size), replace=False)
    return [tuple(int(i) for i in np.unravel_index(f, array.shape)) for f in flat]


def check_layer(layer, forward, inputs, rng, input_names=("x",)):
    """Compare analytic and numerical gradients for all parameters and the named inputs."""
    out = forward()
    weights = rng.normal(size=out.shape)
    loss = lambda: float((forward() * weights).sum())   # noqa: E731
    forward()
    d_inputs = layer.backward(weights.copy())
    worst = 0.0
    for name, value in layer.params.items():
        entries = sample_entries(value, rng)
        numeric = numerical_gradient(loss, value, entries)
        analytic = np.array([layer.grads[name][e] for e in entries])
        worst = max(worst, relative_error(analytic, numeric))
    if d_inputs is not None:
        d_inputs = d_inputs if isinstance(d_inputs, tuple) else (d_inputs,)
        for name, grad in zip(input_names, d_inputs):
            entries = sample_entries(inputs[name], rng)
            numeric = numerical_gradient(loss, inputs[name], entries)
            worst = max(worst, relative_error(np.array([grad[e] for e in entries]), numeric))
    return worst


def batch(rng, b=4, n=5, d=8):
    x = rng.normal(size=(b, n, d))
    mask = np.ones((b, n), bool)
    mask[0, 3:] = False          # a composition with 3 elements
    mask[1, 1:] = False          # a single element
    mask[2, 4:] = False
    fractions = rng.random((b, n)) * mask
    fractions /= fractions.sum(axis=1, keepdims=True)
    return x, mask, fractions


def test_linear():
    rng = np.random.default_rng(0)
    layer = L.Linear(6, 4, rng, F64)
    layer.params["b"][:] = rng.normal(size=4)
    inputs = {"x": rng.normal(size=(3, 5, 6))}
    assert check_layer(layer, lambda: layer.forward(inputs["x"]), inputs, rng) < TOLERANCE


def test_embedding_and_padding_row():
    rng = np.random.default_rng(1)
    layer = L.Embedding(7, 5, rng, F64)
    index = np.array([[1, 2, 0], [3, 3, 6]])
    out = layer.forward(index)
    assert np.all(out[0, 2] == 0)                          # the padding row is zero
    weights = rng.normal(size=out.shape)
    loss = lambda: float((layer.forward(index) * weights).sum())   # noqa: E731
    layer.backward(weights)
    entries = [(1, 0), (3, 2), (6, 4), (2, 1)]
    numeric = numerical_gradient(loss, layer.params["E"], entries)
    assert relative_error(np.array([layer.grads["E"][e] for e in entries]), numeric) < TOLERANCE
    assert np.all(layer.grads["E"][0] == 0)                # and never receives a gradient
    assert np.allclose(layer.grads["E"][3], weights[1, 0] + weights[1, 1])   # repeated index accumulates


def test_layernorm():
    rng = np.random.default_rng(2)
    layer = L.LayerNorm(6, F64)
    layer.params["gamma"][:] = rng.uniform(0.5, 1.5, size=6)
    layer.params["beta"][:] = rng.normal(size=6)
    inputs = {"x": rng.normal(size=(3, 4, 6))}
    assert check_layer(layer, lambda: layer.forward(inputs["x"]), inputs, rng) < TOLERANCE
    out = layer.forward(inputs["x"])
    plain = (out - layer.params["beta"]) / layer.params["gamma"]
    assert np.allclose(plain.mean(axis=-1), 0, atol=1e-9) and np.allclose(plain.std(axis=-1), 1, atol=1e-4)


def test_silu():
    rng = np.random.default_rng(3)
    layer = L.SiLU()
    inputs = {"x": rng.normal(size=(4, 7)) * 3}
    assert check_layer(layer, lambda: layer.forward(inputs["x"]), inputs, rng) < TOLERANCE


def test_dropout_scales_in_training_and_is_identity_in_eval():
    rng = np.random.default_rng(4)
    layer = L.Dropout(0.3, np.random.default_rng(5))
    x = np.ones((200, 50))
    out = layer.forward(x)
    assert set(np.unique(out)) <= {0.0, 1.0 / 0.7}
    assert abs(out.mean() - 1.0) < 0.05                    # expectation is preserved
    grad = layer.backward(np.ones_like(x))
    assert np.array_equal(grad, out)                       # same mask on the way back
    layer.training = False
    assert layer.forward(x) is x and layer.backward(x) is x
    assert rng is not None


@pytest.mark.parametrize("heads", [1, 2, 4])
def test_self_attention(heads):
    rng = np.random.default_rng(10 + heads)
    x, mask, fractions = batch(rng)
    layer = L.SelfAttention(8, heads, rng, F64)
    for name in ("bq", "bv", "bo"):
        layer.params[name][:] = rng.normal(size=8) * 0.1
    bias = np.log(np.where(mask, fractions, 1.0))
    inputs = {"x": x}
    assert check_layer(layer, lambda: layer.forward(inputs["x"], mask, bias), inputs, rng) < TOLERANCE


def test_self_attention_has_no_key_bias_because_its_gradient_is_always_zero():
    rng = np.random.default_rng(15)
    x, mask, _ = batch(rng)
    layer = L.SelfAttention(8, 2, rng, F64)
    assert "bk" not in layer.params
    out = layer.forward(x, mask)
    shifted = L.masked_softmax((layer._q @ (layer._k + 3.7).transpose(0, 1, 3, 2)) * layer._scale,
                               mask[:, None, None, :])
    assert np.allclose(shifted, layer._attention)          # a constant added to every key changes nothing
    assert out.shape == x.shape


def test_self_attention_ignores_padding():
    rng = np.random.default_rng(20)
    x, mask, _ = batch(rng)
    layer = L.SelfAttention(8, 2, rng, F64)
    out = layer.forward(x, mask)
    assert np.all(out[~mask] == 0)                         # padded positions output nothing
    changed = x.copy()
    changed[~mask] = rng.normal(size=changed[~mask].shape) * 100
    assert np.allclose(layer.forward(changed, mask), out)  # and what sits in padding cannot leak in
    grad = layer.backward(np.ones_like(out))
    assert np.all(grad[~mask] == 0)


@pytest.mark.parametrize("heads", [1, 3])
def test_attention_pool(heads):
    rng = np.random.default_rng(30 + heads)
    x, mask, fractions = batch(rng)
    layer = L.AttentionPool(8, heads, rng, F64)
    layer.params["b"][:] = rng.normal(size=heads)
    bias = np.log(np.where(mask, fractions, 1.0))
    inputs = {"x": x}
    assert check_layer(layer, lambda: layer.forward(inputs["x"], mask, bias), inputs, rng) < TOLERANCE
    out = layer.forward(x, mask, bias)
    assert out.shape == (4, heads * 8)
    assert np.allclose(layer._weights.sum(axis=-1), 1.0)   # weights are a proper average
    assert np.all(layer._weights[0, :, 3:] == 0)           # with nothing on padding
    assert np.allclose(out[1].reshape(heads, 8), x[1, 0])  # one element: the pool returns it


def test_weighted_mean_pool():
    rng = np.random.default_rng(40)
    x, mask, fractions = batch(rng)
    weights = rng.normal(size=(4, 8))
    loss = lambda: float((L.weighted_mean_forward(x, fractions) * weights).sum())   # noqa: E731
    grad = L.weighted_mean_backward(weights, fractions)
    entries = sample_entries(x, rng, 20)
    assert relative_error(np.array([grad[e] for e in entries]), numerical_gradient(loss, x, entries)) < TOLERANCE


def test_masked_softmax_handles_empty_rows_and_large_scores():
    scores = np.array([[1e4, 0.0, -1e4], [3.0, 2.0, 1.0]])
    valid = np.array([[True, True, False], [False, False, False]])
    out = L.masked_softmax(scores, valid)
    assert np.allclose(out[0], [1.0, 0.0, 0.0]) and np.all(out[1] == 0) and np.isfinite(out).all()


def test_sigmoid_is_stable():
    x = np.array([-1000.0, -5.0, 0.0, 5.0, 1000.0])
    out = L.sigmoid(x)
    assert np.isfinite(out).all() and out[0] == 0.0 and out[-1] == 1.0 and out[2] == 0.5


def test_bce_with_logits():
    rng = np.random.default_rng(50)
    logits = rng.normal(size=9) * 4
    target = (rng.random(9) > 0.5).astype(F64)
    weight = np.array([1, 1, 0, 2, 1, 0, 1, 1, 1], F64)   # two rows carry no label
    loss, grad = L.bce_with_logits(logits, target, weight)
    probability = L.sigmoid(logits)
    expected = -(weight * (target * np.log(probability) + (1 - target) * np.log(1 - probability))).sum() / weight.sum()
    assert loss == pytest.approx(expected)
    entries = [(i,) for i in range(9)]
    numeric = numerical_gradient(lambda: L.bce_with_logits(logits, target, weight)[0], logits, entries)
    assert relative_error(grad, numeric) < TOLERANCE
    assert grad[2] == 0 and grad[5] == 0
    big, _ = L.bce_with_logits(np.array([800.0, -800.0]), np.array([0.0, 1.0]), np.ones(2))
    assert np.isfinite(big) and big == pytest.approx(800.0)
    assert L.bce_with_logits(logits, target, np.zeros(9)) [0] == 0.0


def test_huber():
    rng = np.random.default_rng(60)
    prediction = rng.normal(size=10) * 2
    target = rng.normal(size=10)
    weight = (rng.random(10) > 0.3).astype(F64)
    loss, grad = L.huber(prediction, target, weight, delta=1.0)
    residual = prediction - target
    expected = np.where(np.abs(residual) <= 1, 0.5 * residual ** 2, np.abs(residual) - 0.5)
    assert loss == pytest.approx((expected * weight).sum() / weight.sum())
    entries = [(i,) for i in range(10)]
    numeric = numerical_gradient(lambda: L.huber(prediction, target, weight)[0], prediction, entries)
    assert relative_error(grad, numeric) < 1e-5
    assert np.all(grad[weight == 0] == 0)


def test_adamw_matches_the_published_update_and_decays_only_named_weights():
    w = np.array([1.0, -2.0])
    b = np.array([0.5])
    grad_w, grad_b = np.array([0.1, -0.3]), np.array([0.2])
    optimizer = AdamW(["w", "b"], lr=0.01, weight_decay=0.1, decay_names={"w"})
    optimizer.step([("w", w, grad_w), ("b", b, grad_b)])
    # First Adam step moves each weight by lr * sign(grad); decay multiplies w by (1 - lr * wd) first.
    assert np.allclose(w, np.array([1.0, -2.0]) * (1 - 0.01 * 0.1) - 0.01 * np.sign(grad_w), atol=1e-7)
    assert np.allclose(b, 0.5 - 0.01, atol=1e-7)
    state = optimizer.state()
    clone = AdamW(["w", "b"], lr=0.01, weight_decay=0.1, decay_names={"w"})
    clone.load_state(state)
    w2, b2 = w.copy(), b.copy()
    optimizer.step([("w", w, grad_w), ("b", b, grad_b)])
    clone.step([("w", w2, grad_w), ("b", b2, grad_b)])
    assert np.array_equal(w, w2) and np.array_equal(b, b2)     # a restored optimizer continues identically


def test_adamw_minimizes_a_quadratic():
    w = np.array([5.0, -3.0])
    optimizer = AdamW(["w"], lr=0.1)
    for _ in range(300):
        optimizer.step([("w", w, 2 * w)])
    assert np.abs(w).max() < 1e-2


def test_schedule_and_clipping():
    assert warmup_cosine(0, 100, 1.0, 10) == pytest.approx(0.1)
    assert warmup_cosine(9, 100, 1.0, 10) == pytest.approx(1.0)
    assert warmup_cosine(55, 100, 1.0, 10) == pytest.approx(0.5)
    assert warmup_cosine(100, 100, 1.0, 10, floor=0.05) == pytest.approx(0.05)
    grads = [np.array([3.0, 0.0]), np.array([4.0])]
    assert clip_by_global_norm(grads, 1.0) == pytest.approx(5.0)
    assert np.sqrt(sum((g ** 2).sum() for g in grads)) == pytest.approx(1.0)
    untouched = [np.array([0.3])]
    clip_by_global_norm(untouched, 1.0)
    assert untouched[0][0] == 0.3
