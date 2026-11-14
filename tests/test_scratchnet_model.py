"""Whole-model checks for the from-scratch composition network."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rt_superconductor.scratchnet import layers as L  # noqa: E402
from rt_superconductor.scratchnet.model import CompositionNet, Config  # noqa: E402
from rt_superconductor.scratchnet.optim import AdamW, numerical_gradient, relative_error  # noqa: E402

F64 = np.float64


def small_config(**changes):
    base = dict(n_elements=12, n_features=5, width=8, blocks=2, heads=2, pool_heads=2, trunk=(10, 6), dropout=0.0)
    base.update(changes)
    return Config(**base)


def toy_batch(rng, config, b=6, n=5):
    features = np.vstack([np.zeros((1, config.n_features)), rng.normal(size=(config.n_elements, config.n_features))])
    index = np.zeros((b, n), np.int64)
    fraction = np.zeros((b, n))
    for row in range(b):
        count = int(rng.integers(1, n + 1))
        index[row, :count] = rng.choice(np.arange(1, config.n_elements + 1), size=count, replace=False)
        amounts = rng.random(count) + 0.05
        fraction[row, :count] = amounts / amounts.sum()
    target_class = (rng.random(b) > 0.5).astype(F64)
    target_tc = rng.normal(size=b)
    has_tc = target_class.copy()
    return features, index, fraction, index > 0, target_class, target_tc, has_tc


def loss_and_backward(model, index, fraction, mask, target_class, target_tc, has_tc, tc_weight=0.7):
    logit, tc = model.forward(index, fraction, mask)
    loss_c, d_logit = L.bce_with_logits(logit, target_class, np.ones_like(target_class))
    loss_r, d_tc = L.huber(tc, target_tc, has_tc)
    model.backward(d_logit, tc_weight * d_tc)
    return loss_c + tc_weight * loss_r


@pytest.mark.parametrize("changes", [
    {},                                        # attention model
    {"blocks": 0},                             # deep-sets model
    {"use_embedding": False},
    {"use_features": False},
    {"fraction_bias": False, "heads": 1, "pool_heads": 1},
])
def test_whole_model_gradient_matches_finite_differences(changes):
    rng = np.random.default_rng(0)
    config = small_config(**changes)
    features, index, fraction, mask, y, tc, has_tc = toy_batch(rng, config)
    model = CompositionNet(config, features, seed=1, dtype=F64)
    model.train(False)
    for _, value, _ in model.parameters():       # move biases and norms off their special initial values
        value += rng.normal(size=value.shape) * 0.05
    if config.use_embedding:
        model.layers["embedding"].params["E"][0] = 0.0
    args = (index, fraction, mask, y, tc, has_tc)
    loss_and_backward(model, *args)
    analytic = {name: grad.copy() for name, _, grad in model.parameters()}

    def loss():
        logit, out = model.forward(index, fraction, mask)
        return L.bce_with_logits(logit, y, np.ones_like(y))[0] + 0.7 * L.huber(out, tc, has_tc)[0]

    worst = 0.0
    for name, value, _ in model.parameters():
        flat = rng.choice(value.size, size=min(6, value.size), replace=False)
        entries = [tuple(int(i) for i in np.unravel_index(f, value.shape)) for f in flat]
        if name == "embedding.E":
            used = np.unique(index[index > 0])
            entries = [(int(rng.choice(used)), int(rng.integers(config.width))) for _ in range(8)]
        numeric = numerical_gradient(loss, value, entries)
        worst = max(worst, relative_error(np.array([analytic[name][e] for e in entries]), numeric))
    assert worst < 1e-5, worst


def test_gradient_with_dropout_when_the_random_mask_is_replayed():
    rng = np.random.default_rng(3)
    config = small_config(dropout=0.3)
    features, index, fraction, mask, y, tc, has_tc = toy_batch(rng, config)
    model = CompositionNet(config, features, seed=2, dtype=F64)
    model.train(True)
    saved = model.dropout_rng.bit_generator.state

    def replay():
        model.dropout_rng.bit_generator.state = saved

    replay()
    loss_and_backward(model, index, fraction, mask, y, tc, has_tc)
    analytic = {name: grad.copy() for name, _, grad in model.parameters()}

    def loss():
        replay()
        logit, out = model.forward(index, fraction, mask)
        return L.bce_with_logits(logit, y, np.ones_like(y))[0] + 0.7 * L.huber(out, tc, has_tc)[0]

    for name in ("block0.attention.Wq", "block1.ff1.W", "trunk0.linear.W", "fraction_proj.W", "pool.W"):
        value = dict((n, v) for n, v, _ in model.parameters())[name]
        entries = [tuple(int(i) for i in np.unravel_index(f, value.shape))
                   for f in rng.choice(value.size, size=6, replace=False)]
        numeric = numerical_gradient(loss, value, entries)
        assert relative_error(np.array([analytic[name][e] for e in entries]), numeric) < 1e-5, name


def test_gradients_match_pytorch_autograd():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    config = small_config()
    features, index, fraction, mask, y, tc, has_tc = toy_batch(rng, config)
    model = CompositionNet(config, features, seed=4, dtype=F64)
    model.train(False)
    for _, value, _ in model.parameters():
        value += rng.normal(size=value.shape) * 0.05
    model.layers["embedding"].params["E"][0] = 0.0
    loss_numpy = loss_and_backward(model, index, fraction, mask, y, tc, has_tc)

    p = {name: torch.tensor(value, dtype=torch.float64, requires_grad=True) for name, value, _ in model.parameters()}
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)   # noqa: E731
    idx, fr, m = torch.tensor(index), t(fraction), torch.tensor(mask)
    mf = m.unsqueeze(-1).double()
    logf = torch.log(torch.clamp(fr, min=1e-6)) * m
    silu = torch.nn.functional.silu

    def norm(x, prefix):
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + 1e-5) * p[prefix + ".gamma"] + p[prefix + ".beta"]

    amount = torch.stack([fr, logf / 4.0, torch.sqrt(fr)], dim=-1)
    x = amount @ p["fraction_proj.W"] + p["fraction_proj.b"] + p["embedding.E"][idx] \
        + t(features)[idx] @ p["feature_proj.W"]
    x = norm(x * mf, "input_norm") * mf
    b, n, w = x.shape
    h, hw = config.heads, config.width // config.heads
    for i in range(config.blocks):
        pre = f"block{i}."
        split = lambda a: a.reshape(b, n, h, hw).permute(0, 2, 1, 3)   # noqa: E731
        q = split(x @ p[pre + "attention.Wq"] + p[pre + "attention.bq"])
        k = split(x @ p[pre + "attention.Wk"])
        v = split(x @ p[pre + "attention.Wv"] + p[pre + "attention.bv"])
        scores = q @ k.transpose(-1, -2) / hw ** 0.5 + logf[:, None, None, :]
        scores = scores.masked_fill(~m[:, None, None, :], float("-inf"))
        attended = (torch.softmax(scores, dim=-1) @ v).permute(0, 2, 1, 3).reshape(b, n, w)
        attended = (attended @ p[pre + "attention.Wo"] + p[pre + "attention.bo"]) * mf
        x = norm(x + attended, pre + "norm1") * mf
        hidden = silu(x @ p[pre + "ff1.W"] + p[pre + "ff1.b"]) @ p[pre + "ff2.W"] + p[pre + "ff2.b"]
        x = norm(x + hidden, pre + "norm2") * mf
    scores = (x @ p["pool.W"] + p["pool.b"] + logf.unsqueeze(-1)).permute(0, 2, 1)
    weights = torch.softmax(scores.masked_fill(~m[:, None, :], float("-inf")), dim=-1)
    pooled = torch.cat([(weights @ x).reshape(b, -1), (x * fr.unsqueeze(-1)).sum(1)], dim=1)
    z = pooled
    for i in range(len(config.trunk)):
        z = silu(z @ p[f"trunk{i}.linear.W"] + p[f"trunk{i}.linear.b"])
    logit = (z @ p["head_class.W"] + p["head_class.b"])[:, 0]
    out = (z @ p["head_tc.W"] + p["head_tc.b"])[:, 0]
    loss_c = torch.nn.functional.binary_cross_entropy_with_logits(logit, t(y))
    loss_r = (torch.nn.functional.huber_loss(out, t(tc), reduction="none") * t(has_tc)).sum() / t(has_tc).sum()
    loss = loss_c + 0.7 * loss_r
    loss.backward()

    assert loss.item() == pytest.approx(loss_numpy, rel=1e-10)
    for name, _, grad in model.parameters():
        theirs = p[name].grad.numpy()
        if name == "embedding.E":
            theirs = theirs.copy()
            theirs[0] = 0.0                       # our padding row is frozen by design
        assert np.allclose(grad, theirs, rtol=1e-8, atol=1e-10), name


def test_output_does_not_depend_on_element_order_or_on_padding():
    rng = np.random.default_rng(7)
    config = small_config()
    features, index, fraction, mask, *_ = toy_batch(rng, config, b=5, n=4)
    model = CompositionNet(config, features, seed=3, dtype=F64)
    model.train(False)
    logit, tc = model.forward(index, fraction, mask)
    order = rng.permutation(index.shape[1])
    logit2, tc2 = model.forward(index[:, order], fraction[:, order], mask[:, order])
    assert np.allclose(logit, logit2) and np.allclose(tc, tc2)
    pad = lambda a: np.concatenate([a, np.zeros((a.shape[0], 3), a.dtype)], axis=1)   # noqa: E731
    logit3, tc3 = model.forward(pad(index), pad(fraction), pad(mask))
    assert np.allclose(logit, logit3) and np.allclose(tc, tc3)


def test_a_tiny_dataset_can_be_memorized():
    rng = np.random.default_rng(9)
    config = small_config(width=16, trunk=(32, 16))
    features, index, fraction, mask, y, tc, has_tc = toy_batch(rng, config, b=16, n=4)
    model = CompositionNet(config, features, seed=5, dtype=F64)
    model.train(True)
    optimizer = AdamW([n for n, _, _ in model.parameters()], lr=5e-3)
    first = None
    for step in range(400):
        loss = loss_and_backward(model, index, fraction, mask, y, tc, has_tc)
        first = loss if first is None else first
        optimizer.step(model.parameters())
    final = loss_and_backward(model, index, fraction, mask, y, tc, has_tc)
    assert final < 0.05 * first and final < 0.05, (first, final)
    logit, _ = model.predict(index, fraction, mask)
    assert np.array_equal(logit > 0, y > 0.5)


def test_save_and_load_reproduce_predictions(tmp_path):
    rng = np.random.default_rng(11)
    config = small_config(dropout=0.2)
    features, index, fraction, mask, *_ = toy_batch(rng, config)
    model = CompositionNet(config, features.astype(np.float32), seed=6)
    before = model.predict(index, fraction, mask)
    model.save(tmp_path / "model.npz", extra={"note": "test", "tc_mean": 1.5})
    clone, extra = CompositionNet.load(tmp_path / "model.npz", features.astype(np.float32))
    after = clone.predict(index, fraction, mask)
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])
    assert extra == {"note": "test", "tc_mean": 1.5} and clone.config == config


def test_same_seed_same_model_and_predict_leaves_training_mode_alone():
    rng = np.random.default_rng(13)
    config = small_config(dropout=0.5)
    features, index, fraction, mask, *_ = toy_batch(rng, config)
    a = CompositionNet(config, features, seed=8, dtype=F64)
    b = CompositionNet(config, features, seed=8, dtype=F64)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a.parameters(), b.parameters()))
    a.train(True)
    first = a.predict(index, fraction, mask)
    second = a.predict(index, fraction, mask)
    assert np.array_equal(first[0], second[0])            # predict() never applies dropout
    assert a.layers["input_drop"].training is True         # and restores the mode it found
    assert a.parameter_count() == sum(v.size for _, v, _ in a.parameters()) > 0
    assert "embedding.E" not in a.decay_names() and "trunk0.linear.W" in a.decay_names()
    assert not any(name.endswith((".b", ".gamma", ".beta")) or ".b" == name[-2:] for name in a.decay_names())


@pytest.mark.parametrize("changes", [
    {"n_aux": 3},
    {"n_global": 4},
    {"n_aux": 2, "n_global": 4, "blocks": 0},
])
def test_gradient_through_the_auxiliary_outputs_and_the_formula_branch(changes):
    rng = np.random.default_rng(3)
    config = small_config(**changes)
    features, index, fraction, mask, y, tc, has_tc = toy_batch(rng, config)
    extra = rng.normal(size=(len(index), config.n_global)) if config.n_global else None
    aux_target = rng.normal(size=(len(index), config.n_aux)) if config.n_aux else None
    aux_known = (rng.random(aux_target.shape) > 0.3).astype(F64) if config.n_aux else None
    model = CompositionNet(config, features, seed=2, dtype=F64)
    model.train(False)
    for _, value, _ in model.parameters():
        value += rng.normal(size=value.shape) * 0.05
    model.layers["embedding"].params["E"][0] = 0.0

    def loss(backward=False):
        logit, out = model.forward(index, fraction, mask, extra)
        loss_c, d_logit = L.bce_with_logits(logit, y, np.ones_like(y))
        loss_r, d_tc = L.huber(out, tc, has_tc)
        total, d_aux = loss_c + 0.7 * loss_r, None
        if config.n_aux:
            loss_a, d_flat = L.huber(model.aux.ravel(), aux_target.ravel(), aux_known.ravel())
            total, d_aux = total + 0.4 * loss_a, 0.4 * d_flat.reshape(aux_target.shape)
        if backward:
            model.backward(d_logit, 0.7 * d_tc, d_aux)
        return total

    loss(backward=True)
    analytic = {name: grad.copy() for name, _, grad in model.parameters()}
    worst = 0.0
    for name, value, _ in model.parameters():
        if name == "embedding.E":
            continue                              # covered by the test above; unused rows have no gradient
        flat = rng.choice(value.size, size=min(8, value.size), replace=False)
        entries = [tuple(int(i) for i in np.unravel_index(f, value.shape)) for f in flat]
        if name == "trunk0.linear.W" and config.n_global:      # make sure the widened rows are probed
            entries += [(value.shape[0] - 1 - i, i % value.shape[1]) for i in range(4)]
        numeric = numerical_gradient(loss, value, entries)
        worst = max(worst, relative_error(np.array([analytic[name][e] for e in entries]), numeric))
    assert worst < 1e-5, worst


def test_optional_parts_do_not_change_how_the_rest_is_initialized():
    rng = np.random.default_rng(0)
    plain_config = small_config()
    features = toy_batch(rng, plain_config)[0]
    plain = CompositionNet(plain_config, features, seed=4).state()
    extended = CompositionNet(small_config(n_aux=2, n_global=3), features, seed=4).state()
    for name, value in plain.items():
        if name == "trunk0.linear.W":
            np.testing.assert_array_equal(extended[name][:value.shape[0]], value)
            assert extended[name].shape[0] == value.shape[0] + plain_config.width
        else:
            np.testing.assert_array_equal(extended[name], value)
    assert {"head_aux.W", "head_aux.b", "global_proj.W", "global_proj.b"} <= set(extended)


def test_model_with_a_formula_branch_refuses_to_run_without_the_features():
    rng = np.random.default_rng(0)
    config = small_config(n_global=3)
    features, index, fraction, mask, *_ = toy_batch(rng, config)
    model = CompositionNet(config, features, seed=0)
    with pytest.raises(ValueError):
        model.forward(index, fraction, mask)
    with pytest.raises(ValueError):
        model.forward(index, fraction, mask, np.zeros((len(index), 2)))


def test_float32_model_stays_float32_all_the_way_through():
    rng = np.random.default_rng(0)
    config = small_config(n_aux=2, n_global=3)
    features, index, fraction, mask, *_ = toy_batch(rng, config)
    model = CompositionNet(config, features, seed=0)
    logit, tc = model.forward(index, fraction.astype(np.float32), mask, np.zeros((len(index), 3), np.float32))
    assert logit.dtype == tc.dtype == model.aux.dtype == np.float32
    model.backward(np.ones(len(index), np.float32), np.ones(len(index), np.float32))
    assert all(grad.dtype == np.float32 for _, _, grad in model.parameters())
    assert not np.any(model.layers["head_aux"].grads["W"])      # no auxiliary gradient was passed in
