"""Training loop and scoring for the from-scratch network."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rt_superconductor.scratchnet import train as T  # noqa: E402
from rt_superconductor.scratchnet.model import CompositionNet, Config  # noqa: E402


def toy(n: int = 240, seed: int = 0) -> tuple[T.Arrays, np.ndarray, Config]:
    """Two elements out of six; pairs drawn from elements 1-4 superconduct, Tc set by how much of 1 and 2."""
    rng = np.random.default_rng(seed)
    features = np.vstack([np.zeros((1, 4)), rng.normal(size=(6, 4))]).astype(np.float32)
    index = np.zeros((n, 3), np.int64)
    fraction = np.zeros((n, 3), np.float32)
    for row in range(n):
        index[row, :2] = rng.choice(np.arange(1, 7), size=2, replace=False)
        first = rng.uniform(0.2, 0.8)
        fraction[row, :2] = first, 1.0 - first
    mask = index > 0
    good = ((index >= 1) & (index <= 4)).sum(1) == 2
    amount_of_one = np.where(index == 1, fraction, 0.0).sum(1)
    amount_of_two = np.where(index == 2, fraction, 0.0).sum(1)
    label = good.astype(float)
    tc = np.where(good, 3.0 + 80.0 * amount_of_one + 25.0 * amount_of_two, np.nan)
    group = np.array(["-".join(map(str, sorted(pair))) for pair in index[:, :2]])
    config = Config(n_elements=6, n_features=4, width=16, blocks=1, heads=2, pool_heads=1, trunk=(24,), dropout=0.0)
    return T.Arrays(index, fraction, mask, label, tc, group), features, config


def test_average_ranks_handles_ties():
    assert T.average_ranks(np.array([10.0, 20.0, 10.0, 30.0])).tolist() == [1.5, 3.0, 1.5, 4.0]


def test_roc_auc_matches_scikit_learn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    label = rng.integers(0, 2, 300)
    score = np.round(rng.normal(size=300) + label, 1)          # rounding creates ties
    assert T.roc_auc(label, score) == pytest.approx(metrics.roc_auc_score(label, score), abs=1e-12)
    assert np.isnan(T.roc_auc(np.ones(5), np.arange(5.0)))      # one class only: undefined


def test_spearman_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(2)
    a = np.round(rng.normal(size=200), 1)
    b = a + rng.normal(size=200)
    assert T.spearman(a, b) == pytest.approx(stats.spearmanr(a, b).statistic, abs=1e-12)
    assert np.isnan(T.spearman(np.ones(10), np.arange(10.0)))


def test_tc_scale_round_trip_and_clipping():
    tc = np.array([0.5, 4.0, 39.0, 92.0, np.nan])
    scale = T.TcScale.fit(tc)
    np.testing.assert_allclose(scale.decode(scale.encode(tc[:4])), tc[:4], rtol=1e-10)
    assert scale.decode(np.array([-1e6]))[0] == 0.0             # never a negative temperature
    assert np.isfinite(scale.decode(np.array([1e6]))[0])        # never an overflow


def test_inner_split_keeps_groups_whole_and_never_holds_a_dominant_group():
    groups = np.array(["big"] * 500 + [f"g{i % 60}" for i in range(600)])
    for seed in range(8):
        fit_rows, valid_rows = T.inner_split(groups, percent=15, seed=seed)
        assert not set(groups[fit_rows]) & set(groups[valid_rows])
        assert "big" not in set(groups[valid_rows])
        assert len(fit_rows) + len(valid_rows) == len(groups) and len(valid_rows) > 0


def test_inner_split_never_cuts_a_group_even_when_every_group_is_large():
    groups = np.repeat(np.array(["a", "b", "c", "d"]), 25)                 # each group is a quarter of the rows
    for seed in range(6):
        fit_rows, valid_rows = T.inner_split(groups, percent=12, seed=seed)
        assert len(valid_rows) > 0 and len(fit_rows) > 0
        assert not set(groups[fit_rows]) & set(groups[valid_rows])
    two = np.array(["a"] * 5 + ["b"] * 5)
    fit_rows, valid_rows = T.inner_split(two, percent=99, seed=0)
    assert len(set(two[fit_rows])) == 1 and len(set(two[valid_rows])) == 1
    fit_rows, valid_rows = T.inner_split(np.array(["only"] * 30), percent=10, seed=0)
    assert len(valid_rows) == 3 and len(fit_rows) == 27                    # one group: every tenth row


def test_score_reports_perfect_and_useless_predictions():
    label = np.array([1.0] * 30 + [0.0] * 30 + [np.nan] * 5)
    tc = np.concatenate([np.linspace(2, 90, 30), np.full(35, np.nan)])
    perfect = T.score(label, tc, np.nan_to_num(label, nan=0.5), np.nan_to_num(tc, nan=1.0))
    assert perfect["auc"] == 1.0 and perfect["tc_mae_k"] == 0.0 and perfect["tc_spearman"] == pytest.approx(1.0)
    assert perfect["labelled_rows"] == 60 and perfect["superconductor_rows"] == 30
    assert perfect["top_decile_recall"] == 1.0
    flat = T.score(label, tc, np.full(65, 0.5), np.full(65, 10.0))
    assert flat["auc"] == 0.5 and np.isnan(flat["tc_spearman"])


def test_fit_learns_the_toy_rule_and_generalizes_to_unseen_rows():
    data, features, config = toy(n=360)
    train, test = data.take(np.arange(0, 180)), data.take(np.arange(180, 360))
    model = CompositionNet(config, features, seed=0)
    settings = T.TrainConfig(epochs=80, batch_size=32, lr=5e-3, patience=80, warmup_epochs=1, seed=0)
    scale = T.TcScale.fit(train.tc_k)
    before = T.evaluate_loss(model, test, scale, settings)["total"]
    result = T.fit(model, train, settings, scale=scale)
    after = T.evaluate_loss(model, test, scale, settings)["total"]
    probability, tc_pred = T.predict(model, test, scale)
    scores = T.score(test.label, test.tc_k, probability, tc_pred)
    assert after < 0.5 * before
    assert scores["auc"] > 0.95 and scores["tc_spearman"] > 0.8
    assert result["epochs_run"] == len(result["history"]) and result["fit_rows"] + result["inner_validation_rows"] == 180


def test_fit_stops_early_and_restores_the_best_weights():
    data, features, config = toy(seed=3)
    train, valid = data.take(np.arange(0, 160)), data.take(np.arange(160, 240))
    model = CompositionNet(config, features, seed=1)
    settings = T.TrainConfig(epochs=400, batch_size=16, lr=2e-2, patience=4, min_epochs=5, warmup_epochs=1, seed=1)
    scale = T.TcScale.fit(train.tc_k)
    result = T.fit(model, train, settings, valid=valid, scale=scale)
    assert result["epochs_run"] < 400                                       # it stopped on its own
    assert result["fit_rows"] == 160 and result["inner_validation_rows"] == 80
    restored = T.evaluate_loss(model, valid, scale, settings)["total"]
    assert restored == pytest.approx(result["best_valid_loss"], rel=1e-5)   # weights are the best epoch's
    assert restored == pytest.approx(result["history"][result["best_epoch"] - 1]["valid_loss"], rel=1e-5)
    assert restored <= min(h["valid_loss"] for h in result["history"]) + 1e-5


def test_fit_is_reproducible_for_a_fixed_seed():
    data, features, config = toy(n=120)
    settings = T.TrainConfig(epochs=4, batch_size=32, min_epochs=1, seed=5)
    runs = []
    for _ in range(2):
        model = CompositionNet(config, features, seed=7)
        T.fit(model, data, settings)
        runs.append(T.predict(model, data, T.TcScale.fit(data.tc_k))[0])
    np.testing.assert_array_equal(runs[0], runs[1])


def test_arrays_take_and_join_carry_the_optional_columns():
    data, _, _ = toy(n=20)
    with_aux = T.Arrays(data.index, data.fraction, data.mask, data.label, data.tc_k, data.group,
                        aux=np.arange(40.0).reshape(20, 2))
    assert with_aux.take(np.array([3, 5])).aux.tolist() == [[6.0, 7.0], [10.0, 11.0]]
    joined = T.Arrays.join([data, with_aux])
    assert len(joined) == 40 and np.isnan(joined.aux[:20]).all() and joined.aux[20:].tolist() == with_aux.aux.tolist()
    assert joined.extra is None
    with_extra = T.Arrays(data.index, data.fraction, data.mask, data.label, data.tc_k, data.group, extra=np.ones((20, 3)))
    with pytest.raises(ValueError):
        T.Arrays.join([data, with_extra])


def test_auxiliary_targets_are_learned_and_do_not_count_toward_early_stopping():
    data, features, config = toy(n=300, seed=4)
    config = Config(**{**config.__dict__, "n_aux": 1})
    amount = np.where(data.index == 2, data.fraction, 0.0).sum(1, keepdims=True)
    aux = (amount - amount.mean()) / amount.std()
    unlabelled = T.Arrays(data.index, data.fraction, data.mask, np.full(300, np.nan), np.full(300, np.nan),
                          data.group, aux=aux)                      # auxiliary targets only: no class, no Tc
    model = CompositionNet(config, features, seed=0)
    settings = T.TrainConfig(epochs=40, batch_size=32, lr=5e-3, patience=40, warmup_epochs=1, aux_weight=1.0)
    scale = T.TcScale(0.0, 1.0)
    result = T.fit(model, unlabelled.take(np.arange(240)), settings, valid=data.take(np.arange(240, 300)), scale=scale)
    model.predict(data.index[240:], data.fraction[240:], data.mask[240:])
    assert np.corrcoef(model.aux[:, 0], aux[240:, 0])[0, 1] > 0.9      # it learned the auxiliary target
    assert result["history"][-1]["train_loss"] < 0.3 * result["history"][0]["train_loss"]
    losses = T.evaluate_loss(model, unlabelled.take(np.arange(240)), scale, settings)
    assert losses == {"total": 0.0, "class": 0.0, "tc": 0.0}            # auxiliary targets never drive early stopping


def test_weight_averaging_returns_the_average_not_the_last_step():
    data, features, config = toy(n=160, seed=5)
    settings = T.TrainConfig(epochs=3, batch_size=16, lr=2e-2, min_epochs=3, patience=9, warmup_epochs=0, seed=2)
    plain = CompositionNet(config, features, seed=3)
    T.fit(plain, data, settings)
    averaged = CompositionNet(config, features, seed=3)
    result = T.fit(averaged, data, T.TrainConfig(**{**settings.to_dict(), "ema_decay": 0.9}))
    first, second = plain.state(), averaged.state()
    assert any(not np.allclose(first[name], second[name]) for name in first)
    scale = T.TcScale(**result["tc_scale"])
    valid = data.take(T.inner_split(data.group, settings.inner_validation_percent, settings.seed)[1])
    assert T.evaluate_loss(averaged, valid, scale, settings)["total"] == pytest.approx(result["best_valid_loss"], rel=1e-5)


def test_class_output_can_start_at_the_training_base_rate():
    data, features, config = toy(n=200, seed=6)
    model = CompositionNet(config, features, seed=0)
    T.fit(model, data, T.TrainConfig(epochs=1, min_epochs=1, lr=1e-9, warmup_epochs=0, init_class_bias=True))
    fit_rows, _ = T.inner_split(data.group, 12, 0)
    rate = data.label[fit_rows].mean()
    assert model.layers["head_class"].params["b"][0] == pytest.approx(np.log(rate / (1 - rate)), abs=1e-4)


def test_top_decile_recall_does_not_depend_on_row_order_when_superconductors_are_rare():
    label = np.array([1.0] * 3 + [0.0] * 97)
    tc = np.array([50.0, 20.0, 5.0] + [np.nan] * 97)
    probability = np.linspace(1.0, 0.0, 100)                    # the three superconductors are ranked first
    tc_pred = np.full(100, 10.0)
    forward = T.score(label, tc, probability, tc_pred)["top_decile_recall"]
    order = np.arange(100)[::-1]                                # the same rows, listed backwards
    backward = T.score(label[order], tc[order], probability[order], tc_pred[order])["top_decile_recall"]
    assert forward == backward == 1.0                           # all three true positives are in the model's top ten
    missed = T.score(label, tc, probability[::-1].copy(), tc_pred)["top_decile_recall"]
    assert missed == 0.0


def test_fit_refuses_validation_rows_without_any_label():
    data, features, config = toy(n=60)
    blank = T.Arrays(data.index, data.fraction, data.mask, np.full(60, np.nan), np.full(60, np.nan), data.group)
    with pytest.raises(ValueError, match="early stopping"):
        T.fit(CompositionNet(config, features, seed=0), data, T.TrainConfig(epochs=1), valid=blank)
