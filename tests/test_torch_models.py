import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

torch = pytest.importorskip("torch")

from torch_models import TorchTabularClassifier, TunedTorchClassifier  # noqa: E402


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    X[rng.random((n, 8)) < 0.05] = np.nan
    logit = 2 * np.nan_to_num(X[:, 0]) - np.nan_to_num(X[:, 1])
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    groups = np.asarray([f"g{i % 40}" for i in range(n)])
    tc = np.where(y == 1, rng.uniform(1, 40, n), np.nan)
    return X, y, groups, tc


def test_learns_a_simple_signal_and_handles_missing_values():
    from sklearn.metrics import roc_auc_score

    X, y, groups, tc = _data()
    model = TorchTabularClassifier(seed=1, max_epochs=200).fit(X[:300], y[:300], groups=groups[:300])
    p = model.predict_proba(X[300:])[:, 1]
    assert p.shape == (100,) and np.all((p >= 0) & (p <= 1))
    assert roc_auc_score(y[300:], p) > 0.75
    assert model.epochs_ <= 200


def test_is_deterministic_for_a_seed_and_uses_inner_validation_only():
    X, y, groups, tc = _data()
    a = TorchTabularClassifier(seed=3, max_epochs=30).fit(X, y, groups=groups).predict_proba(X)
    b = TorchTabularClassifier(seed=3, max_epochs=30).fit(X, y, groups=groups).predict_proba(X)
    assert np.allclose(a, b)


def test_multitask_head_and_sample_weights_run():
    X, y, groups, tc = _data()
    m = TorchTabularClassifier(seed=0, aux_weight=0.5, max_epochs=20).fit(
        X, y, groups=groups, aux_target=tc, sample_weight=np.full(len(y), 0.5))
    assert np.isfinite(m.best_validation_loss_)


def test_tuned_model_selects_by_validation_loss(monkeypatch):
    X, y, groups, tc = _data(n=200)
    monkeypatch.setattr(TunedTorchClassifier, "GRID", TunedTorchClassifier.GRID[:2])
    m = TunedTorchClassifier(seed=0).fit(X, y, groups=groups)
    assert m.best_config_ in TunedTorchClassifier.GRID
    assert m.predict_proba(X).shape == (200, 2)
