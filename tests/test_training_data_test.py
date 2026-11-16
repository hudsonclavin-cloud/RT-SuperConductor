"""The training-data test: fold assignment, label disagreement and the interval helper."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("pymatgen")

import training_data_test as D  # noqa: E402


def test_a_composition_s_fold_depends_only_on_its_element_system():
    assert D.system_of("Nb:0.750;Sn:0.250") == D.system_of("Nb:0.500;Sn:0.500") == "Nb-Sn"
    assert D.fold_of("Nb-Sn") == D.fold_of(D.system_of("Nb:0.750;Sn:0.250")) and 0 <= D.fold_of("Nb-Sn") < D.FOLDS
    assert len({D.fold_of(f"El{i}-Xx") for i in range(200)}) == D.FOLDS


def test_disagreement_counts_shared_compositions_with_different_labels():
    record = [{"key": "a", "label": 1}, {"key": "b", "label": 0}, {"key": "c", "label": 1}, {"key": "d", "label": 0}]
    nims = [{"key": "a", "label": 1}, {"key": "b", "label": 1}, {"key": "c", "label": 0}, {"key": "e", "label": 0}]
    out = D.disagreement(record, nims)
    assert out["shared_compositions"] == 3 and out["labels_differ"] == 2 and out["fraction"] == round(2 / 3, 4)
    assert out["kinds"] == {"record negative, NIMS superconductor": 1, "record superconductor, NIMS null": 1}


def test_group_interval_is_positive_for_a_better_score_and_zero_for_the_same():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    noise = rng.normal(size=300)
    groups = np.array([f"s{i // 3}" for i in range(300)])
    better = D.group_interval(y, y * 0.3 + noise, y * 2.0 + noise, groups, (0.5, 99.5), resamples=200)
    assert better[0] > 0
    assert D.group_interval(y, noise, noise, groups, (0.5, 99.5), resamples=200) == [0.0, 0.0]


def test_cross_fit_never_trains_on_a_held_out_element_system(monkeypatch):
    """Spy on the fit: no training group may share a fold with the rows it then scores."""
    seen = []

    class Stub:
        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    def fake_fit(X, y, groups, settings, seed=0):
        seen.append((seed, set(groups.tolist())))
        return Stub()

    monkeypatch.setattr(D.N, "fit_boost", fake_fit)
    systems = [("Aa", "Bb"), ("Cc", "Dd"), ("Ee", "Ff"), ("Gg", "Hh"), ("Ii", "Jj"), ("Kk", "Ll"), ("Mm", "Nn")]
    monkeypatch.setattr(D, "system_of", lambda key: key.split("|")[0])
    rows = [{"key": f"{a}-{b}|{i}", "label": i % 2, "family": "Other"} for a, b in systems for i in range(2)]
    features = {r["key"]: np.zeros(3) for r in rows}
    out = D.cross_fit({"record": rows[:8], "nims": rows[6:]}, features)
    for fold, groups in seen:
        assert all(D.fold_of(g) != fold for g in groups)
    assert not np.isnan(out["scores"]["record"]["both"]).any() and not np.isnan(out["scores"]["nims"]["record"]).any()
