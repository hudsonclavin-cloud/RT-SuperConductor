import numpy as np
import pytest

from rtsc2 import evaluate
from rtsc2.models import Blend, DeepEnsemble, TreeBag, _group_bootstrap


def _auc(model, toy, train, test):
    from sklearn.metrics import roc_auc_score

    y = toy.label.astype(int)
    model.fit(toy.X[train], y[train], groups=toy.groups[train], tc=toy.tc[train],
              tokens=(toy.el_index[train], toy.el_fraction[train]))
    mean, members = model.predict(toy.X[test], tokens=(toy.el_index[test], toy.el_fraction[test]))
    assert mean.shape == (len(test),) and members.shape[1] == len(test)
    assert np.allclose(mean, members.mean(axis=0)) and ((mean >= 0) & (mean <= 1)).all()
    return roc_auc_score(y[test], mean), members


def test_group_bootstrap_keeps_groups_whole():
    groups = np.asarray(["a", "a", "b", "c", "c", "c"])
    rows = _group_bootstrap(groups, np.random.default_rng(1))
    for g in set(groups):
        assert (groups[rows] == g).sum() % (groups == g).sum() == 0


def test_tree_bag_learns_and_members_disagree(toy):
    train, test = evaluate.folds(toy, 0)[0]
    auc, members = _auc(TreeBag(members=4, max_iter=50, early_stopping=False), toy, train, test)
    assert auc > 0.75 and members.std(axis=0).max() > 0


def test_folds_never_share_an_element_system(toy):
    for train, test in evaluate.folds(toy, 0):
        assert not set(toy.groups[train]) & set(toy.groups[test])


def test_deep_ensemble_learns_and_uses_elements(toy):
    pytest.importorskip("torch")
    train, test = evaluate.folds(toy, 0)[0]
    auc, members = _auc(DeepEnsemble(members=4, max_epochs=40, patience=40, batch_size=64), toy, train, test)
    assert auc > 0.75 and members.shape[0] == 4
    with pytest.raises(ValueError):
        DeepEnsemble(members=2).fit(toy.X, toy.label, tokens=None)


def test_deep_ensemble_validation_masks_hold_out_whole_groups(toy):
    pytest.importorskip("torch")
    masks = DeepEnsemble(members=3)._validation_masks(toy.label.astype(int), toy.groups)
    for mask in masks:
        assert 0 < mask.sum() < len(mask)
        assert not set(toy.groups[mask]) & set(toy.groups[~mask])
    assert not (masks[0] == masks[1]).all()  # members stop on different splits


def test_deep_ensemble_is_reproducible(toy):
    pytest.importorskip("torch")
    tokens = (toy.el_index, toy.el_fraction)
    runs = [DeepEnsemble(members=2, max_epochs=5, seed=3).fit(toy.X, toy.label, groups=toy.groups, tc=toy.tc,
                                                               tokens=tokens).predict(toy.X, tokens=tokens)[0]
            for _ in range(2)]
    assert np.allclose(runs[0], runs[1], atol=1e-6)


def test_blend_is_the_plain_average(toy):
    tokens = (toy.el_index, toy.el_fraction)
    a = TreeBag(members=2, max_iter=20, early_stopping=False, seed=1)
    b = TreeBag(members=2, max_iter=20, early_stopping=False, seed=2)
    mean, members = Blend(a, b).fit(toy.X, toy.label, groups=toy.groups).predict(toy.X, tokens=tokens)
    assert np.allclose(mean, (a.predict(toy.X)[0] + b.predict(toy.X)[0]) / 2) and len(members) == 4


def test_metrics_and_calibration():
    y = np.asarray([0, 0, 1, 1] * 25)
    perfect = evaluate.metrics(y, y.astype(float))
    assert perfect["auc"] == 1.0 and perfect["brier"] == 0.0 and perfect["calibration_error"] == 0.0
    assert evaluate.calibration_error(y, np.full(100, 0.9)) == pytest.approx(0.4)
