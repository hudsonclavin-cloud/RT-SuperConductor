"""Generations 8 to 10: the breeding rules, the halves, the weights and the once-only lockbox."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import children_evolve as E  # noqa: E402


def test_every_system_has_one_fixed_half():
    systems = [f"A{i}-B{i}" for i in range(400)]
    halves = [E.half(s) for s in systems]
    assert halves == [E.half(s) for s in systems]
    assert set(halves) == {"selection", "lockbox"}
    assert 150 < halves.count("lockbox") < 250


def test_generation_8_is_the_planned_set():
    children = E.generation_8()
    singles = [c for c in children if len(c) == 1]
    assert {next(iter(c)) for c in singles} == set(E.GENES)
    removals = [c for c in children if len(c) == 4]
    assert all(c < E.OLD_CHAMPION for c in removals) and len(removals) == 5
    assert E.add_gene(E.OLD_CHAMPION, "proxy2") in children
    assert len(children) == 16 and len(set(children)) == 16


def test_the_two_proxies_never_meet():
    assert E.add_gene(frozenset({"proxy", "alloy"}), "proxy2") == frozenset({"proxy2", "alloy"})
    assert E.add_gene(frozenset({"proxy2"}), "proxy") == frozenset({"proxy"})
    assert E.merge(frozenset({"proxy2", "bag"}), frozenset({"proxy", "alloy"})) == frozenset({"proxy2", "bag", "alloy"})
    assert E.merge(frozenset({"bag"}), frozenset({"proxy"})) == frozenset({"bag", "proxy"})
    assert not E.valid(frozenset({"proxy", "proxy2"}))
    ranked = [frozenset({"tc-size"}), frozenset({"elements", "proxy"}), frozenset({"proxy2"})]
    children = E.next_generation(ranked)
    assert all(E.valid(c) for c in children) and len(children) == len(set(children))
    assert frozenset({"tc-size", "elements", "proxy"}) in children            # a merged pair
    assert frozenset({"tc-size", "leaves63"}) in children                     # one more gene


def test_only_the_three_fittest_breed():
    ranked = [frozenset({"tc-size"}), frozenset({"alloy"}), frozenset({"bag"}), frozenset({"recent"})]
    assert not any("recent" in c and len(c) == 2 and "tc-size" not in c and "alloy" not in c and "bag" not in c
                   for c in E.next_generation(ranked))


def test_ranking_breaks_ties_towards_fewer_genes():
    fitness = {frozenset({"alloy", "bag"}): 0.7, frozenset({"bag"}): 0.7, frozenset(): 0.6, frozenset({"alloy"}): 0.7}
    assert E.rank(fitness) == [frozenset({"alloy"}), frozenset({"bag"}), frozenset({"alloy", "bag"}), frozenset()]


def test_names_round_trip():
    for genes in (frozenset(), frozenset({"bag"}), frozenset({"tc-size", "elements", "proxy2"})):
        name = E.name_of(genes)
        assert (frozenset(name.split("+")) if name != "baseline" else frozenset()) == genes


def test_weights():
    y = np.array([1, 1, 1, 0])
    age = np.array([0.0, 15.0, 30.0, 0.0])
    assert E.weights_of(frozenset({"alloy"}), y, age) is None
    balanced = E.weights_of(frozenset({"balance"}), y, age)
    assert balanced[y == 1].sum() == pytest.approx(balanced[y == 0].sum())
    assert balanced.sum() == pytest.approx(4)
    recent = E.weights_of(frozenset({"recent"}), y, age)
    assert recent[1] / recent[0] == pytest.approx(np.exp(-1))
    assert recent[2] / recent[0] == pytest.approx(np.exp(-2))
    both = E.weights_of(frozenset({"recent", "balance"}), y, age)
    assert both[y == 1].sum() == pytest.approx(both[y == 0].sum())
    assert both[1] / both[0] == pytest.approx(np.exp(-1))


def test_the_weighted_fit_runs_and_the_unweighted_one_matches_the_project_fit():
    import children_knowledge as G
    import children_nims as N

    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6))
    y = (X[:, 0] + 0.5 * rng.normal(size=600) > 0).astype(int)
    groups = np.array([f"g{i % 60}" for i in range(600)])
    mine = E.fit("classifier", X, y, groups, dict(G.BOOST), None, 0).predict_proba(X)[:, 1]
    theirs = N.fit_boost(X, y, groups, G.BOOST, seed=0).predict_proba(X)[:, 1]
    assert np.allclose(mine, theirs)
    weights = E.weights_of(frozenset({"balance"}), y, np.zeros(600))
    weighted = E.fit("classifier", X, y, groups, dict(G.BOOST), weights, 0).predict_proba(X)[:, 1]
    assert weighted.shape == (600,) and np.isfinite(weighted).all()
    regressor = E.fit("regressor", X, X[:, 1], groups, dict(G.BOOST), weights, 0)
    assert np.isfinite(regressor.predict(X)).all()
    plain = E.fit("regressor", X, X[:, 1], groups, dict(G.BOOST), None, 0)
    assert np.allclose(plain.predict(X), G.fit_regressor(X, X[:, 1], groups, seed=0).predict(X))


def test_the_lockbox_opens_once(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "DOCS", tmp_path)
    (tmp_path / "evolve-lockbox.json").write_text("{}")
    with pytest.raises(SystemExit, match="opened once"):
        E.cmd_lockbox()


def test_the_lockbox_waits_for_generation_10(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "DOCS", tmp_path / "docs")
    monkeypatch.setattr(E, "WORK", tmp_path / "work")
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "state.json").write_text(json.dumps({"fitness": {"baseline": 0.7}, "generation": {"8": []}}))
    with pytest.raises(SystemExit, match="generation 10"):
        E.cmd_lockbox()


def test_a_generation_needs_the_one_before(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "WORK", tmp_path)
    with pytest.raises(SystemExit, match="generation 8 has not been run"):
        E.run_generation(9)


def test_fitness_reads_the_selection_half_only():
    data = {"half": np.array(["selection"] * 4 + ["lockbox"] * 4),
            "test_family": np.array(["Other"] * 8),
            "y_test": np.array([1, 0, 1, 0, 0, 1, 0, 1])}
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2])
    assert E.auc_on(data, scores, "selection") == 1.0
    assert E.auc_on(data, scores, "lockbox") == 0.0


def test_warming_the_cache_skips_children_already_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "WORK", tmp_path)
    children = [frozenset({"alloy"}), frozenset({"bag", "alloy"})]
    for child in children:
        for fold in E.FOLDS:
            E.cache_file(child, fold).parent.mkdir(parents=True, exist_ok=True)
            np.save(E.cache_file(child, fold), np.zeros(3))
    called = []
    monkeypatch.setattr(E, "_warm_fold", lambda fold, names: called.append((fold, names)))
    E.warm(children, jobs=4)                                   # everything cached: no worker starts
    assert called == []
    assert E.cache_file(children[0], "A") != E.cache_file(children[1], "A")
    assert E.cache_file(children[0], "A") != E.cache_file(children[0], "B")
