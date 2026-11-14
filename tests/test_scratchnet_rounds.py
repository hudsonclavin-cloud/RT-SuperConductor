"""The round runner: splits, comparison models, scoring and the locked-test guard.

Everything here runs on a small made-up table, so it needs neither the SuperCon exports nor a trained model.
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("sklearn")
pytest.importorskip("scipy")
C = pytest.importorskip("scratchnet_common")
R = pytest.importorskip("run_scratchnet")
build = pytest.importorskip("build_scratchnet_data")

from rt_superconductor.scratchnet import data as D  # noqa: E402

PAIRS = [("Nb", "Sn"), ("Nb", "Ti"), ("Nb", "Ge"), ("V", "Si"), ("V", "Ga"), ("Mg", "B"), ("Pb", "Bi"), ("Na", "Cl"),
         ("K", "Cl"), ("Li", "F"), ("Cu", "O"), ("Fe", "Se"), ("Fe", "As"), ("La", "O"), ("Al", "O"), ("Si", "O")]
SUPERCONDUCTING = {"Nb", "V", "Pb", "Mg", "Fe"}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> C.Dataset:
    """16 two-element systems x 12 compositions; pairs with Nb, V, Pb, Mg or Fe superconduct."""
    rows = []
    for number, (a, b) in enumerate(PAIRS):
        split, fold = ("train_only", -1) if number == 0 else ("lockbox", -1) if number in (1, 2) else ("dev", number % 5)
        for step in range(12):
            x = 0.2 + 0.05 * step
            positive = bool({a, b} & SUPERCONDUCTING)
            rows.append({"composition": D.composition_key({a: x, b: 1 - x}), "system": f"{a}-{b}",
                         "family": "Cuprates" if (a, b) == ("Cu", "O") else "Other", "major_group": f"{a}-{b}",
                         "group": f"{a}-{b}", "group_rows": 12, "n_elements": 2,
                         "label": "" if (number == 15 and step < 3) else int(positive),
                         "tc_k": round(4.0 + 30.0 * x, 3) if positive else "", "split": split, "fold": fold})
    folder = tmp_path_factory.mktemp("scratchnet")
    D.write_table(folder / "compositions.csv.gz", rows, build.COMPOSITION_FIELDS)
    return C.load_dataset(folder / "compositions.csv.gz", report=None)


def test_load_dataset_refuses_a_table_that_does_not_match_the_recorded_checksum(tmp_path, dataset):
    table = tmp_path / "compositions.csv.gz"
    D.write_table(table, dataset.rows, build.COMPOSITION_FIELDS)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"output_sha256": "0" * 64}))
    with pytest.raises(SystemExit):
        C.load_dataset(table, report=report)
    report.write_text(json.dumps({"output_sha256": hashlib.sha256(table.read_bytes()).hexdigest()}))
    assert len(C.load_dataset(table, report=report).rows) == len(dataset.rows)


def test_dataset_arrays_line_up_with_the_table(dataset):
    assert len(dataset.arrays) == 192 and dataset.dense.shape == (192, 91)
    np.testing.assert_allclose(dataset.dense.sum(axis=1), 1.0, atol=2e-3)
    assert np.isnan(dataset.arrays.label).sum() == 3
    assert np.isfinite(dataset.arrays.tc_k).sum() == (dataset.arrays.label == 1).sum()


def test_fold_rows_keep_test_groups_out_of_training_and_never_touch_the_lockbox(dataset):
    lockbox = set(np.flatnonzero(dataset.split == "lockbox").tolist())
    for fold in range(5):
        fit, valid, test = R.fold_rows(dataset, fold)
        groups = dataset.arrays.group
        assert not set(groups[test]) & (set(groups[fit]) | set(groups[valid]))
        assert not set(groups[fit]) & set(groups[valid])
        assert not lockbox & (set(fit.tolist()) | set(valid.tolist()) | set(test.tolist()))
        assert set(dataset.split[test]) == {"dev"}
        assert "Nb-Sn" in set(groups[fit]) | set(groups[valid])                         # the train-only group is used


def test_final_rows_test_on_the_lockbox_and_rehearsals_never_see_it(dataset):
    fit, valid, test = R.final_rows(dataset, valid_fold=1, rehearse=False)
    assert set(dataset.split[test]) == {"lockbox"} and "lockbox" not in set(dataset.split[np.concatenate([fit, valid])])
    assert set(dataset.fold[valid]) == {1} and not set(fit.tolist()) & set(valid.tolist())
    fit, valid, test = R.final_rows(dataset, valid_fold=1, rehearse=True)
    assert "lockbox" not in set(dataset.split[np.concatenate([fit, valid, test])])
    assert set(dataset.fold[test]) == {0} and 0 not in set(dataset.fold[fit]) | set(dataset.fold[valid])


def test_final_claims_the_locked_test_set_before_training_and_refuses_a_second_run(tmp_path, monkeypatch, dataset):
    table = tmp_path / "compositions.csv.gz"
    D.write_table(table, dataset.rows, build.COMPOSITION_FIELDS)
    sha = hashlib.sha256(table.read_bytes()).hexdigest()
    marker = tmp_path / "docs" / "scratchnet-lockbox.json"
    marker.parent.mkdir()
    monkeypatch.setattr(C, "DATA", table)
    monkeypatch.setattr(R, "LOCKBOX_REPORT", marker)

    def crash(variants):
        raise RuntimeError("training crashed")

    monkeypatch.setattr(R, "prepare", crash)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    # A selection made on a different table is refused before the locked set is claimed.
    with pytest.raises(SystemExit, match="changed places"):
        R.run_final("x", ["tuned"], [0], 1, 0.5, False, elsewhere, selection_sha256="0" * 64)
    assert not marker.exists()
    # A rehearsal never claims it.
    with pytest.raises(RuntimeError):
        R.run_final("x", ["tuned"], [0], 1, 0.5, True, elsewhere, selection_sha256=sha)
    assert not marker.exists()
    # A real run claims the fixed path first, even with another output folder and even if it then crashes.
    with pytest.raises(RuntimeError):
        R.run_final("x", ["tuned"], [0], 1, 0.5, False, elsewhere, selection_sha256=sha)
    assert json.loads(marker.read_text())["status"] == "started" and not list(elsewhere.iterdir())
    with pytest.raises(SystemExit, match="already been used"):
        R.run_final("again", ["tuned"], [0], 1, 0.5, False, elsewhere, selection_sha256=sha)


def test_formula_text_and_blend():
    assert C.formula_text({"Mg": 1 / 3, "B": 2 / 3}) == "B0.667Mg0.333"
    first, second = (np.array([0.8]), np.array([99.0])), (np.array([0.2]), np.array([9.0]))
    assert C.blend(first, second, 1.0)[0][0] == 0.8 and C.blend(first, second, 0.0)[1][0] == pytest.approx(9.0)
    p, tc = C.blend(first, second, 0.5)
    assert p[0] == pytest.approx(0.5) and tc[0] == pytest.approx(np.sqrt(100.0 * 10.0) - 1.0)     # mean on the log scale


def test_scale_features_handles_gaps_and_extremes():
    X = np.array([[1.0, 5.0], [3.0, np.nan], [2.0, 5.0]])
    scaler = C.feature_scaler(X)
    scaled = C.scale_features(np.array([[2.0, np.nan], [1e9, 5.0]]), scaler)
    assert scaled.dtype == np.float32 and scaled[0].tolist() == [0.0, 0.0] and scaled[1].tolist() == [6.0, 0.0]


def test_group_bootstrap_resamples_whole_groups_and_is_reproducible():
    groups = np.repeat(np.arange(20), 5)
    values = np.repeat(np.arange(20.0), 5)
    seen = []

    def statistic(sample: np.ndarray) -> float:
        seen.append(sample)
        return float(values[sample].mean())

    first = C.group_bootstrap(groups, statistic, resamples=50, seed=1)
    assert all(len(sample) == 100 and (np.bincount(groups[sample], minlength=20) % 5 == 0).all() for sample in seen)
    assert first == C.group_bootstrap(groups, statistic, resamples=50, seed=1)
    assert first["low"] < 9.5 < first["high"]


def test_neighbour_lookup_and_distance(dataset):
    train = np.flatnonzero(dataset.split != "lockbox")
    probability, tc = C.neighbour_predict(dataset, train, dataset.dense[train][:5], k=1)
    assert probability.tolist() == dataset.arrays.label[train][:5].tolist()        # each row finds itself
    distance, _ = C.nearest_distance(D_dense := dataset.dense[:1], np.array([np.roll(D_dense[0], 1)]))
    assert distance[0] == pytest.approx(1.0, abs=2e-3)                              # no atoms in common


def test_boosting_separates_the_made_up_families_and_differences_have_the_right_sign(dataset):
    X = dataset.dense                                    # element fractions are enough for this toy rule
    dev = np.flatnonzero(dataset.split == "dev")
    held = np.isin(dataset.arrays.group, ["Pb-Bi", "K-Cl"])                 # one superconducting system, one not
    fit, valid = np.flatnonzero((dataset.split != "lockbox") & ~held), np.flatnonzero(held)
    settings = {**C.BOOSTING_DEFAULT, "min_samples_leaf": 2}
    p, tc, info = C.boosting_predict(X, dataset.arrays, fit, valid, X[dev],
                                     classifier_settings=settings, regressor_settings=settings)
    assert info["classifier_trees"] >= 1 and info["regressor_trees"] >= 1
    good = (p, tc)
    scores = C.score_table(dataset, dev, np.zeros(len(dev)), {"boosting": good})
    assert scores["all"]["boosting"]["auc"] > 0.9 and "Cuprates" in scores and "not cuprates" in scores
    useless = (np.full(len(dev), 0.5), np.full(len(dev), 1.0))
    gap = C.paired_differences(dataset, dev, good, useless, resamples=40)
    assert gap["auc"]["median"] > 0.3 and gap["tc_log_mae"]["median"] < 0           # better AUC, smaller Tc error
    only_classifier, _ = C.boosting_fit(X, dataset.arrays, fit, valid, parts="classifier", classifier_settings=settings)
    assert only_classifier[1] is None and C.boosting_apply(only_classifier, X[:3])[1] is None
    no_superconductors = np.flatnonzero(dataset.arrays.group == "K-Cl")
    with pytest.raises(ValueError, match="early stopping"):
        C.boosting_fit(X, dataset.arrays, fit, no_superconductors, parts="regressor")


def test_choose_picks_the_better_network_then_the_best_blend_weight(dataset):
    dev = np.flatnonzero(dataset.split == "dev")
    label = np.nan_to_num(dataset.arrays.label[dev], nan=0.0)
    tc = np.nan_to_num(dataset.arrays.tc_k[dev], nan=1.0)
    rng = np.random.default_rng(0)
    predictions = {"boosting": (label * 0.6 + 0.2 + rng.normal(0, 0.3, len(dev)), tc),
                   "network good": (label * 0.9 + 0.05, tc), "network bad": (rng.random(len(dev)), tc)}
    picked = R.choose(dataset, dev, predictions, ["bad", "good"])
    assert picked["variant"] == "good" and picked["blend_weight"] == 1.0
    assert set(picked["by_variant_and_weight_on_network"]["good"]) == {"0", "0.25", "0.5", "0.75", "1"}
    predictions["network good"], predictions["boosting"] = predictions["boosting"], predictions["network good"]
    assert R.choose(dataset, dev, predictions, ["good"])["blend_weight"] == 0.0      # boosting alone is better


def test_load_aux_takes_the_lowest_energy_structure_and_skips_unknown_elements(tmp_path, dataset):
    header = ["mp_id", "formula", "num_sites", "energy_above_hull_ev_per_atom", "formation_energy_ev_per_atom",
              "band_gap_ev", "is_metal", "volume_a3", "total_magnetization_mub"]
    lines = [["mp-1", "NaCl", "2", "0.30", "-1.0", "0.0", "True", "40", "0"],       # a high-energy polymorph
             ["mp-2", "NaCl", "2", "0.00", "-2.0", "5.0", "False", "44", "0"],      # the ground state
             ["mp-3", "Nb3Sn", "8", "0.00", "-0.2", "0.0", "True", "150", "0"],
             ["mp-4", "Fe", "2", "0.00", "0.0", "0.0", "True", "23", "4.4"],
             ["mp-5", "HeNe", "2", "0.00", "0.0", "9.0", "False", "30", "0"]]       # not in the element table
    path = tmp_path / "candidates.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(",".join(line) for line in [header] + lines) + "\n")
    aux = C.load_aux(dataset.table, path)
    assert aux.formulas == ["Cl0.500Na0.500", "Fe1.000", "Nb0.750Sn0.250"]
    assert aux.arrays.aux.shape == (3, len(C.AUX_TARGETS)) and np.isnan(aux.arrays.label).all()
    np.testing.assert_allclose(aux.arrays.aux.mean(axis=0), 0.0, atol=1e-9)
    is_metal = aux.arrays.aux[:, C.AUX_TARGETS.index("is_metal")]
    assert is_metal[0] < 0 < is_metal[1]                                             # NaCl took the insulating entry


def test_every_variant_builds_a_model_and_a_training_configuration(dataset, monkeypatch):
    monkeypatch.setitem(R._STATE, "X", np.zeros((len(dataset.rows), 7)))
    for name in R.VARIANTS:
        config = R.model_config(dataset, name)
        assert config.n_elements == 90 and config.n_features == 28
        assert config.n_global == (7 if R.VARIANTS[name].get("formula") else 0)
        assert config.n_aux == (len(C.AUX_TARGETS) if R.VARIANTS[name].get("aux") else 0)
        assert R.train_config(name, 3).seed == 3
    assert R.train_config("tuned", 0).ema_decay == 0.995 and R.train_config("attention", 0).ema_decay == 0.0


def test_one_network_trains_and_predicts_through_the_runner_path(dataset, tmp_path):
    fit, valid, test = R.fold_rows(dataset, 0)
    X = np.random.default_rng(0).normal(size=(len(dataset.rows), 4))
    config = C.Config(n_elements=90, n_features=28, width=8, blocks=1, heads=2, pool_heads=1, trunk=(8,), n_global=4)
    settings = C.T.TrainConfig(epochs=2, min_epochs=1, batch_size=16)
    (prediction,), info = C.network_fit_predict(dataset, fit, valid, [test], config, settings, X=X,
                                                save=tmp_path / "member.npz")
    assert prediction[0].shape == prediction[1].shape == (len(test),) and info["epochs_run"] == 2
    model, scale, scaler = C.load_network(tmp_path / "member.npz", dataset.table)
    again = C.T.predict(model, C.with_extra(dataset.arrays.take(test), X[test], scaler), scale)
    np.testing.assert_allclose(again[0], prediction[0], rtol=1e-5)
    # Elements with no training row are left with no learned vector, instead of their random starting one.
    table = model.layers["embedding"].params["E"]
    trained = np.unique(dataset.arrays.index[fit])
    assert "Xe" in info["elements_never_trained"] and not table[dataset.table.index["Xe"]].any()
    assert all(table[i].any() for i in trained if i > 0)


def test_capped_keeps_the_order_and_limits_each_element_system():
    ranked = [{"rank": i, "element_system": system} for i, system in enumerate("AAAABBCAD", start=1)]
    kept = R.capped(ranked, top=5, per_system=2)
    assert [row["rank"] for row in kept] == [1, 2, 5, 6, 7]
    assert len(R.capped(ranked, top=50, per_system=1)) == 4


def test_ranking_scores_pick_the_model_that_sets_the_order():
    network, boosting = (np.array([0.9]), np.array([80.0])), (np.array([0.3]), np.array([8.0]))
    assert R.ranking_scores(network, boosting, 0.5, "boosting") is boosting
    assert R.ranking_scores(network, boosting, 0.5, "network") is network
    mixed = R.ranking_scores(network, boosting, 0.5, "mix")
    assert mixed[0][0] == pytest.approx(0.6) and mixed[1][0] == pytest.approx(np.sqrt(81.0 * 9.0) - 1.0)
    with pytest.raises(ValueError):
        R.ranking_scores(network, boosting, 0.5, "best")


@pytest.fixture(scope="module")
def ranked_toy(dataset, tmp_path_factory):
    """A tiny saved ensemble (two networks with a formula branch, two boosting models) and a candidate pool."""
    pytest.importorskip("matminer")
    joblib = pytest.importorskip("joblib")
    folder = tmp_path_factory.mktemp("rank")
    model_dir = folder / "work" / "models" / "toy"
    model_dir.mkdir(parents=True)
    X, _ = C.formula_features([C.formula_text(C.fractions_of(row["composition"])) for row in dataset.rows])
    held = np.isin(dataset.arrays.group, ["Pb-Bi", "K-Cl"])
    fit, valid = np.flatnonzero((dataset.split != "lockbox") & ~held), np.flatnonzero(held)
    config = C.Config(n_elements=90, n_features=28, width=8, blocks=1, heads=2, pool_heads=1, trunk=(8,),
                      n_global=X.shape[1])
    settings = {**C.BOOSTING_DEFAULT, "min_samples_leaf": 2}
    for fold in (1, 2):
        C.network_fit_predict(dataset, fit, valid, [], config, C.T.TrainConfig(epochs=3, min_epochs=1, batch_size=16),
                              X=X, save=model_dir / f"network-toy-fold{fold}-seed0.npz")
        models, _ = C.boosting_fit(X, dataset.arrays, fit, valid, seed=fold, classifier_settings=settings,
                                   regressor_settings=settings)
        joblib.dump(models, model_dir / f"boosting-fold{fold}.joblib")
    # A stray member file that is not in the recorded ensemble must be ignored.
    (model_dir / "network-toy-fold9-seed0.npz").write_bytes(b"not a model")
    (model_dir / "ensemble.json").write_text(json.dumps(
        {"variants": ["toy"], "seeds": [0], "folds": [1, 2], "blend_weight": 0.5, "data_sha256": dataset.sha256}))
    header = ["mp_id", "formula", "energy_above_hull_ev_per_atom", "known_match", "is_metal", "band_gap_ev", "theoretical"]
    lines = [["mp-1", "Nb3Al", "0.0", "", "True", "0", "False"], ["mp-2", "NbN", "0.2", "", "True", "0", "False"],
             ["mp-3", "NbN", "0.0", "", "True", "0", "False"], ["mp-4", "V3Ge", "0.0", "", "True", "0", "False"],
             ["mp-5", "NaF", "0.0", "", "False", "6", "False"], ["mp-6", "KBr", "0.0", "", "False", "5", "False"],
             ["mp-7", "MgB2", "0.0", "exact", "True", "0", "False"], ["mp-8", "PbTe", "0.0", "", "False", "0.2", "False"],
             ["mp-9", "HeNe", "0.0", "", "False", "9", "True"], ["mp-10", "Nb3Ga", "0.0", "", "True", "0", "False"],
             ["mp-11", "Nb5Al", "0.0", "", "True", "0", "False"], ["mp-12", "Nb7Al", "0.0", "", "True", "0", "False"],
             ["mp-13", "Nb9Al", "0.0", "", "True", "0", "False"], ["mp-14", "CuF2", "0.0", "", "True", "0", "False"],
             ["mp-15", "Cu2OF2", "0.0", "", "False", "1", "False"]]
    pool = folder / "candidates.csv.gz"
    with gzip.open(pool, "wt", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(",".join(line) for line in [header] + lines) + "\n")
    return folder, pool


def run_toy_rank(ranked_toy, dataset, monkeypatch, order):
    folder, pool = ranked_toy
    monkeypatch.setattr(C, "WORK", folder / "work")
    monkeypatch.setattr(C, "load_dataset", lambda *args, **kwargs: dataset)
    out = folder / f"out-{order}"
    out.mkdir(exist_ok=True)
    summary = R.run_rank("toy", pool, top=20, out_dir=out, per_system=2, order=order)
    return summary, out


def test_chemistry_class_marks_the_two_unconventional_families():
    assert R.chemistry_class({"Ba", "Cu", "O", "Y"}) == "copper oxide"
    assert R.chemistry_class({"Fe", "Se"}) == "iron pnictide or chalcogenide"
    assert R.chemistry_class({"Cu", "F", "Rb"}) == "other" and R.chemistry_class({"Nb", "Sn"}) == "other"


def test_network_view_has_three_levels_set_by_the_ratio_of_the_two_figures():
    assert [R.network_view(network, 40.0) for network in (40.0, 20.0, 19.9, 12.0, 11.9, 0.0)] == \
        ["agrees", "agrees", "lukewarm", "lukewarm", "rejects", "rejects"]
    assert R.network_view(90.0, 40.0) == "agrees"                    # the network may also be the more optimistic one
    assert R.network_view(5.0, 40.0, agree_ratio=0.1, reject_ratio=0.05) == "agrees"
    assert R.network_view(5.0, 0.0) == "lukewarm"                    # nothing to compare with


def test_rank_defaults_to_boosting_when_no_order_is_given(ranked_toy, dataset, monkeypatch):
    """The owner's decision: boosting sets the order unless someone asks for something else."""
    folder, pool = ranked_toy
    monkeypatch.setattr(C, "WORK", folder / "work")
    monkeypatch.setattr(C, "load_dataset", lambda *args, **kwargs: dataset)
    out = folder / "out-default"
    out.mkdir(exist_ok=True)
    monkeypatch.setattr(sys, "argv", ["run_scratchnet.py", "rank", "--name", "toy", "--candidates", str(pool),
                                      "--out-dir", str(out), "--top", "20"])
    assert R.main() == 0
    assert sorted(path.name for path in out.iterdir()) == [
        "scratchnet-candidates-boosting-top-metals.csv", "scratchnet-candidates-boosting-top-not-cuprates.csv",
        "scratchnet-candidates-boosting-top.csv", "scratchnet-candidates-boosting.json"]
    top = D.read_table(out / "scratchnet-candidates-boosting-top.csv")
    assert {row["ranked_by"] for row in top} == {"boosting"} and all(row["rank"] == row["boosting_rank"] for row in top)
    summary = json.loads((out / "scratchnet-candidates-boosting.json").read_text())
    assert summary["ranked_by"] == "boosting"
    # Defaults of the second opinion: "agrees" at half of boosting's figure or more, "rejects" below 0.3 of it.
    assert "at 0.5 of it or more" in summary["second_opinion"] and "below 0.3 of it" in summary["second_opinion"]
    direct = R.run_rank("toy", pool, top=20, out_dir=out)                 # the function's own default, too
    assert direct["ranked_by"] == "boosting"


def test_rank_orders_by_boosting_with_the_network_as_a_second_opinion(ranked_toy, dataset, monkeypatch):
    summary, out = run_toy_rank(ranked_toy, dataset, monkeypatch, "boosting")
    assert sorted(path.name for path in out.iterdir()) == [
        "scratchnet-candidates-boosting-top-metals.csv", "scratchnet-candidates-boosting-top-not-cuprates.csv",
        "scratchnet-candidates-boosting-top.csv", "scratchnet-candidates-boosting.json"]
    full = D.read_table(ranked_toy[0] / "work" / "toy-candidates-boosting.csv.gz")
    # One row per composition, known materials and unreadable elements left out.
    formulas = [row["formula"] for row in full]
    assert sorted(formulas) == sorted(["Nb3Al", "NbN", "V3Ge", "NaF", "KBr", "PbTe", "Nb3Ga", "Nb5Al", "Nb7Al", "Nb9Al",
                                       "CuF2", "Cu2OF2"])
    nbn = next(row for row in full if row["formula"] == "NbN")
    assert nbn["mp_id"] == "mp-3" and nbn["structures_with_this_formula"] == "2"      # the lower-energy structure
    # The order, and the headline numbers, are boosting's.
    expected = [float(row["expected_tc_k"]) for row in full]
    assert expected == sorted(expected, reverse=True)
    for row in full:
        assert row["ranked_by"] == "boosting" and row["rank"] == row["boosting_rank"]
        assert row["p_superconductor"] == row["boosting_p"] and row["tc_if_superconductor_k"] == row["boosting_tc_k"]
        assert float(row["expected_tc_k"]) == pytest.approx(float(row["boosting_p"]) * float(row["boosting_tc_k"]), abs=0.06)
        assert row["network_view"] == R.network_view(float(row["network_expected_tc_k"]), float(row["expected_tc_k"]))
        assert float(row["network_expected_tc_k"]) == pytest.approx(float(row["network_p"]) * float(row["network_tc_k"]), abs=0.06)
    assert sorted(int(row["network_rank"]) for row in full) == list(range(1, 13))
    assert {row["network_view"] for row in full} <= {"agrees", "lukewarm", "rejects"}
    # The top list keeps at most two rows per set of elements (four Al-Nb compositions are in the pool).
    top = D.read_table(out / "scratchnet-candidates-boosting-top.csv")
    assert sum(row["element_system"] == "Al-Nb" for row in top) == 2 and len(top) == 10
    metals = D.read_table(out / "scratchnet-candidates-boosting-top-metals.csv")
    assert metals and all(row["is_metal"] == "True" for row in metals)
    assert summary["ranked_by"] == "boosting" and summary["network_members"] == 2 and summary["boosting_members"] == 2
    assert summary["untested_scored"] == 12 and summary["distinct_compositions_scored"] == 13
    # Every count in the summary can be recomputed from the list it describes.
    def views(rows):
        return {view: sum(row["network_view"] == view for row in rows) for view in ("agrees", "lukewarm", "rejects")}

    assert summary["top_network_view"] == views(top) and summary["metals_list"]["network_view"] == views(metals)
    assert sum(views(top).values()) == len(top) and sum(views(metals).values()) == len(metals)
    by_distance = summary["top_network_view_by_distance"]
    assert set(by_distance) == {"near", "middle", "far"}
    assert sum(sum(counts.values()) for counts in by_distance.values()) == len(top)
    assert sum(summary["top_network_view_copper_oxides"].values()) == 1              # Cu2OF2 is the one copper oxide
    assert summary["top_rejected_fluorides"] == sum(row["network_view"] == "rejects" and "F" in
                                                    row["element_system"].split("-") for row in top)
    systems = [set(row["element_system"].split("-")) for row in top]
    assert summary["top_containing_copper"] == sum("Cu" in system for system in systems) == 2
    assert summary["top_copper_fluorides_without_oxygen"] == 1                       # CuF2 but not Cu2OF2
    assert summary["top_most_common_elements"]["Nb"] == sum("Nb" in system for system in systems)
    expected = np.array([float(row["expected_tc_k"]) for row in top])
    by_network = np.array([float(row["network_expected_tc_k"]) for row in top])
    assert summary["top_median_expected_tc_k"] == pytest.approx(np.median(expected))
    assert summary["top_median_network_expected_tc_k"] == pytest.approx(np.median(by_network))
    assert summary["top_expected_tc_at_least_30k"] == int((expected >= 30).sum())
    assert summary["top_network_expected_tc_at_least_30k"] == int((by_network >= 30).sum())
    assert summary["top_network_expected_below_half_of_ranking_model"] == int((by_network < 0.5 * expected).sum())
    assert summary["top_network_expected_below_0.3_of_ranking_model"] == int((by_network < 0.3 * expected).sum())
    assert summary["top_not_metals_by_dft"] == sum(row["is_metal"] == "False" for row in top)
    assert summary["expected_tc_of_last_listed_row_k"] == float(top[-1]["expected_tc_k"])


def test_rank_by_the_mix_keeps_the_original_file_names_and_differs_from_boosting(ranked_toy, dataset, monkeypatch):
    run_toy_rank(ranked_toy, dataset, monkeypatch, "boosting")                        # so this test stands alone
    _, out = run_toy_rank(ranked_toy, dataset, monkeypatch, "mix")
    assert sorted(path.name for path in out.iterdir()) == [
        "scratchnet-candidates-top-metals.csv", "scratchnet-candidates-top-not-cuprates.csv",
        "scratchnet-candidates-top.csv", "scratchnet-candidates.json"]
    full = D.read_table(ranked_toy[0] / "work" / "toy-candidates-mix.csv.gz")
    assert {row["ranked_by"] for row in full} == {"mix"}
    assert any(row["p_superconductor"] != row["boosting_p"] for row in full)          # the mix is not boosting
    # The second-opinion columns do not depend on which model sets the order.
    by_boosting = {row["formula"]: row for row in D.read_table(ranked_toy[0] / "work" / "toy-candidates-boosting.csv.gz")}
    for row in full:
        twin = by_boosting[row["formula"]]
        assert (row["network_rank"], row["boosting_rank"], row["network_p"]) == \
            (twin["network_rank"], twin["boosting_rank"], twin["network_p"])


def rejection_toy():
    """105 rows; the first model ranks them 0, 1, 2, ...; five unlabelled rows sit at the very top of its order."""
    n = 105
    label = np.ones(n)
    label[[15, 16, 17, 18, 19, 22]] = 0.0            # rows 15-19 and row 22 do not superconduct
    label[:5] = np.nan                               # no answer known: never among the picks
    tc = np.where(label == 1, 10.0, np.nan)
    tc[5:10] = 30.0                                  # the first five labelled rows are the high-Tc ones
    first = (np.ones(n), np.linspace(200.0, 1.0, n))
    second_score = np.linspace(100.0, 1.0, n)
    second_score[15:20] = 0.5                        # the second model rejects rows 15-19 ...
    second_score[60:65] = 150.0                      # ... and prefers five rows the first model ranks low
    groups = np.arange(n) // 2
    return label, tc, groups, first, (np.ones(n), second_score)


def test_rejection_check_compares_kept_and_rejected_picks_with_a_placebo():
    label, tc, groups, first, second = rejection_toy()
    family = np.where(np.arange(len(label)) % 2 == 0, "even", "odd")
    check = C.rejection_check(label, tc, groups, first, second, first_share=0.2, second_share=0.2, resamples=300,
                              family=family, minimum_rejected=5)
    assert check["rejected_by_family"] == {"even": {"rows": 2, "superconduct": 0}, "odd": {"rows": 3, "superconduct": 0}}
    # 100 labelled rows: the picks are rows 5-24; the second model keeps all but rows 15-19.
    assert (check["picks"], check["kept"], check["rejected"]) == (20, 15, 5)
    superconducts = check["outcomes"]["superconducts"]
    assert superconducts["kept_share"] == pytest.approx(14 / 15) and superconducts["rejected_share"] == 0.0
    assert superconducts["gap"] == pytest.approx(14 / 15)
    # Placebo for "superconducts": reject the five picks the first model itself found least probable. In this
    # toy every probability is 1, so ties leave the first five picks (rows 5-9), all superconductors.
    assert superconducts["placebo_rejected_share"] == 1.0
    assert superconducts["placebo_gap"] == pytest.approx(9 / 15 - 1.0)
    assert superconducts["gap_minus_placebo"] == pytest.approx(14 / 15 - (9 / 15 - 1.0))
    # Placebo for "reaches 20 K": reject the first model's own last five picks (rows 20-24).
    high = check["outcomes"]["true_tc_at_least_20k"]
    assert high["kept_share"] == pytest.approx(5 / 15) and high["rejected_share"] == 0.0
    assert high["gap"] == pytest.approx(1 / 3) and high["placebo_gap"] == pytest.approx(1 / 3)
    assert high["gap_minus_placebo"] == pytest.approx(0.0)

    # The intervals, recomputed here from row numbers with the same random draws: whole groups are resampled.
    picks, rejected = np.arange(5, 25), set(range(15, 20))
    names = np.unique(groups[picks])
    rng = np.random.default_rng(0)
    gaps = []
    for _ in range(300):
        rows = np.concatenate([picks[groups[picks] == names[i]] for i in rng.integers(0, len(names), len(names))])
        yes = [label[row] for row in rows if row not in rejected]
        no = [label[row] for row in rows if row in rejected]
        if yes and no:
            gaps.append(np.mean(yes) - np.mean(no))
    low, high_end = np.percentile(gaps, [2.5, 97.5])
    assert superconducts["gap_interval"] == {"low": pytest.approx(low), "high": pytest.approx(high_end)}
    assert superconducts["gap_interval"]["low"] < 14 / 15 <= superconducts["gap_interval"]["high"] + 1e-9


def test_rejection_check_with_too_few_rejections_gives_no_interval():
    label, tc, groups, first, second = rejection_toy()
    everything = C.rejection_check(label, tc, groups, first, second, first_share=0.2, second_share=1.0)
    assert everything["rejected"] == 0 and everything["outcomes"]["superconducts"]["rejected_share"] is None
    assert everything["outcomes"]["superconducts"]["gap_interval"] is None
    one = C.rejection_check(label, tc, groups, first, second, first_share=0.2, second_share=0.99)
    assert one["rejected"] == 1 and one["outcomes"]["superconducts"]["rejected_share"] == 0.0
    assert one["outcomes"]["superconducts"]["gap_interval"] is None          # one rejection is not evidence
    with pytest.raises(ValueError, match="exactly one"):
        C.rejection_check(label, tc, groups, first, second, first_share=0.2)
    with pytest.raises(ValueError, match="exactly one"):
        C.rejection_check(label, tc, groups, first, second, first_share=0.2, second_share=0.2, second_ratio=0.5)
    five = C.rejection_check(label, tc, groups, first, second, first_share=0.2, second_share=0.2)
    assert five["rejected"] == 5 and five["outcomes"]["superconducts"]["gap_interval"] is None   # under the default floor of ten
    same = C.rejection_check(label, tc, groups, first, first, first_share=0.2, second_share=0.2)
    assert same["kept"] == 20 and same["rejected"] == 0                     # a model never rejects its own picks


def test_rejection_by_size_rejects_picks_the_second_model_scores_far_lower():
    label, tc, groups, first, second = rejection_toy()
    # The first model's scores for the picks (rows 5-24) run from about 190 down to 154; the second model gives
    # rows 15-19 a score of 0.5 and the others between 95 and 77, so only rows 15-19 fall below 0.3 of the first.
    by_size = C.rejection_check(label, tc, groups, first, second, first_share=0.2, second_ratio=0.3, minimum_rejected=5)
    by_rank = C.rejection_check(label, tc, groups, first, second, first_share=0.2, second_share=0.2, minimum_rejected=5)
    assert (by_size["kept"], by_size["rejected"], by_size["second_ratio"], by_size["second_share"]) == (15, 5, 0.3, None)
    assert by_size["outcomes"] == by_rank["outcomes"]                       # here both rules reject the same five picks
    assert C.rejection_check(label, tc, groups, first, second, first_share=0.2, second_ratio=0.6)["rejected"] == 20
    assert by_size["rejected_also_rejected_by_probability_placebo"] == 0    # the placebo took rows 5-9 instead
    # Picks only the second model rejects (rows 15-19, none superconducts) against picks neither model doubts
    # (rows 10-14 and 20-24, of which row 22 does not superconduct).
    only_against_neither = by_size["rejected_by_second_model_only_against_doubted_by_neither"]
    assert only_against_neither["second_model_only"] == {"rows": 5, "superconduct": 0}
    assert only_against_neither["neither"] == {"rows": 10, "superconduct": 9}
    assert only_against_neither["gap"] == pytest.approx(0.9)
    assert only_against_neither["gap_interval"]["low"] <= 0.9 <= only_against_neither["gap_interval"]["high"]


def test_rejection_placebo_for_superconducts_uses_the_first_models_own_probability():
    """If the second model only rejects what the first model already doubted, nothing is left beyond the placebo."""
    n = 60
    label = np.ones(n)
    label[40:50] = 0.0
    tc = np.where(label == 1, 10.0, np.nan)
    probability = np.ones(n)
    probability[40:50] = 0.3                              # the first model already doubts exactly these rows
    first = (probability, np.full(n, 100.0) - np.arange(n) * 0.01 + np.where(probability < 1, 300.0, 0.0))
    second_score = np.arange(n, 0, -1).astype(float)
    second_score[40:50] = -1.0                            # and the second model rejects exactly the same rows
    check = C.rejection_check(label, tc, np.arange(n), first, (np.ones(n), second_score), first_share=0.5,
                              second_share=0.5, resamples=200)
    superconducts = check["outcomes"]["superconducts"]
    assert check["rejected"] == 10 and superconducts["rejected_share"] == 0.0 and superconducts["gap"] == 1.0
    assert superconducts["placebo_gap"] == 1.0 and superconducts["gap_minus_placebo"] == 0.0
    assert superconducts["gap_minus_placebo_interval"] == {"low": 0.0, "high": 0.0}


def test_second_opinion_command_reads_the_saved_development_predictions(dataset, tmp_path, monkeypatch):
    dev = np.flatnonzero(dataset.split == "dev")
    label = np.nan_to_num(dataset.arrays.label[dev], nan=0.0)
    tc = np.nan_to_num(dataset.arrays.tc_k[dev], nan=1.0)
    rng = np.random.default_rng(0)
    distance = np.where(np.arange(len(dev)) % 3 == 0, 0.5, 0.05)             # a third of the rows are far
    work = tmp_path / "work"
    work.mkdir()
    np.savez(work / "toy-oof.npz", rows=dev, distance=distance,
             **{"p::boosting": label * 0.6 + 0.2, "tc::boosting": tc * 2.0,
                "p::network x": label * 0.5 + 0.25 + rng.normal(0, 0.05, len(dev)), "tc::network x": tc,
                "p::network y": np.zeros(len(dev)), "tc::network y": np.zeros(len(dev))})
    (tmp_path / "scratchnet-toy.json").write_text(json.dumps(
        {"data_sha256": dataset.sha256, "selection": {"variant": "x"}}))
    monkeypatch.setattr(C, "WORK", work)
    monkeypatch.setattr(C, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(sys, "argv", ["run_scratchnet.py", "second-opinion", "--name", "toy", "--out-dir",
                                      str(tmp_path), "--size-threshold", "20"])
    assert R.main() == 0
    out = json.loads((tmp_path / "scratchnet-second-opinion.json").read_text())
    assert out["first_model"] == "boosting" and out["second_model"] == "network x"       # the selected variant
    assert out["labelled_rows"] == int(np.isfinite(dataset.arrays.label[dev]).sum()) == 153
    assert [(c["first_share"], c["second_share"]) for c in out["rejection_checks"]] == \
        [tuple(rule) for rule in R.SECOND_OPINION_RULES]
    assert all(c["kept"] + c["rejected"] == c["picks"] for c in out["rejection_checks"])
    assert all(sum(f["rows"] for f in c["rejected_by_family"].values()) == c["rejected"] for c in out["rejection_checks"])
    far_picks = max(1, round(0.1 * int(((distance >= 0.3) & np.isfinite(dataset.arrays.label[dev])).sum())))
    assert [c["picks"] for c in out["far_band_rejection_checks"]] == [far_picks] * len(R.FAR_BAND_RULES)
    assert set(out["network_expected_tc_at_the_edge_of_its_top_share"]) == {"0.1", "0.2", "0.3"}
    assert [(c["first_share"], c["second_ratio"]) for c in out["size_rejection_checks"]] == \
        [tuple(rule) for rule in R.SIZE_RULES]
    # The toy network's figure is about half of boosting's: below 0.3 of it for nobody.
    assert out["size_rejection_checks"][1]["rejected"] == 0
    assert sum(out["boosting_top_tenth_by_distance"].values()) == out["rejection_checks"][0]["picks"] == 15
    labelled = np.isfinite(dataset.arrays.label[dev])
    far = (distance >= 0.3) & labelled
    truth = np.where(dataset.arrays.label[dev] == 1, np.nan_to_num(dataset.arrays.tc_k[dev]), 0.0)
    assert out["far_rows"]["labelled"] == int(far.sum())
    assert out["far_rows"]["true_tc_at_least_10k"] == int((truth[far] >= 10).sum())
    assert set(out["far_rows"]["high_predictions"]) == {f"{model} expects at least {t} K" for model in
                                                        ("boosting", "network") for t in (10, 20, 30)}
    expected = (label * 0.6 + 0.2) * tc * 2.0
    confident = labelled & (expected >= 20)
    assert out["size_agreement"]["rows"] == int(confident.sum()) and out["size_agreement"]["threshold_k"] == 20
    assert 0.3 < out["size_agreement"]["median_network_over_boosting"] < 0.7     # the toy network says about half
    assert set(out["boosting_order_inside_each_bands_top_tenth"]) == {"near", "middle", "far"}
    assert out["boosting_order_inside_each_bands_top_tenth"]["middle"] == {"rows": 0}
    # A report computed on another table is refused.
    (tmp_path / "scratchnet-toy.json").write_text(json.dumps({"data_sha256": "0" * 64, "selection": {"variant": "x"}}))
    with pytest.raises(SystemExit, match="different composition table"):
        R.run_second_opinion("toy", tmp_path)
