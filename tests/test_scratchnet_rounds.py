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


def test_final_refuses_to_score_the_locked_test_set_twice(tmp_path, monkeypatch):
    used = tmp_path / "scratchnet-lockbox.json"
    used.write_text("{}")
    monkeypatch.setattr(R, "LOCKBOX_REPORT", used)
    with pytest.raises(SystemExit, match="already been used"):
        R.run_final("again", ["tuned"], [0], 1, 0.5, False, tmp_path)


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
