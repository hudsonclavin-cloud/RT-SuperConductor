"""The one-time locked-row scoring of the blend: no label leaves, the laptop's run predicts exactly the locked rows,
the reading follows the plan, and a second scoring is refused."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("scipy")
pytest.importorskip("sklearn")
import foundation_gpu_train as F  # noqa: E402
import local_gpu as G  # noqa: E402
import lockbox_blend as LB  # noqa: E402
from test_local_gpu import fake_additions, fake_dataset  # noqa: E402

NAMES = np.array([f"f{i}" for i in range(6)])


def files(tmp_path):
    ds, X = fake_dataset(n_dev=240, n_train=90, n_lock=40)
    bundle = G.build_bundle(ds, X, fake_additions(), {"pool_X": X[:5], "pool_keys": np.array(list("abcde"))})
    bundle_path, locked_path = tmp_path / "bundle.npz", tmp_path / "locked.npz"
    np.savez_compressed(bundle_path, feature_names=NAMES, **bundle)
    np.savez_compressed(locked_path, feature_names=NAMES, **LB.locked_features(ds, X))
    return ds, X, bundle_path, locked_path


def test_export_holds_only_the_locked_rows_features_and_keys():
    ds, X = fake_dataset()
    out = LB.locked_features(ds, X)
    assert set(out) == {"candidates_locked_X", "candidates_locked_keys"}
    locked = np.flatnonzero(ds.split == "lockbox")
    assert list(out["candidates_locked_keys"]) == [ds.rows[i]["composition"] for i in locked]
    assert np.array_equal(out["candidates_locked_X"], X[locked])


def test_runner_predicts_exactly_the_locked_rows(tmp_path):
    ds, _, bundle_path, locked_path = files(tmp_path)
    out = tmp_path / "run"
    assert F.main([str(bundle_path), "--model", "hgb", "--out", str(out), "--extra-candidates", str(locked_path),
                   "--candidates", "locked"]) == 0
    with np.load(out / "candidates_locked.npz") as saved:
        keys = [ds.rows[i]["composition"] for i in np.flatnonzero(ds.split == "lockbox")]
        p, tc = LB.aligned(keys, saved["keys"], saved["p"], saved["tc"])
    assert len(p) == len(keys) and np.all((p >= 0) & (p <= 1)) and np.all(tc >= 0)
    assert not (out / "candidates_pool.npz").exists()                      # only the list asked for
    run = json.loads((out / "run.json").read_text())
    assert run["extra_candidates"]["lists"] == {"locked": len(keys)}
    assert len(run["extra_candidates"]["sha256"]) == 64


def test_runner_refuses_an_extra_file_with_labels_or_training_rows(tmp_path):
    ds, X, bundle_path, locked_path = files(tmp_path)
    with np.load(locked_path) as good:
        arrays = dict(good)
    labelled = tmp_path / "labelled.npz"
    np.savez_compressed(labelled, label=np.ones(3), **arrays)
    with pytest.raises(SystemExit):
        F.main([str(bundle_path), "--model", "hgb", "--out", str(tmp_path / "a"), "--extra-candidates", str(labelled)])
    renamed = tmp_path / "renamed.npz"
    np.savez_compressed(renamed, **{**arrays, "feature_names": np.array([f"g{i}" for i in range(6)])})
    with pytest.raises(SystemExit):
        F.main([str(bundle_path), "--model", "hgb", "--out", str(tmp_path / "b"), "--extra-candidates", str(renamed)])
    with np.load(bundle_path) as bundle:
        train_keys = bundle["keys"][:3]
    leaky = tmp_path / "leaky.npz"
    np.savez_compressed(leaky, feature_names=NAMES, candidates_locked_X=X[:3], candidates_locked_keys=train_keys)
    with pytest.raises(SystemExit):
        F.main([str(bundle_path), "--model", "hgb", "--out", str(tmp_path / "c"), "--extra-candidates", str(leaky)])


def test_alignment_refuses_missing_or_extra_compositions():
    keys = ["a", "b", "c"]
    p, tc = LB.aligned(keys, np.array(["c", "a", "b"]), np.array([0.3, 0.1, 0.2]), np.array([3.0, 1.0, 2.0]))
    assert list(p) == [0.1, 0.2, 0.3] and list(tc) == [1.0, 2.0, 3.0]
    with pytest.raises(SystemExit):
        LB.aligned(keys, np.array(["a", "b"]), np.zeros(2), np.zeros(2))
    with pytest.raises(SystemExit):
        LB.aligned(keys, np.array(["a", "b", "c", "d"]), np.zeros(4), np.zeros(4))


def test_reading_follows_the_plan():
    ds, _ = fake_dataset(n_dev=60, n_train=20, n_lock=200, seed=3)
    rows = np.flatnonzero(ds.split == "lockbox")
    label = np.nan_to_num(ds.arrays.label[rows], nan=0.0)
    rng = np.random.default_rng(0)
    tc = np.full(len(rows), 5.0)
    good = (np.clip(0.5 * label + rng.uniform(0, 0.5, len(rows)), 0, 1), tc)
    same = LB.score_locked(ds, rows, good, good)                          # blend equals the record: passes (at least)
    assert same["reading"]["blend_passes"]
    worse = (1.0 - good[0], tc)                                           # an anti-correlated partner drags the blend down
    score = LB.score_locked(ds, rows, good, worse)
    assert not score["reading"]["blend_passes"]
    assert score["reading"]["blend_auc_not_cuprates"] < score["reading"]["record_auc_not_cuprates"]
    assert set(score["paired"]["not cuprates"]) == {"blend_minus_record", "tabicl_minus_record"}


def test_second_scoring_is_refused(tmp_path):
    done = tmp_path / "lockbox-blend.json"
    LB.refuse_if_scored(done)                                             # not scored yet: fine
    done.write_text("{}")
    with pytest.raises(SystemExit):
        LB.refuse_if_scored(done)


def test_learning_curve_runs_and_its_full_fraction_is_the_plain_fit(tmp_path):
    import learning_curve as LC

    _, _, bundle_path, _ = files(tmp_path)
    with np.load(bundle_path) as saved:
        bundle = {k: saved[k] for k in saved.files}
    points = LC.curve(bundle, fractions=(0.5, 1.0), seeds=2)
    assert set(points) == {"0.5", "1.0"} and len(points["0.5"]["aucs"]) == 2 and len(points["1.0"]["aucs"]) == 1
    assert points["0.5"]["training_rows"] < points["1.0"]["training_rows"]
    assert all(0.0 <= a <= 1.0 for p in points.values() for a in p["aucs"])
    rising = {"a": {"training_rows": 100, "auc_mean": 0.70}, "b": {"training_rows": 200, "auc_mean": 0.73}}
    assert abs(LC.slope_per_doubling(rising) - 0.03) < 1e-9


def test_blend_ranking_refuses_a_different_candidate_set_and_ranks_highest_first():
    import blend_ranking as BR

    record = (np.array([0.2, 0.9]), np.array([1.0, 2.0]))
    p, tc = BR.blended(np.array(["x", "y"]), record, {"x": (0.4, 1.0), "y": (0.7, 2.0)})
    assert np.allclose(p, [0.3, 0.8]) and list(BR.ranks(p)) == [2, 1]
    with pytest.raises(SystemExit):
        BR.blended(np.array(["x", "y"]), record, {"x": (0.4, 1.0)})


def test_novelty_classes():
    import novelty_check as N

    held = {"Nb:1.000": {"Nb": 1.0}, "Ge:0.250;Nb:0.750": {"Ge": 0.25, "Nb": 0.75}}
    new = {"Nb:1.000": {"Nb": 1.0},                                       # held
           "Ge:0.260;Nb:0.740": {"Ge": 0.26, "Nb": 0.74},               # a doping step away
           "Ge:0.400;Nb:0.600": {"Ge": 0.4, "Nb": 0.6},                 # same major elements
           "Mg:0.333;B:0.667": {"Mg": 1 / 3, "B": 2 / 3}}                # elements we do not hold
    got = N.classify(held, new)
    assert got == {"Nb:1.000": "held", "Ge:0.260;Nb:0.740": "near twin", "Ge:0.400;Nb:0.600": "known parent",
                   "Mg:0.333;B:0.667": "new system"}


def test_nemad_parsing():
    import nemad_download as ND

    assert ND.tc_value("30 K") == 30.0 and ND.tc_value("4.2") == 4.2
    assert ND.tc_value("20-30 K") is None and ND.tc_value("<2 K") is None and ND.tc_value(None) is None
    assert ND.label_of({"Material_Type": "SC"}) == 1 and ND.label_of({"Material_Type": "NSC"}) == 0
    assert ND.label_of({"Material_Type": "?"}) is None
    assert ND.ambient("ambient") is True and ND.ambient("5 GPa") is False and ND.ambient(None) is None
    text, fractions = ND.formula_of({"Material_Name": "MgB2 (Tc~39K)"})
    assert text == "MgB2" and abs(fractions["B"] - 2 / 3) < 1e-9
    assert ND.formula_of({"Material_Name": "High-Tc superconductor family"}) == (None, None)


def test_nemad_name_cleaning():
    import nemad_download as ND

    assert ND.clean_name("Nb₃Sn") == "Nb3Sn"
    assert ND.clean_name("YBa2Cu3O7−δ (untreated)") == "YBa2Cu3O7"
    assert ND.clean_name("Bi2Sr2CaCu2O8+δ") == "Bi2Sr2CaCu2O8"
    assert ND.clean_name("2H-Li0.096TaS2") == "Li0.096TaS2"
    assert ND.clean_name("HfV2 (polycrystalline, pc1)") == "HfV2"
    assert ND.clean_name("La2-xSrxCuO4 (x=0.15)") == "La1.85Sr0.15CuO4"
    assert ND.clean_name("(Hg0.7Tl0.3)2Ba2CuO4") == "(Hg0.7Tl0.3)2Ba2CuO4"       # a leading bracket is formula
    assert ND.formula_of({"Material_Name": "BaFe2-xNixAs2"}) == (None, None)       # x unknown: not usable
    assert ND.formula_of({"Material_Name": "Lead Bismuth"}) == (None, None)
    assert ND.formula_of({"Material_Name": "Cu1234"}) == (None, None)                # a nickname, not copper
    assert ND.formula_of({"Material_Name": "Bi2212"}) == (None, None)
    assert ND.formula_of({"Material_Name": "BaFe2As2 (K-doped)"}) == (None, None)    # the note changes the compound
    assert ND.formula_of({"Material_Name": "YbBa2Cu3O6+x"}) == (None, None)          # x is not small here
    assert ND.formula_of({"Material_Name": "Nb"})[1] == {"Nb": 1.0}                  # a plain element still parses
    text, fractions = ND.formula_of({"Material_Name": "La2-xSrxCuO4 (x=0.15)"})
    assert abs(fractions["Sr"] - 0.15 / 7) < 1e-9


def test_shuffled_label_placebo_keeps_rows_and_balance():
    import run_learning_cycle as L

    extra = {"source": "x", "label": np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0]), "tc_k": np.array([5.0, 6.0, np.nan, 7.0, np.nan, 8.0]),
             "origin": np.array(["nims", "nims", "nemad", "nemad", "nemad", "nemad"]), "score": True}
    out = L.permute_labels(extra, "nemad", seed=3)
    assert list(out["label"][:2]) == [1.0, 1.0] and list(out["tc_k"][:2]) == [5.0, 6.0]   # other sources untouched
    assert sorted(out["label"][2:]) == sorted(extra["label"][2:])                         # same balance
    assert out["score"] is False and "permuted" in out["source"]
    for l, t in zip(out["label"], out["tc_k"]):                                           # label and Tc move together
        assert (l == 1.0) == np.isfinite(t)
    with pytest.raises(SystemExit):
        L.permute_labels(extra, "absent", seed=0)


def test_added_rows_scored_by_source():
    import run_learning_cycle as L

    rng = np.random.default_rng(0)
    n = 80
    y = (rng.uniform(size=n) < 0.6).astype(float)
    extra = {"label": y, "p_with": np.clip(y * 0.6 + rng.uniform(0, 0.4, n), 0, 1), "p_without": rng.uniform(size=n),
             "group": np.array([f"g{i}" for i in range(n)]), "cuprate": np.zeros(n, bool),
             "origin": np.array(["nims"] * 30 + ["nemad"] * 50)}
    out = L.score_added_rows(extra, resamples=50)
    assert {"all", "not cuprates", "source nemad", "source nims"} <= set(out)
    assert out["source nemad"]["all"]["rows"] == 50 and out["source nims"]["all"]["rows"] == 30


def test_outside_test_reading_and_once_only(tmp_path):
    import nemad_outside_test as O

    rng = np.random.default_rng(1)
    keys, labels = [], {}
    for i in range(160):
        a = 0.2 + 0.6 * (i % 40) / 40
        key = f"{['Nb', 'Mo', 'Ta', 'V'][i // 40]}:{a:.3f};Sn:{1 - a:.3f}"
        keys.append(key)
        y = int(rng.uniform() < 0.7)
        labels[key] = (y, 5.0 if y else None)
    keys = sorted(set(keys))
    y = np.array([labels[k][0] for k in keys], float)
    tc = np.full(len(keys), 5.0)
    good = (np.clip(0.5 * y + rng.uniform(0, 0.5, len(keys)), 0, 1), tc)
    same = O.score_rows(keys, labels, good, good)                 # the blend equals cycle 6: "at least"
    assert same["reading"]["blend_at_least_cycle6"] and same["rows"] == len(keys)
    assert set(same["scores"]) == {"all", "not cuprates", "intermetallic"}
    worse = O.score_rows(keys, labels, good, (1.0 - good[0], tc))  # an anti-correlated TabICLv2 drags the blend down
    assert not worse["reading"]["blend_at_least_cycle6"]
    assert worse["paired"]["all"]["blend_minus_cycle6"]["auc"]["median"] < 0
    done = tmp_path / "nemad-outside-test.json"
    O.refuse_if_scored(done)
    done.write_text("{}")
    with pytest.raises(SystemExit):
        O.refuse_if_scored(done)


def test_outside_test_refuses_runs_unlike_the_record():
    import nemad_outside_test as O

    manifest = {"folds": [0, 1, 2, 3, 4], "content_sha256": "abc"}
    good = {"model": "tabicl", "complete": True, "bundle_content_sha256": "abc",
            "folds": [{"fold": f} for f in range(5)], "settings": {"estimators": 8, "max_context": 0, "seed": 0}}
    O.check_run(good, manifest)
    for bad in ({**good, "folds": [{"fold": 0}, {"fold": 1}]},                  # a --folds run
                {**good, "settings": {**good["settings"], "estimators": 4}},  # other settings
                {**good, "model": "tabpfn"}):
        with pytest.raises(SystemExit):
            O.check_run(bad, manifest)
    # near twins across a major-element line are one group, as in the table
    groups = O.independence_groups(["La:0.200;Ni:0.250;O:0.500;Sr:0.050", "La:0.210;Ni:0.250;O:0.500;Sr:0.040",
                                    "Nb:0.750;Sn:0.250"])
    assert groups[0] == groups[1] != groups[2]
