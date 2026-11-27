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
