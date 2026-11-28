"""The local-GPU bridge: the bundle holds no locked rows and the cycle 6 folds; the trainer runs; compare is paired."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("scipy")
import local_gpu as G  # noqa: E402
import local_gpu_train as TR  # noqa: E402
from scratchnet_common import T  # noqa: E402


def fake_dataset(n_dev=90, n_train=30, n_lock=20, seed=0):
    """A small table in the real layout: dev rows in three folds, train-only rows, locked rows, many groups."""
    rng = np.random.default_rng(seed)
    n = n_dev + n_train + n_lock
    split = np.array(["dev"] * n_dev + ["train_only"] * n_train + ["lockbox"] * n_lock)
    fold = np.array([i % 3 for i in range(n_dev)] + [-1] * (n_train + n_lock))
    group = np.array([f"g{i // 2}" for i in range(n)])
    label = rng.integers(0, 2, n).astype(float)
    label[::17] = np.nan
    tc = np.where(label == 1, rng.uniform(1, 30, n), np.nan)
    rows = [{"composition": f"C{i}"} for i in range(n)]
    arrays = SimpleNamespace(label=label, tc_k=tc, group=group)
    family = np.array(["Cuprates" if i % 9 == 0 else "Other" for i in range(n)])
    return SimpleNamespace(rows=rows, split=split, fold=fold, arrays=arrays, family=family), rng.normal(size=(n, 6))


def fake_additions(n=8, seed=1):
    rng = np.random.default_rng(seed)
    return {"keys": [f"A{i}" for i in range(n)], "X": rng.normal(size=(n, 6)), "label": np.ones(n),
            "tc_k": rng.uniform(1, 10, n), "fold": [i % 3 for i in range(n)],
            "tied": [{f"g{i}"} if i % 2 else set() for i in range(n)]}


def test_bundle_has_no_locked_rows_and_keeps_each_folds_rows():
    ds, X = fake_dataset()
    extra = fake_additions()
    bundle = G.build_bundle(ds, X, extra)
    assert "lockbox" not in set(bundle["split"].tolist())
    locked_keys = {ds.rows[i]["composition"] for i in np.flatnonzero(ds.split == "lockbox")}
    assert not locked_keys & set(bundle["keys"].tolist())
    n_table = int((~bundle["is_added_row"]).sum())
    assert len(bundle["keys"]) == n_table + len(extra["keys"])
    for fold in bundle["folds"]:
        fit, valid, test = (bundle[f"fold{fold}_{part}"] for part in ("fit", "valid", "test"))
        assert set(bundle["split"][test]) == {"dev"}
        expected = {ds.rows[i]["composition"] for i in np.flatnonzero((ds.split == "dev") & (ds.fold == fold))}
        assert set(bundle["keys"][test]) == expected
        assert not (set(fit) & set(test)) and not (set(valid) & set(test)) and not (set(fit) & set(valid))
        added = fit[fit >= n_table] - n_table
        assert all(extra["fold"][i] != fold for i in added)        # no added row of the held-out fold


def test_compare_of_identical_predictions_is_a_tie():
    ds, _ = fake_dataset()
    rng = np.random.default_rng(3)
    dev = np.flatnonzero(ds.split == "dev")
    preds = {ds.rows[i]["composition"]: (float(rng.uniform()), float(rng.uniform(1, 20))) for i in dev}
    out = G.compare(ds, preds, preds)
    assert out["rows_compared"] == len(dev)
    gap = out["paired"]["not cuprates"]["network_minus_record"]["auc"]
    assert gap["low"] == pytest.approx(0.0) and gap["high"] == pytest.approx(0.0)
    assert out["reading"]["network_beats_record"] is False


def test_preprocessing_is_fitted_on_the_fit_rows_only():
    X = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0]])
    prep = TR.preprocess_fit(X[:2])
    out = TR.preprocess_apply(np.array([[100.0, np.nan]]), prep)
    assert out.shape == (1, 3)                                     # two scaled columns plus one missing flag
    assert out[0, 0] == pytest.approx((100.0 - 2.0) / 1.0)         # mean and spread of the first two rows only
    assert out[0, 2] == 1.0


def write_bundle(tmp_path, ds, X, extra, candidates=None):
    bundle = G.build_bundle(ds, X, extra, candidates)
    path = tmp_path / "bundle.npz"
    np.savez_compressed(path, **bundle)
    return path


def test_trainer_runs_on_a_tiny_bundle_and_records_its_checksum(tmp_path):
    pytest.importorskip("torch")
    ds, X = fake_dataset()
    candidates = {"pool_X": np.random.default_rng(2).normal(size=(5, 6)), "pool_keys": np.array(list("abcde"))}
    path = write_bundle(tmp_path, ds, X, fake_additions(), candidates)
    out = tmp_path / "results"
    assert TR.main([str(path), "--out", str(out), "--device", "cpu", "--seeds", "1", "--epochs", "3",
                    "--patience", "2", "--hidden", "8"]) == 0
    with np.load(out / "oof.npz") as oof:
        assert len(oof["compositions"]) == int((ds.split == "dev").sum())
        assert np.all((oof["p"] >= 0) & (oof["p"] <= 1)) and np.all(oof["tc"] >= 0)
    with np.load(out / "candidates_pool.npz") as pool:
        assert list(pool["keys"]) == list("abcde")
    run = json.loads((out / "run.json").read_text())
    assert run["bundle_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert [f["fold"] for f in run["folds"]] == [0, 1, 2]


def test_trainer_refuses_a_bundle_with_locked_rows(tmp_path):
    pytest.importorskip("torch")
    ds, X = fake_dataset()
    bundle = G.build_bundle(ds, X, None)
    bundle["split"] = bundle["split"].copy()
    bundle["split"][0] = "lockbox"
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **bundle)
    with pytest.raises(SystemExit):
        TR.main([str(path), "--out", str(tmp_path / "r"), "--device", "cpu", "--seeds", "1", "--epochs", "1"])


def test_score_helper_still_takes_the_arguments_compare_passes():
    out = T.score(np.array([0.0, 1.0, 1.0]), np.array([np.nan, 5.0, 9.0]), np.array([0.2, 0.7, 0.9]),
                  np.array([1.0, 4.0, 8.0]))
    assert out["auc"] == pytest.approx(1.0)


def test_content_checksum_ignores_how_the_file_was_written(tmp_path):
    ds, X = fake_dataset()
    first = write_bundle(tmp_path, ds, X, fake_additions())
    second = tmp_path / "again.npz"
    np.savez(second, **G.build_bundle(ds, X, fake_additions()))            # uncompressed this time
    with np.load(first) as a, np.load(second) as b:
        assert TR.content_sha256(a) == TR.content_sha256(b)
    assert G.same_bundle({"bundle_content_sha256": "x", "bundle_sha256": "f"}, {"content_sha256": "x", "sha256": "g"})
    assert not G.same_bundle({"bundle_content_sha256": "y"}, {"content_sha256": "x"})
    assert G.same_bundle({"bundle_sha256": "f"}, {"sha256": "f"})                # runs made before the content checksum
