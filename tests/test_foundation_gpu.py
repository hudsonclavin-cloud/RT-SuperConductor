"""The runner for off-the-shelf tabular learners: same folds as cycle 6, no test row in training, compare-ready output."""
import hashlib
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
import local_gpu_train as TR  # noqa: E402
from test_local_gpu import fake_additions, fake_dataset  # noqa: E402


def write_bundle(tmp_path, n_dev=240, n_train=90):
    ds, X = fake_dataset(n_dev=n_dev, n_train=n_train, n_lock=30)
    X = X.copy()
    X[::11, 2] = np.nan                                         # the learners must cope with missing values
    candidates = {"pool_X": np.random.default_rng(5).normal(size=(7, 6)), "pool_keys": np.array(list("abcdefg"))}
    path = tmp_path / "bundle.npz"
    np.savez_compressed(path, **G.build_bundle(ds, X, fake_additions(), candidates))
    return ds, path


def test_in_context_models_get_fit_and_validation_rows_but_never_test_rows(tmp_path):
    _, path = write_bundle(tmp_path)
    bundle = np.load(path)
    for fold in bundle["folds"]:
        fit, valid, test = (bundle[f"fold{fold}_{p}"] for p in ("fit", "valid", "test"))
        labelled = np.isfinite(bundle["label"])
        together = F.fold_arrays(bundle, int(fold), "tabicl", cap=0, seed=0)
        assert together["counts"]["classifier_rows"] == int(labelled[fit].sum() + labelled[valid].sum())
        assert together["counts"]["validation_rows"] == 0
        apart = F.fold_arrays(bundle, int(fold), "hgb", cap=0, seed=0)
        assert apart["counts"]["classifier_rows"] == int(labelled[fit].sum())
        assert apart["counts"]["validation_rows"] == int(labelled[valid].sum())
        assert list(together["test"]) == list(test) and not set(test) & (set(fit) | set(valid))


def test_a_bundle_with_a_test_row_in_training_is_refused(tmp_path):
    _, path = write_bundle(tmp_path)
    bundle = dict(np.load(path))
    bundle["fold0_fit"] = np.concatenate([bundle["fold0_fit"], bundle["fold0_test"][:1]])
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **bundle)
    with pytest.raises(SystemExit):
        F.fold_arrays(np.load(bad), 0, "hgb", cap=0, seed=0)


def test_context_cap_keeps_the_label_balance():
    rng = np.random.default_rng(0)
    label = (rng.uniform(size=1000) < 0.7).astype(float)
    rows = np.arange(1000)
    kept = F.capped(rows, label, 200, seed=1)
    assert 190 <= len(kept) <= 210 and len(set(kept.tolist())) == len(kept)
    assert abs(label[kept].mean() - label.mean()) < 0.02
    assert list(F.capped(rows, label, 0, seed=1)) == list(rows)          # 0 means no cap


@pytest.mark.parametrize("model", ["hgb", "logistic"])
def test_runner_writes_output_that_compare_accepts(tmp_path, model):
    ds, path = write_bundle(tmp_path)
    out = tmp_path / model
    assert F.main([str(path), "--model", model, "--out", str(out)]) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["complete"] and run["bundle_kind"] == "formula" and run["model"] == model
    with np.load(path) as bundle:
        assert run["bundle_content_sha256"] == TR.content_sha256(bundle)
    assert run["bundle_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert G.same_bundle(run, {"content_sha256": run["bundle_content_sha256"], "sha256": "other"})
    with np.load(out / "oof.npz") as oof:
        dev = {r["composition"] for r, s in zip(ds.rows, ds.split) if s == "dev"}
        assert set(oof["compositions"].tolist()) == dev
        assert np.all((oof["p"] >= 0) & (oof["p"] <= 1)) and np.all(oof["tc"] >= 0)
        predictions = {str(k): (float(p), float(t)) for k, p, t in zip(oof["compositions"], oof["p"], oof["tc"])}
    with np.load(out / "candidates_pool.npz") as pool:
        assert list(pool["keys"]) == list("abcdefg")
    report = G.compare(ds, predictions, predictions)                    # the judge reads this output as it stands
    assert report["rows_compared"] == len(dev)


def test_missing_library_fails_before_any_work(tmp_path, monkeypatch):
    _, path = write_bundle(tmp_path)
    monkeypatch.setitem(sys.modules, "tabicl", None)                    # importing it now raises ImportError
    with pytest.raises(ImportError):
        F.main([str(path), "--model", "tabicl", "--out", str(tmp_path / "x")])
    assert not (tmp_path / "x").exists()
