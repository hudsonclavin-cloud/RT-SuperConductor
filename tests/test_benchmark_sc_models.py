import csv
import gzip
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
CANDIDATES = ROOT / "data/candidates/mp-candidates.csv.gz"

pytest.importorskip("matminer")


def test_every_benchmark_model_builds_and_predicts_probabilities():
    import numpy as np
    from benchmark_sc_models import model_factories

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6))
    X[::7, 2] = np.nan  # models must cope with missing values
    y = (X[:, 0] + rng.normal(scale=0.5, size=120) > 0).astype(int)
    for name, factory in model_factories(0).items():
        p = factory().fit(X, y).predict_proba(X)[:, 1]
        assert p.shape == (120,) and np.all((0 <= p) & (p <= 1)), name


@pytest.mark.skipif(not CANDIDATES.is_file(), reason="candidate table not present")
def test_benchmark_runs_on_a_real_slice(tmp_path):
    from benchmark_sc_models import run

    with gzip.open(CANDIDATES, "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pick = ([r for r in rows if r["known_status"] == "superconductor"][:60]
            + [r for r in rows if r["known_status"] == "tested_non_superconductor"
               and r.get("negative_training_use", "use") == "use"][:60])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(pick)
    small = tmp_path / "small.csv.gz"
    small.write_bytes(gzip.compress(buffer.getvalue().encode()))
    report = run(small, tmp_path / "bench.json", repeats=1, models=["logistic_regression", "gradient_boosting"])
    assert report["rows"] == 120 and set(report["models"]) == {"logistic_regression", "gradient_boosting"}
    assert report["best_by_auc"] in report["models"]
