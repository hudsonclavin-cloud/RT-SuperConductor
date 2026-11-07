import csv
import gzip
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
CANDIDATES = ROOT / "data/candidates/mp-candidates.csv.gz"

pytest.importorskip("matminer")


@pytest.mark.skipif(not CANDIDATES.is_file(), reason="candidate table not present")
def test_classifier_runs_end_to_end_on_a_real_slice(tmp_path):
    from train_mp_sc_classifier import run

    with gzip.open(CANDIDATES, "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pick = ([r for r in rows if r["known_status"] == "superconductor"][:40]
            + [r for r in rows if r["known_status"] == "tested_non_superconductor"
               and r.get("negative_training_use", "use") == "use"][:40]
            + [r for r in rows if r["known_status"] == "untested_by_formula"][:40])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(pick)
    small = tmp_path / "small.csv.gz"
    small.write_bytes(gzip.compress(buffer.getvalue().encode()))
    report = run(small, tmp_path / "scores.csv.gz", tmp_path / "top.csv", tmp_path / "summary.json", top=10)
    assert report["labelled"] == 80
    assert 0.0 <= report["held_out_quality"]["auc"] <= 1.0
    scored = list(csv.DictReader(gzip.open(tmp_path / "scores.csv.gz", "rt")))
    assert len(scored) == 120 and all(0.0 <= float(r["p_superconductor"]) <= 1.0 for r in scored)
    ranked = list(csv.DictReader((tmp_path / "top.csv").open()))
    assert all(r["epw_eligible"] == "True" and r["known_status"] == "untested_by_formula" for r in ranked)
    probabilities = [float(r["p_superconductor"]) for r in ranked]
    assert probabilities == sorted(probabilities, reverse=True)
    assert json.loads((tmp_path / "summary.json").read_text())["materials"] == 120
