import csv
import gzip
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
CORE = ROOT / "data/candidates/mp-candidates.csv.gz"
WIDE = ROOT / "data/candidates/mp-wide-labelled.csv.gz"

pytest.importorskip("matminer")


def _slice(path, n_each, tmp, name, skip=0):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pick = ([r for r in rows if r["known_status"] == "superconductor"][skip:skip + n_each]
            + [r for r in rows if r["known_status"] == "tested_non_superconductor"
               and r.get("negative_training_use", "use") == "use"][skip:skip + n_each])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(pick)
    out = tmp / name
    out.write_bytes(gzip.compress(buffer.getvalue().encode()))
    return out


@pytest.mark.skipif(not (CORE.is_file() and WIDE.is_file()), reason="label tables not present")
def test_conditions_are_scored_on_identical_core_rows(tmp_path):
    from eval_label_expansion import CONDITIONS, run

    core = _slice(CORE, 40, tmp_path, "core.csv.gz")
    wide = _slice(WIDE, 70, tmp_path, "wide.csv.gz")
    report = run(core, wide, tmp_path / "out.json", models=["logistic_regression"], repeats=1)
    assert report["core_labels"] == 80 and report["extra_labels"] > 0
    result = report["results"]["logistic_regression"]
    assert set(result) == set(CONDITIONS)
    assert result["core_only"]["paired_gain_vs_core_only"] == 0.0


def test_committed_wide_labels_are_only_labelled_rows():
    if not WIDE.is_file():
        pytest.skip("wide labels not present")
    with gzip.open(WIDE, "rt") as handle:
        statuses = {r["known_status"] for r in csv.DictReader(handle)}
    assert statuses == {"superconductor", "tested_non_superconductor"}
