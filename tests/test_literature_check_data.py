"""The literature-check data files: checksums, groups, verdicts, and that the committed summaries match them."""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import literature_check_summary as L  # noqa: E402

DATA = ROOT / "data" / "benchmarks" / "literature-check"
CASES = {
    "made-metals.csv": ({"top": 60, "random": 60, "bottom": 40},
                        "docs/literature-check-made-metals.json",
                        [("top", "random"), ("top", "bottom"), ("random", "bottom")]),
    "negatives-sample.csv": ({"suspect": 40, "random": 40},
                             "docs/literature-check-negatives.json",
                             [("suspect", "random")]),
}


def rows_of(name):
    with open(DATA / name, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_checksums_match_the_provenance_table():
    table = dict(re.findall(r"^\| (\S+\.csv) \| ([0-9a-f]{64}) \|$", (DATA / "PROVENANCE.md").read_text(), re.M))
    assert set(table) == set(CASES)
    for name, digest in table.items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("name", sorted(CASES))
def test_groups_verdicts_and_second_reading(name):
    sizes = CASES[name][0]
    rows = rows_of(name)
    assert len({row["case"] for row in rows}) == len(rows)
    assert {group: sum(row["group"] == group for row in rows) for group in sizes} == sizes
    assert {row["verdict"] for row in rows} <= set(L.VERDICTS)
    for row in rows:
        claimed = row["first_reader_verdict"] in L.VERDICTS[:2]
        assert not claimed or row["second_reader"] == "yes", row["case"]


@pytest.mark.parametrize("name", sorted(CASES))
def test_committed_summary_matches_the_data(name):
    _, summary_path, comparisons = CASES[name]
    expected = L.rounded({"cases_file": name, **L.summarise(rows_of(name), comparisons)})
    assert json.loads((ROOT / summary_path).read_text()) == expected


def test_negatives_sample_is_development_only_and_has_no_local_paths():
    rows = rows_of("negatives-sample.csv")
    assert {row["fold"] for row in rows} <= {"0", "1", "2", "3", "4"}
    assert not any("/tmp/" in value or "scratchpad" in value for row in rows for value in row.values())
    suspects = [float(row["oof_boosting_p"]) for row in rows if row["group"] == "suspect"]
    assert min(suspects) >= 0.93
