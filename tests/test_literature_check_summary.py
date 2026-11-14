"""The literature-check summary: counts, intervals and the exact test."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import literature_check_summary as L  # noqa: E402


def test_wilson_interval_matches_known_values():
    low, high = L.wilson(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4) and high == pytest.approx(0.7634, abs=1e-4)
    low, high = L.wilson(0, 20)
    assert low == 0.0 and high == pytest.approx(0.1611, abs=1e-4)
    assert L.wilson(0, 0) == (None, None)


def test_fisher_exact_matches_textbook_cases():
    # Fisher's tea-tasting table: 3 of 4 against 1 of 4.
    assert L.fisher_two_sided(3, 4, 1, 4) == pytest.approx(0.4857, abs=1e-4)
    assert L.fisher_two_sided(10, 10, 0, 10) == pytest.approx(2 / 184756, rel=1e-6)
    assert L.fisher_two_sided(2, 10, 2, 10) == pytest.approx(1.0)
    scipy_stats = pytest.importorskip("scipy.stats")
    for table in ((7, 12, 2, 15), (0, 9, 4, 11), (5, 5, 5, 6)):
        a, n, b, m = table
        assert L.fisher_two_sided(a, n, b, m) == pytest.approx(
            scipy_stats.fisher_exact([[a, n - a], [b, m - b]])[1], rel=1e-9)


def rows():
    out = [{"group": "top", "verdict": "superconductor_ambient"}] * 6
    out += [{"group": "top", "verdict": "measured_no_superconductivity"}] * 2
    out += [{"group": "top", "verdict": "no_low_temperature_data_found"}] * 2
    out += [{"group": "random", "verdict": "superconductor_ambient"}] * 1
    out += [{"group": "random", "verdict": "superconductor_only_pressure_doping_or_film"}] * 1
    out += [{"group": "random", "verdict": "magnetic_or_insulating_no_superconductivity"}] * 5
    out += [{"group": "random", "verdict": "no_low_temperature_data_found"}] * 3
    return out


def test_summary_counts_shares_and_comparison():
    summary = L.summarise(rows(), [("top", "random")])
    top, random = summary["groups"]["top"], summary["groups"]["random"]
    assert top["rows"] == 10 and top["with_low_temperature_data"] == 8 and random["with_low_temperature_data"] == 7
    assert top["superconductor_of_all"]["share"] == 0.6 and top["superconductor_of_measured"]["share"] == 0.75
    assert random["superconductor_of_all"]["hits"] == 1 and random["superconducts_somehow_of_all"]["hits"] == 2
    assert sum(top["verdicts"].values()) == 10 and sum(random["verdicts"].values()) == 10
    compared = summary["comparisons"][0]
    assert compared["superconductor_of_all"]["difference"] == pytest.approx(0.5)
    assert compared["superconductor_of_all"]["fisher_two_sided_p"] == pytest.approx(L.fisher_two_sided(6, 10, 1, 10))
    assert compared["superconductor_of_measured"]["difference"] == pytest.approx(0.75 - 1 / 7)
    with pytest.raises(SystemExit):
        L.summarise(rows(), [("top", "missing")])
    with pytest.raises(SystemExit):
        L.summarise(rows() + [{"group": "top", "verdict": "maybe"}], [])


def test_command_writes_the_summary(tmp_path, monkeypatch):
    cases = tmp_path / "cases.csv"
    cases.write_text("group,verdict\n" + "\n".join(f"{r['group']},{r['verdict']}" for r in rows()) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["literature_check_summary.py", "--cases", str(cases), "--compare", "top", "random",
                                      "--out", str(tmp_path / "out.json")])
    assert L.main() == 0
    written = json.loads((tmp_path / "out.json").read_text())
    assert written["rows"] == 20 and written["cases_file"] == "cases.csv" and len(written["comparisons"]) == 1
