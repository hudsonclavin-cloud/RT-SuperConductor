"""Scoring outside cases: what counts as never seen, and that the summary matches the rows."""
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
E = pytest.importorskip("score_external_cases")

from test_scratchnet_rounds import dataset, ranked_toy  # noqa: E402,F401  (fixtures)

from rt_superconductor.scratchnet import data as D  # noqa: E402


def test_auc_interval_and_percentile():
    label = np.array([1, 1, 1, 0, 0, 0, 1, 0], dtype=float)
    perfect = E.auc_with_interval(label * 2.0 + 0.1, label, resamples=200)
    assert perfect["auc"] == 1.0 and perfect["low"] == 1.0 and perfect["superconductors"] == 4
    chance = E.auc_with_interval(np.array([1, 0, 1, 0, 1, 0, 0, 1], dtype=float), label, resamples=200)
    assert 0.0 <= chance["low"] <= chance["auc"] <= chance["high"] <= 1.0
    assert E.auc_with_interval(np.ones(3), np.ones(3))["auc"] is None           # one class only: no AUC
    assert E.percentile_in(np.array([1.0, 2.0, 3.0, 4.0]), 3.5) == 25.0
    assert E.percentile_in(np.array([1.0, 2.0]), 9.0) == 0.0
    # Resampling whole groups: each group holds one case of each class, ranked right, so every resample gives 1.
    grouped = E.auc_with_interval(np.array([3.0, 0.0, 2.0, 1.0]), np.array([1.0, 0.0, 1.0, 0.0]), resamples=200,
                                  groups=np.array(["a", "a", "b", "b"]))
    assert grouped["auc"] == 1.0 and grouped["low"] == grouped["high"] == 1.0 and grouped["resampled"] == "whole groups"
    rng = np.random.default_rng(3)
    score, truth = rng.random(40), (rng.random(40) < 0.5).astype(float)
    paper = np.repeat(np.arange(8), 5)
    by_case = E.auc_with_interval(score, truth, resamples=400)
    by_paper = E.auc_with_interval(score, truth, resamples=400, groups=paper)
    assert by_case["auc"] == by_paper["auc"] and by_case["resampled"] == "cases"
    assert by_paper["low"] <= by_paper["auc"] <= by_paper["high"]


def test_cases_are_scored_and_only_unseen_rows_count_as_clean(ranked_toy, dataset, monkeypatch, tmp_path):
    folder, pool = ranked_toy
    monkeypatch.setattr(C, "WORK", folder / "work")
    monkeypatch.setattr(C, "load_dataset", lambda *args, **kwargs: dataset)
    in_table = dataset.rows[40]                                    # a row of the made-up table itself
    twin = C.formula_text(C.fractions_of(in_table["composition"]))
    cases = [{"formula": "Nb3Al", "outcome": "superconductor", "tc_k": "18.0", "source": "a"},
             {"formula": "V3Ge", "outcome": "superconductor", "tc_k": "6.0", "source": "b"},
             {"formula": "NaF", "outcome": "not_superconductor", "tc_k": "", "source": "c"},
             {"formula": "KBr", "outcome": "not_superconductor", "tc_k": "", "source": "d"},
             {"formula": twin, "outcome": "superconductor" if in_table["label"] == "0" else "not_superconductor",
              "tc_k": "", "source": "e"},
             {"formula": "PbTe", "outcome": "unknown", "tc_k": "", "source": "f"},
             {"formula": "Xx2O", "outcome": "superconductor", "tc_k": "3", "source": "g"},
             {"formula": "Og2O", "outcome": "superconductor", "tc_k": "3", "source": "h"}]
    rows, summary = E.score_cases(cases, "toy", group_column="source")
    assert summary["clean"]["groups"] == 4 and summary["clean"]["boosting"]["by_probability"]["resampled"] == "whole groups"
    with pytest.raises(SystemExit):
        E.score_cases(cases, "toy", group_column="paper")
    assert [r["scored"] for r in rows[:6]] == ["yes"] * 6 and all(r["scored"].startswith("no") for r in rows[6:])
    assert summary["rows"] == 8 and summary["scored"] == 6 and summary["with_known_outcome"] == 5
    assert rows[4]["in_table"] == "yes" and rows[4]["clean"] == "no" and rows[4]["table_split"] == in_table["split"]
    assert summary["in_table"] == {"rows": 1, "table_label_disagrees_with_outcome": [twin]}
    clean = [r for r in rows if r["clean"] == "yes" and r["outcome"] in E.OUTCOMES]
    assert summary["clean"]["rows"] == len(clean) == 4 and summary["all"]["rows"] == 5
    counts = summary["clean"]["boosting"]["by_probability"]
    assert (counts["superconductors"], counts["not_superconductors"]) == (2, 2) and 0.0 <= counts["auc"] <= 1.0
    assert summary["clean"]["boosting"]["tc_rows"] == 2
    for r in rows[:6]:
        assert float(r["boosting_expected_tc_k"]) == pytest.approx(float(r["boosting_p"]) * float(r["boosting_tc_k"]), abs=0.06)
        assert r["network_view"] in ("agrees", "lukewarm", "rejects") and r["source"] in "abcdef"
    views = summary["clean"]["network_view_superconductors"]
    assert sum(views.values()) == 2
    # With a ranked pool, each case gets its place in it.
    import run_scratchnet as R
    R.run_rank("toy", pool, top=20, out_dir=tmp_path)
    rows, summary = E.score_cases(cases, "toy", pool=folder / "work" / "toy-candidates-boosting.csv.gz")
    assert summary["pool"]["rows"] == 12 and all(0.0 <= float(r["pool_percentile_by_p"]) <= 100.0 for r in rows[:6])
    assert summary["clean_superconductors_median_pool_percentile_by_p"] is not None


def test_command_writes_rows_and_summary(ranked_toy, dataset, monkeypatch, tmp_path):
    folder, _ = ranked_toy
    monkeypatch.setattr(C, "WORK", folder / "work")
    monkeypatch.setattr(C, "load_dataset", lambda *args, **kwargs: dataset)
    cases = tmp_path / "cases.csv"
    cases.write_text("formula,outcome,tc_k,note\nNb3Al,superconductor,18,x\nNaF,not_superconductor,,y\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["score_external_cases.py", "--cases", str(cases), "--name", "toy",
                                      "--out-rows", str(tmp_path / "rows.csv"), "--out", str(tmp_path / "s.json")])
    assert E.main() == 0
    rows = D.read_table(tmp_path / "rows.csv")
    assert [r["formula"] for r in rows] == ["Nb3Al", "NaF"] and rows[0]["note"] == "x" and rows[0]["boosting_p"] != ""
    assert json.loads((tmp_path / "s.json").read_text())["cases_file"] == "cases.csv"
    bad = tmp_path / "bad.csv"
    bad.write_text("name,result\nNb3Al,yes\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        E.read_cases(bad)


def test_a_composition_near_a_table_row_is_not_clean_even_when_it_is_not_in_the_table(ranked_toy, dataset, monkeypatch):
    folder, _ = ranked_toy
    monkeypatch.setattr(C, "WORK", folder / "work")
    monkeypatch.setattr(C, "load_dataset", lambda *args, **kwargs: dataset)
    assert "Si:0.600;V:0.400" in {row["composition"] for row in dataset.rows}     # V0.4Si0.6 is a table row
    cases = [{"formula": "V0.39Si0.61", "outcome": "superconductor", "tc_k": ""},     # 0.01 from it, not one of the rows
             {"formula": "V0.375Si0.625", "outcome": "superconductor", "tc_k": ""}]   # halfway between two rows: 0.025
    rows, summary = E.score_cases(cases, "toy")
    assert rows[0]["in_table"] == "no" and rows[0]["clean"] == "no"
    assert rows[0]["distance_to_nearest_table_row"] == pytest.approx(0.01)
    assert rows[1]["in_table"] == "no" and rows[1]["clean"] == "yes"
    assert rows[1]["distance_to_nearest_table_row"] == pytest.approx(0.025)
    assert summary["all"]["rows"] == 2 and summary["clean"]["rows"] == 1
    rows, _ = E.score_cases(cases, "toy", clean_distance=0.005)
    assert [r["clean"] for r in rows] == ["yes", "yes"]


def test_a_pool_not_ordered_by_boosting_is_refused(ranked_toy, dataset, monkeypatch, tmp_path):
    import gzip
    import run_scratchnet as R
    folder, pool = ranked_toy
    monkeypatch.setattr(C, "WORK", folder / "work")
    monkeypatch.setattr(C, "load_dataset", lambda *args, **kwargs: dataset)
    R.run_rank("toy", pool, top=20, out_dir=tmp_path, order="mix")
    cases = [{"formula": "Nb3Al", "outcome": "superconductor", "tc_k": "18"}]
    with pytest.raises(SystemExit, match="--order boosting"):
        E.score_cases(cases, "toy", pool=folder / "work" / "toy-candidates-mix.csv.gz")
    unlabelled = tmp_path / "old.csv.gz"                                   # an older file with no ranked_by column
    with gzip.open(unlabelled, "wt", encoding="utf-8") as handle:
        handle.write("formula,expected_tc_k,p_superconductor\nNbN,3.0,0.9\n")
    with pytest.raises(SystemExit):
        E.score_cases(cases, "toy", pool=unlabelled)
