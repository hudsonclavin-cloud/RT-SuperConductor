"""The blind test: record checks, the table rules and the once-only scoring guard."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("pymatgen")

import blind_test as B  # noqa: E402

VALID = {"Nb", "Ir", "B", "La", "Ru", "Si", "Mg", "Cu", "O", "Fe", "As"}


def test_identifier_accepts_a_doi_or_an_arxiv_number_in_any_spelling():
    assert B.identifier("10.1103/PhysRevB.103.214501") == "10.1103/physrevb.103.214501"
    assert B.identifier("https://doi.org/10.1088/1361-6668/abf7cf") == "10.1088/1361-6668/abf7cf"
    assert B.identifier("2101.00862") == B.identifier("arXiv:2101.00862v2") == "arxiv:2101.00862"
    assert B.identifier("10.48550/arxiv.2603.13111") == "arxiv:2603.13111"
    assert B.identifier("Phys. Rev. B 103, 214501") is None and B.identifier("") is None


def test_check_record_keeps_a_good_line_and_names_why_a_bad_one_is_dropped():
    good = ["NbIr2B2", "superconductor", "7.2", "2021", "2101.00862", "yes", "evidence"]
    record, reason = B.check_record(good, 2021, VALID)
    assert reason == "" and record["class"] == "superconductor" and record["identifier"] == "arxiv:2101.00862"
    cases = {"wrong_column_count": good[:6], "bad_class": [*good[:1], "maybe", *good[2:]],
             "temperature_not_a_number": [*good[:2], "about 7", *good[3:]],
             "temperature_out_of_range": [*good[:2], "0", *good[3:]],
             "year_not_the_cell_s": [*good[:3], "2020", *good[4:]],
             "no_doi_or_arxiv_number": [*good[:4], "none", *good[5:]],
             "formula_not_usable": ["NbIr2Xx2", *good[1:]]}
    for expected, line in cases.items():
        assert B.check_record(line, 2021, VALID) == (None, expected)
    null = ["LaRu3Si2", "null", "0", "2021", "10.1000/x", "maybe", "e"]
    record, _ = B.check_record(null, 2021, VALID)
    assert record["temperature_k"] == 0.0 and record["first_report"] == "unclear"


def records():
    base = {"year": "2022", "identifier": "10.1/a", "first_report": "yes"}
    return [{**base, "record_id": "r1", "formula": "NbIr2B2", "class": "superconductor", "temperature_k": "7.2"},
            {**base, "record_id": "r2", "formula": "NbIr2B2", "class": "null", "temperature_k": "1.8", "year": "2023"},
            {**base, "record_id": "r3", "formula": "LaRu3Si2", "class": "null", "temperature_k": "1.8"},
            {**base, "record_id": "r4", "formula": "LaRu3Si2", "class": "null", "temperature_k": "0.4", "first_report": "no"},
            {**base, "record_id": "r5", "formula": "MgB2", "class": "superconductor", "temperature_k": "39"},
            {**base, "record_id": "r6", "formula": "La2Ru3Si", "class": "null", "temperature_k": "4.2"},
            {**base, "record_id": "r7", "formula": "MgCu2", "class": "superconductor", "temperature_k": "3"},
            {**base, "record_id": "r8", "formula": "FeAs", "class": "null", "temperature_k": "0.3"}]


def test_table_follows_the_plan_s_rules():
    verdicts = {f"r{i}": {"verdict": "confirmed", "note": ""} for i in range(1, 9)}
    verdicts["r7"] = {"verdict": "wrong_value", "note": ""}
    rows, steps = B.build_table(records(), verdicts, nims_systems={frozenset({"Mg", "B"})})
    assert steps == {"records": 8, "confirmed_by_the_second_reader": 7, "nulls_measured_to_2_k_or_lower": 6,
                     "compositions": 4, "in_a_new_element_system": 3}
    by_formula = {r["formula"]: r for r in rows}
    assert set(by_formula) == {"NbIr2B2", "LaRu3Si2", "FeAs"}          # MgB2's system is in NIMS; MgCu2 was rejected
    nb = by_formula["NbIr2B2"]
    assert nb["label"] == 1 and nb["tc_k"] == 7.2 and nb["first_year"] == 2022 and nb["family"] == "Other"
    la = by_formula["LaRu3Si2"]
    assert la["label"] == 0 and la["null_lowest_k"] == 0.4 and la["record_ids"] == "r3 r4"
    assert by_formula["FeAs"]["family"] == "Ferrites"
    assert {r["first_harvest"] for r in rows} == {"yes"}               # records without a cell count as first harvest


def test_a_composition_found_only_by_the_second_harvest_is_marked():
    base = {"year": "2022", "identifier": "10.1/a", "first_report": "yes", "class": "null", "temperature_k": "0.4"}
    recs = [{**base, "record_id": "r1", "formula": "LaRu3Si2", "cell": "y2022-t7"},
            {**base, "record_id": "r2", "formula": "NbIr2B2", "cell": "y2022-t6"},
            {**base, "record_id": "r3", "formula": "NbIr2B2", "cell": "y2022-t4"}]
    verdicts = {f"r{i}": {"verdict": "confirmed", "note": ""} for i in (1, 2, 3)}
    rows, _ = B.build_table(recs, verdicts, nims_systems=set())
    assert {r["formula"]: r["first_harvest"] for r in rows} == {"LaRu3Si2": "no", "NbIr2B2": "yes"}


def test_a_null_measured_only_above_2_k_is_not_a_null():
    verdicts = {"r6": {"verdict": "confirmed", "note": ""}}
    rows, steps = B.build_table([records()[5]], verdicts, nims_systems=set())
    assert rows == [] and steps["nulls_measured_to_2_k_or_lower"] == 0


def test_scoring_refuses_a_second_run(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "DOCS", tmp_path)
    (tmp_path / "score.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="scored once"):
        B.cmd_score()
