import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from enrich_materials_project import enrich_csv, select_lowest_energy_polymorph
from generate_qe_inputs import generate_inputs
from rt_superconductor.harmonized import load_harmonized


def test_harmonized_fixture_has_all_sources_and_valid_rows():
    records = load_harmonized(ROOT / "data/training/harmonized_supercon.csv")
    assert len(records) == 21
    assert {record.source_db for record in records} == {"supercon2", "3dsc", "supermat", "uci"}
    assert all(0 < record.tc_kelvin <= 300 for record in records)


def test_materials_project_selects_lowest_energy_polymorph():
    documents = [
        SimpleNamespace(material_id="mp-high", energy_above_hull=0.12, symmetry=SimpleNamespace(number=12)),
        SimpleNamespace(material_id="mp-low", energy_above_hull=0.01, symmetry=SimpleNamespace(number=225)),
    ]
    match = select_lowest_energy_polymorph(documents)
    assert match.mp_id == "mp-low"
    assert match.energy_above_hull == pytest.approx(0.01)
    assert match.spacegroup_number == 225


def test_materials_project_enrichment_queries_unique_formulas(tmp_path):
    source = ROOT / "data/training/harmonized_supercon.csv"
    output = tmp_path / "enriched.csv"

    class Summary:
        def __init__(self):
            self.formulas = []

        def search(self, *, formula, fields):
            self.formulas.append(formula)
            return [SimpleNamespace(material_id="mp-test", energy_above_hull=0.02, symmetry=SimpleNamespace(number=1))]

    summary = Summary()
    client = SimpleNamespace(materials=SimpleNamespace(summary=summary))
    report = enrich_csv(source, output, client=client)
    assert report["unique_formula_count"] == len(summary.formulas)
    assert report["errors"] == {}
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["mp_id"] == "mp-test"
    assert row["energy_above_hull"] == "0.02"
    assert row["spacegroup_number"] == "1"


def test_qe_generator_applies_screen_and_writes_epw_inputs(tmp_path):
    candidates = tmp_path / "candidates.csv"
    candidates.write_text(
        "record_id,formula,predicted_tc_kelvin,energy_above_hull,review_status\n"
        "pass,H3S,180,0.01,confirmed\n"
        "fail,Fe,90,0.08,confirmed\n",
        encoding="utf-8",
    )
    summary = generate_inputs(candidates, tmp_path / "qe")
    assert summary["selected_count"] == 1
    candidate = tmp_path / "qe" / "pass"
    assert (candidate / "scf.in").is_file()
    assert (candidate / "ph.in").is_file()
    assert (candidate / "epw.in").is_file()
    scf = (candidate / "scf.in").read_text(encoding="utf-8")
    assert "conv_thr = 1.0d-10" in scf
    assert "ATOMIC_POSITIONS crystal" in scf
