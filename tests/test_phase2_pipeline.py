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
from rt_superconductor.harmonized import bridge_reviewed_records, load_harmonized
from train_pressure_aware import select_training_records, train_pressure_aware


def test_harmonized_fixture_has_all_sources_and_valid_rows():
    records = load_harmonized(ROOT / "data/training/harmonized_supercon.csv")
    assert len(records) == 26
    assert {record.source_db for record in records} == {"supercon2", "3dsc", "supermat", "uci"}
    assert all(0 < record.tc_kelvin <= 300 for record in records)
    assert sum(record.is_synthetic for record in records) == 21
    assert sum(record.provenance == "literature" for record in records) == 5
    empirical = select_training_records(records)
    assert len(empirical) == 5
    assert all(record.provenance == "literature" and record.is_synthetic == 0 for record in empirical)


def test_reviewed_bridge_is_idempotent(tmp_path):
    source = ROOT / "data/training/harmonized_supercon.csv"
    target = tmp_path / "harmonized.csv"
    lines = source.read_text(encoding="utf-8").splitlines()
    target.write_text("\n".join(lines[:22]) + "\n", encoding="utf-8")
    reviewed = ROOT / "data/training/reviewed/reviewed_records.csv"

    first = bridge_reviewed_records(target, reviewed)
    second = bridge_reviewed_records(target, reviewed)
    assert first["bridged_record_count"] == 5
    assert second["bridged_record_count"] == 0
    assert sum(record.provenance == "literature" for record in load_harmonized(target)) == 5


def test_training_blocks_without_two_empirical_families(tmp_path):
    report = train_pressure_aware(
        input_path=ROOT / "data/training/harmonized_supercon.csv",
        output_dir=tmp_path / "model",
        allow_insufficient_data=True,
    )
    assert report["training_status"] == "blocked_insufficient_empirical_data"
    assert report["empirical_training_record_count"] == 5
    assert report["synthetic_record_count"] == 21
    assert report["training_filter"] == "review_status == 'confirmed' and is_synthetic == 0"


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
