import csv
import gzip
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("pymatgen.core")

from build_mp_candidates import (  # noqa: E402
    NEGATIVE_TIERS, TRAINING_TIERS, build, fractions, known_index, negative_tier, training_label,
)
from fetch_mp_snapshot import flatten, write_snapshot  # noqa: E402

MTG_TABLE = ROOT / "data/training/external/supercon-mtg/SuperCon_MTG.csv"


def _reports(*items):
    return [(tc, source) for tc, source in items]


def test_tier_rules():
    assert negative_tier("exact", _reports((5.0, "supercon_stanev2018")), 0.0) == ""
    assert negative_tier("exact", _reports((0.0, "supercon_stanev2018"), (0.0, "3dsc_mp")), 0.0) == "C_database_zero"
    assert negative_tier("exact", _reports((0.0, "supercon_mtg")), 0.0) == "C_database_zero"
    assert negative_tier("exact", _reports((0.0, "supercon_stanev2018"), (0.0, "hosono_mtg")), 0.0) == "B_corroborated_zero"
    assert negative_tier("exact", _reports((0.0, "hosono_mtg")), 0.0) == "H_unsuccessful_search"
    assert negative_tier("near", _reports((0.0, "supercon_stanev2018")), 0.0) == "D_suspect"
    assert negative_tier("exact", _reports((0.0, "supercon_stanev2018")), 2.3) == "D_suspect"
    assert negative_tier("exact", _reports((0.0, "audit:10.1000/x")), 2.3) == "A_referenced_null"
    assert set(NEGATIVE_TIERS) >= TRAINING_TIERS


def test_training_label_respects_tiers_and_legacy_tables():
    assert training_label({"known_status": "superconductor"}) == 1
    assert training_label({"known_status": "tested_non_superconductor", "negative_training_use": "use"}) == 0
    assert training_label({"known_status": "tested_non_superconductor", "negative_training_use": "exclude"}) is None
    assert training_label({"known_status": "tested_non_superconductor"}) == 0  # table without tiers
    assert training_label({"known_status": "untested_by_formula"}) is None


def test_mtg_prefixes_become_lineages(tmp_path):
    mtg = tmp_path / "mtg.csv"
    mtg.write_text("Material-ID,Chemical_Formula,Temp_critical\nS-1,Nb3Sn,18\nH-1,MgFeAsH,0\nSM-1,Pb,7.2\nX-1,Bad,0\n")
    index = known_index(tmp_path / "absent.csv", tmp_path / "absent2.csv", mtg, None)
    sources = {source for reports in index.values() for _, _, source in reports}
    assert sources == {"supercon_mtg", "hosono_mtg", "supermat_mtg"}


def test_audit_file_overrides_to_tier_a(tmp_path):
    audit = tmp_path / "audit.csv"
    audit.write_text("formula,publication_doi,superconductivity_observed,tc_k,lowest_measured_temperature_k\n"
                     "BaTiO3,10.1000/example,no,,1.8\n")
    index = known_index(tmp_path / "a", tmp_path / "b", None, audit)
    (frac, tc, source), = index[frozenset(fractions("BaTiO3"))]
    assert tc == 0.0 and source == "audit:10.1000/example"
    assert negative_tier("exact", [(tc, source)], 2.5) == "A_referenced_null"


def _doc(i, formula, elements, gap, metal):
    return {"material_id": f"mp-{i}", "formula_pretty": formula, "elements": elements, "nelements": len(elements),
            "nsites": 4, "energy_above_hull": 0.0, "formation_energy_per_atom": -0.5, "is_stable": True,
            "theoretical": False, "symmetry": {"crystal_system": "Cubic", "number": 221, "symbol": "Pm-3m"},
            "volume": 60.0, "density": 6.0, "is_metal": metal, "band_gap": gap, "is_magnetic": False,
            "ordering": "NM", "total_magnetization": 0.0, "total_magnetization_normalized_formula_units": 0.0,
            "num_magnetic_sites": 0, "database_IDs": {}}


def test_build_writes_tiers_and_training_use(tmp_path):
    snap = tmp_path / "snap.csv.gz"
    rows = [flatten(_doc(1, "Nb3Sn", ["Nb", "Sn"], 0.0, True)), flatten(_doc(2, "BaTiO3", ["Ba", "Ti", "O"], 2.3, False)),
            flatten(_doc(3, "ZrRh3", ["Zr", "Rh"], 0.0, True)), flatten(_doc(4, "MgFeAsH", ["Mg", "Fe", "As", "H"], 0.0, True))]
    write_snapshot(rows, {}, snap, tmp_path / "m.json")
    stanev = tmp_path / "stanev.csv"
    stanev.write_text("name,Tc\nNb3Sn,17.8\nBaTiO3,0\nZrRh3,0\n")
    mtg = tmp_path / "mtg.csv"
    mtg.write_text("Material-ID,Chemical_Formula,Temp_critical\nH-1,MgFeAsH,0\nS-2,ZrRh3,0\n")
    report = build(snap, tmp_path / "out.csv.gz", threedsc=tmp_path / "none.csv", stanev=stanev, mtg=mtg, audit=None)
    out = {r["formula"]: r for r in csv.DictReader(gzip.open(tmp_path / "out.csv.gz", "rt"))}
    assert out["Nb3Sn"]["known_status"] == "superconductor" and out["Nb3Sn"]["negative_training_use"] == ""
    assert out["BaTiO3"]["negative_evidence_tier"] == "D_suspect" and out["BaTiO3"]["negative_training_use"] == "exclude"
    assert out["ZrRh3"]["negative_evidence_tier"] == "C_database_zero" and out["ZrRh3"]["negative_training_use"] == "use"
    assert out["ZrRh3"]["known_lineages"] == "supercon" and out["ZrRh3"]["known_sources"] == "supercon_mtg supercon_stanev2018"
    assert out["MgFeAsH"]["negative_evidence_tier"] == "H_unsuccessful_search"
    assert report["counts"]["negative_tier_D_suspect"] == 1


@pytest.mark.skipif(not MTG_TABLE.is_file(), reason="vendored SuperCon-MTG table not present")
def test_vendored_mtg_table_matches_provenance():
    import hashlib

    digest = hashlib.sha256(MTG_TABLE.read_bytes()).hexdigest()
    assert digest == "9743f0b000ee59b4ef1efb9b9ff2c1e967d45abebe9310da69f5d273c398423e"
    with MTG_TABLE.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 13731
    assert sum(r["Material-ID"].startswith("H-") for r in rows) == 493


def _quarantine_fixture(tmp_path):
    """3DSC and SuperCon rows for the suspect mp-510625 family plus one ordinary superconductor."""
    threedsc = tmp_path / "3dsc.csv"
    threedsc.write_text(
        '# header comment\n'
        'formula_sc,tc,material_id_2\n'
        'Cr1Cu1O2,132.0,mp-510625\n'
        'Nb3Sn,17.8,mp-1\n')
    stanev = tmp_path / "stanev.csv"
    stanev.write_text("name,Tc\nCu1Cr1O2,132\nNb3Sn,17.8\n")
    return threedsc, stanev


def test_quarantined_label_is_removed_from_every_source(tmp_path):
    from build_mp_candidates import match_known

    threedsc, stanev = _quarantine_fixture(tmp_path)
    held: list = []
    index = known_index(threedsc, stanev, None, None, quarantine_queue=tmp_path / "queue.csv", quarantined_out=held)
    assert match_known("CrCuO2", index) == ("", [])          # neither the 3DSC nor the SuperCon copy survives
    assert match_known("Nb3Sn", index)[0] == "exact"
    assert {h["source"] for h in held} == {"3dsc_mp", "supercon_stanev2018"}


def test_reviewer_retain_decision_releases_a_quarantined_label(tmp_path):
    from build_mp_candidates import match_known

    threedsc, stanev = _quarantine_fixture(tmp_path)
    queue = tmp_path / "queue.csv"
    queue.write_text("record_id,formula,tc_kelvin,audit_status,review_decision\n"
                     "abc,CrCuO2,132,reviewed,retain\n")
    index = known_index(threedsc, stanev, None, None, quarantine_queue=queue)
    assert match_known("CrCuO2", index)[0] == "exact"


def test_build_reports_quarantined_labels(tmp_path):
    threedsc, stanev = _quarantine_fixture(tmp_path)
    snap = tmp_path / "snap.csv.gz"
    write_snapshot([flatten(_doc(1, "CuCrO2", ["Cr", "Cu", "O"], 0.0, True))], {}, snap, tmp_path / "m.json")
    report = build(snap, tmp_path / "out.csv.gz", threedsc=threedsc, stanev=stanev, mtg=None, audit=None,
                   quarantine_queue=tmp_path / "queue.csv")
    row = next(csv.DictReader(gzip.open(tmp_path / "out.csv.gz", "rt")))
    assert row["known_status"] == "untested_by_formula"
    assert report["quarantined_reports"] == 2
