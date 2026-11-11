import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("matminer")
pytest.importorskip("sklearn")

import score_pool_physics  # noqa: E402
from score_pool_physics import (  # noqa: E402
    RANKING_CLASSIFIER, RANKING_CONSENSUS, _consensus, default_candidates, fit_formula_model,
    formula_feature_rows, run, score,
)
from train_physics_targets import build_characteristics  # noqa: E402

TABLE = ROOT / "data/training/physics-target-characteristics.csv"


def _small_table(tmp_path: Path, n: int = 80) -> Path:
    with TABLE.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [row for _, row in zip(range(n), reader)]
        fields = reader.fieldnames
    out = tmp_path / "table.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out


def test_formula_features_match_the_trainer_exactly():
    from pymatgen.core import Lattice, Structure

    s = Structure(Lattice.cubic(5.29), ["Nb"] * 6 + ["Sn"] * 2,
                  [[0, .25, .5], [0, .75, .5], [.5, 0, .25], [.5, 0, .75], [.25, .5, 0], [.75, .5, 0],
                   [0, 0, 0], [.5, .5, .5]])
    record = {"jid": "hazel-nb3sn", "lamb": 1.0, "wlog": 150.0, "atoms": {
        "lattice_mat": s.lattice.matrix.tolist(), "coords": s.frac_coords.tolist(),
        "elements": [str(x.specie) for x in s], "cartesian": False}}
    trainer_rows, names, _ = build_characteristics([record], None)
    ours = formula_feature_rows(["Nb3Sn"])[0]
    for name in names:
        assert ours[name] == pytest.approx(trainer_rows[0][name]), name


def test_grouped_formulas_are_flattened_like_the_trainer():
    row = formula_feature_rows(["VGa(CuO2)2"])[0]
    assert "(" not in row["formula"]


def test_consensus_is_the_lower_model():
    assert _consensus(14.1, "11.8") == pytest.approx(11.8)
    assert _consensus(7.1, "16.1") == pytest.approx(7.1)
    assert _consensus(5.0, "") is None


def test_eligibility_and_ranking(tmp_path):
    models, meta = fit_formula_model(_small_table(tmp_path), None, evaluate=False)
    assert meta["targets"].startswith("log(lambda)")
    candidates = [
        {"mp_id": "a", "formula": "Nb3Sn", "chemical_family": "Other", "num_sites": "8", "predicted_tc_kelvin": "15"},
        {"mp_id": "b", "formula": "V3Si", "chemical_family": "Other", "num_sites": "8", "predicted_tc_kelvin": "1"},
        {"mp_id": "c", "formula": "YBa2Cu3O7", "chemical_family": "Cuprates", "num_sites": "13", "predicted_tc_kelvin": "90"},
        {"mp_id": "d", "formula": "FeSe", "chemical_family": "Other", "num_sites": "4", "predicted_tc_kelvin": "8"},
        {"mp_id": "e", "formula": "Nb3Sn", "chemical_family": "Other", "num_sites": "32", "predicted_tc_kelvin": "15"},
    ]
    rows = {r["mp_id"]: r for r in score(candidates, models)}
    assert rows["a"]["epw_eligible"] and rows["b"]["epw_eligible"]
    assert "not_phonon_mediated_route" in rows["c"]["epw_ineligible_reasons"]
    assert "magnetic_element_heuristic" in rows["d"]["epw_ineligible_reasons"]
    assert "more_than_20_sites" in rows["e"]["epw_ineligible_reasons"]
    ranked = [r for r in score(candidates, models) if r["epw_eligible"]]
    assert [r["epw_rank"] for r in ranked] == [1, 2]
    assert ranked[0]["consensus_tc_k"] >= ranked[1]["consensus_tc_k"]


def test_empty_candidate_pool_writes_valid_empty_outputs_without_fitting(tmp_path):
    candidates = tmp_path / "empty.csv"
    candidates.write_text(
        "mp_id,formula,chemical_family,num_sites,predicted_tc_kelvin\n",
        encoding="utf-8",
    )
    output = tmp_path / "rescored.csv"
    summary = tmp_path / "rescored.json"

    report = run(
        candidates,
        output,
        summary,
        model_path=None,
        table=tmp_path / "missing-training-table.csv",
        hazel_records=tmp_path / "missing-hazel-records.json",
    )

    assert report["status"] == "empty_candidate_pool"
    assert report["candidates_scored"] == 0
    assert report["epw_eligible"] == 0
    assert report["rows_written"] == 0
    assert report["model"]["feature_mode"] == "not_run_empty_candidate_pool"
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert "epw_eligible" in reader.fieldnames
        assert list(reader) == []
    assert json.loads(summary.read_text(encoding="utf-8"))["status"] == "empty_candidate_pool"


def test_default_candidates_skip_empty_lists(tmp_path, monkeypatch):
    """A header-only strict list must not leave the loop with nothing to score."""
    strict, pool, classifier = tmp_path / "strict.csv", tmp_path / "pool.csv", tmp_path / "classifier.csv"
    header = "mp_id,formula,chemical_family,num_sites\n"
    strict.write_text(header, encoding="utf-8")
    pool.write_text(header, encoding="utf-8")
    classifier.write_text(header + "x,Nb3Sn,Other,8\n", encoding="utf-8")
    sources = ((strict, "strict"), (pool, "pool"), (classifier, "classifier"))
    monkeypatch.setattr(score_pool_physics, "CANDIDATE_SOURCES", sources)
    assert default_candidates() == (classifier, "classifier")
    pool.write_text(header + "y,V3Si,Other,8\n", encoding="utf-8")
    assert default_candidates() == (pool, "pool")
    strict.write_text(header + "z,Nb3Ge,Other,8\n", encoding="utf-8")
    assert default_candidates() == (strict, "strict")


def test_default_candidates_report_the_empty_strict_list_when_nothing_has_rows(tmp_path, monkeypatch):
    missing = tmp_path / "missing.csv"
    monkeypatch.setattr(score_pool_physics, "CANDIDATE_SOURCES", ((missing, "strict"),))
    monkeypatch.setattr(score_pool_physics, "ELIGIBLE_LIST", missing)
    path, reason = default_candidates()
    assert path == missing and "no candidate list has rows" in reason


def test_classifier_rows_keep_the_classifier_order(tmp_path):
    """The physics model is a second opinion there; it must not reorder the list."""
    table = _small_table(tmp_path)
    candidates = tmp_path / "classifier.csv"
    candidates.write_text(
        "mp_id,formula,chemical_family,num_sites,p_superconductor,is_metal,is_magnetic\n"
        "low,Nb3Sn,Other,8,0.40,True,False\n"
        "high,NaCl,Other,2,0.95,True,False\n"
        "mid,V3Si,Other,8,0.70,True,False\n",
        encoding="utf-8",
    )
    report = run(candidates, tmp_path / "out.csv", tmp_path / "out.json", model_path=None,
                 table=table, hazel_records=tmp_path / "absent.json", source_reason="test")
    assert report["ranking"] == RANKING_CLASSIFIER
    assert report["candidates_source_reason"] == "test"
    assert [row["mp_id"] for row in report["top_eligible"]] == ["high", "mid", "low"]
    with (tmp_path / "out.csv").open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert [row["p_superconductor"] for row in written] == ["0.95", "0.7", "0.4"]
    assert all(row["physics_tc_k"] for row in written)


def test_rows_without_a_classifier_probability_rank_by_consensus(tmp_path):
    table = _small_table(tmp_path)
    candidates = tmp_path / "pool.csv"
    candidates.write_text("mp_id,formula,chemical_family,num_sites,predicted_tc_kelvin\n"
                          "a,Nb3Sn,Other,8,15\nb,V3Si,Other,8,1\n", encoding="utf-8")
    report = run(candidates, tmp_path / "out.csv", tmp_path / "out.json", model_path=None,
                 table=table, hazel_records=tmp_path / "absent.json")
    assert report["ranking"] == RANKING_CONSENSUS


def test_a_reviewed_hazel_record_changes_the_model(tmp_path):
    """The loop is closed only if a new Hazel result moves the predictions."""
    from pymatgen.core import Lattice, Structure

    table = _small_table(tmp_path)
    candidates = tmp_path / "cands.csv"
    candidates.write_text("mp_id,formula,chemical_family,num_sites,predicted_tc_kelvin\n"
                          "x,Nb3Sn,Other,8,15\n", encoding="utf-8")
    run(candidates, tmp_path / "before.csv", tmp_path / "before.json",
        model_path=None, table=table, hazel_records=tmp_path / "absent.json")
    s = Structure(Lattice.cubic(5.29), ["Nb"] * 6 + ["Sn"] * 2,
                  [[0, .25, .5], [0, .75, .5], [.5, 0, .25], [.5, 0, .75], [.25, .5, 0], [.75, .5, 0],
                   [0, 0, 0], [.5, .5, .5]])
    records = [{"jid": "hazel-nb3sn", "lamb": 3.0, "wlog": 400.0, "atoms": {
        "lattice_mat": s.lattice.matrix.tolist(), "coords": s.frac_coords.tolist(),
        "elements": [str(x.specie) for x in s], "cartesian": False}}]
    (tmp_path / "hazel.json").write_text(json.dumps(records), encoding="utf-8")
    report = run(candidates, tmp_path / "after.csv", tmp_path / "after.json",
                 model_path=None, table=table, hazel_records=tmp_path / "hazel.json")
    assert report["model"]["hazel_records_used"] == ["hazel-nb3sn"]
    before = next(csv.DictReader((tmp_path / "before.csv").open()))
    after = next(csv.DictReader((tmp_path / "after.csv").open()))
    assert float(after["physics_tc_k"]) != pytest.approx(float(before["physics_tc_k"]))
