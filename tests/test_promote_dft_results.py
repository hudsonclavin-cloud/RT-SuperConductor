import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from promote_dft_results import allen_dynes_tc_k, evaluate_directory, parse_targets, promote  # noqa: E402

pymatgen = pytest.importorskip("pymatgen.core")


def _write_run(directory: Path, *, role="candidate", formula="Ta", epw_text=None, review_extra=None,
               with_review=True, element="Ta"):
    from pymatgen.core import Lattice, Structure

    directory.mkdir(parents=True)
    Structure(Lattice.cubic(3.30), [element, element], [[0, 0, 0], [0.5, 0.5, 0.5]]).to(
        filename=str(directory / "structure.cif"))
    (directory / "epw.out").write_text(epw_text or "lambda = 0.75\nomega_log = 160.0 K\n", encoding="utf-8")
    if with_review:
        review = {"role": role, "formula": formula, "converged": True, "accept_for_training": True,
                  "reviewer": "test", "mu_star": 0.1}
        review.update(review_extra or {})
        (directory / "review.json").write_text(json.dumps(review), encoding="utf-8")


def test_allen_dynes_matches_tantalum_worked_example():
    assert allen_dynes_tc_k(0.7572, 164.99, 0.1) == pytest.approx(6.88, abs=0.02)
    assert allen_dynes_tc_k(0.05, 100.0, 0.1) is None


def test_fallback_parser_reads_epw_summary_wording():
    text = ("Electron-phonon coupling strength =   0.7621\n"
            "Estimated w_log in McMillan Tc =   14.20 meV\n")
    lam, wlog, parser = parse_targets(text)
    assert lam == pytest.approx(0.7621)
    assert wlog == pytest.approx(14.20 * 11.60451812155)
    assert parser == "epw_summary_fallback_unconfirmed"


def test_accepted_candidate_becomes_jarvis_format_training_record(tmp_path):
    _write_run(tmp_path / "results" / "ta-candidate")
    report = promote(tmp_path / "results", tmp_path / "records.json", tmp_path / "report.json")
    records = json.loads((tmp_path / "records.json").read_text())
    assert report["training_records_written"] == 1
    record = records[0]
    assert record["jid"] == "hazel-ta-candidate"
    assert record["lamb"] == pytest.approx(0.75) and record["wlog"] == pytest.approx(160.0)
    assert set(record["atoms"]) == {"lattice_mat", "coords", "elements", "cartesian"}
    assert record["atoms"]["elements"] == ["Ta", "Ta"]


def test_benchmark_is_compared_not_trained(tmp_path):
    _write_run(tmp_path / "results" / "ta-benchmark", role="benchmark",
               review_extra={"reference": {"lambda": 0.7572, "omega_log_k": 164.99, "source": "JVASP-1014"}})
    report = promote(tmp_path / "results", tmp_path / "records.json", tmp_path / "report.json")
    assert report["training_records_written"] == 0
    assert report["benchmarks"] == 1
    outcome = report["outcomes"][0]
    assert outcome["passes_suggested_tolerance"] is True
    assert outcome["lambda_relative_error"] == pytest.approx(abs(0.75 - 0.7572) / 0.7572)


@pytest.mark.parametrize("kwargs,reason", [
    ({"with_review": False}, "missing_review_json"),
    ({"formula": "Nb"}, "review_formula_does_not_match_cif"),
    ({"epw_text": "nothing useful here\n"}, "lambda_or_omega_log_not_parsed"),
    ({"review_extra": {"converged": False}}, "review_does_not_confirm_convergence"),
    ({"review_extra": {"accept_for_training": False}}, "candidate_not_accepted_for_training"),
])
def test_unreviewed_or_inconsistent_runs_are_rejected(tmp_path, kwargs, reason):
    _write_run(tmp_path / "run", **kwargs)
    outcome = evaluate_directory(tmp_path / "run")
    assert outcome["outcome"] == "rejected"
    assert outcome["reason"] == reason


def test_trainer_loads_and_validates_extra_records(tmp_path):
    pytest.importorskip("sklearn")
    from train_physics_targets import load_extra_records

    assert load_extra_records(tmp_path / "absent.json") == []
    good = [{"jid": "hazel-x", "atoms": {}, "lamb": 0.8, "wlog": 200.0}]
    (tmp_path / "good.json").write_text(json.dumps(good))
    assert load_extra_records(tmp_path / "good.json") == good
    bad = [{"jid": "JVASP-1", "atoms": {}, "lamb": 0.8, "wlog": 200.0}]
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_extra_records(tmp_path / "bad.json")
