#!/usr/bin/env python3
"""Close the loop: turn reviewed Hazel QE/EPW results into training records.

This is the step that sends physics results back into the model. It reads one
directory per calculation under ``--results-root`` and sorts each into one of
three outcomes:

* ``tools_check`` -> Stage 0 only: lambda and omega_log must be read and the
  recomputed Tc must match the Tc EPW printed. Never compared, never trained.
* ``benchmark``  -> compared with its published reference value and written to
  the benchmark report. Benchmarks validate the method; they never become
  training rows, because the reference values are already in JARVIS.
* ``candidate`` -> if a person has reviewed it and marked it converged and
  accepted, written as a JARVIS-format record (``jid``, ``atoms``, ``lamb``,
  ``wlog``) that ``train_physics_targets.py --extra-records`` appends to the
  training data.
* ``rejected``  -> anything missing a review, a structure, a parsed lambda or
  omega_log, or a matching formula. The reason is recorded; nothing is guessed.

Each results directory must contain ``epw.out``, exactly one ``*.cif`` and a
``review.json`` written by a person after checking the calculation::

    {
      "role": "candidate",              # or "benchmark" or "tools_check"
      "formula": "Nb3Sn",
      "converged": true,
      "accept_for_training": true,      # candidates only
      "reviewer": "Hudson Clavin",
      "mu_star": 0.1,
      "reference": {"lambda": 0.7572, "omega_log_k": 164.99,
                    "source": "JARVIS-SuperconDB JVASP-1014"},  # benchmarks only
      "notes": "k/q meshes and cutoffs recorded in run-manifest.json"
    }

No DFT is run here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_dft_results import parse_epw_targets  # noqa: E402

JID_PREFIX = "hazel-"
SOURCE_LABEL = "hazel-epw (reviewed local calculation)"
DEFAULT_TOLERANCE = 0.25  # suggested benchmark tolerance: 25% relative error

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?"
# EPW prints its own McMillan-form Tc; the formula below reproduces it exactly
# (checked against the QE test-suite epw_super benchmark: 26.4082 K at
# muc=0.16 and 28.4619 K at muc=0.15). EPW's separately printed
# "Allen-Dynes modified" value adds f1*f2 strong-coupling factors we omit.
_EPW_MCMILLAN_TC = re.compile(
    rf"Estimated Tc using McMillan expression\s*=\s*({_FLOAT})\s*K(?:\s*for muc\s*=\s*({_FLOAT}))?", re.I)
_EPW_MUC_USED = re.compile(rf"muc\s*=\s*({_FLOAT})\s*is used", re.I)
_EPW_SUMMARY = re.compile(r"Electron-phonon coupling strength\s*=|Eliashberg Spectral Function", re.I)
# a2f-only runs print one line per mu after "Estimated Allen-Dynes Tc":
# "mu =   0.10 Tc =         0.806700595152 K"
_EPW_A2F_TC = re.compile(rf"\bmu\s*=\s*({_FLOAT})\s+Tc\s*=\s*({_FLOAT})\s*K", re.I)


def allen_dynes_tc_k(lam: float, omega_log_k: float, mu_star: float = 0.1) -> float | None:
    """McMillan/Allen-Dynes Tc without strong-coupling corrections, in K."""
    denominator = lam - mu_star * (1.0 + 0.62 * lam)
    if lam <= 0 or omega_log_k <= 0 or denominator <= 0:
        return None
    return omega_log_k / 1.2 * math.exp(-1.04 * (1.0 + lam) / denominator)


def parse_targets(text: str) -> tuple[float | None, float | None, str]:
    """Return (lambda, omega_log in K, which kind of line they came from)."""
    lam, wlog = parse_epw_targets(text)
    return lam, wlog, "epw_summary" if _EPW_SUMMARY.search(text) else "generic_lambda_line"


_EPW_DOS_AT_EF = re.compile(r"DOS\s*=\s*([-+0-9.Ee]+)\s*states/spin/eV/Unit Cell at Ef")
_EPW_LAMBDA_OVERFLOW = re.compile(r"^\s*lambda(?:___\(\s*tot\s*\))?\s*[:=]\s*\*+", re.MULTILINE)


def epw_output_problems(text: str) -> list[str]:
    """Known signs that an EPW run finished but its numbers are meaningless.

    Both were seen on a real run (QE 7.6, Pb, fine k-mesh 8x8x8 with 0.05 eV
    smearing): EPW found no states at the Fermi level, every lambda divided by
    zero, and the job still ended normally with a Tc printed.
    """
    problems = []
    dos = [float(value) for value in _EPW_DOS_AT_EF.findall(text)]
    if dos and min(dos) <= 0.0:
        problems.append("zero_dos_at_fermi_level: EPW found no electronic states at the Fermi level on its "
                        "fine k-mesh, so lambda is a division by zero; use a finer nkf mesh or a larger degaussw")
    if _EPW_LAMBDA_OVERFLOW.search(text):
        problems.append("lambda_overflow: EPW printed asterisks for lambda, a number too large to print")
    return problems


def epw_printed_mcmillan(text: str) -> tuple[float | None, float | None]:
    """EPW's own printed McMillan Tc (K) and the muc it says it used, if present."""
    a2f = _EPW_A2F_TC.findall(text)
    if a2f and not _EPW_MCMILLAN_TC.search(text):
        # Prefer the mu = 0.10 line, the project's default mu*.
        chosen = next((pair for pair in a2f if abs(float(pair[0]) - 0.10) < 1e-9), a2f[0])
        return float(chosen[1]), float(chosen[0])
    tc = _EPW_MCMILLAN_TC.search(text)
    # Prefer "muc = X is used": in the epw_super benchmark the McMillan line's
    # own "for muc" label can disagree with it, and the printed Tc follows the
    # "is used" value.
    used = _EPW_MUC_USED.search(text)
    muc = used.group(1) if used else (tc.group(2) if tc and tc.group(2) else None)
    return (float(tc.group(1)) if tc else None, float(muc) if muc else None)


def _atoms_from_cif(path: Path) -> tuple[dict[str, Any], str]:
    from pymatgen.core import Structure

    structure = Structure.from_file(str(path))
    atoms = {
        "lattice_mat": [list(map(float, row)) for row in structure.lattice.matrix],
        "coords": [list(map(float, site.frac_coords)) for site in structure],
        "elements": [str(site.specie) for site in structure],
        "cartesian": False,
    }
    return atoms, structure.composition.reduced_formula


def _same_formula(a: str, b: str) -> bool:
    from pymatgen.core import Composition

    return Composition(a).reduced_formula == Composition(b).reduced_formula


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_directory(directory: Path, tolerance: float = DEFAULT_TOLERANCE) -> dict[str, Any]:
    """Sort one results directory into benchmark, candidate or rejected."""
    outcome: dict[str, Any] = {"structure_id": directory.name, "directory": str(directory)}
    review_path = directory / "review.json"
    epw_path = directory / "epw.out"
    cifs = sorted(directory.glob("*.cif"))
    if not review_path.is_file():
        return {**outcome, "outcome": "rejected", "reason": "missing_review_json"}
    review = json.loads(review_path.read_text(encoding="utf-8"))
    role = review.get("role")
    if role not in {"benchmark", "candidate", "tools_check"}:
        return {**outcome, "outcome": "rejected", "reason": "review_role_must_be_benchmark_candidate_or_tools_check"}
    if not epw_path.is_file():
        return {**outcome, "outcome": "rejected", "reason": "missing_epw_out"}
    if len(cifs) != 1:
        return {**outcome, "outcome": "rejected", "reason": f"expected_one_cif_found_{len(cifs)}"}
    if review.get("converged") is not True:
        return {**outcome, "outcome": "rejected", "reason": "review_does_not_confirm_convergence"}
    epw_text = epw_path.read_text(encoding="utf-8", errors="replace")
    problems = epw_output_problems(epw_text)
    if problems:
        return {**outcome, "outcome": "rejected", "reason": "epw_output_has_a_known_failure",
                "epw_output_problems": problems}
    lam, wlog, parser = parse_targets(epw_text)
    if lam is None or wlog is None:
        return {**outcome, "outcome": "rejected", "reason": "lambda_or_omega_log_not_parsed",
                "parsed_lambda": lam, "parsed_omega_log_k": wlog}
    atoms, structure_formula = _atoms_from_cif(cifs[0])
    claimed = str(review.get("formula", ""))
    if not claimed or not _same_formula(claimed, structure_formula):
        return {**outcome, "outcome": "rejected", "reason": "review_formula_does_not_match_cif",
                "review_formula": claimed, "cif_formula": structure_formula}
    mu_star = float(review.get("mu_star", 0.1))
    epw_tc, epw_muc = epw_printed_mcmillan(epw_text)
    check: dict[str, Any] = {"epw_printed_mcmillan_tc_k": epw_tc, "epw_reported_muc": epw_muc}
    if epw_tc is not None and epw_muc is not None:
        recomputed = allen_dynes_tc_k(lam, wlog, epw_muc)
        check["recomputed_tc_at_epw_muc_k"] = recomputed
        check["matches_epw_printed_tc"] = recomputed is not None and abs(recomputed - epw_tc) <= 0.01 * epw_tc
    evidence = {
        "formula": structure_formula, "lambda": lam, "omega_log_k": wlog, "mu_star": mu_star,
        "allen_dynes_tc_k": allen_dynes_tc_k(lam, wlog, mu_star), "parser": parser,
        "epw_out_sha256": _sha256(epw_path), "cif_sha256": _sha256(cifs[0]),
        "reviewer": review.get("reviewer", ""), "notes": review.get("notes", ""),
        "epw_consistency": check,
    }
    if check.get("matches_epw_printed_tc") is False:
        return {**outcome, **evidence, "outcome": "rejected", "reason": "recomputed_tc_disagrees_with_epw_output"}
    if role == "tools_check":
        # Stage 0: proves QE/EPW run end to end on Hazel and that we read their
        # output correctly. Never compared with a reference, never trained on.
        return {**outcome, **evidence, "outcome": "tools_check",
                "passes_tools_check": check.get("matches_epw_printed_tc") is True}
    if role == "benchmark":
        reference = review.get("reference") or {}
        ref_lam, ref_wlog = reference.get("lambda"), reference.get("omega_log_k")
        if ref_lam is None or ref_wlog is None:
            return {**outcome, **evidence, "outcome": "rejected", "reason": "benchmark_missing_reference_values"}
        lam_error = abs(lam - float(ref_lam)) / float(ref_lam)
        wlog_error = abs(wlog - float(ref_wlog)) / float(ref_wlog)
        return {**outcome, **evidence, "outcome": "benchmark", "reference": reference,
                "lambda_relative_error": lam_error, "omega_log_relative_error": wlog_error,
                "tolerance": tolerance,
                "passes_suggested_tolerance": lam_error <= tolerance and wlog_error <= tolerance}
    if review.get("accept_for_training") is not True:
        return {**outcome, **evidence, "outcome": "rejected", "reason": "candidate_not_accepted_for_training"}
    record = {
        "jid": JID_PREFIX + directory.name, "atoms": atoms, "lamb": lam, "wlog": wlog,
        "Tc": evidence["allen_dynes_tc_k"], "press": 0, "source_dataset_doi": SOURCE_LABEL,
        "provenance": {k: evidence[k] for k in ("parser", "epw_out_sha256", "cif_sha256", "reviewer", "mu_star")},
    }
    return {**outcome, **evidence, "outcome": "candidate", "training_record": record}


def promote(results_root: Path, records_out: Path, report_out: Path,
            tolerance: float = DEFAULT_TOLERANCE) -> dict[str, Any]:
    directories = sorted(p for p in results_root.iterdir() if p.is_dir()) if results_root.is_dir() else []
    outcomes = [evaluate_directory(d, tolerance) for d in directories]
    records = [o.pop("training_record") for o in outcomes if o["outcome"] == "candidate"]
    benchmarks = [o for o in outcomes if o["outcome"] == "benchmark"]
    report = {
        "results_root": str(results_root),
        "directories_checked": len(outcomes),
        "benchmarks": len(benchmarks),
        "benchmarks_passing_suggested_tolerance": sum(o["passes_suggested_tolerance"] for o in benchmarks),
        "tools_checks": sum(o["outcome"] == "tools_check" for o in outcomes),
        "tools_checks_passing": sum(bool(o.get("passes_tools_check")) for o in outcomes),
        "training_records_written": len(records),
        "rejected": sum(o["outcome"] == "rejected" for o in outcomes),
        "suggested_tolerance": tolerance,
        "outcomes": outcomes,
        "note": ("Benchmarks validate the method and never enter training. Candidate records enter "
                 "training only through train_physics_targets.py --extra-records."),
    }
    records_out.parent.mkdir(parents=True, exist_ok=True)
    records_out.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--records-out", type=Path, default=Path("data/training/hazel-epw-records.json"))
    parser.add_argument("--report-out", type=Path, default=Path("data/processed/dft-results/promotion-report.json"))
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()
    report = promote(args.results_root, args.records_out, args.report_out, args.tolerance)
    print(json.dumps({k: v for k, v in report.items() if k != "outcomes"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
