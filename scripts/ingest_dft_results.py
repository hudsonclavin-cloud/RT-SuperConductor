#!/usr/bin/env python3
"""Validate external QE/EPW result files and extract explicit parsed values.

File presence is not scientific validation: output rows retain convergence or
completion evidence and hashes. No result is inferred from a missing file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

EXPECTED = {
    "qe_relax": "relax.out", "qe_vc_relax": "vc-relax.out", "qe_scf": "scf.out",
    "qe_nscf": "nscf.out", "qe_ph": "ph.out", "qe_q2r": "q2r.out",
    "qe_matdyn": "matdyn.out", "bader": "ACF.dat", "elf": "elf.cube",
    "cohp": "ICOHP.lobster", "wannier": "wannier90.wout", "epw": "epw.out",
}
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?"
_FREQUENCY = re.compile(
    rf"\b(?:freq|omega)\s*\(\s*\d+\s*\)\s*=\s*({_FLOAT})\s*\[\s*(THz|cm\s*(?:\^\s*-?1|-1))\s*\]",
    re.I,
)
_LAMBDA = re.compile(rf"\blambda\s*(?:=|:)\s*({_FLOAT})", re.I)
_OMEGA_LOG = re.compile(
    rf"\b(?:omega[_ ]?log|wlog)\s*(?:=|:)\s*({_FLOAT})\s*(?:\[\s*)?(K|kelvin|meV|eV|THz|cm\s*(?:\^\s*-?1|-1))\b\s*\]?",
    re.I,
)


def _float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def parse_matdyn_frequencies(text: str) -> list[float]:
    """Return all explicit matdyn mode frequencies tagged in THz."""
    values: list[float] = []
    for match in _FREQUENCY.finditer(text):
        value = _float(match.group(1))
        unit = re.sub(r"\s|\^", "", match.group(2)).casefold()
        values.append(value if unit == "thz" else value / 33.35640951981521)
    return values


def parse_epw_targets(text: str) -> tuple[float | None, float | None]:
    """Extract the first explicit EPW lambda and omega_log (in K)."""
    lam_match = _LAMBDA.search(text)
    omega_match = _OMEGA_LOG.search(text)
    lam = _float(lam_match.group(1)) if lam_match else None
    omega = None
    if omega_match:
        omega = _float(omega_match.group(1))
        unit = re.sub(r"\s|\^", "", omega_match.group(2)).casefold()
        to_kelvin = {"k": 1.0, "kelvin": 1.0, "mev": 11.60451812155,
                     "ev": 11604.51812155, "thz": 47.9924307337,
                     "cm-1": 1.43877687750393, "cm-^1": 1.43877687750393}
        omega *= to_kelvin[unit]
    return lam, omega


def completion_status(stage: str, text: str) -> tuple[str, str]:
    """Stage-aware completion check; NSCF does not require an SCF iteration marker."""
    normalized = text.casefold()
    job_done = "job done." in normalized
    if stage in {"qe_relax", "qe_vc_relax"}:
        relax_end = any(marker in normalized for marker in (
            "end of bfgs geometry optimization", "bfgs converged", "end final coordinates",
        ))
        return ("completed", "job_done_and_relaxation_end") if job_done and relax_end else ("not_converged", "missing_job_done_or_relaxation_end")
    if stage == "qe_scf":
        scf_converged = "convergence has been achieved" in normalized
        return ("completed", "scf_convergence_and_job_done") if scf_converged and job_done else ("not_converged", "missing_scf_convergence_or_job_done")
    if stage == "qe_nscf":
        return ("completed", "job_done") if job_done else ("not_converged", "missing_job_done")
    if stage in {"qe_ph", "qe_q2r", "qe_matdyn"}:
        return ("completed", "job_done") if job_done else ("not_converged", "missing_job_done")
    return ("unverified", "no_stage_parser")


def ingest_results(results_root: Path, output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for directory in sorted(path for path in results_root.iterdir() if path.is_dir()):
        for stage, filename in EXPECTED.items():
            result_file = directory / filename
            exists = result_file.is_file()
            text = result_file.read_text(errors="ignore") if exists else ""
            if not exists:
                status, completion = "blocked", "file_missing"
            elif stage.startswith("qe_"):
                status, completion = completion_status(stage, text)
            else:
                status, completion = "present_unverified", "no_stage_parser"
            frequencies = parse_matdyn_frequencies(text) if stage == "qe_matdyn" and exists else []
            lam, omega_log = parse_epw_targets(text) if stage == "epw" and exists else (None, None)
            rows.append({
                "structure_id": directory.name, "stage": stage, "status": status,
                "completion_evidence": completion,
                "file": str(result_file) if exists else "",
                "sha256": hashlib.sha256(result_file.read_bytes()).hexdigest() if exists else "",
                "phonon_frequency_count": len(frequencies),
                "phonon_frequencies_thz_json": json.dumps(frequencies, separators=(",", ":")),
                "minimum_phonon_frequency_thz": min(frequencies) if frequencies else "",
                "epw_lambda": lam if lam is not None else "",
                "epw_omega_log_k": omega_log if omega_log is not None else "",
                "software_version": "must be supplied in job manifest",
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["structure_id", "stage", "status", "completion_evidence", "file", "sha256",
              "phonon_frequency_count", "phonon_frequencies_thz_json", "minimum_phonon_frequency_thz",
              "epw_lambda", "epw_omega_log_k", "software_version"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {"result_files_checked": len(rows), "completed": sum(row["status"] == "completed" for row in rows),
              "phonon_frequency_records": sum(int(row["phonon_frequency_count"]) for row in rows),
              "epw_rows_with_lambda": sum(row["epw_lambda"] != "" for row in rows),
              "note": "Completion markers and parsed values are evidence only; convergence and scientific review remain required."}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(ingest_results(args.results_root, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
