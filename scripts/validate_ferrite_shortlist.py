#!/usr/bin/env python3
"""Cross-check known ferrite shortlist rows against paper-oriented exports.

This command is deliberately a review-queue builder, not a label importer.  It
copies measured values and DOI strings from the supplied SuperCon2 and
SuperMat exports, but every output row remains ``pending_paper_review``.
Missing exports are reported as a blocked source-access condition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scipy.stats import spearmanr

from rt_superconductor.datasets import _parse_supermat_tc, iter_supercon2_records
from rt_superconductor.formula import canonicalize_formula


QUEUE_FIELDS = (
    "candidate_rank", "candidate_id", "formula", "predicted_tc_kelvin", "source",
    "source_record_id", "measured_tc_kelvin", "doi", "review_status", "match_basis", "notes",
)


def _canonical(value: str) -> str | None:
    result = canonicalize_formula(value.strip())
    return result.canonical if result.status == "parsed" else None


def _read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"rank", "candidate_id", "formula", "predicted_tc_kelvin", "known_in_database"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path}: shortlist must include {sorted(required)}")
    return rows


def _read_supercon2(path: Path) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for record in iter_supercon2_records(path):
        formula = _canonical(record.composition)
        if formula is None or record.reported_tc_k is None:
            continue
        doi = record.publication_reference or ""
        evidence.append({
            "formula": formula,
            "source": "supercon2",
            "source_record_id": record.record_id,
            "tc_kelvin": float(record.reported_tc_k),
            "doi": doi if doi.startswith("10.") else "",
            "notes": "SuperCon2 extraction requires paper verification; DOI copied only when source field is DOI-shaped.",
        })
    return evidence


def _read_supermat(path: Path) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for source_row, row in enumerate(csv.DictReader(handle), start=2):
            formula = _canonical(row.get("material", "") or row.get("formula", ""))
            tc, tc_note = _parse_supermat_tc(row.get("tcValue", ""))
            if formula is None or tc is None:
                continue
            doi = (row.get("doi") or row.get("DOI") or "").strip()
            notes = "SuperMat annotation requires paper verification."
            if tc_note:
                notes += f" {tc_note}"
            if not doi:
                notes += " The supplied SuperMat row has no DOI field; no DOI was invented."
            evidence.append({
                "formula": formula,
                "source": "supermat",
                "source_record_id": row.get("id", "").strip() or f"supermat:{source_row}",
                "tc_kelvin": float(tc),
                "doi": doi if doi.startswith("10.") else "",
                "notes": notes,
            })
    return evidence


def _write_queue(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def validate_shortlist(
    *,
    candidates_path: Path,
    output_path: Path,
    report_path: Path,
    supercon2_path: Path | None = None,
    supermat_path: Path | None = None,
) -> dict[str, object]:
    candidates = _read_candidates(candidates_path)
    known = [row for row in candidates if row.get("known_in_database") == "known"]
    top10 = sorted(known, key=lambda row: (int(row["rank"]), row["candidate_id"]))[:10]
    source_status = {
        "supercon2": "available" if supercon2_path and supercon2_path.is_file() else "missing",
        "supermat": "available" if supermat_path and supermat_path.is_file() else "missing",
    }
    evidence: list[dict[str, object]] = []
    if source_status["supercon2"] == "available":
        evidence.extend(_read_supercon2(supercon2_path))
    if source_status["supermat"] == "available":
        evidence.extend(_read_supermat(supermat_path))
    evidence_by_formula: dict[str, list[dict[str, object]]] = {}
    for item in evidence:
        evidence_by_formula.setdefault(str(item["formula"]), []).append(item)

    queue: list[dict[str, object]] = []
    candidate_measurements: list[tuple[float, float]] = []
    matched_candidates = 0
    known_superconductors = 0
    for candidate in top10:
        formula = _canonical(candidate["formula"]) or candidate["formula"]
        matches = evidence_by_formula.get(formula, [])
        if matches:
            matched_candidates += 1
            if any(float(item["tc_kelvin"]) > 0 for item in matches):
                known_superconductors += 1
            measured = sum(float(item["tc_kelvin"]) for item in matches) / len(matches)
            candidate_measurements.append((float(candidate["predicted_tc_kelvin"]), measured))
        for item in matches:
            queue.append({
                "candidate_rank": candidate["rank"],
                "candidate_id": candidate["candidate_id"],
                "formula": candidate["formula"],
                "predicted_tc_kelvin": candidate["predicted_tc_kelvin"],
                "source": item["source"],
                "source_record_id": item["source_record_id"],
                "measured_tc_kelvin": item["tc_kelvin"],
                "doi": item["doi"],
                "review_status": "pending_paper_review",
                "match_basis": "canonical_formula",
                "notes": item["notes"],
            })

    rho: float | None = None
    if len(candidate_measurements) >= 2:
        value = float(spearmanr(
            [item[0] for item in candidate_measurements],
            [item[1] for item in candidate_measurements],
        ).statistic)
        rho = value if math.isfinite(value) else None
    if source_status["supercon2"] == "missing" and source_status["supermat"] == "missing":
        status = "blocked_missing_supercon2_and_supermat_exports"
    elif not top10:
        status = "blocked_no_top10_known_compounds"
    elif not queue:
        status = "ready_no_literature_matches"
    else:
        status = "ready"
    report: dict[str, object] = {
        "status": status,
        "candidates": str(candidates_path),
        "candidate_dataset_sha256": hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
        "top10_known_compound_count": len(top10),
        "matched_candidate_count": matched_candidates,
        "known_superconductor_count": known_superconductors,
        "literature_match_row_count": len(queue),
        "ordering_spearman_rho": rho,
        "ordering_agreement": None if rho is None else rho > 0,
        "source_status": source_status,
        "source_paths": {
            "supercon2": str(supercon2_path) if supercon2_path else None,
            "supermat": str(supermat_path) if supermat_path else None,
        },
        "interpretation": "Rows are paper-review leads only. No match is a confirmed training label.",
    }
    _write_queue(output_path, queue)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/candidates/ferrites_top25.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/training/review_queue_ferrite_shortlist.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/processed/candidates/ferrite-literature-validation.json"))
    parser.add_argument("--supercon2", type=Path)
    parser.add_argument("--supermat", type=Path)
    args = parser.parse_args()
    report = validate_shortlist(
        candidates_path=args.candidates,
        output_path=args.output,
        report_path=args.report,
        supercon2_path=args.supercon2,
        supermat_path=args.supermat,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
