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


def _reduced_formula(value: str) -> str | None:
    """Return the canonical key for a pymatgen-reduced formula."""

    try:
        from pymatgen.core import Composition

        reduced = Composition(value.strip()).reduced_formula
    except Exception:
        reduced = value
    return _canonical(reduced)


def _element_set(formula: str) -> frozenset[str]:
    from pymatgen.core import Composition

    return frozenset(str(element) for element in Composition(formula).elements)


def _evidence(
    *,
    formula: str,
    source: str,
    source_record_id: str,
    tc_kelvin: float,
    doi: str = "",
    notes: str = "",
) -> dict[str, object] | None:
    reduced = _reduced_formula(formula)
    if reduced is None or not math.isfinite(float(tc_kelvin)):
        return None
    return {
        "formula": reduced,
        "element_set": _element_set(reduced),
        "source": source,
        "source_record_id": source_record_id,
        "tc_kelvin": float(tc_kelvin),
        "doi": doi if doi.startswith("10.") else "",
        "notes": notes,
    }


def _read_uci(path: Path) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for source_row, row in enumerate(csv.DictReader(handle), start=2):
            formula = row.get("material", "")
            raw_tc = (row.get("critical_temp") or "").strip()
            try:
                tc = float(raw_tc)
            except ValueError:
                continue
            item = _evidence(
                formula=formula,
                source="uci",
                source_record_id=f"uci:unique_m:{source_row}",
                tc_kelvin=tc,
                notes="UCI unique_m.csv has no DOI or row-level method/pressure metadata.",
            )
            if item is not None:
                evidence.append(item)
    return evidence


def _read_stanev(path: Path) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for source_row, row in enumerate(csv.DictReader(handle), start=2):
            try:
                tc = float((row.get("Tc") or "").strip())
            except ValueError:
                continue
            item = _evidence(
                formula=row.get("name", ""),
                source="3dsc_stanev",
                source_record_id=f"3dsc_stanev:{source_row}",
                tc_kelvin=tc,
                notes="3DSC repository Supercon_data_by_2018_Stanev.csv; source row has no DOI field.",
            )
            if item is not None:
                evidence.append(item)
    return evidence


def _read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        rows = list(reader)
    required = {"rank", "candidate_id", "formula", "predicted_tc_kelvin", "known_in_database"}
    if not required.issubset(fields):
        raise ValueError(f"{path}: shortlist must include {sorted(required)}")
    return rows


def _training_formula_set(path: Path | None) -> tuple[set[str], str]:
    if path is None or not path.is_file():
        return set(), "missing"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        formulas = {
            reduced
            for row in rows
            if (reduced := _reduced_formula(row.get("formula", ""))) is not None
        }
    return formulas, "available"


def _read_supercon2(path: Path) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for record in iter_supercon2_records(path):
        formula = _reduced_formula(record.composition)
        if formula is None or record.reported_tc_k is None:
            continue
        doi = record.publication_reference or ""
        item = _evidence(
            formula=formula,
            source="supercon2",
            source_record_id=record.record_id,
            tc_kelvin=float(record.reported_tc_k),
            doi=doi,
            notes="SuperCon2 extraction requires paper verification; DOI copied only when source field is DOI-shaped.",
        )
        if item is not None:
            evidence.append(item)
    return evidence


def _read_supermat(path: Path) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for source_row, row in enumerate(csv.DictReader(handle), start=2):
            formula = _reduced_formula(row.get("material", "") or row.get("formula", ""))
            tc, tc_note = _parse_supermat_tc(row.get("tcValue", ""))
            if formula is None or tc is None:
                continue
            doi = (row.get("doi") or row.get("DOI") or "").strip()
            notes = "SuperMat annotation requires paper verification."
            if tc_note:
                notes += f" {tc_note}"
            if not doi:
                notes += " The supplied SuperMat row has no DOI field; no DOI was invented."
            item = _evidence(
                formula=formula,
                source="supermat",
                source_record_id=row.get("id", "").strip() or f"supermat:{source_row}",
                tc_kelvin=float(tc),
                doi=doi,
                notes=notes,
            )
            if item is not None:
                evidence.append(item)
    return evidence


def load_measurement_evidence(
    *,
    uci_path: Path | None = None,
    stanev_path: Path | None = None,
    supercon2_path: Path | None = None,
    supermat_path: Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Load available measured-Tc sources without treating missing files as empty data."""

    readers = {
        "uci": (uci_path, _read_uci),
        "3dsc_stanev": (stanev_path, _read_stanev),
        "supercon2": (supercon2_path, _read_supercon2),
        "supermat": (supermat_path, _read_supermat),
    }
    evidence: list[dict[str, object]] = []
    status: dict[str, str] = {}
    for source, (path, reader) in readers.items():
        if path is None or not path.is_file():
            status[source] = "missing"
            continue
        status[source] = "available"
        evidence.extend(reader(path))
    return evidence, status


def match_measurement_evidence(
    formula: str,
    evidence: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return exact reduced-formula matches and non-exact same-element-set matches."""

    reduced = _reduced_formula(formula)
    if reduced is None:
        return [], []
    elements = _element_set(reduced)
    exact = [item for item in evidence if item["formula"] == reduced]
    same_elements = [
        item for item in evidence
        if item["formula"] != reduced and item["element_set"] == elements
    ]
    return exact, same_elements


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
    training_path: Path | None = None,
    uci_path: Path | None = None,
    stanev_path: Path | None = None,
    supercon2_path: Path | None = None,
    supermat_path: Path | None = None,
) -> dict[str, object]:
    candidates = _read_candidates(candidates_path)
    known = [row for row in candidates if row.get("known_in_database") == "known"]
    top10 = sorted(known, key=lambda row: (int(row["rank"]), row["candidate_id"]))[:10]
    evidence, source_status = load_measurement_evidence(
        uci_path=uci_path,
        stanev_path=stanev_path,
        supercon2_path=supercon2_path,
        supermat_path=supermat_path,
    )
    training_formulas, training_status = _training_formula_set(training_path)

    queue: list[dict[str, object]] = []
    exact_measurements_not_in_training: list[tuple[float, float]] = []
    matched_candidates = 0
    known_superconductors = 0
    exact_match_candidates = 0
    same_element_set_candidates = 0
    for candidate in top10:
        exact_matches, same_element_matches = match_measurement_evidence(candidate["formula"], evidence)
        matches = exact_matches + same_element_matches
        if matches:
            matched_candidates += 1
            if any(float(item["tc_kelvin"]) > 0 for item in matches):
                known_superconductors += 1
        if exact_matches:
            exact_match_candidates += 1
            if same_element_matches:
                same_element_set_candidates += 1
            if _reduced_formula(candidate["formula"]) not in training_formulas:
                measured = sum(float(item["tc_kelvin"]) for item in exact_matches) / len(exact_matches)
                exact_measurements_not_in_training.append((float(candidate["predicted_tc_kelvin"]), measured))
        elif same_element_matches:
            same_element_set_candidates += 1
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
                "match_basis": "exact_reduced_formula" if item in exact_matches else "same_element_set",
                "notes": item["notes"],
            })

    exact_mae: float | None = None
    exact_rho: float | None = None
    if exact_measurements_not_in_training:
        exact_mae = sum(abs(predicted - measured) for predicted, measured in exact_measurements_not_in_training) / len(exact_measurements_not_in_training)
    if len(exact_measurements_not_in_training) >= 2:
        value = float(spearmanr(
            [item[0] for item in exact_measurements_not_in_training],
            [item[1] for item in exact_measurements_not_in_training],
        ).statistic)
        exact_rho = value if math.isfinite(value) else None
    if all(value == "missing" for value in source_status.values()):
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
        "exact_match_candidate_count": exact_match_candidates,
        "same_element_set_candidate_count": same_element_set_candidates,
        "exact_match_not_in_training_count": len(exact_measurements_not_in_training),
        "exact_match_not_in_training_mae_k": exact_mae,
        "exact_match_not_in_training_spearman_rho": exact_rho,
        "ordering_spearman_rho": exact_rho,
        "ordering_agreement": None if exact_rho is None else exact_rho > 0,
        "training_status": training_status,
        "training_path": str(training_path) if training_path else None,
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
    parser.add_argument("--training", type=Path, help="harmonized training table used to exclude leakage from metrics")
    parser.add_argument("--uci", type=Path, help="UCI unique_m.csv measured-Tc source")
    parser.add_argument("--stanev", type=Path, help="3DSC Supercon_data_by_2018_Stanev.csv measured-Tc source")
    parser.add_argument("--supercon2", type=Path)
    parser.add_argument("--supermat", type=Path)
    args = parser.parse_args()
    report = validate_shortlist(
        candidates_path=args.candidates,
        output_path=args.output,
        report_path=args.report,
        training_path=args.training,
        uci_path=args.uci,
        stanev_path=args.stanev,
        supercon2_path=args.supercon2,
        supermat_path=args.supermat,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
