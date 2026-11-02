#!/usr/bin/env python3
"""Build a three-stage paper-review campaign from extracted literature leads.

This is not model training, candidate inference, or a closed learning cycle.
The input Tc values are copied only into fields explicitly named "extracted".
No prediction or uncertainty is emitted. A prediction campaign is blocked until
a suitable validated model and candidate feature matrix exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _priority(row: dict[str, str]) -> tuple[int, str, str]:
    """Prioritize verification risk, not predicted Tc or model uncertainty."""

    risks = {item.strip() for item in row.get("review_risk", "").split(";") if item.strip()}
    if "unusually_high_extracted_tc" in risks:
        label = "critical_extracted_outlier"
        rank = 0
    elif "title_may_be_indirect_evidence" in risks:
        label = "source_context_risk"
        rank = 1
    elif "missing_measurement_method" in risks or "missing_sample_form" in risks:
        label = "incomplete_evidence_context"
        rank = 2
    else:
        label = "standard_paper_review"
        rank = 3
    return rank, label, row.get("review_id", "")


def run_campaign(input_path: Path, output_path: Path, batch_size: int) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "review_id", "source_record_id", "formula", "extracted_tc_kelvin",
        "extracted_pressure_gpa", "source_doi", "source_title", "review_status",
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"review queue missing required fields: {sorted(missing)}")
    revision = hashlib.sha256(input_path.read_bytes()).hexdigest()
    pending = [row for row in rows if row.get("review_status") == "pending_paper_review"]
    prioritized = sorted(pending, key=_priority)
    selected = prioritized[:batch_size]

    stage_1 = {
        "name": "review_queue_inventory",
        "input_count": len(rows),
        "pending_paper_review_count": len(pending),
        "extracted_outlier_count": sum(
            "unusually_high_extracted_tc" in row.get("review_risk", "") for row in pending
        ),
        "learning_cycles_completed": 0,
    }
    stage_2 = {
        "name": "evidence_risk_prioritization",
        "prioritization_basis": "explicit review-risk flags only; not Tc prediction or calibrated uncertainty",
        "review_batch_size": len(selected),
        "records": [
            {
                "review_id": row["review_id"],
                "review_priority": _priority(row)[1],
                "formula_as_extracted": row["formula"],
                "extracted_tc_kelvin": row["extracted_tc_kelvin"],
                "extracted_pressure_gpa": row["extracted_pressure_gpa"],
                "source_doi": row["source_doi"],
                "source_title": row["source_title"],
                "review_risk": row.get("review_risk", ""),
            }
            for row in selected
        ],
    }
    stage_3 = {
        "name": "paper_review_task_emission",
        "task_count": len(selected),
        "tasks": [
            {
                "task_id": hashlib.sha256(
                    f"paper-review|{row['review_id']}|{revision}".encode()
                ).hexdigest()[:24],
                "task_type": "paper_review",
                "review_id": row["review_id"],
                "source_record_id": row["source_record_id"],
                "source_doi": row["source_doi"],
                "source_title": row["source_title"],
                "extracted_tc_kelvin": row["extracted_tc_kelvin"],
                "extracted_pressure_gpa": row["extracted_pressure_gpa"],
                "requested_review_fields": [
                    "transition_type_and_criterion",
                    "formula_phase_and_sample_identity",
                    "measurement_method_and_conditions",
                    "measurement_pressure_and_calibration",
                    "page_table_figure_or_quoted_source_location",
                    "review_decision_and_reviewer",
                ],
                "status": "pending_manual_review",
            }
            for row in selected
        ],
    }
    report = {
        "pipeline_version": 2,
        "campaign_type": "literature_review_triage",
        "input": str(input_path),
        "input_sha256": revision,
        "model_inference_status": "blocked_no_validated_candidate_model",
        "model_inference_blocker": (
            "No target-appropriate validated model and candidate feature matrix are available. "
            "Extracted literature Tc is not a prediction."
        ),
        "learning_cycles_completed": 0,
        "stages": [stage_1, stage_2, stage_3],
        "interpretation": (
            "This report prioritizes source verification only. It emits no Tc predictions, "
            "no prediction uncertainty, and no DFT or experimental tasks. Confirmed review "
            "decisions must be added to the reviewed evidence store before retraining."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    report = run_campaign(args.input, args.output, args.batch_size)
    print(json.dumps({
        "campaign_type": report["campaign_type"],
        "learning_cycles_completed": report["learning_cycles_completed"],
        "model_inference_status": report["model_inference_status"],
        "stages": [item["name"] for item in report["stages"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
