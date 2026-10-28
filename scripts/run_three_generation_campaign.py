#!/usr/bin/env python3
"""Run three transparent virtual-campaign generations over a review queue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from rt_superconductor.search import CandidateHypothesis, make_simulation_tasks, propose_batch
from rt_superconductor.targets import ENGINEERING_PRESSURE_CEILING_GPA


def _candidate(row: dict[str, str]) -> CandidateHypothesis:
    risks = [item for item in row["review_risk"].split(";") if item]
    uncertainty = 30.0 + 25.0 * len(risks)
    if "unusually_high_extracted_tc" in risks:
        uncertainty += 30.0
    return CandidateHypothesis(
        candidate_id=row["review_id"],
        composition=row["formula"],
        predicted_tc_k=float(row["extracted_tc_kelvin"]),
        tc_uncertainty_k=uncertainty,
        pressure_gpa=float(row["extracted_pressure_gpa"]),
        structural_plausibility=0.2 if row["material_class"] else 0.0,
        synthesis_feasibility=0.2 if row["sample_form"] else 0.0,
        novelty=0.1,
        evidence_record_ids=(row["source_record_id"],),
        provenance=("source:supercon2", "status:pending_paper_review"),
    )


def run_campaign(input_path: Path, output_path: Path, batch_size: int) -> dict[str, object]:
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    candidates = [_candidate(row) for row in rows]
    input_revision = hashlib.sha256(input_path.read_bytes()).hexdigest()

    generation_1 = {
        "name": "review_queue_triage",
        "input_count": len(rows),
        "pending_review_count": sum(row["review_status"] == "pending_paper_review" for row in rows),
        "high_risk_count": sum("unusually_high_extracted_tc" in row["review_risk"] for row in rows),
    }
    proposals = propose_batch(
        candidates,
        batch_size=batch_size,
        pressure_ceiling_gpa=ENGINEERING_PRESSURE_CEILING_GPA,
        allow_unknown_pressure=False,
    )
    generation_2 = {
        "name": "pressure_target_acquisition",
        "input_count": len(candidates),
        "proposal_count": len(proposals),
        "pressure_ceiling_gpa": ENGINEERING_PRESSURE_CEILING_GPA,
        "proposals": [
            {
                "rank": rank,
                "candidate_id": proposal.candidate.candidate_id,
                "composition": proposal.candidate.composition,
                "predicted_tc_k": proposal.candidate.predicted_tc_k,
                "uncertainty_k": proposal.candidate.tc_uncertainty_k,
                "pressure_gpa": proposal.candidate.pressure_gpa,
                "score": proposal.score,
                "reasons": proposal.reasons,
                "review_required": True,
            }
            for rank, proposal in enumerate(proposals, start=1)
        ],
    }
    tasks = make_simulation_tasks(proposals, input_revision=input_revision)
    generation_3 = {
        "name": "composition_screen_task_emission",
        "task_count": len(tasks),
        "tasks": [
            {
                "task_id": task.task_id,
                "candidate_id": task.candidate_id,
                "stage": task.stage.value,
                "input_revision": task.input_revision,
                "requested_outputs": task.requested_outputs,
                "status": task.status.value,
                "review_required": True,
            }
            for task in tasks
        ],
    }
    report = {
        "pipeline_version": 1,
        "input": str(input_path),
        "input_sha256": input_revision,
        "generations": [generation_1, generation_2, generation_3],
        "interpretation": (
            "Virtual ranking and task emission only. Pending literature records are not training labels "
            "and no experiment is authorized by this report."
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
    print(json.dumps({"generations": [item["name"] for item in report["generations"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
