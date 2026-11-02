#!/usr/bin/env python3
"""Prepare, but never execute, blocked ferrite DFT input jobs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from fetch_materials_project_structures import fetch_structures
from generate_qe_inputs import generate_inputs


def _eligible(row: dict[str, str]) -> bool:
    try:
        hull = float(row.get("energy_above_hull", ""))
        prediction = float(row.get("predicted_tc_kelvin", ""))
    except ValueError:
        return False
    band_gap = row.get("band_gap_ev", "").strip()
    is_metal = row.get("is_metal", "").strip().lower()
    return (
        hull < 0.05
        and prediction >= 0
        and is_metal == "true"
        and (not band_gap or float(band_gap) <= 0.1)
        and bool(row.get("mp_id", "").strip())
    )


def prepare_dft(
    *,
    candidates_path: Path,
    output_dir: Path,
    api_key: str | None = None,
    top_n: int = 5,
) -> dict[str, object]:
    with candidates_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    ferrite_rows = [row for row in rows if row.get("chemical_family") == "Ferrites"]
    eligible = [row for row in ferrite_rows if _eligible(row)][:top_n]
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "dft-selected-candidates.csv"
    selected_fields = ("record_id", "formula", "predicted_tc_kelvin", "energy_above_hull", "mp_id", "is_metal", "band_gap_ev", "review_status")
    with selected_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_fields)
        writer.writeheader()
        for row in eligible:
            writer.writerow({
                "record_id": row.get("candidate_id", ""),
                "formula": row.get("formula", ""),
                "predicted_tc_kelvin": row.get("predicted_tc_kelvin", ""),
                "energy_above_hull": row.get("energy_above_hull", ""),
                "mp_id": row.get("mp_id", ""),
                "is_metal": row.get("is_metal", ""),
                "band_gap_ev": row.get("band_gap_ev", ""),
                "review_status": "pending",
            })

    if not api_key:
        generation = generate_inputs(
            selected_path,
            output_dir / "qe-inputs",
            pseudopotential_catalog=ROOT / "data/pseudopotentials/pslibrary-catalog.csv",
            allow_placeholder=False,
            min_predicted_tc=0,
        )
        report = {
            "status": "degraded_no_mp_api_key",
            "mp_status": "degraded_no_mp_api_key",
            "top_n": top_n,
            "ferrite_candidate_count": len(ferrite_rows),
            "metallic_hull_stable_mp_candidate_count": len(eligible),
            "dft_ready_count": 0,
            "blocked_job_count": len(eligible),
            "selected_candidates": [row.get("formula", "") for row in eligible],
            "blocked_reason": "MP_API_KEY is absent; MP structures cannot be retrieved. No QE inputs were generated and no DFT was run.",
            "generation": generation,
        }
    else:
        structure_dir = output_dir / "structures"
        structure_report = fetch_structures(selected_path, structure_dir, api_key=api_key)
        generation = generate_inputs(
            selected_path,
            output_dir / "qe-inputs",
            structure_dir=structure_dir,
            structure_manifest=structure_dir / "structure-manifest.json",
            pseudopotential_catalog=ROOT / "data/pseudopotentials/pslibrary-catalog.csv",
            allow_placeholder=False,
            min_predicted_tc=0,
        )
        report = {
            "status": "ready_with_blocked_jobs" if generation.get("generated_count", 0) else "blocked_no_retrieved_structures",
            "mp_status": "ready",
            "top_n": top_n,
            "ferrite_candidate_count": len(ferrite_rows),
            "metallic_hull_stable_mp_candidate_count": len(eligible),
            "dft_ready_count": generation.get("generated_count", 0),
            "blocked_job_count": generation.get("blocked_count", 0),
            "selected_candidates": [row.get("formula", "") for row in eligible],
            "structure_retrieval": structure_report,
            "generation": generation,
            "blocked_reason": "Pseudopotential UPF files and Hazel access remain prerequisites; no DFT was run.",
        }
    (output_dir / "ferrite-dft-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/candidates/ferrites_top25.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/candidates/ferrite-dft"))
    parser.add_argument("--api-key", default=os.environ.get("MP_API_KEY"))
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(prepare_dft(candidates_path=args.candidates, output_dir=args.output_dir, api_key=args.api_key, top_n=args.top_n), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
