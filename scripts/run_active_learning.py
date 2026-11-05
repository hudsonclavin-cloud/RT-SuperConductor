#!/usr/bin/env python3
"""Run one reproducible download -> train -> evaluate -> rank cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bridge_reviewed_records import bridge_reviewed_records
from download_training_data import download_training_data
from evaluate_structure_features import evaluate_structure_features
from evaluate_3dsc_structure_spin import run as evaluate_3dsc_structure_spin
from ingest_3dsc import ingest_3dsc
from rank_candidates import rank_candidates
from train_baseline import train_baseline
from train_pressure_aware import train_pressure_aware
from rt_superconductor.harmonized import load_harmonized


def _blocked(path: Path, step: str, exc: Exception) -> dict[str, Any]:
    report = {
        "status": "blocked",
        "step": step,
        "error": f"{type(exc).__name__}: {exc}",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _run_or_block(step: str, report_path: Path, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return function()
    except Exception as exc:
        return _blocked(report_path, step, exc)


def run_active_learning(
    *,
    data_root: Path = Path("data/raw"),
    harmonized_path: Path = Path("data/training/harmonized_supercon.csv"),
    output_dir: Path = Path("data/processed/active-learning"),
    cif_limit: int = 128,
    all_cifs: bool = False,
    batch_size: int = 25,
    max_candidates: int = 3000,
    require_hull: bool = False,
    api_key: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    download_report = _run_or_block(
        "download",
        output_dir / "download-blocked.json",
        lambda: download_training_data(
            data_root=data_root,
            cif_limit=cif_limit,
            all_cifs=all_cifs,
            report_path=output_dir / "download-report.json",
        ),
    )
    reviewed_report = _run_or_block(
        "reviewed_bridge",
        output_dir / "reviewed-bridge-blocked.json",
        lambda: bridge_reviewed_records(
            harmonized_path,
            ROOT / "data/training/reviewed/reviewed_records.csv",
        ),
    )
    ingestion_report = _run_or_block(
        "3dsc_ingestion",
        output_dir / "3dsc-ingestion-blocked.json",
        lambda: ingest_3dsc(
            input_path=data_root / "3dsc/MP/3DSC_MP.csv",
            output_path=harmonized_path,
            structure_root=data_root / "3dsc/MP/cifs",
            report_path=output_dir / "3dsc-ingestion-report.json",
            structure_index_path=output_dir / "3dsc-structure-index.csv",
        ),
    )
    validation_report: dict[str, Any]
    try:
        records = load_harmonized(harmonized_path)
        validation_report = {
            "status": "ready",
            "record_count": len(records),
            "dataset_sha256": hashlib.sha256(harmonized_path.read_bytes()).hexdigest(),
        }
        (output_dir / "harmonized-validation.json").write_text(
            json.dumps(validation_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception as exc:
        validation_report = _blocked(output_dir / "harmonized-validation-blocked.json", "harmonized_validation", exc)

    uci_path = data_root / "uci/extracted/unique_m.csv"
    baseline_dir = output_dir / "baseline"
    baseline_report = _run_or_block(
        "uci_baseline",
        baseline_dir / "baseline-metrics.json",
        lambda: train_baseline(input_path=uci_path, output_dir=baseline_dir),
    )
    pressure_dir = output_dir / "pressure-aware"
    pressure_report = _run_or_block(
        "pressure_aware_training",
        pressure_dir / "pressure-aware-report.json",
        lambda: train_pressure_aware(
            input_path=harmonized_path,
            output_dir=pressure_dir,
            allow_insufficient_data=True,
        ),
    )
    structure_report = _run_or_block(
        "structure_features",
        output_dir / "structure-features-report.json",
        lambda: evaluate_structure_features(
            input_path=harmonized_path,
            structure_index=output_dir / "3dsc-structure-index.csv",
            output_path=output_dir / "structure-features-report.json",
        ),
    )
    structure_spin_report = _run_or_block(
        "3dsc_structure_spin_ablation",
        output_dir / "3dsc-structure-spin-ablation-blocked.json",
        lambda: evaluate_3dsc_structure_spin(
            harmonized=harmonized_path,
            source_csv=data_root / "3dsc/MP/3DSC_MP.csv",
            structure_index=output_dir / "3dsc-structure-index.csv",
            output_dir=output_dir,
        ),
    )
    candidate_path = output_dir.parent / "candidates/top_k.csv"
    candidate_report = _run_or_block(
        "candidate_ranking",
        output_dir / "candidate-ranking-report.json",
        lambda: rank_candidates(
            input_path=harmonized_path,
            model_dir=pressure_dir,
            output_path=candidate_path,
            batch_size=batch_size,
            max_candidates=max_candidates,
            require_hull=require_hull,
            api_key=api_key,
        ),
    )
    status = "ready" if pressure_report.get("training_status") == "ready" and candidate_report.get("ranked_count", 0) else "blocked"
    report = {
        "status": status,
        "interpretation": "Candidate rows are reviewable model hypotheses, not claims or labels of superconductivity.",
        "download": download_report,
        "reviewed_bridge": reviewed_report,
        "3dsc_ingestion": ingestion_report,
        "validation": validation_report,
        "baseline": baseline_report,
        "pressure_aware": pressure_report,
        "structure_features": structure_report,
        "structure_spin_ablation": structure_spin_report,
        "candidate_ranking": candidate_report,
        "artifacts": {
            "baseline_metrics": str(baseline_dir / "baseline-metrics.json"),
            "pressure_metrics": str(pressure_dir / "pressure-aware-report.json"),
            "structure_metrics": str(output_dir / "structure-features-report.json"),
            "structure_spin_metrics": str(output_dir / "3dsc-structure-spin-ablation.json"),
            "top_k": str(candidate_path),
        },
    }
    (output_dir / "active-learning-run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--harmonized", type=Path, default=Path("data/training/harmonized_supercon.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/active-learning"))
    parser.add_argument("--cif-limit", type=int, default=128)
    parser.add_argument("--all-cifs", action="store_true")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-candidates", type=int, default=3000)
    parser.add_argument("--require-hull", action="store_true")
    parser.add_argument("--api-key")
    args = parser.parse_args()
    report = run_active_learning(
        data_root=args.data_root,
        harmonized_path=args.harmonized,
        output_dir=args.output_dir,
        cif_limit=args.cif_limit,
        all_cifs=args.all_cifs,
        batch_size=args.batch_size,
        max_candidates=args.max_candidates,
        require_hull=args.require_hull,
        api_key=args.api_key,
    )
    print(json.dumps({"status": report["status"], "top_k": report["artifacts"]["top_k"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
