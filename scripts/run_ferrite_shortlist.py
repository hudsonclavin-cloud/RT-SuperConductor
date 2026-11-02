#!/usr/bin/env python3
"""Run one reproducible Ferrites within-family shortlist cycle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from download_training_data import download_training_data
from ingest_3dsc import ingest_3dsc
from prepare_ferrite_dft import prepare_dft
import promotion_gate
from rank_candidates import rank_candidates
from train_pressure_aware import train_pressure_aware
from validate_ferrite_shortlist import validate_shortlist


def _run_or_block(name: str, report_path: Path, action: Callable[[], dict[str, object]]) -> dict[str, object]:
    try:
        return action()
    except Exception as exc:
        report = {"status": "blocked", "blocked_reason": f"{type(exc).__name__}: {exc}"}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report


def _evaluate_gate(predictions_path: Path, model_path: Path, report_path: Path) -> dict[str, object]:
    report = promotion_gate.evaluate_gate(predictions_path=predictions_path, model_path=model_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_cycle(
    *,
    data_root: Path = Path("data/raw"),
    output_root: Path = Path("data/processed/ferrite-shortlist"),
    candidate_path: Path = Path("data/processed/candidates/ferrites_top25.csv"),
    supercon2_path: Path | None = None,
    supermat_path: Path | None = None,
    api_key: str | None = None,
    skip_download: bool = False,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    download = {"status": "skipped"}
    if not skip_download:
        download = _run_or_block(
            "download",
            output_root / "download-report.json",
            lambda: download_training_data(data_root=data_root, cif_limit=0, report_path=output_root / "download-report.json"),
        )
    input_table = data_root / "3dsc/MP/3DSC_MP.csv"
    harmonized = output_root / "harmonized.csv"
    ingestion = _run_or_block(
        "ingestion",
        output_root / "ingestion-report.json",
        lambda: ingest_3dsc(
            input_path=input_table,
            output_path=harmonized,
            structure_root=data_root / "3dsc/MP/cifs",
            report_path=output_root / "ingestion-report.json",
            structure_index_path=output_root / "structure-index.csv",
        ),
    )
    training = _run_or_block(
        "training",
        output_root / "model/pressure-aware-report.json",
        lambda: train_pressure_aware(input_path=harmonized, output_dir=output_root / "model"),
    )
    gate = _run_or_block(
        "promotion_gate",
        output_root / "model/promotion-gate.json",
        lambda: _evaluate_gate(
            predictions_path=output_root / "model/promotion_predictions.csv",
            model_path=output_root / "model/pressure-aware-model.joblib",
            report_path=output_root / "model/promotion-gate.json",
        ),
    )
    ranking = _run_or_block(
        "ranking",
        candidate_path.with_name("ferrite-ranking-report.json"),
        lambda: rank_candidates(
            input_path=harmonized,
            model_dir=output_root / "model",
            output_path=candidate_path,
            batch_size=25,
            max_candidates=3000,
            family="Ferrites",
            api_key=api_key or os.environ.get("MP_API_KEY"),
            uci_path=data_root / "uci/extracted/unique_m.csv",
            supercon2_path=supercon2_path,
            supermat_path=supermat_path,
        ),
    )
    validation = _run_or_block(
        "literature_validation",
        output_root / "ferrite-literature-validation.json",
        lambda: validate_shortlist(
            candidates_path=candidate_path,
            output_path=Path("data/training/review_queue_ferrite_shortlist.csv"),
            report_path=output_root / "ferrite-literature-validation.json",
            supercon2_path=supercon2_path,
            supermat_path=supermat_path,
        ),
    )
    dft = _run_or_block(
        "dft_preparation",
        output_root / "dft/ferrite-dft-report.json",
        lambda: prepare_dft(
            candidates_path=candidate_path,
            output_dir=output_root / "dft",
            api_key=api_key or os.environ.get("MP_API_KEY"),
            top_n=5,
        ),
    )
    ranking_count = int(ranking.get("ranked_count", 0) or 0)
    training_ready = training.get("training_status") == "ready"
    overall = "ready" if training_ready and ranking_count else "blocked"
    if ranking.get("status") == "degraded_no_mp_api_key":
        overall = "degraded_no_mp_api_key"
    report = {
        "status": overall,
        "family": "Ferrites",
        "download": download,
        "ingestion": ingestion,
        "training": training,
        "promotion_gate": gate,
        "ranking": ranking,
        "literature_validation": validation,
        "dft_preparation": dft,
        "candidate_path": str(candidate_path),
        "interpretation": "All shortlist rows are hypotheses. No candidate is a claim of superconductivity and no DFT was executed.",
    }
    (output_root / "ferrite-shortlist-run.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/ferrite-shortlist"))
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/candidates/ferrites_top25.csv"))
    parser.add_argument("--supercon2", type=Path)
    parser.add_argument("--supermat", type=Path)
    parser.add_argument("--api-key", default=os.environ.get("MP_API_KEY"))
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_cycle(
        data_root=args.data_root,
        output_root=args.output_root,
        candidate_path=args.candidates,
        supercon2_path=args.supercon2,
        supermat_path=args.supermat,
        api_key=args.api_key,
        skip_download=args.skip_download,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
