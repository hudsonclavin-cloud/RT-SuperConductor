#!/usr/bin/env python3
"""Run the validated harmonize -> enrich -> train -> gate -> QE workflow."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from enrich_materials_project import enrich_csv
from generate_qe_inputs import generate_inputs
from promotion_gate import evaluate_gate
from rt_superconductor.harmonized import load_harmonized
from train_pressure_aware import train_pressure_aware


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/training/harmonized_supercon.csv"))
    parser.add_argument("--enriched-output", type=Path, default=Path("data/training/harmonized_supercon_enriched.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/pressure-aware"))
    parser.add_argument("--qe-output-dir", type=Path, default=Path("data/processed/qe-inputs"))
    parser.add_argument("--skip-enrichment", action="store_true", help="do not call the Materials Project API")
    args = parser.parse_args()

    print("--- Step 1: Validating harmonized multi-source data ---")
    records = load_harmonized(args.input)
    print(f"Validated {len(records)} rows from {len({record.source_db for record in records})} source databases.")

    training_input = args.input
    if not args.skip_enrichment and os.environ.get("MP_API_KEY"):
        print("\n--- Step 2: Querying Materials Project API ---")
        report = enrich_csv(args.input, args.enriched_output, api_key=os.environ["MP_API_KEY"])
        print(f"Enriched {report['matched_formula_count']} of {report['unique_formula_count']} unique formulas.")
        if report["errors"]:
            print(f"Materials Project errors retained in report: {report['errors']}")
        else:
            training_input = args.enriched_output
    else:
        print("\n--- Step 2: Materials Project enrichment skipped (set MP_API_KEY to enable) ---")

    print("\n--- Step 3: Training family-grouped pressure-aware surrogate ---")
    train_pressure_aware(input_path=training_input, output_dir=args.output_dir)
    predictions = args.output_dir / "promotion_predictions.csv"
    model = args.output_dir / "pressure-aware-model.joblib"

    print("\n--- Step 4: Applying strict promotion gate ---")
    gate = evaluate_gate(predictions_path=predictions, model_path=model)
    print(f"Promotion gate: {gate['status']}")

    print("\n--- Step 5: Generating Quantum ESPRESSO and EPW inputs ---")
    summary = generate_inputs(predictions, args.qe_output_dir)
    print(f"Generated starter inputs for {summary['selected_count']} screened candidates.")
    return 0 if gate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
