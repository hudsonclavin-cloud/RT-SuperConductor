#!/usr/bin/env python3
"""Compute auditable formula, shell, orbital, and bonding-status descriptors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rt_superconductor.descriptors import descriptor_names, formula_descriptor_record
from rt_superconductor.harmonized import load_harmonized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/training/harmonized_supercon.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/formula-descriptors.csv"))
    args = parser.parse_args()
    records = load_harmonized(args.input)
    fields = ["record_id", "formula", "review_status", "provenance", "is_synthetic", *descriptor_names(),
              "descriptor_status", "elemental_electron_configurations", "lewis_dot_proxy_outer_electrons",
              "lewis_structure_status", "sigma_bond_count", "pi_bond_count", "bond_descriptor_status",
              "oxidation_state_status", "shell_percentage_definition"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {
                "record_id": record.record_id,
                "formula": record.formula,
                "review_status": record.review_status,
                "provenance": record.provenance,
                "is_synthetic": record.is_synthetic,
            }
            row.update(formula_descriptor_record(record.formula))
            writer.writerow(row)
    report = {
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output": str(args.output),
        "record_count": len(records),
        "descriptor_count": len(descriptor_names()),
        "bonding_fields": "status-only for formula-only inputs",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
