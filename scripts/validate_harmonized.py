#!/usr/bin/env python3
"""Validate the Phase 2 harmonized superconductivity CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rt_superconductor.harmonized import REQUIRED_COLUMNS, load_harmonized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/training/harmonized_supercon.csv"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    records = load_harmonized(args.input)
    report = {
        "path": str(args.input),
        "record_count": len(records),
        "required_columns": list(REQUIRED_COLUMNS),
        "source_counts": {source: sum(record.source_db == source for record in records) for source in sorted({record.source_db for record in records})},
        "review_status_counts": {status: sum(record.review_status == status for record in records) for status in sorted({record.review_status for record in records})},
        "provenance_counts": {provenance: sum(record.provenance == provenance for record in records) for provenance in sorted({record.provenance for record in records})},
        "synthetic_record_count": sum(record.is_synthetic for record in records),
        "empirical_confirmed_record_count": sum(record.review_status == "confirmed" and record.is_synthetic == 0 for record in records),
        "chemical_family_count": len({record.chemical_family for record in records}),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
