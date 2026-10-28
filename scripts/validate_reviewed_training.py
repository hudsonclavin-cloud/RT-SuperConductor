#!/usr/bin/env python3
"""Validate human-reviewed records before they can enter training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rt_superconductor.reviewed import load_reviewed_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/training/reviewed"))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    records = load_reviewed_records(args.input_dir)
    report = {
        "input_dir": str(args.input_dir),
        "record_count": len(records),
        "confirmed_count": len(records),
        "known_pressure_count": sum(record.pressure_condition == "known" for record in records),
        "ambient_or_low_pressure_count": sum(
            record.pressure_condition == "ambient"
            or (record.pressure_gpa is not None and record.pressure_gpa <= 0.1)
            for record in records
        ),
        "record_ids": [record.record_id for record in records],
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
