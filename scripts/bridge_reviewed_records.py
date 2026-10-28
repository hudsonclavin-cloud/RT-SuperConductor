#!/usr/bin/env python3
"""Bridge confirmed reviewed-record evidence into the harmonized CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rt_superconductor.harmonized import bridge_reviewed_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmonized", type=Path, default=Path("data/training/harmonized_supercon.csv"))
    parser.add_argument("--reviewed", type=Path, default=Path("data/training/reviewed/reviewed_records.csv"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = bridge_reviewed_records(args.harmonized, args.reviewed)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
