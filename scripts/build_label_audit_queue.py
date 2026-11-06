#!/usr/bin/env python3
"""Refresh the label-audit queue without modifying source observations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rt_superconductor.harmonized import load_harmonized
from rt_superconductor.label_quarantine import write_label_audit_queue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/training/harmonized_supercon.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/training/label_audit_queue.csv"))
    args = parser.parse_args()
    counts = write_label_audit_queue(load_harmonized(args.input), args.output)
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
