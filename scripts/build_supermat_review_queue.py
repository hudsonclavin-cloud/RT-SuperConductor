#!/usr/bin/env python3
"""Make a paper-review queue from SuperMat annotations, never training labels.

SuperMat's CSV is an extraction/annotation source with free-text fields. This
script preserves source text and only converts unambiguous scalar Tc/pressure
values. Every row remains pending until a person checks the cited paper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rt_superconductor.datasets import _parse_supermat_pressure, _parse_supermat_tc
from rt_superconductor.targets import ENGINEERING_PRESSURE_CEILING_GPA


FIELDS = (
    "review_id", "source", "source_row", "material_text", "tc_raw",
    "tc_kelvin", "pressure_raw", "pressure_gpa", "pressure_bucket",
    "measurement_method", "paper_filename", "review_status", "reviewer_notes",
)


def build_queue(source: Path, output: Path, manifest: Path) -> dict[str, object]:
    rows_out: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"material", "tcValue", "pressure", "me_method", "filename"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"SuperMat input missing fields: {sorted(missing)}")
        for source_row, row in enumerate(reader, start=2):
            tc, tc_note = _parse_supermat_tc(row.get("tcValue", ""))
            if tc is None or not 0.0 <= tc <= 350.0:
                counts["skipped_missing_or_ambiguous_tc"] += 1
                continue
            pressure, pressure_note = _parse_supermat_pressure(row.get("pressure", ""))
            if pressure is None:
                bucket = "pressure_unknown_or_ambiguous"
            elif pressure <= ENGINEERING_PRESSURE_CEILING_GPA:
                bucket = "within_provisional_engineering_ceiling"
            else:
                bucket = "high_pressure_control"
            material = (row.get("material") or "").strip()
            filename = (row.get("filename") or "").strip()
            source_key = f"{source_row}|{material}|{row.get('tcValue','')}|{row.get('pressure','')}|{filename}"
            review_id = "supermat-review-" + hashlib.sha256(source_key.encode()).hexdigest()[:12]
            notes = ["Verify transition assignment, units, pressure, and sample context in source paper."]
            if tc_note:
                notes.append(tc_note)
            if pressure_note:
                notes.append(pressure_note)
            rows_out.append({
                "review_id": review_id,
                "source": "SuperMat 2.0",
                "source_row": str(source_row),
                "material_text": material,
                "tc_raw": (row.get("tcValue") or "").strip(),
                "tc_kelvin": f"{tc:.8g}",
                "pressure_raw": (row.get("pressure") or "").strip(),
                "pressure_gpa": "" if pressure is None else f"{pressure:.8g}",
                "pressure_bucket": bucket,
                "measurement_method": (row.get("me_method") or "").strip(),
                "paper_filename": filename,
                "review_status": "pending_paper_review",
                "reviewer_notes": " ".join(notes),
            })
            counts[bucket] += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)
    report = {
        "source": "SuperMat 2.0",
        "source_file": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "candidate_rows": len(rows_out),
        "counts": dict(sorted(counts.items())),
        "output": str(output),
        "training_eligible": False,
        "reason": "automated annotations are review leads; none enter empirical training without paper-level confirmation",
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_queue(args.input, args.output, args.manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
