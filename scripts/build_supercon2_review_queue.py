#!/usr/bin/env python3
"""Build a conservative paper-review queue from a SuperCon2 CSV export.

The output is a review queue, never training data.  Rows with missing formulas,
DOIs, or titles are omitted because they cannot be responsibly verified from a
paper reference alone.  A reviewer must still inspect the cited source before
copying any row into ``data/training/reviewed/``.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rt_superconductor.datasets import _parse_supercon2_pressure, _parse_supercon2_tc


OUTPUT_COLUMNS = (
    "review_id",
    "source_record_id",
    "formula",
    "extracted_tc_kelvin",
    "extracted_pressure_gpa",
    "raw_pressure",
    "source_doi",
    "source_title",
    "source_year",
    "measurement_method",
    "sample_form",
    "material_class",
    "review_status",
    "review_risk",
    "reviewer_notes",
)


def _parse_scalar(parser, value: str) -> float | None:
    if not value or not value.strip():
        return None
    return parser(value)[0]


def _risk(row: dict[str, str], tc_kelvin: float) -> str:
    risks: list[str] = []
    title = row.get("title", "").casefold()
    if tc_kelvin >= 200:
        risks.append("unusually_high_extracted_tc")
    if any(word in title for word in ("review", "theory", "magnetic", "model", "optical")):
        risks.append("title_may_be_indirect_evidence")
    if not row.get("criticalTemperatureMeasurementMethod", "").strip():
        risks.append("missing_measurement_method")
    if not row.get("shape", "").strip():
        risks.append("missing_sample_form")
    return ";".join(risks) or "context_requires_verification"


def build_queue(input_path: Path, output_path: Path, limit: int) -> int:
    candidates: list[dict[str, str | float]] = []
    seen_formulas: set[str] = set()
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            tc_kelvin = _parse_scalar(_parse_supercon2_tc, row.get("criticalTemperature", ""))
            pressure_gpa = _parse_scalar(_parse_supercon2_pressure, row.get("appliedPressure", ""))
            formula = row.get("formula", "").strip()
            doi = row.get("doi", "").strip()
            title = row.get("title", "").strip()
            if tc_kelvin is None or pressure_gpa is None or tc_kelvin < 70 or tc_kelvin > 350:
                continue
            if pressure_gpa > 0.1 or not formula or not doi or not title:
                continue
            formula_key = re.sub(r"\s+", "", formula).casefold()
            if formula_key in seen_formulas:
                continue
            seen_formulas.add(formula_key)
            candidates.append(
                {
                    "source_record_id": row.get("id", "").strip(),
                    "formula": formula,
                    "extracted_tc_kelvin": tc_kelvin,
                    "extracted_pressure_gpa": pressure_gpa,
                    "raw_pressure": row.get("appliedPressure", "").strip(),
                    "source_doi": doi,
                    "source_title": title,
                    "source_year": row.get("year", "").strip(),
                    "measurement_method": row.get("criticalTemperatureMeasurementMethod", "").strip(),
                    "sample_form": row.get("shape", "").strip(),
                    "material_class": row.get("materialClass", "").strip(),
                    "review_status": "pending_paper_review",
                    "review_risk": _risk(row, tc_kelvin),
                    "reviewer_notes": "Verify the cited paper, transition evidence, pressure, and sample context.",
                }
            )
    candidates.sort(key=lambda item: (-float(item["extracted_tc_kelvin"]), str(item["formula"])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for index, candidate in enumerate(candidates[:limit], start=1):
            writer.writerow({"review_id": f"supercon2-review-{index:03d}", **candidate})
    return min(len(candidates), limit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    print(f"Wrote {build_queue(args.input, args.output, args.limit)} pending review records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
