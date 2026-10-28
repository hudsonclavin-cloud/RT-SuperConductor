#!/usr/bin/env python3
"""Screen SuperCon2 and optionally enrich a small batch through Materials Project."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from rt_superconductor.datasets import _parse_supercon2_pressure, _parse_supercon2_tc
from rt_superconductor.targets import ENGINEERING_PRESSURE_CEILING_GPA


OUTPUT_FIELDS = (
    "record_id",
    "formula",
    "rawMaterial",
    "tc_kelvin",
    "criticalTemperature",
    "appliedPressure",
    "pressure_gpa",
    "pressure_status",
    "criticalTemperatureMeasurementMethod",
    "doi",
    "title",
    "year",
    "materialClass",
    "section",
    "subsection",
    "parse_notes",
)


def _read_candidates(source: Path, *, minimum_tc_k: float) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    review_rows: list[dict[str, str]] = []
    known_rows: list[dict[str, str]] = []
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            tc, tc_note = _parse_supercon2_tc(row["criticalTemperature"])
            pressure, pressure_note = _parse_supercon2_pressure(row["appliedPressure"])
            if tc is None or tc < minimum_tc_k:
                continue
            if pressure is not None and pressure > ENGINEERING_PRESSURE_CEILING_GPA:
                continue
            notes = [note for note in (tc_note, pressure_note) if note]
            pressure_status = "unknown" if pressure is None else "known_within_engineering_limit"
            candidate = {
                "record_id": f"supercon2:{row['id'].strip()}",
                "formula": row["formula"].strip(),
                "rawMaterial": row["rawMaterial"].strip(),
                "tc_kelvin": str(tc),
                "criticalTemperature": row["criticalTemperature"].strip(),
                "appliedPressure": row["appliedPressure"].strip(),
                "pressure_gpa": "" if pressure is None else str(pressure),
                "pressure_status": pressure_status,
                "criticalTemperatureMeasurementMethod": row[
                    "criticalTemperatureMeasurementMethod"
                ].strip(),
                "doi": row["doi"].strip(),
                "title": row["title"].strip(),
                "year": row["year"].strip(),
                "materialClass": row["materialClass"].strip(),
                "section": row["section"].strip(),
                "subsection": row["subsection"].strip(),
                "parse_notes": "; ".join(notes),
            }
            review_rows.append(candidate)
            if pressure is not None:
                known_rows.append(candidate)
    return review_rows, known_rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _query_materials_project(rows: list[dict[str, str]], output: Path, limit: int) -> str:
    api_key = os.environ.get("MP_API_KEY") or os.environ.get("Materials_Project_API")
    if not api_key:
        return "Materials Project skipped: no MP_API_KEY or Materials_Project_API runtime secret."
    try:
        from mp_api.client import MPRester
    except ImportError:
        return "Materials Project skipped: install mp-api in the runtime first."

    formulas = []
    for row in rows:
        formula = row["formula"]
        if formula and formula not in formulas:
            formulas.append(formula)
    matches: list[dict[str, object]] = []
    with MPRester(api_key) as mpr:
        for formula in formulas[:limit]:
            try:
                docs = mpr.materials.summary.search(formula=formula)
                for doc in docs:
                    matches.append(
                        {
                            "query_formula": formula,
                            "material_id": str(doc.material_id),
                            "formula_pretty": doc.formula_pretty,
                            "energy_above_hull": doc.energy_above_hull,
                        }
                    )
            except Exception as error:  # keep one malformed formula from aborting the batch
                matches.append({"query_formula": formula, "error": str(error)})
    output.write_text(json.dumps(matches, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return f"Materials Project wrote {len(matches)} matches to {output}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/supercon2/supercon2_v22.12.03.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/low-budget-screen"))
    parser.add_argument("--minimum-tc-k", type=float, default=70.0)
    parser.add_argument("--mp-limit", type=int, default=5)
    args = parser.parse_args()

    review_rows, known_rows = _read_candidates(args.source, minimum_tc_k=args.minimum_tc_k)
    review_path = args.output_dir / "candidates_to_verify.csv"
    known_path = args.output_dir / "candidates_known_low_pressure.csv"
    _write_csv(review_path, review_rows)
    _write_csv(known_path, known_rows)
    print(f"Found {len(review_rows)} high-Tc records with unknown or <=0.1 GPa pressure.")
    print(f"Found {len(known_rows)} high-Tc records with known <=0.1 GPa pressure.")
    print(f"Wrote {review_path}")
    print(f"Wrote {known_path}")
    print(_query_materials_project(known_rows, args.output_dir / "materials_project_matches.json", args.mp_limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
