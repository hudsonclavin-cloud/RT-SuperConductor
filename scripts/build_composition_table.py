#!/usr/bin/env python3
"""Build a UCI-style composition table from a two-column ``formula,Tc`` export.

The UCI Superconductivity ``unique_m.csv`` file is a table of elemental amounts
per formula unit plus ``critical_temp`` and ``material``. This script produces
the same layout from any curated ``name,Tc`` CSV (for example the SuperCon
export distributed with the 3DSC repository) so ``train_baseline.py`` can run
when the UCI archive itself is unavailable. Formulas that the conservative
parser rejects are skipped and counted; nothing is guessed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rt_superconductor.descriptors import ATOMIC_NUMBERS
from rt_superconductor.formula import ELEMENTS, TOKEN, canonicalize_formula


def parse_amounts(formula: str) -> dict[str, float] | None:
    """Return element amounts for a formula the conservative parser accepts."""

    if canonicalize_formula(formula).status != "parsed":
        return None
    amounts: dict[str, float] = {}
    for match in TOKEN.finditer("".join(formula.split())):
        element = match.group("element")
        if element not in ELEMENTS:
            return None
        count_text = match.group("count")
        try:
            count = float(Decimal(count_text)) if count_text else 1.0
        except InvalidOperation:
            return None
        if count > 0:
            amounts[element] = amounts.get(element, 0.0) + count
    return amounts or None


def build_table(*, input_path: Path, output_path: Path, name_column: str, tc_column: str) -> dict[str, object]:
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {name_column, tc_column}.issubset(reader.fieldnames):
            raise ValueError(f"{input_path}: expected columns {name_column!r} and {tc_column!r}")
        rows = list(reader)

    parsed: list[tuple[str, float, dict[str, float]]] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        formula = (row[name_column] or "").strip()
        try:
            tc = float(row[tc_column])
        except (TypeError, ValueError):
            skipped["non_numeric_tc"] += 1
            continue
        if tc < 0:
            skipped["negative_tc"] += 1
            continue
        amounts = parse_amounts(formula)
        if amounts is None:
            skipped["unparseable_formula"] += 1
            continue
        parsed.append((formula, tc, amounts))

    elements = sorted({element for _, _, amounts in parsed for element in amounts}, key=ATOMIC_NUMBERS.__getitem__)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*elements, "critical_temp", "material"])
        for formula, tc, amounts in parsed:
            writer.writerow([f"{amounts.get(element, 0.0):g}" for element in elements] + [f"{tc:g}", formula])

    return {
        "input": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output": str(output_path),
        "input_rows": len(rows),
        "output_rows": len(parsed),
        "skipped": dict(skipped),
        "element_columns": len(elements),
        "rows_with_zero_tc": sum(tc == 0 for _, tc, _ in parsed),
        "layout": "UCI unique_m.csv style: elemental amounts, critical_temp, material",
        "interpretation": (
            "Composition table derived from a curated database export. Pressure and "
            "measurement method are not recorded by the source; rows are not paper-verified."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name-column", default="name")
    parser.add_argument("--tc-column", default="Tc")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = build_table(
        input_path=args.input, output_path=args.output, name_column=args.name_column, tc_column=args.tc_column
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
