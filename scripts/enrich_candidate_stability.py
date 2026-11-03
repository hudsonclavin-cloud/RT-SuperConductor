#!/usr/bin/env python3
"""Query Materials Project stability for exact-formula search candidates.

Multiple phases remain ambiguous; a parent/prototype hull value is never copied
to a substituted candidate.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

from pymatgen.core import Composition

from train_physics_targets import _write_csv


def _signature(formula: str) -> tuple[tuple[str, float], ...]:
    comp = Composition(formula).fractional_composition
    return tuple(sorted((str(el), round(float(amount), 8)) for el, amount in comp.items()))


def enrich(input_csv: Path, output_csv: Path, api_key: str | None = None, rester_factory: Any = None) -> dict[str, Any]:
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        candidates = list(csv.DictReader(handle))
    api_key = api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        rows = [{"formula": row["formula"], "status": "blocked_missing_mp_api_key",
                 "materials_project_id": "", "energy_above_hull_ev_atom": ""} for row in candidates]
        _write_csv(output_csv, rows)
        return {"rows": len(rows), "status": "blocked_missing_mp_api_key", "output": str(output_csv)}
    if rester_factory is None:
        from mp_api.client import MPRester
        rester_factory = MPRester
    rows = []
    failure: str | None = None
    try:
        with rester_factory(api_key) as mpr:
            for candidate in candidates:
                formula = candidate["formula"]
                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=["material_id", "formula_pretty", "energy_above_hull", "density", "band_gap", "is_metal"],
                )
                exact = [doc for doc in docs if _signature(str(doc.formula_pretty)) == _signature(formula)]
                if len(exact) == 1:
                    doc = exact[0]
                    rows.append({
                        "formula": formula, "status": "exact_composition_single_mp_phase_structure_review_required",
                        "materials_project_id": str(doc.material_id),
                        "energy_above_hull_ev_atom": getattr(doc, "energy_above_hull", ""),
                        "density_g_cm3": getattr(doc, "density", ""),
                        "band_gap_ev": getattr(doc, "band_gap", ""),
                        "is_metal": getattr(doc, "is_metal", ""),
                    })
                else:
                    rows.append({
                        "formula": formula,
                        "status": "no_exact_composition_match" if not exact else "ambiguous_multiple_mp_phases",
                        "materials_project_id": "", "energy_above_hull_ev_atom": "",
                        "density_g_cm3": "", "band_gap_ev": "", "is_metal": "",
                    })
    except Exception as exc:
        failure = f"api_unavailable_{'403' if '403' in str(exc) else type(exc).__name__}"
    if failure:
        rows = [{"formula": row["formula"], "status": failure,
                 "materials_project_id": "", "energy_above_hull_ev_atom": "",
                 "density_g_cm3": "", "band_gap_ev": "", "is_metal": ""} for row in candidates]
    _write_csv(output_csv, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {"rows": len(rows), "status_counts": counts, "output": str(output_csv)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs/physics-target-top20.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/jarvis-supercond/mp-candidate-stability.csv"))
    args = parser.parse_args()
    import json
    print(json.dumps(enrich(args.input, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
