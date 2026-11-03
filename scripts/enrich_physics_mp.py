#!/usr/bin/env python3
"""Add unambiguous Materials Project summary/electronic-structure data.

Exact-formula matches with multiple MP phases are deliberately retained as
ambiguous instead of assigning one polymorph's properties to another.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from train_physics_targets import _write_csv


def enrich(input_csv: Path, output_csv: Path, api_key: str | None = None, rester_factory: Any = None) -> dict[str, Any]:
    api_key = api_key or os.environ.get("MP_API_KEY")
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
    if not api_key:
        raise RuntimeError("MP_API_KEY is required to enrich Materials Project electronic fields")
    if rester_factory is None:
        from mp_api.client import MPRester
        rester_factory = MPRester
    from pymatgen.core import Composition

    def signature(formula: str) -> tuple[tuple[str, float], ...]:
        composition = Composition(formula).fractional_composition
        return tuple(sorted((str(el), round(float(amount), 8)) for el, amount in composition.items()))

    results = []
    api_unavailable: str | None = None
    with rester_factory(api_key) as mpr:
        for row in source_rows:
            formula = row["formula"]
            try:
                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=["material_id", "formula_pretty", "density", "band_gap", "is_metal"],
                )
            except Exception as exc:
                api_unavailable = f"api_unavailable_{'403' if '403' in str(exc) else type(exc).__name__}"
                break
            exact = [doc for doc in docs if signature(str(doc.formula_pretty)) == signature(formula)]
            if len(exact) != 1:
                results.append({
                    "jid": row["jid"], "formula": formula, "mp_match_status":
                    "no_exact_match" if not exact else "ambiguous_multiple_mp_phases",
                    "mp_material_id": "", "mp_dos_ef_states_per_ev_per_cell": "",
                    "mp_band_gap_ev": "", "mp_is_metal": "", "mp_density_g_cm3": "",
                })
                continue
            doc = exact[0]
            material_id = str(doc.material_id)
            dos_ef = ""
            dos_status = "dos_unavailable"
            try:
                e_docs = mpr.materials.electronic_structure.search(
                    material_ids=[material_id], fields=["material_id", "dos", "efermi"]
                )
                if e_docs:
                    electronic = e_docs[0]
                    dos = electronic.dos
                    if dos is not None:
                        energies = dos.energies
                        densities = dos.get_densities()
                        idx = min(range(len(energies)), key=lambda i: abs(float(energies[i]) - float(electronic.efermi)))
                        dos_ef = float(densities[idx])
                        dos_status = "ready"
            except Exception as exc:  # records an honest failure per material
                dos_status = f"query_failed_{type(exc).__name__}"
            results.append({
                "jid": row["jid"], "formula": formula, "mp_match_status": dos_status,
                "mp_material_id": material_id, "mp_dos_ef_states_per_ev_per_cell": dos_ef,
                "mp_band_gap_ev": getattr(doc, "band_gap", ""),
                "mp_is_metal": getattr(doc, "is_metal", ""),
                "mp_density_g_cm3": getattr(doc, "density", ""),
            })
    if api_unavailable:
        results = [{
            "jid": row["jid"], "formula": row["formula"], "mp_match_status": api_unavailable,
            "mp_material_id": "", "mp_dos_ef_states_per_ev_per_cell": "",
            "mp_band_gap_ev": "", "mp_is_metal": "", "mp_density_g_cm3": "",
        } for row in source_rows]
    _write_csv(output_csv, results)
    counts: dict[str, int] = {}
    for row in results:
        counts[row["mp_match_status"]] = counts.get(row["mp_match_status"], 0) + 1
    return {"rows": len(results), "status_counts": counts, "output": str(output_csv)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/physics-target/jarvis-supercond-characteristics.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/jarvis-supercond/mp-characteristics.csv"))
    args = parser.parse_args()
    import json
    print(json.dumps(enrich(args.input, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
