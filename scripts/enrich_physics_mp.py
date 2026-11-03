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
        formulae = list(dict.fromkeys(row["formula"] for row in source_rows))
        docs_by_formula: dict[str, list[Any]] = {formula: [] for formula in formulae}
        try:
            for start in range(0, len(formulae), 30):
                batch = formulae[start:start + 30]
                docs = mpr.materials.summary.search(
                    formula=batch,
                    fields=["material_id", "formula_pretty", "density", "band_gap", "is_metal"],
                )
                for doc in docs:
                    pretty = str(doc.formula_pretty)
                    exact_formulas = [formula for formula in batch if signature(formula) == signature(pretty)]
                    for formula in exact_formulas:
                        docs_by_formula[formula].append(doc)
        except Exception as exc:
            api_unavailable = f"api_unavailable_{'403' if '403' in str(exc) else type(exc).__name__}"

        exact_doc_by_formula = {
            formula: docs[0] for formula, docs in docs_by_formula.items() if len(docs) == 1
        }
        dos_by_id: dict[str, tuple[float | str, str]] = {}
        if not api_unavailable and exact_doc_by_formula:
            material_ids = list(dict.fromkeys(str(doc.material_id) for doc in exact_doc_by_formula.values()))
            from enrich_deep_mp import dos_at_fermi,complete_dos_with_task
            import math
            for material_id in material_ids:
                dos_rester=getattr(mpr.materials,'electronic_structure',None)
                original_mode=getattr(dos_rester,'use_document_model',None)
                try:
                    if dos_rester is not None:dos_rester.use_document_model=False
                    try:dos,_task_id=complete_dos_with_task(mpr,material_id)
                    finally:
                        if dos_rester is not None:dos_rester.use_document_model=original_mode
                    value=dos_at_fermi(dos) if dos is not None else None
                    dos_by_id[material_id]=(value,'ready') if value is not None and math.isfinite(value) else ('','dos_unavailable')
                except Exception as exc:
                    failure=f"dos_query_unavailable_{'403' if '403' in str(exc) else type(exc).__name__}"
                    dos_by_id[material_id]=('',failure)

        for row in source_rows:
            formula = row["formula"]
            matches = docs_by_formula.get(formula, [])
            if api_unavailable:
                status, doc, dos_ef = api_unavailable, None, ""
            elif len(matches) == 0:
                status, doc, dos_ef = "no_exact_match", None, ""
            elif len(matches) > 1:
                status, doc, dos_ef = "ambiguous_multiple_mp_phases", None, ""
            else:
                doc = matches[0]
                material_id = str(doc.material_id)
                dos_ef, status = dos_by_id.get(material_id, ("", "dos_unavailable"))
            results.append({
                "jid": row["jid"], "formula": formula, "mp_match_status": status,
                "mp_material_id": "" if doc is None else str(doc.material_id),
                "mp_dos_ef_states_per_ev_per_cell": dos_ef,
                "mp_band_gap_ev": "" if doc is None else getattr(doc, "band_gap", ""),
                "mp_is_metal": "" if doc is None else getattr(doc, "is_metal", ""),
                "mp_density_g_cm3": "" if doc is None else getattr(doc, "density", ""),
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
