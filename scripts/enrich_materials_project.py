#!/usr/bin/env python3
"""Attach the lowest-energy Materials Project polymorph to each formula.

The script reads the validated harmonized table and writes a new CSV with
``mp_id``, ``energy_above_hull`` (eV/atom), ``spacegroup_number``, and optional
formation energy, band gap, metallicity, magnetization, density, and volume.
It is safe to rerun: API misses preserve existing structure links and other
known values, while provenance and review fields are retained.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rt_superconductor.harmonized import load_harmonized
from rt_superconductor.materials_project import audit_materials_project_queries, normalize_fixed_formula, validated_candidates


@dataclass(frozen=True, slots=True)
class MaterialsProjectMatch:
    mp_id: str = ""
    energy_above_hull: float | None = None
    spacegroup_number: int | None = None
    formation_energy_per_atom: float | None = None
    band_gap_ev: float | None = None
    is_metal: bool | None = None
    total_magnetization_muB: float | None = None
    density_g_cm3: float | None = None
    volume_angstrom3: float | None = None


def _value(document: Any, name: str, default: Any = None) -> Any:
    if isinstance(document, dict):
        return document.get(name, default)
    return getattr(document, name, default)


def _spacegroup_number(document: Any) -> int | None:
    direct = _value(document, "spacegroup_number")
    symmetry = _value(document, "symmetry")
    nested = _value(symmetry, "number") if symmetry is not None else None
    value = direct if direct is not None else nested
    return int(value) if value is not None else None


def select_lowest_energy_polymorph(documents: Iterable[Any]) -> MaterialsProjectMatch:
    """Select the polymorph with minimum finite energy above hull."""

    docs = list(documents)
    usable = [document for document in docs if _value(document, "energy_above_hull") is not None]
    if not usable:
        return MaterialsProjectMatch()
    # The explicit min is important: a formula may have many polymorphs.
    selected = min(usable, key=lambda document: float(_value(document, "energy_above_hull")))
    mp_id = _value(selected, "material_id", _value(selected, "mp_id", ""))
    return MaterialsProjectMatch(
        mp_id=str(mp_id) if mp_id is not None else "",
        energy_above_hull=float(_value(selected, "energy_above_hull")),
        spacegroup_number=_spacegroup_number(selected),
        formation_energy_per_atom=_optional_float(_value(selected, "formation_energy_per_atom")),
        band_gap_ev=_optional_float(_value(selected, "band_gap")),
        is_metal=_optional_bool(_value(selected, "is_metal")),
        total_magnetization_muB=_optional_float(_value(selected, "total_magnetization")),
        density_g_cm3=_optional_float(_value(selected, "density")),
        volume_angstrom3=_optional_float(_value(selected, "volume")),
    )


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _optional_bool(value: Any) -> bool | None:
    return bool(value) if value is not None else None


def _make_client(api_key: str | None):
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise RuntimeError("mp-api is required; install the project extra with `pip install -e '.[mp]'`") from exc
    if not api_key:
        raise RuntimeError("Materials Project API key is required via --api-key or MP_API_KEY")
    return MPRester(api_key)


def enrich_csv(
    input_path: Path,
    output_path: Path,
    *,
    api_key: str | None = None,
    client: Any | None = None,
    request_delay_seconds: float = 0.0,
) -> dict[str, object]:
    """Enrich every unique formula; ``client`` is injectable for tests."""

    # Validate before making remote calls so malformed rows cannot be emitted.
    records = load_harmonized(input_path)
    formulas = sorted({record.formula for record in records})
    # Preserve links already in the source on API misses or when no usable
    # polymorph is returned, so transient lookup failures do not erase lineage.
    existing_matches: dict[str, MaterialsProjectMatch] = {}
    for record in records:
        if record.mp_id and record.formula not in existing_matches:
            existing_matches[record.formula] = MaterialsProjectMatch(
                mp_id=record.mp_id,
                energy_above_hull=record.energy_above_hull,
                spacegroup_number=record.spacegroup_number,
            )
    own_client = client is None
    context_manager = None
    if client is None:
        client = _make_client(api_key or os.environ.get("MP_API_KEY"))
        context_manager = client
        client = client.__enter__()
    matches: dict[str, MaterialsProjectMatch] = {}
    errors: dict[str, str] = {}
    audit: dict[str, object]
    try:
        audit = audit_materials_project_queries(
            [{"record_id": record.record_id, "formula": record.formula} for record in records],
            summary_client=client.materials.summary,
            error_secrets=(api_key or os.environ.get("MP_API_KEY") or "",),
            request_delay_seconds=request_delay_seconds,
        )
    finally:
        if context_manager is not None:
            context_manager.__exit__(None, None, None)

    safe_by_formula = validated_candidates(audit)
    query_by_formula = {
        query.get("normalized_formula"): query
        for query in audit.get("query_records", [])
        if query.get("normalized_formula")
    }
    for formula in formulas:
        parsed = normalize_fixed_formula(formula)
        alternatives = safe_by_formula.get(parsed.normalized or "", [])
        if alternatives:
            selected = min(
                alternatives,
                key=lambda item: float(item["energy_above_hull"])
                if item.get("energy_above_hull") is not None else float("inf"),
            )
            matches[formula] = MaterialsProjectMatch(
                mp_id=str(selected.get("material_id", "")),
                energy_above_hull=selected.get("energy_above_hull"),
                spacegroup_number=selected.get("spacegroup_number"),
                formation_energy_per_atom=selected.get("formation_energy_per_atom"),
                band_gap_ev=selected.get("band_gap_ev"),
                is_metal=selected.get("is_metal"),
                total_magnetization_muB=selected.get("total_magnetization_muB"),
                density_g_cm3=selected.get("density_g_cm3"),
                volume_angstrom3=selected.get("volume_angstrom3"),
            )
        else:
            matches[formula] = existing_matches.get(formula, MaterialsProjectMatch())
        query = query_by_formula.get(parsed.normalized or "")
        if query and query.get("query_status") == "query_error":
            errors[formula] = str(query.get("error_detail") or "Materials Project query failed")

    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    output_fields = (
        "mp_id", "energy_above_hull", "spacegroup_number", "formation_energy_per_atom",
        "band_gap_ev", "is_metal", "total_magnetization_muB", "density_g_cm3", "volume_angstrom3",
        "mp_query_status", "mp_validation_status", "mp_candidate_structures",
    )
    for field in output_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    for row in rows:
        match = matches[row["formula"]]
        parsed = normalize_fixed_formula(row["formula"])
        query = query_by_formula.get(parsed.normalized or "")
        row["mp_query_status"] = query.get("query_status", "ambiguous_input") if query else "ambiguous_input"
        row["mp_validation_status"] = query.get("validation_status", "not_applicable") if query else "not_applicable"
        row["mp_candidate_structures"] = json.dumps(
            safe_by_formula.get(parsed.normalized or "", []), sort_keys=True
        )
        if match.mp_id:
            row["mp_id"] = match.mp_id
        if match.energy_above_hull is not None:
            row["energy_above_hull"] = f"{match.energy_above_hull:.8g}"
        if match.spacegroup_number is not None:
            row["spacegroup_number"] = str(match.spacegroup_number)
        for field in output_fields[3:9]:
            value = getattr(match, field)
            if value is not None:
                row[field] = str(value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "unique_formula_count": len(formulas),
        "matched_formula_count": sum(bool(matches[formula].mp_id) for formula in formulas),
        "errors": errors,
        "query_status_counts": {
            status: sum(query.get("query_status") == status for query in audit.get("query_records", []))
            for status in ("matched", "no_match", "ambiguous_input", "query_error", "not_queried")
        },
        "safe_match_count": audit.get("safe_match_count", 0),
        "rejected_mismatch_count": audit.get("rejected_mismatch_count", 0),
        "audit": audit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/training/harmonized_supercon.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/training/harmonized_supercon_enriched.csv"))
    parser.add_argument("--api-key", default=os.environ.get("MP_API_KEY"))
    parser.add_argument("--request-delay", type=float, default=0.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = enrich_csv(
        args.input,
        args.output,
        api_key=args.api_key,
        request_delay_seconds=args.request_delay,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
