#!/usr/bin/env python3
"""Attach the lowest-energy Materials Project polymorph to each formula.

The script reads the validated harmonized table and writes a new CSV with
``mp_id``, ``energy_above_hull`` (eV/atom), and ``spacegroup_number``.  It is
safe to rerun: existing enrichment columns are replaced from the current API
response, while all provenance and review fields are retained.
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
    # Preserve IDs already attached to the harmonized source.  A temporary API
    # miss must not erase a known structure link and leave the next workflow
    # stage with nothing to retrieve.
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
    try:
        for formula in formulas:
            try:
                documents = client.materials.summary.search(
                    formula=formula,
                    fields=[
                        "material_id", "energy_above_hull", "formation_energy_per_atom",
                        "symmetry", "band_gap", "is_metal", "total_magnetization",
                        "density", "volume",
                    ],
                )
                selected = select_lowest_energy_polymorph(documents)
                matches[formula] = selected if selected.mp_id else existing_matches.get(formula, MaterialsProjectMatch())
            except Exception as exc:  # preserve other formulas and report the failure
                errors[formula] = f"{type(exc).__name__}: {exc}"
                matches[formula] = existing_matches.get(formula, MaterialsProjectMatch())
            if request_delay_seconds:
                time.sleep(request_delay_seconds)
    finally:
        if context_manager is not None:
            context_manager.__exit__(None, None, None)

    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    output_fields = (
        "mp_id", "energy_above_hull", "spacegroup_number", "formation_energy_per_atom",
        "band_gap_ev", "is_metal", "total_magnetization_muB", "density_g_cm3", "volume_angstrom3",
    )
    for field in output_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    for row in rows:
        match = matches[row["formula"]]
        if match.mp_id:
            row["mp_id"] = match.mp_id
        if match.energy_above_hull is not None:
            row["energy_above_hull"] = f"{match.energy_above_hull:.8g}"
        if match.spacegroup_number is not None:
            row["spacegroup_number"] = str(match.spacegroup_number)
        for field in output_fields[3:]:
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
