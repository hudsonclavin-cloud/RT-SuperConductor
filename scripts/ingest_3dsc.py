#!/usr/bin/env python3
"""Ingest the 3DSC Materials Project table into the harmonized boundary.

3DSC supplies curated database matches, including explicit ``Tc=0`` negative
examples. Its table does not contain a measurement pressure, so this importer
records the atmospheric-pressure assumption in the report rather than hiding
it in a source-specific parser. CIF-derived space-group numbers are used when
the referenced structure is available; missing structures remain missing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rt_superconductor.formula import canonicalize_formula
from rt_superconductor.harmonized import HarmonizedRecord, load_harmonized, write_harmonized
from rt_superconductor.targets import ATMOSPHERIC_PRESSURE_GPA


THREEDSC_DOI = "10.6084/m9.figshare.c.6914407"
SOURCE_DB = "3dsc"
REVIEW_STATUS = "curated_database"
FAMILY_MAP = {
    "Cuprate": "Cuprates",
    "Oxide": "Oxides",
    "Ferrite": "Ferrites",
    "Heavy_fermion": "Heavy-fermion",
    "Chevrel": "Chevrel",
    "Carbon": "Carbon-based",
    "Other": "Other",
    "Heavy_fermionChevrel": "Heavy-fermion/Chevrel",
    "OxideHeavy_fermion": "Oxides/Heavy-fermion",
}


def _field(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "")
        if value is not None and value.strip():
            return value.strip()
    return ""


def _float(value: str, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc


def _record_id(row: dict[str, str], row_number: int, formula: str, tc: float) -> str:
    mp_id = _field(row, "material_id_2", "material_id", "mp_id")
    payload = "|".join((SOURCE_DB, mp_id, formula, f"{tc:.8g}", str(row_number)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _cif_candidates(structure_root: Path | None, value: str) -> list[Path]:
    if structure_root is None or not value:
        return []
    raw = Path(value.replace("\\", "/"))
    candidates = [structure_root / raw.name, structure_root / raw]
    normalized = value.replace("\\", "/")
    for prefix in (
        "data/final/MP/cifs/",
        "superconductors_3D/data/final/MP/cifs/",
    ):
        if prefix in normalized:
            candidates.append(structure_root / normalized.split(prefix, 1)[1])
    return list(dict.fromkeys(candidates))


def _spacegroup_from_cif(path: Path) -> int | None:
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    structure = Structure.from_file(path)
    return int(SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number())


@lru_cache(maxsize=64)
def _spacegroup_symbol_number(symbol: str) -> int | None:
    if not symbol:
        return None
    try:
        from pymatgen.symmetry.groups import SpaceGroup

        number = int(SpaceGroup(symbol.replace(" ", "")).int_number)
        return number if 1 <= number <= 230 else None
    except Exception:
        return None


def _spacegroup(row: dict[str, str], cif_path: Path | None) -> tuple[int | None, str]:
    for name in ("spacegroup_number_2", "spacegroup_number"):
        value = row.get(name, "").strip()
        if value:
            try:
                number = int(float(value))
            except ValueError:
                break
            if 1 <= number <= 230:
                return number, "table"
    table_symbol = _field(row, "spacegroup_2", "spacegroup")
    if cif_path is not None:
        try:
            return _spacegroup_from_cif(cif_path), "cif"
        except Exception:
            if table_symbol:
                number = _spacegroup_symbol_number(table_symbol)
                return (number, "table_symbol") if number is not None else (None, "cif_unreadable")
            return None, "cif_unreadable"
    if table_symbol:
        number = _spacegroup_symbol_number(table_symbol)
        if number is not None:
            return number, "table_symbol"
    return None, "missing_cif"


def _read_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        first = handle.readline()
        if not first:
            return []
        if not first.lstrip().startswith("#"):
            handle.seek(0)
        return list(csv.DictReader(handle, skipinitialspace=True))


def ingest_3dsc(
    *,
    input_path: Path,
    output_path: Path,
    structure_root: Path | None = None,
    report_path: Path | None = None,
    structure_index_path: Path | None = None,
) -> dict[str, Any]:
    rows = _read_rows(input_path)
    existing = load_harmonized(output_path) if output_path.is_file() else []
    existing_ids = {record.record_id for record in existing}
    records: list[HarmonizedRecord] = []
    structure_index: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    family_counts: Counter[str] = Counter()
    spacegroup_sources: Counter[str] = Counter()
    cif_found = 0
    zero_tc = 0

    for row_number, row in enumerate(rows, start=2):
        try:
            raw_formula = _field(row, "formula_sc", "formula")
            formula_result = canonicalize_formula(raw_formula)
            if formula_result.status != "parsed" or not formula_result.canonical:
                raise ValueError(f"formula is not parseable: {raw_formula!r}")
            formula = formula_result.canonical
            tc = _float(_field(row, "tc", "tc_kelvin"), "tc")
            if not 0 <= tc <= 300:
                raise ValueError(f"tc must satisfy 0 <= Tc <= 300 K, got {tc}")
            source_class = _field(row, "sc_class")
            family = FAMILY_MAP.get(source_class, f"3DSC:{source_class or 'Unknown'}")
            record_id = _record_id(row, row_number, formula, tc)
            if record_id in existing_ids:
                cif_path = next((path for path in _cif_candidates(structure_root, _field(row, "cif")) if path.is_file()), None)
                if cif_path is not None:
                    structure_index.append({"record_id": record_id, "cif_path": str(cif_path)})
                continue
            cif_path = next((path for path in _cif_candidates(structure_root, _field(row, "cif")) if path.is_file()), None)
            if cif_path is not None:
                cif_found += 1
            spacegroup, source = _spacegroup(row, cif_path)
            spacegroup_sources[source] += 1
            record = HarmonizedRecord(
                record_id=record_id,
                source_db=SOURCE_DB,
                formula=formula,
                tc_kelvin=tc,
                applied_pressure_gpa=ATMOSPHERIC_PRESSURE_GPA,
                measurement_method="unknown",
                sample_form="unknown",
                doi=THREEDSC_DOI,
                review_status=REVIEW_STATUS,
                provenance="curated",
                is_synthetic=0,
                chemical_family=family,
                mp_id=_field(row, "material_id_2", "material_id", "mp_id"),
                energy_above_hull=_float(_field(row, "e_above_hull_2", "energy_above_hull"), "e_above_hull")
                if _field(row, "e_above_hull_2", "energy_above_hull") else None,
                spacegroup_number=spacegroup,
            )
            records.append(record)
            existing_ids.add(record_id)
            family_counts[family] += 1
            if tc == 0:
                zero_tc += 1
            if cif_path is not None:
                structure_index.append({"record_id": record_id, "cif_path": str(cif_path)})
        except (KeyError, ValueError) as exc:
            skipped.append({"row": str(row_number), "reason": str(exc)})

    if records:
        write_harmonized(output_path, existing + records)
    elif not output_path.is_file():
        write_harmonized(output_path, [])
    if structure_index_path is not None:
        structure_index_path.parent.mkdir(parents=True, exist_ok=True)
        with structure_index_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("record_id", "cif_path"))
            writer.writeheader()
            writer.writerows(structure_index)

    report: dict[str, Any] = {
        "status": "ready" if not skipped else "ready_with_skips",
        "input": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output": str(output_path),
        "input_row_count": len(rows),
        "ingested_row_count": len(records),
        "skipped_row_count": len(skipped),
        "skipped_rows": skipped[:100],
        "zero_tc_row_count": zero_tc,
        "family_counts": dict(sorted(family_counts.items())),
        "cif_found_count": cif_found,
        "spacegroup_sources": dict(sorted(spacegroup_sources.items())),
        "structure_index": str(structure_index_path) if structure_index_path else None,
        "pressure_assumption": {
            "applied_pressure_gpa": ATMOSPHERIC_PRESSURE_GPA,
            "reason": "3DSC_MP does not provide measurement pressure; ambient pressure is an explicit screening assumption, not an observed label.",
        },
        "review_status": REVIEW_STATUS,
        "provenance": "curated",
        "source_doi": THREEDSC_DOI,
        "training_policy": "eligible for empirical training; SuperCon2 automatic extraction is excluded by the trainer",
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/3dsc/MP/3DSC_MP.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/training/harmonized_supercon.csv"))
    parser.add_argument("--structure-root", type=Path, default=Path("data/raw/3dsc/MP/cifs"))
    parser.add_argument("--report", type=Path, default=Path("data/processed/3dsc-ingestion-report.json"))
    parser.add_argument("--structure-index", type=Path, default=Path("data/processed/3dsc-structure-index.csv"))
    args = parser.parse_args()
    report = ingest_3dsc(
        input_path=args.input,
        output_path=args.output,
        structure_root=args.structure_root if args.structure_root.exists() else None,
        report_path=args.report,
        structure_index_path=args.structure_index,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
