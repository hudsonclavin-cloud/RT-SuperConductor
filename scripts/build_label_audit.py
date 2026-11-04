#!/usr/bin/env python3
"""Build a fixed-seed, residual-blind 30-record human label-audit sample."""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path
from typing import Iterable

from pymatgen.core import Composition

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from rt_superconductor.formula import canonicalize_formula
FAMILIES = ("Ferrites", "Other", "Cuprates")
SEED = 20261104
HUMAN_FIELDS = (
    "source_doi_checked", "exact_sample", "phase", "tc_definition",
    "measurement_pressure_gpa", "synthesis_pressure_gpa",
    "cif_matches_sample_yes_no", "reviewer_notes", "verdict_keep_fix_quarantine",
)
FIELDS = (
    "record_id", "formula", "family", "database_tc_kelvin", "mp_id",
    "cif_parent_formula", "synthetic_doping_flag", "doi_available",
    "selection_seed", *HUMAN_FIELDS,
)


def fractional_formula(formula: str) -> bool:
    try:
        return any(abs(float(amount) - round(float(amount))) > 1e-7
                   for amount in Composition(formula).get_el_amt_dict().values())
    except Exception:
        return False


def _distinct_key(row: dict[str, str]) -> str:
    return (row.get("mp_id") or "").strip() or Composition(row["formula"]).reduced_formula


def select_audit_sample(records: Iterable[dict[str, str]], *, seed: int = SEED) -> list[dict[str, str]]:
    """Select 10 rows/family with zero and fractional coverage before any model use."""
    records = list(records)
    result: list[dict[str, str]] = []
    for family_index, family in enumerate(FAMILIES):
        pool = sorted((row for row in records if row.get("chemical_family") == family),
                      key=lambda row: row["record_id"])
        if len(pool) < 10:
            raise ValueError(f"{family} has only {len(pool)} rows; 10 required")
        rng = random.Random(seed + family_index)
        rng.shuffle(pool)
        chosen: list[dict[str, str]] = []
        used: set[str] = set()

        def take(rows: Iterable[dict[str, str]], count: int) -> None:
            if count <= 0:
                return
            for row in rows:
                key = _distinct_key(row)
                if key in used:
                    continue
                chosen.append(row)
                used.add(key)
                if count > 0:
                    count -= 1
                if count == 0:
                    return

        take((row for row in pool if row.get("tc_kelvin") not in (None, "") and float(row["tc_kelvin"]) == 0), 2)
        fractional_count = sum(fractional_formula(row["formula"]) for row in chosen)
        if fractional_count < 3:
            take((row for row in pool if fractional_formula(row["formula"])), 3 - fractional_count)
            fractional_count = sum(fractional_formula(row["formula"]) for row in chosen)
        if sum(float(row["tc_kelvin"]) == 0 for row in chosen) < 2:
            raise ValueError(f"{family} lacks two distinct zero-Tc records")
        if fractional_count < 3:
            raise ValueError(f"{family} lacks three distinct fractional-stoichiometry records")
        take((row for row in pool if row not in chosen), 10 - len(chosen))
        if len(chosen) != 10:
            raise ValueError(f"Could not select 10 distinct records for {family}")
        result.extend(chosen)
    return result


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _reduced(value: str) -> str | None:
    try:
        return Composition(value).reduced_formula
    except Exception:
        return None


def _source_maps(source_3dsc: Path, supercon2: Path, supermat: Path) -> tuple[dict[str, dict], dict[str, set[str]]]:
    by_record: dict[str, dict] = {}
    with source_3dsc.open(newline="", encoding="utf-8-sig") as handle:
        first = handle.readline()
        if not first.lstrip().startswith("#"):
            handle.seek(0)
        for row_number, row in enumerate(csv.DictReader(handle, skipinitialspace=True), start=2):
            formula = row.get("formula_sc", "").strip()
            mp_id = (row.get("material_id_2") or "").strip()
            try:
                tc = float(row.get("tc", ""))
                parsed = canonicalize_formula(formula)
                if parsed.status != "parsed" or not parsed.canonical:
                    continue
                canonical = parsed.canonical
            except Exception:
                continue
            record_id = hashlib.sha256(
                "|".join(("3dsc", mp_id, canonical, f"{tc:.8g}", str(row_number))).encode()
            ).hexdigest()[:24]
            by_record[record_id] = row
    doi_by_formula: dict[str, set[str]] = {}
    for source_name, path, formula_field, doi_field in (
        ("SuperCon2", supercon2, "formula", "doi"),
        ("SuperMat", supermat, "material", "doi"),
    ):
        for row in _load_csv(path):
            formula = _reduced(row.get(formula_field, ""))
            doi = (row.get(doi_field) or "").strip()
            if formula and doi:
                doi_by_formula.setdefault(formula, set()).add((source_name, doi))
    return by_record, doi_by_formula


def build_rows(
    records: list[dict[str, str]], *, source_3dsc: Path, supercon2: Path, supermat: Path,
    structure_index: Path,
) -> list[dict[str, str]]:
    from pymatgen.core import Structure

    source_records, source_dois = _source_maps(source_3dsc, supercon2, supermat)
    cif_paths = {row["record_id"]: row["cif_path"] for row in _load_csv(structure_index)}
    output = []
    for row in select_audit_sample(records):
        source = source_records.get(row["record_id"], {})
        doi_entries: set[str] = set()
        raw_doi = (source.get("doi_2") or "").strip()
        if raw_doi:
            doi_entries.add("3DSC doi_2=" + raw_doi)
        formula_key = _reduced(row["formula"])
        if formula_key:
            doi_entries.update(source_name + "=" + doi for source_name, doi in source_dois.get(formula_key, set()))
        cif_path_text = cif_paths.get(row["record_id"], "")
        if cif_path_text and not Path(cif_path_text).is_absolute():
            cif_path = ROOT / cif_path_text
        else:
            cif_path = Path(cif_path_text) if cif_path_text else None
        parent_formula = ""
        if cif_path and cif_path.is_file():
            try:
                parent_formula = Structure.from_file(cif_path).composition.reduced_formula
            except Exception:
                parent_formula = ""
        result = {
            "record_id": row["record_id"], "formula": row["formula"],
            "family": row["chemical_family"], "database_tc_kelvin": row.get("tc_kelvin", ""),
            "mp_id": row.get("mp_id", ""), "cif_parent_formula": parent_formula,
            "synthetic_doping_flag": source.get("synth_doped", ""),
            "doi_available": "; ".join(sorted(doi_entries)), "selection_seed": str(SEED),
        }
        result.update({field: "" for field in HUMAN_FIELDS})
        output.append(result)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/training/harmonized_supercon.csv"))
    parser.add_argument("--source-3dsc", type=Path, default=Path("data/raw/3dsc/MP/3DSC_MP.csv"))
    parser.add_argument("--supercon2", type=Path, default=Path("data/raw/supercon2/supercon2_v22.12.03.csv"))
    parser.add_argument("--supermat", type=Path, default=Path("data/raw/supermat/SuperMat-2.0.csv"))
    parser.add_argument("--structure-index", type=Path, default=Path("data/processed/3dsc-structure-index.csv"))
    parser.add_argument("--output", type=Path, default=Path("docs/label-audit-template.csv"))
    args = parser.parse_args()
    records = _load_csv(args.input)
    available = {row.get("chemical_family") for row in records}
    missing = sorted(set(FAMILIES) - available)
    if missing:
        parser.error(
            f"--input must be the full 3DSC harmonized table; missing families {missing}. "
            "Run scripts/ingest_3dsc.py to a working directory and pass its --output file."
        )
    rows = build_rows(records=records, source_3dsc=args.source_3dsc,
                      supercon2=args.supercon2, supermat=args.supermat,
                      structure_index=args.structure_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows; seed={SEED}; families={','.join(FAMILIES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
