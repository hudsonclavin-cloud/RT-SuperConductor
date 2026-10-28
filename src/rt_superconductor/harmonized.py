"""Schema and validation for the harmonized superconductivity table.

The table is deliberately strict at its boundary.  Source-specific parsing
belongs upstream; once a row enters the harmonized file, downstream modeling
can rely on stable names, units, and controlled vocabularies.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .formula import canonicalize_formula
from .reviewed import ReviewedRecord, load_reviewed_records


REQUIRED_COLUMNS = (
    "record_id",
    "source_db",
    "formula",
    "tc_kelvin",
    "applied_pressure_gpa",
    "measurement_method",
    "sample_form",
    "doi",
    "review_status",
    "provenance",
    "is_synthetic",
)
OPTIONAL_COLUMNS = (
    "chemical_family",
    "mp_id",
    "energy_above_hull",
    "spacegroup_number",
)
ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
SOURCE_DATABASES = {"supercon2", "3dsc", "supermat", "uci"}
MEASUREMENT_METHODS = {
    "resistivity",
    "magnetic_susceptibility",
    "heat_capacity",
    "meissner",
    "unknown",
}
SAMPLE_FORMS = {"single_crystal", "polycrystalline", "thin_film", "powder", "unknown"}
REVIEW_STATUSES = {"pending", "confirmed", "excluded"}
PROVENANCE_VALUES = {"literature", "synthetic_demo", "curated"}
HASH_RE = re.compile(r"^[0-9a-f]{16,64}$")


@dataclass(frozen=True, slots=True)
class HarmonizedRecord:
    record_id: str
    source_db: str
    formula: str
    tc_kelvin: float
    applied_pressure_gpa: float
    measurement_method: str
    sample_form: str
    doi: str
    review_status: str
    provenance: str
    is_synthetic: int
    chemical_family: str = "Unknown"
    mp_id: str = ""
    energy_above_hull: float | None = None
    spacegroup_number: int | None = None


def stable_record_id(source_db: str, formula: str, doi: str, tc_kelvin: float, pressure_gpa: float) -> str:
    """Create a deterministic row identity from source provenance and values."""

    payload = "|".join((source_db, formula, doi, f"{tc_kelvin:.8g}", f"{pressure_gpa:.8g}"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _float(value: str, field: str, row_number: int, *, allow_blank: bool = False) -> float | None:
    value = value.strip()
    if not value and allow_blank:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field} must be numeric") from exc


def _optional_int(value: str, field: str, row_number: int) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field} must be an integer") from exc
    return parsed


def _binary(value: str, field: str, row_number: int) -> int:
    normalized = value.strip().lower()
    if normalized in {"0", "false"}:
        return 0
    if normalized in {"1", "true"}:
        return 1
    raise ValueError(f"row {row_number}: {field} must be 0/1 or false/true")


def validate_row(row: dict[str, str], row_number: int) -> HarmonizedRecord:
    missing = [column for column in REQUIRED_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    record_id = row["record_id"].strip().lower()
    if not HASH_RE.fullmatch(record_id):
        raise ValueError(f"row {row_number}: record_id must be a hexadecimal string hash")
    source_db = row["source_db"].strip().lower()
    if source_db not in SOURCE_DATABASES:
        raise ValueError(f"row {row_number}: source_db must be one of {sorted(SOURCE_DATABASES)}")
    formula_result = canonicalize_formula(row["formula"])
    if formula_result.status != "parsed" or not formula_result.canonical:
        raise ValueError(f"row {row_number}: formula must be normalized elemental formula")
    formula = row["formula"].strip()
    if any(character.isspace() for character in formula):
        raise ValueError(f"row {row_number}: formula must not contain whitespace")
    tc_kelvin = _float(row["tc_kelvin"], "tc_kelvin", row_number)
    assert tc_kelvin is not None
    if not 0 < tc_kelvin <= 300:
        raise ValueError(f"row {row_number}: tc_kelvin must satisfy 0 < Tc <= 300 K")
    pressure = _float(row["applied_pressure_gpa"], "applied_pressure_gpa", row_number)
    assert pressure is not None
    if pressure < 0:
        raise ValueError(f"row {row_number}: applied_pressure_gpa cannot be negative")
    method = row["measurement_method"].strip().lower()
    if method not in MEASUREMENT_METHODS:
        raise ValueError(f"row {row_number}: invalid measurement_method {method!r}")
    sample_form = row["sample_form"].strip().lower()
    if sample_form not in SAMPLE_FORMS:
        raise ValueError(f"row {row_number}: invalid sample_form {sample_form!r}")
    doi = row["doi"].strip()
    if not doi:
        raise ValueError(f"row {row_number}: doi is required, use a source identifier when no DOI exists")
    review_status = row["review_status"].strip().lower()
    if review_status not in REVIEW_STATUSES:
        raise ValueError(f"row {row_number}: invalid review_status {review_status!r}")
    provenance = row["provenance"].strip().lower()
    if provenance not in PROVENANCE_VALUES:
        raise ValueError(f"row {row_number}: invalid provenance {provenance!r}")
    is_synthetic = _binary(row["is_synthetic"], "is_synthetic", row_number)
    if (provenance == "synthetic_demo") != bool(is_synthetic):
        raise ValueError(
            f"row {row_number}: provenance={provenance!r} must agree with is_synthetic={is_synthetic}"
        )
    hull = _float(row.get("energy_above_hull", ""), "energy_above_hull", row_number, allow_blank=True)
    if hull is not None and hull < 0:
        raise ValueError(f"row {row_number}: energy_above_hull cannot be negative")
    spacegroup = _optional_int(row.get("spacegroup_number", ""), "spacegroup_number", row_number)
    if spacegroup is not None and not 1 <= spacegroup <= 230:
        raise ValueError(f"row {row_number}: spacegroup_number must be in [1, 230]")
    return HarmonizedRecord(
        record_id=record_id,
        source_db=source_db,
        formula=formula,
        tc_kelvin=tc_kelvin,
        applied_pressure_gpa=pressure,
        measurement_method=method,
        sample_form=sample_form,
        doi=doi,
        review_status=review_status,
        provenance=provenance,
        is_synthetic=is_synthetic,
        chemical_family=row.get("chemical_family", "").strip() or "Unknown",
        mp_id=row.get("mp_id", "").strip(),
        energy_above_hull=hull,
        spacegroup_number=spacegroup,
    )


def load_harmonized(path: Path, *, require_unique_ids: bool = True) -> list[HarmonizedRecord]:
    """Load and validate a harmonized CSV, including its exact required header."""

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")
        records = [validate_row(row, row_number) for row_number, row in enumerate(reader, start=2)]
    if require_unique_ids:
        ids = [record.record_id for record in records]
        duplicates = sorted({record_id for record_id in ids if ids.count(record_id) > 1})
        if duplicates:
            raise ValueError(f"{path}: duplicate record_id values: {duplicates}")
    return records


def write_harmonized(path: Path, records: list[HarmonizedRecord]) -> None:
    """Write a schema-compliant harmonized table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ALL_COLUMNS))
        writer.writeheader()
        for record in records:
            writer.writerow({
                "record_id": record.record_id,
                "source_db": record.source_db,
                "formula": record.formula,
                "tc_kelvin": f"{record.tc_kelvin:g}",
                "applied_pressure_gpa": f"{record.applied_pressure_gpa:g}",
                "measurement_method": record.measurement_method,
                "sample_form": record.sample_form,
                "doi": record.doi,
                "review_status": record.review_status,
                "provenance": record.provenance,
                "is_synthetic": record.is_synthetic,
                "chemical_family": record.chemical_family,
                "mp_id": record.mp_id,
                "energy_above_hull": "" if record.energy_above_hull is None else f"{record.energy_above_hull:g}",
                "spacegroup_number": "" if record.spacegroup_number is None else record.spacegroup_number,
            })


_REVIEWED_FORMULA_OVERRIDES = {
    "supercon2-review-016": "YBa2Cu3O7",
    "supercon2-review-017": "YBa2Cu3O6.94",
    "supercon2-review-018": "YBa2Cu3O7",
    "supercon2-review-019": "YBa2Cu4O8",
    "supercon2-review-020": "YBa2Cu4O8",
}


def _reviewed_formula(record: ReviewedRecord) -> str:
    try:
        return _REVIEWED_FORMULA_OVERRIDES[record.record_id]
    except KeyError as exc:
        raise ValueError(f"no conservative formula mapping for reviewed record {record.record_id!r}") from exc


def _reviewed_method(value: str) -> str:
    normalized = value.casefold()
    if "suscept" in normalized or "quadrupole" in normalized:
        return "magnetic_susceptibility"
    if "resist" in normalized or "transport" in normalized:
        return "resistivity"
    return "unknown"


def _reviewed_sample_form(value: str) -> str:
    normalized = value.casefold()
    if "single crystal" in normalized:
        return "single_crystal"
    if "polycrystal" in normalized or "ceramic" in normalized:
        return "polycrystalline"
    if "powder" in normalized:
        return "powder"
    return "unknown"


def reviewed_to_harmonized(record: ReviewedRecord) -> HarmonizedRecord:
    """Convert a confirmed reviewed row without inventing missing metadata."""

    formula = _reviewed_formula(record)
    if record.pressure_gpa is None:
        raise ValueError(f"reviewed record {record.record_id!r} has no numeric pressure for the harmonized schema")
    pressure = record.pressure_gpa
    return HarmonizedRecord(
        record_id=stable_record_id("supercon2", formula, record.source_doi, record.tc_kelvin, pressure),
        source_db="supercon2",
        formula=formula,
        tc_kelvin=record.tc_kelvin,
        applied_pressure_gpa=pressure,
        measurement_method=_reviewed_method(record.measurement_method),
        sample_form=_reviewed_sample_form(record.sample_form),
        doi=record.source_doi,
        review_status="confirmed",
        provenance="literature",
        is_synthetic=0,
        chemical_family="Cuprates",
    )


def bridge_reviewed_records(harmonized_path: Path, reviewed_path: Path) -> dict[str, object]:
    """Idempotently append confirmed reviewed records to the harmonized table."""

    records = load_harmonized(harmonized_path)
    reviewed = load_reviewed_records(reviewed_path.parent)
    unsupported = sorted({record.record_id for record in reviewed if record.record_id not in _REVIEWED_FORMULA_OVERRIDES})
    if unsupported:
        raise ValueError(
            "reviewed records need an explicit conservative formula mapping before ingestion: "
            + ", ".join(unsupported)
        )
    existing_ids = {record.record_id for record in records}
    bridged = [
        reviewed_to_harmonized(record)
        for record in reviewed
        if stable_record_id(
            "supercon2",
            _reviewed_formula(record),
            record.source_doi,
            record.tc_kelvin,
            record.pressure_gpa if record.pressure_gpa is not None else 0.0,
        ) not in existing_ids
    ]
    if bridged:
        write_harmonized(harmonized_path, records + bridged)
    return {
        "harmonized_path": str(harmonized_path),
        "reviewed_path": str(reviewed_path),
        "reviewed_record_count": len(reviewed),
        "bridged_record_count": len(bridged),
        "bridged_record_ids": [record.record_id for record in bridged],
    }
