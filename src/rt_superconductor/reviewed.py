"""Validation for human-reviewed superconductivity training records."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path


REQUIRED_COLUMNS = (
    "record_id",
    "formula",
    "tc_kelvin",
    "pressure_gpa",
    "pressure_condition",
    "measurement_method",
    "sample_form",
    "synthesis_conditions",
    "source_doi",
    "source_record_id",
    "reviewer",
    "review_date",
    "evidence_status",
    "notes",
)
PRESSURE_CONDITIONS = {"ambient", "known", "unknown", "not_reported"}


@dataclass(frozen=True)
class ReviewedRecord:
    """A validated record that is eligible for later model training."""

    record_id: str
    formula: str
    tc_kelvin: float
    pressure_gpa: float | None
    pressure_condition: str
    measurement_method: str
    sample_form: str
    synthesis_conditions: str
    source_doi: str
    source_record_id: str
    reviewer: str
    review_date: str
    evidence_status: str
    notes: str


def _required(value: str, field: str, row_number: int) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"row {row_number}: {field} is required")
    return value


def _parse_review_date(value: str, row_number: int) -> str:
    value = _required(value, "review_date", row_number)
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: review_date must be YYYY-MM-DD") from exc
    return value


def _parse_pressure(value: str, condition: str, row_number: int) -> float | None:
    value = value.strip()
    if not value:
        if condition == "known":
            raise ValueError(f"row {row_number}: known pressure requires pressure_gpa")
        return None
    try:
        pressure = float(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: pressure_gpa must be numeric") from exc
    if pressure < 0:
        raise ValueError(f"row {row_number}: pressure_gpa cannot be negative")
    if condition == "ambient":
        if pressure > 0.1:
            raise ValueError(f"row {row_number}: ambient pressure must be <= 0.1 GPa")
        return pressure
    if condition != "known":
        raise ValueError(f"row {row_number}: pressure_gpa requires pressure_condition=known")
    return pressure


def validate_row(row: dict[str, str], row_number: int) -> ReviewedRecord:
    """Validate one CSV row and return a typed record."""

    missing = [column for column in REQUIRED_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    status = _required(row["evidence_status"], "evidence_status", row_number).lower()
    if status != "confirmed":
        raise ValueError(
            f"row {row_number}: evidence_status must be confirmed for training, got {status!r}"
        )
    condition = _required(row["pressure_condition"], "pressure_condition", row_number).lower()
    if condition not in PRESSURE_CONDITIONS:
        allowed = ", ".join(sorted(PRESSURE_CONDITIONS))
        raise ValueError(f"row {row_number}: pressure_condition must be one of {allowed}")
    tc_text = _required(row["tc_kelvin"], "tc_kelvin", row_number)
    try:
        tc_kelvin = float(tc_text)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: tc_kelvin must be numeric") from exc
    if not 0 <= tc_kelvin <= 1000:
        raise ValueError(f"row {row_number}: tc_kelvin must be between 0 and 1000 K")
    return ReviewedRecord(
        record_id=_required(row["record_id"], "record_id", row_number),
        formula=_required(row["formula"], "formula", row_number),
        tc_kelvin=tc_kelvin,
        pressure_gpa=_parse_pressure(row["pressure_gpa"], condition, row_number),
        pressure_condition=condition,
        measurement_method=_required(row["measurement_method"], "measurement_method", row_number),
        sample_form=_required(row["sample_form"], "sample_form", row_number),
        synthesis_conditions=_required(row["synthesis_conditions"], "synthesis_conditions", row_number),
        source_doi=_required(row["source_doi"], "source_doi", row_number),
        source_record_id=_required(row["source_record_id"], "source_record_id", row_number),
        reviewer=_required(row["reviewer"], "reviewer", row_number),
        review_date=_parse_review_date(row["review_date"], row_number),
        evidence_status=status,
        notes=row["notes"].strip(),
    )


def load_reviewed_records(input_dir: Path) -> list[ReviewedRecord]:
    """Load and validate every CSV in a reviewed-data directory."""

    records: list[ReviewedRecord] = []
    seen_ids: set[str] = set()
    for path in sorted(input_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(REQUIRED_COLUMNS):
                raise ValueError(f"{path}: columns must exactly match the reviewed training schema")
            for row_number, row in enumerate(reader, start=2):
                record = validate_row(row, row_number)
                if record.record_id in seen_ids:
                    raise ValueError(f"{path}:{row_number}: duplicate record_id {record.record_id!r}")
                seen_ids.add(record.record_id)
                records.append(record)
    return records
