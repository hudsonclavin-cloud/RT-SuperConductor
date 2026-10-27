"""Import the first received superconductivity datasets into shared evidence types."""

import csv
import re
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

from .evidence import EvidenceKind, EvidenceRecord
from .targets import ATMOSPHERIC_PRESSURE_GPA


UCI_SOURCE = "doi:10.24432/C53P47"
THREEDSC_SOURCE = "doi:10.6084/m9.figshare.c.6914407"
SUPERMAT_SOURCE = "doi:10.1080/27660400.2021.1918396"


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def _parse_supermat_tc(value: str) -> tuple[float | None, str | None]:
    """Parse only an unambiguous scalar Tc from SuperMat free text."""

    text = value.strip().replace("−", "-")
    if not text:
        return None, None
    if any(token in text.lower() for token in ("<", ">", "-", " to ", "–", "±")):
        return None, f"raw_tc_value={value!r}; non-scalar value retained for review"
    matches = re.findall(r"(?<![\d.])(\d+(?:\.\d*)?|\.\d+)", text)
    if len(matches) != 1:
        return None, f"raw_tc_value={value!r}; could not parse one scalar"
    scale = 1.0
    if re.search(r"\bmk\b", text, flags=re.IGNORECASE):
        scale = 0.001
    elif re.search(r"(?:µ|μ)k\b", text, flags=re.IGNORECASE):
        scale = 0.000001
    return float(Decimal(matches[0])) * scale, None


def _parse_supermat_pressure(value: str) -> tuple[float | None, str | None]:
    """Parse pressure to GPa while preserving bounds and ranges as caveats."""

    text = value.strip().replace("−", "-")
    if not text:
        return None, None
    lowered = text.casefold()
    if lowered in {"ambient", "ambient pressure", "ambient p", "normal pressure", "ap"}:
        return ATMOSPHERIC_PRESSURE_GPA, None
    if any(token in lowered for token in ("<", ">", "up to", "under", "high pressure", " and ", " to ", "-", "–")):
        return None, f"raw_pressure={value!r}; bound or range retained for review"
    match = re.search(r"(?<![\d.])(\d+(?:\.\d*)?|\.\d+)\s*(gpa|kbar|mpa|bar)?", lowered)
    if match is None:
        return None, f"raw_pressure={value!r}; could not parse one scalar"
    amount = Decimal(match.group(1))
    unit = match.group(2) or "gpa"
    factor = {
        "gpa": Decimal(1),
        "kbar": Decimal("0.1"),
        "mpa": Decimal("0.001"),
        "bar": Decimal("0.0001"),
    }[unit]
    caveat = f"raw_pressure={value!r}; approximate value" if any(marker in lowered for marker in ("~", "∼")) else None
    return float(amount * factor), caveat


def iter_uci_records(path: str | Path) -> Iterator[EvidenceRecord]:
    """Yield composition and reported-Tc records from UCI ``unique_m.csv``.

    The UCI file has no per-row pressure or measurement metadata. Those fields
    intentionally remain unknown rather than being inferred as ambient pressure.
    """

    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"material", "critical_temp"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"UCI file is missing columns: {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            yield EvidenceRecord(
                record_id=f"uci:unique_m:{row_number}",
                composition=row["material"],
                source_reference=UCI_SOURCE,
                evidence_kind=EvidenceKind.EXPERIMENTAL,
                reported_tc_k=_optional_float(row["critical_temp"]),
                caveats=("The source does not provide row-level pressure or method metadata.",),
            )


def iter_3dsc_mp_records(path: str | Path) -> Iterator[EvidenceRecord]:
    """Yield records from ``3DSC_MP.csv``.

    3DSC_MP starts with a comment/preamble line. ``csv.DictReader`` is retained
    for multiline quoted BibTeX fields so records are not split on embedded
    newlines.
    """

    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        preamble = handle.readline()
        if not preamble.startswith("#"):
            raise ValueError("3DSC_MP CSV must begin with its comment preamble")
        reader = csv.DictReader(handle)
        required = {"formula_sc", "tc", "cif", "material_id_2"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"3DSC_MP file is missing columns: {sorted(required)}")
        for row_number, row in enumerate(reader, start=3):
            caveats = [
                "The source combines reported superconductivity data with computed structure metadata."
            ]
            if not row["cif"]:
                caveats.append("No crystal-structure path was supplied for this record.")
            yield EvidenceRecord(
                record_id=f"3dsc_mp:{row['material_id_2']}:{row_number}",
                composition=row["formula_sc"],
                source_reference=row.get("doi_2") or f"{THREEDSC_SOURCE}:{row['material_id_2']}",
                evidence_kind=EvidenceKind.EXPERIMENTAL,
                crystal_structure=row["cif"] or None,
                material_family=row.get("sc_class") or None,
                reported_tc_k=_optional_float(row["tc"]),
                caveats=tuple(caveats),
            )


def iter_supermat_records(path: str | Path) -> Iterator[EvidenceRecord]:
    """Yield pressure-aware literature annotations from SuperMat 2.0.

    SuperMat values are free text extracted from publications. Scalar values are
    normalized, while bounds, ranges, and ambiguous values remain unknown with
    their original text recorded in caveats.
    """

    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"material", "tcValue", "pressure", "me_method", "filename"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"SuperMat file is missing columns: {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            material = row["material"].strip()
            caveats = ["SuperMat is literature annotation data; verify claims against the cited paper."]
            if not material:
                material = f"unknown-material:supermat:{row_number}"
                caveats.append("The source row has no material annotation.")
            tc, tc_caveat = _parse_supermat_tc(row["tcValue"])
            pressure, pressure_caveat = _parse_supermat_pressure(row["pressure"])
            if tc_caveat:
                caveats.append(tc_caveat)
            if pressure_caveat:
                caveats.append(pressure_caveat)
            yield EvidenceRecord(
                record_id=f"supermat:{row_number}",
                composition=material,
                source_reference=SUPERMAT_SOURCE,
                evidence_kind=EvidenceKind.EXPERIMENTAL,
                reported_tc_k=tc,
                measurement_pressure_gpa=pressure,
                measurement_method=row["me_method"].strip() or None,
                sample_identifier=row["filename"].strip() or None,
                caveats=tuple(caveats),
            )
