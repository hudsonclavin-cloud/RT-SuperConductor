"""Import the first received superconductivity datasets into shared evidence types."""

import csv
from collections.abc import Iterator
from pathlib import Path

from .evidence import EvidenceKind, EvidenceRecord


UCI_SOURCE = "doi:10.24432/C53P47"
THREEDSC_SOURCE = "doi:10.6084/m9.figshare.c.6914407"


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


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
