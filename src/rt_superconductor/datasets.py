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
KAGGLE_SUPERCON_SOURCE = "https://www.kaggle.com/datasets/chaozhuang/supercon-dataset"
SUPERCON2_SOURCE = "https://mdr.nims.go.jp/datasets/2169a31f-ca73-4391-aceb-7acf15066f78"


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


def _parse_kaggle_tc(value: str, unit: str) -> tuple[float | None, str | None]:
    """Parse the scalar temperature in the Kaggle SuperCon snapshot."""

    text = value.strip().replace("−", "-")
    if not text:
        return None, None
    if unit.strip().casefold() != "k":
        return None, f"raw_tc_unit={unit!r}; unsupported unit retained for review"
    if any(token in text.casefold() for token in ("<", ">", " to ", "–", "±", "-")):
        return None, f"raw_tc_value={value!r}; non-scalar value retained for review"
    if not re.fullmatch(r"\d+(?:\.\d*)?|\.\d+", text):
        return None, f"raw_tc_value={value!r}; could not parse one scalar"
    return float(Decimal(text)), None


def _parse_supercon2_tc(value: str) -> tuple[float | None, str | None]:
    """Parse one extracted Kelvin value while flagging ambiguous text."""

    text = value.strip().replace("−", "-").replace("–", "-")
    if not text:
        return None, None
    if not re.search(r"k\b", text, flags=re.IGNORECASE):
        return None, f"raw_tc_value={value!r}; missing Kelvin unit"
    if any(token in text.casefold() for token in ("<", ">", " to ", "±", "-")):
        return None, f"raw_tc_value={value!r}; non-scalar value retained for review"
    matches = re.findall(r"(?<![\d.])(\d+(?:\.\d*)?|\.\d+)", text)
    if len(matches) != 1:
        return None, f"raw_tc_value={value!r}; could not parse one scalar"
    scale = 0.001 if re.search(r"mk\b", text, flags=re.IGNORECASE) else 1.0
    caveat = f"raw_tc_value={value!r}; approximate value" if "~" in text else None
    return float(Decimal(matches[0]) * Decimal(str(scale))), caveat


def _parse_supercon2_pressure(value: str) -> tuple[float | None, str | None]:
    """Parse scalar SuperCon2 pressure text into GPa conservatively."""

    text = " ".join(value.strip().replace("−", "-").split())
    if not text:
        return None, None
    lowered = text.casefold().replace("- ", "-").replace(" -", "-")
    ambient = {
        "ambient",
        "ambient pressure",
        "ambient conditions",
        "ambient pressures",
        "atmospheric pressure",
        "normal pressure",
        "zero pressure",
        "ambient-pressure",
        "ambient pres",
        "ambient pres- sure",
        "atmo- spheric pressure",
    }
    if lowered in ambient:
        pressure = 0.0 if lowered == "zero pressure" else ATMOSPHERIC_PRESSURE_GPA
        return pressure, None
    if any(token in lowered for token in ("<", ">", " to ", " and ", ",", "~", "∼")):
        return None, f"raw_pressure={value!r}; bound or range retained for review"
    if re.search(r"\d\s*-\s*\d", lowered):
        return None, f"raw_pressure={value!r}; range retained for review"
    match = re.fullmatch(
        r"(?:pressure\s*)?([0-9]+(?:\.\d*)?|\.\d+)\s*"
        r"(gpa|mpa|kbar|mbar|bar|mtorr|torr|atm|pa)?(?:\s*pressure)?",
        lowered,
    )
    if match is None:
        return None, f"raw_pressure={value!r}; could not parse one scalar"
    amount = Decimal(match.group(1))
    unit = match.group(2)
    if unit is None:
        return None, f"raw_pressure={value!r}; missing pressure unit"
    factor = {
        "gpa": Decimal(1),
        "mpa": Decimal("0.001"),
        "kbar": Decimal("0.1"),
        "bar": Decimal("0.0001"),
        "mbar": Decimal("0.0000001"),
        "mtorr": Decimal("0.000000000133322"),
        "torr": Decimal("0.000000133322"),
        "atm": Decimal("0.000101325"),
        "pa": Decimal("0.000000001"),
    }[unit]
    return float(amount * factor), None


def _supercon2_synthesis(row: dict[str, str]) -> str | None:
    parts = []
    for label in ("fabrication", "shape", "substrate", "doping", "variables"):
        value = row.get(label, "").strip()
        if value:
            parts.append(f"{label}={value}")
    return "; ".join(parts) or None


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


def iter_kaggle_supercon_records(path: str | Path) -> Iterator[EvidenceRecord]:
    """Yield rows from Kaggle's raw SuperCon literature snapshot.

    The file has three header rows followed by tab-separated literature records.
    It contains no row-level pressure field, so pressure remains unknown even
    when a material is known to have been measured under pressure elsewhere.
    """

    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        # The first two rows are display metadata; the third supplies keys.
        for _ in range(2):
            handle.readline()
        fieldnames = handle.readline().rstrip("\r\n").split("\t")
        required = {"num", "name", "element", "str3", "utc", "tc", "journal"}
        if not required.issubset(fieldnames):
            raise ValueError(f"Kaggle SuperCon file is missing columns: {sorted(required)}")
        reader = csv.DictReader(handle, fieldnames=fieldnames, delimiter="\t")
        for row_number, row in enumerate(reader, start=4):
            composition = row["element"].strip()
            caveats = [
                "Kaggle snapshot of the SuperCon literature table; verify the row against the original publication.",
                "The source does not provide row-level pressure or measurement-method metadata.",
            ]
            if not composition:
                composition = f"unknown-material:kaggle-supercon:{row_number}"
                caveats.append("The source row has no chemical formula.")
            tc, tc_caveat = _parse_kaggle_tc(row["tc"], row["utc"])
            if tc_caveat:
                caveats.append(tc_caveat)
            if row["name"].strip():
                caveats.append(f"common_material_name={row['name'].strip()!r}")
            yield EvidenceRecord(
                record_id=f"kaggle_supercon:{row['num'].strip() or row_number}",
                composition=composition,
                source_reference=KAGGLE_SUPERCON_SOURCE,
                evidence_kind=EvidenceKind.EXPERIMENTAL,
                publication_reference=row["journal"].strip() or None,
                material_family=row["str3"].strip() or row["name"].strip() or None,
                reported_tc_k=tc,
                sample_identifier=row["num"].strip() or None,
                caveats=tuple(caveats),
            )


def iter_supercon2_records(path: str | Path) -> Iterator[EvidenceRecord]:
    """Yield conservatively normalized records from the NIMS SuperCon2 export."""

    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "id",
            "rawMaterial",
            "materialId",
            "name",
            "formula",
            "materialClass",
            "fabrication",
            "criticalTemperature",
            "criticalTemperatureMeasurementMethod",
            "appliedPressure",
            "section",
            "subsection",
            "title",
            "doi",
            "journal",
            "year",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"SuperCon2 file is missing columns: {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            raw_material = row["rawMaterial"].strip()
            composition = row["formula"].strip() or raw_material
            caveats = [
                "SuperCon2 is automatically extracted literature data; verify the claim against the original paper.",
            ]
            if not composition:
                composition = f"unknown-material:supercon2:{row_number}"
                caveats.append("The source row has no material expression or formula.")
            if not row["formula"].strip() and raw_material:
                caveats.append(f"raw_material_expression={raw_material!r}")
            tc, tc_caveat = _parse_supercon2_tc(row["criticalTemperature"])
            pressure, pressure_caveat = _parse_supercon2_pressure(row["appliedPressure"])
            if tc_caveat:
                caveats.append(tc_caveat)
            if pressure_caveat:
                caveats.append(pressure_caveat)
            if row["section"].strip() or row["subsection"].strip():
                caveats.append(
                    f"extracted_from={row['section'].strip()}/{row['subsection'].strip()}"
                )
            if not row["doi"].strip():
                caveats.append("The source row has no DOI link.")
            yield EvidenceRecord(
                record_id=f"supercon2:{row['id'].strip() or row_number}",
                composition=composition,
                source_reference=SUPERCON2_SOURCE,
                evidence_kind=EvidenceKind.EXPERIMENTAL,
                publication_reference=(row["doi"].strip() or row["title"].strip() or None),
                material_family=row["materialClass"].strip() or row["name"].strip() or None,
                reported_tc_k=tc,
                measurement_pressure_gpa=pressure,
                synthesis_conditions=_supercon2_synthesis(row),
                measurement_method=row["criticalTemperatureMeasurementMethod"].strip() or None,
                sample_identifier=row["materialId"].strip() or row["id"].strip() or None,
                caveats=tuple(caveats),
            )
