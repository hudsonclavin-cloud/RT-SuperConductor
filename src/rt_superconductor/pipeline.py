"""Deterministic normalization and quality reporting for local source data."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .datasets import (
    iter_3dsc_mp_records,
    iter_kaggle_supercon_records,
    iter_supermat_records,
    iter_uci_records,
)
from .evidence import EvidenceRecord
from .formula import FormulaResult, canonicalize_formula


def composition_key(composition: str) -> str:
    """Return a conservative key for exact textual composition matching.

    This deliberately does not reorder elements or infer stoichiometry. Formula
    normalization is a later scientific task that needs a reviewed parser.
    """

    return re.sub(r"\s+", "", composition).casefold()


@dataclass(frozen=True, slots=True)
class PipelineSummary:
    total_records: int
    source_counts: dict[str, int]
    records_with_tc: int
    records_without_tc: int
    records_with_structure: int
    records_without_structure: int
    records_with_known_pressure: int
    duplicate_composition_groups: int
    duplicate_record_count: int
    canonical_duplicate_composition_groups: int
    canonical_duplicate_record_count: int
    unparseable_formula_records: int
    tc_min_k: float | None
    tc_max_k: float | None
    tc_mean_k: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _summary(records: list[EvidenceRecord]) -> PipelineSummary:
    by_composition: defaultdict[str, list[EvidenceRecord]] = defaultdict(list)
    by_canonical: defaultdict[str, list[EvidenceRecord]] = defaultdict(list)
    formula_results = [canonicalize_formula(record.composition) for record in records]
    for record in records:
        by_composition[composition_key(record.composition)].append(record)
    for record, formula in zip(records, formula_results):
        if formula.canonical is not None:
            by_canonical[formula.canonical].append(record)
    tc_values = [record.reported_tc_k for record in records if record.reported_tc_k is not None]
    duplicate_groups = [group for group in by_composition.values() if len(group) > 1]
    canonical_duplicate_groups = [group for group in by_canonical.values() if len(group) > 1]
    return PipelineSummary(
        total_records=len(records),
        source_counts=dict(
            sorted(
                Counter(record.record_id.split(":", maxsplit=1)[0] for record in records).items()
            )
        ),
        records_with_tc=len(tc_values),
        records_without_tc=len(records) - len(tc_values),
        records_with_structure=sum(record.crystal_structure is not None for record in records),
        records_without_structure=sum(record.crystal_structure is None for record in records),
        records_with_known_pressure=sum(
            record.measurement_pressure_gpa is not None for record in records
        ),
        duplicate_composition_groups=len(duplicate_groups),
        duplicate_record_count=sum(len(group) for group in duplicate_groups),
        canonical_duplicate_composition_groups=len(canonical_duplicate_groups),
        canonical_duplicate_record_count=sum(len(group) for group in canonical_duplicate_groups),
        unparseable_formula_records=sum(result.canonical is None for result in formula_results),
        tc_min_k=min(tc_values) if tc_values else None,
        tc_max_k=max(tc_values) if tc_values else None,
        tc_mean_k=(sum(tc_values) / len(tc_values)) if tc_values else None,
    )


def _record_dict(record: EvidenceRecord, structure_root: Path | None) -> dict[str, object]:
    value = asdict(record)
    formula = canonicalize_formula(record.composition)
    value["evidence_kind"] = record.evidence_kind.value
    value["caveats"] = list(record.caveats)
    value["composition_key"] = composition_key(record.composition)
    value["canonical_composition"] = formula.canonical
    value["formula_parse_status"] = formula.status
    value["formula_parse_reason"] = formula.reason
    if structure_root is not None and record.crystal_structure is not None:
        local_path = structure_root / Path(record.crystal_structure).name
        value["local_structure_path"] = str(local_path) if local_path.is_file() else None
    return value


def _review_rows(records: list[EvidenceRecord]) -> list[dict[str, object]]:
    """Build a review queue without merging or deleting source records."""

    groups: defaultdict[str, list[tuple[EvidenceRecord, FormulaResult]]] = defaultdict(list)
    rows: list[dict[str, object]] = []
    for record in records:
        result = canonicalize_formula(record.composition)
        if result.canonical is None:
            rows.append(
                {
                    "review_type": "formula_review",
                    "reason": result.reason,
                    "status": result.status,
                    "record_id": record.record_id,
                    "composition": record.composition,
                    "source_reference": record.source_reference,
                }
            )
        else:
            groups[result.canonical].append((record, result))
    for canonical, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        rows.append(
            {
                "review_type": "duplicate_composition",
                "canonical_composition": canonical,
                "records": [
                    {
                        "record_id": record.record_id,
                        "composition": record.composition,
                        "source_reference": record.source_reference,
                        "reported_tc_k": record.reported_tc_k,
                        "measurement_pressure_gpa": record.measurement_pressure_gpa,
                        "crystal_structure": record.crystal_structure,
                    }
                    for record, _ in group
                ],
            }
        )
    return rows


def run_pipeline(
    *,
    uci_path: str | Path,
    threedsc_path: str | Path,
    supermat_path: str | Path | None = None,
    kaggle_supercon_path: str | Path | None = None,
    output_path: str | Path,
    report_path: str | Path,
    structure_root: str | Path | None = None,
    review_path: str | Path | None = None,
) -> PipelineSummary:
    """Normalize both supplied datasets and write JSONL plus a report."""

    records = [
        *iter_uci_records(uci_path),
        *iter_3dsc_mp_records(threedsc_path),
    ]
    if supermat_path is not None:
        records.extend(iter_supermat_records(supermat_path))
    if kaggle_supercon_path is not None:
        records.extend(iter_kaggle_supercon_records(kaggle_supercon_path))
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("source records contain duplicate record IDs")

    output = Path(output_path)
    report = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    local_root = Path(structure_root) if structure_root is not None else None
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(_record_dict(record, local_root), sort_keys=True))
            handle.write("\n")

    summary = _summary(records)
    report.write_text(
        json.dumps(
            {
                "pipeline_version": 1,
                "inputs": {
                    "uci": str(uci_path),
                    "3dsc_mp": str(threedsc_path),
                    "supermat": str(supermat_path) if supermat_path is not None else None,
                    "kaggle_supercon": (
                        str(kaggle_supercon_path) if kaggle_supercon_path is not None else None
                    ),
                },
                "summary": summary.as_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if review_path is not None:
        review = Path(review_path)
        review.parent.mkdir(parents=True, exist_ok=True)
        with review.open("w", encoding="utf-8", newline="\n") as handle:
            for row in _review_rows(records):
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uci", required=True, type=Path, help="UCI unique_m.csv")
    parser.add_argument(
        "--3dsc-mp",
        dest="threedsc_mp",
        required=True,
        type=Path,
        help="3DSC_MP.csv",
    )
    parser.add_argument("--supermat", type=Path, help="optional SuperMat-2.0.csv")
    parser.add_argument(
        "--kaggle-supercon",
        type=Path,
        help="optional raw.tsv from Kaggle's SuperCon dataset",
    )
    parser.add_argument("--output", required=True, type=Path, help="normalized JSONL output")
    parser.add_argument("--report", required=True, type=Path, help="quality report JSON output")
    parser.add_argument(
        "--structure-root",
        type=Path,
        help="local directory containing extracted 3DSC CIF files",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        help="optional JSONL duplicate and formula-review queue",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_pipeline(
        uci_path=args.uci,
        threedsc_path=args.threedsc_mp,
        supermat_path=args.supermat,
        kaggle_supercon_path=args.kaggle_supercon,
        output_path=args.output,
        report_path=args.report,
        structure_root=args.structure_root,
        review_path=args.review_output,
    )
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
