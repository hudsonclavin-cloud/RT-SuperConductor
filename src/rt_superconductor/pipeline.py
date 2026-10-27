"""Deterministic normalization and quality reporting for local source data."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .datasets import iter_3dsc_mp_records, iter_uci_records
from .evidence import EvidenceRecord


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
    tc_min_k: float | None
    tc_max_k: float | None
    tc_mean_k: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _summary(records: list[EvidenceRecord]) -> PipelineSummary:
    by_composition: defaultdict[str, list[EvidenceRecord]] = defaultdict(list)
    for record in records:
        by_composition[composition_key(record.composition)].append(record)
    tc_values = [record.reported_tc_k for record in records if record.reported_tc_k is not None]
    duplicate_groups = [group for group in by_composition.values() if len(group) > 1]
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
        tc_min_k=min(tc_values) if tc_values else None,
        tc_max_k=max(tc_values) if tc_values else None,
        tc_mean_k=(sum(tc_values) / len(tc_values)) if tc_values else None,
    )


def _record_dict(record: EvidenceRecord, structure_root: Path | None) -> dict[str, object]:
    value = asdict(record)
    value["evidence_kind"] = record.evidence_kind.value
    value["caveats"] = list(record.caveats)
    value["composition_key"] = composition_key(record.composition)
    if structure_root is not None and record.crystal_structure is not None:
        local_path = structure_root / Path(record.crystal_structure).name
        value["local_structure_path"] = str(local_path) if local_path.is_file() else None
    return value


def run_pipeline(
    *,
    uci_path: str | Path,
    threedsc_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    structure_root: str | Path | None = None,
) -> PipelineSummary:
    """Normalize both supplied datasets and write JSONL plus a report."""

    records = [
        *iter_uci_records(uci_path),
        *iter_3dsc_mp_records(threedsc_path),
    ]
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
                },
                "summary": summary.as_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
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
    parser.add_argument("--output", required=True, type=Path, help="normalized JSONL output")
    parser.add_argument("--report", required=True, type=Path, help="quality report JSON output")
    parser.add_argument(
        "--structure-root",
        type=Path,
        help="local directory containing extracted 3DSC CIF files",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_pipeline(
        uci_path=args.uci,
        threedsc_path=args.threedsc_mp,
        output_path=args.output,
        report_path=args.report,
        structure_root=args.structure_root,
    )
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
