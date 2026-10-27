"""Generate reviewable high-value composition shortlists from normalized evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path


def _groups(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        key = record.get("canonical_composition") or record["composition_key"]
        grouped[str(key)].append(record)
    result = []
    for key, group in grouped.items():
        temperatures = [
            float(record["reported_tc_k"])
            for record in group
            if record.get("reported_tc_k") is not None
        ]
        sources = sorted({str(record["record_id"]).split(":", 1)[0] for record in group})
        structures = [record for record in group if record.get("crystal_structure")]
        result.append(
            {
                "canonical_composition": key,
                "records": group,
                "record_count": len(group),
                "source_count": len(sources),
                "sources": sources,
                "structure_count": len(structures),
                "max_tc_k": max(temperatures) if temperatures else None,
                "min_tc_k": min(temperatures) if temperatures else None,
                "tc_spread_k": (max(temperatures) - min(temperatures)) if temperatures else None,
            }
        )
    return result


def _score(group: dict[str, object], pass_name: str, target_tc_k: float) -> float:
    max_tc = float(group["max_tc_k"] or 0.0) / target_tc_k
    structure_fraction = float(group["structure_count"]) / float(group["record_count"])
    source_bonus = 1.0 if int(group["source_count"]) >= 2 else 0.0
    spread = float(group["tc_spread_k"] or 0.0) / target_tc_k
    if pass_name == "highest_tc":
        return max_tc
    if pass_name == "highest_tc_with_structure":
        return max_tc + 0.35 * structure_fraction
    if pass_name == "cross_source_evidence":
        return max_tc + 0.35 * source_bonus + 0.15 * structure_fraction
    if pass_name == "structured_cross_source":
        return max_tc + 0.35 * source_bonus + 0.35 * structure_fraction
    if pass_name == "information_value":
        return 0.45 * max_tc + 0.55 * spread + 0.15 * source_bonus
    raise ValueError(f"unknown triage pass: {pass_name}")


PASS_NAMES = (
    "highest_tc",
    "highest_tc_with_structure",
    "cross_source_evidence",
    "information_value",
    "structured_cross_source",
)


def generate_triage_runs(
    records: Iterable[dict[str, object]], *, limit: int = 25, target_tc_k: float = 300.0
) -> list[dict[str, object]]:
    """Generate four deterministic ranking passes over composition groups."""

    groups = _groups(records)
    runs = []
    for pass_name in PASS_NAMES:
        pool = groups
        if pass_name == "highest_tc_with_structure":
            pool = [group for group in groups if int(group["structure_count"]) > 0]
        elif pass_name == "cross_source_evidence":
            pool = [group for group in groups if int(group["source_count"]) >= 2]
        elif pass_name == "structured_cross_source":
            pool = [
                group
                for group in groups
                if int(group["source_count"]) >= 2 and int(group["structure_count"]) > 0
            ]
        ranked = sorted(
            pool,
            key=lambda group: (
                -_score(group, pass_name, target_tc_k),
                -(float(group["max_tc_k"] or 0.0)),
                str(group["canonical_composition"]),
            ),
        )
        candidates = []
        for rank, group in enumerate(ranked[:limit], start=1):
            candidates.append(
                {
                    "rank": rank,
                    "score": round(_score(group, pass_name, target_tc_k), 8),
                    "canonical_composition": group["canonical_composition"],
                    "record_count": group["record_count"],
                    "source_count": group["source_count"],
                    "sources": group["sources"],
                    "structure_count": group["structure_count"],
                    "max_tc_k": group["max_tc_k"],
                    "min_tc_k": group["min_tc_k"],
                    "tc_spread_k": group["tc_spread_k"],
                    "record_ids": [record["record_id"] for record in group["records"]],
                }
            )
        runs.append({"pass": pass_name, "target_tc_k": target_tc_k, "candidates": candidates})
    return runs


def run_triage(
    *, input_path: str | Path, output_path: str | Path, limit: int = 25, target_tc_k: float = 300.0
) -> list[dict[str, object]]:
    with Path(input_path).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    runs = generate_triage_runs(records, limit=limit, target_tc_k=target_tc_k)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"pipeline_version": 1, "input": str(input_path), "passes": runs},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return runs


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="normalized evidence JSONL")
    parser.add_argument("--output", required=True, type=Path, help="triage JSON output")
    parser.add_argument("--limit", type=int, default=25, help="candidates per pass")
    args = parser.parse_args(argv)
    runs = run_triage(input_path=args.input, output_path=args.output, limit=args.limit)
    print(json.dumps({"passes": [run["pass"] for run in runs], "limit": args.limit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
