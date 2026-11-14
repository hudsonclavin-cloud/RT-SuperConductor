#!/usr/bin/env python3
"""Count the outcomes of a blinded literature check by group and compare the groups.

The input is a CSV in which every row is one material that was looked up in the published literature without
the reader knowing which group it belonged to. It needs a ``group`` column and a ``verdict`` column holding
one of VERDICTS. The summary gives, for each group, how many materials fall under each verdict and the share
that are reported superconductors at ambient pressure - both out of all materials and out of those for which
any low-temperature measurement was found - with 95% Wilson intervals, and a Fisher exact test between any
two groups named with ``--compare``.

    python scripts/literature_check_summary.py --cases data/benchmarks/literature-check/made-metals.csv \
        --compare top random --compare top bottom --out docs/literature-check-made-metals.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

VERDICTS = ("superconductor_ambient", "superconductor_only_pressure_doping_or_film", "measured_no_superconductivity",
            "magnetic_or_insulating_no_superconductivity", "no_low_temperature_data_found")
HIT = "superconductor_ambient"
UNDECIDED = "no_low_temperature_data_found"


def wilson(hits: int, total: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    """95% Wilson score interval for a share."""
    if total == 0:
        return None, None
    share = hits / total
    centre = (share + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - half), min(1.0, centre + half)


def fisher_two_sided(a_hits: int, a_total: int, b_hits: int, b_total: int) -> float:
    """Two-sided Fisher exact test for two shares: the summed probability of tables no likelier than this one."""
    hits, total = a_hits + b_hits, a_total + b_total

    def probability(k: int) -> float:
        return math.comb(a_total, k) * math.comb(b_total, hits - k) / math.comb(total, hits)

    observed = probability(a_hits)
    low, high = max(0, hits - b_total), min(a_total, hits)
    return min(1.0, sum(probability(k) for k in range(low, high + 1) if probability(k) <= observed * (1 + 1e-9)))


def share(hits: int, total: int) -> dict:
    low, high = wilson(hits, total)
    return {"hits": hits, "of": total, "share": hits / total if total else None, "low": low, "high": high}


def summarise(rows: list[dict], comparisons: list[tuple[str, str]]) -> dict:
    unknown = sorted({row["verdict"] for row in rows} - set(VERDICTS))
    if unknown:
        raise SystemExit(f"unknown verdicts: {unknown}")
    groups: dict[str, dict] = {}
    for name in sorted({row["group"] for row in rows}):
        members = [row for row in rows if row["group"] == name]
        counts = {verdict: sum(row["verdict"] == verdict for row in members) for verdict in VERDICTS}
        decided = len(members) - counts[UNDECIDED]
        groups[name] = {"rows": len(members), "verdicts": counts, "with_low_temperature_data": decided,
                        "superconductor_of_all": share(counts[HIT], len(members)),
                        "superconductor_of_measured": share(counts[HIT], decided),
                        "superconducts_somehow_of_all": share(counts[HIT] + counts[VERDICTS[1]], len(members))}
    out: dict = {"rows": len(rows), "groups": groups, "comparisons": []}
    for first, second in comparisons:
        if first not in groups or second not in groups:
            raise SystemExit(f"cannot compare {first!r} with {second!r}: groups are {sorted(groups)}")
        a, b = groups[first], groups[second]
        entry = {"first": first, "second": second}
        for key in ("superconductor_of_all", "superconductor_of_measured"):
            x, y = a[key], b[key]
            entry[key] = {"difference": (x["share"] - y["share"]) if x["of"] and y["of"] else None,
                          "fisher_two_sided_p": fisher_two_sided(x["hits"], x["of"], y["hits"], y["of"])
                          if x["of"] and y["of"] else None}
        out["comparisons"].append(entry)
    return out


def rounded(value, digits: int = 4):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item, digits) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--compare", nargs=2, action="append", default=[], metavar=("FIRST", "SECOND"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with open(args.cases, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "group" not in rows[0] or "verdict" not in rows[0]:
        raise SystemExit(f"{args.cases} needs the columns 'group' and 'verdict'")
    summary = {"cases_file": args.cases.name, **summarise(rows, [tuple(pair) for pair in args.compare])}
    args.out.write_text(json.dumps(rounded(summary), indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rounded(summary), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
