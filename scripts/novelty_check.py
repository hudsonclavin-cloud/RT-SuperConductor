#!/usr/bin/env python3
"""How much of a list of formulas is new to this project (docs/learning-curve.md: only new compositions add training value).

    python novelty.py FILE.csv --column formula [--year year] [--sep ,] [--out report.json]

"What we hold" is every composition in the ranking of record's table (all splits; only the compositions of locked
rows are read, never their labels) plus every composition in the NIMS file. Each unique composition in FILE is put
in the first class that fits, using the table's own grouping rules (scripts/build_scratchnet_data.py):

    held            the same composition, to 3 decimals of atomic fraction
    near twin       within 0.02 atomic fraction of a held composition on every element (a doping step)
    known parent    the same set of major elements (5% or more each) as a held composition
    new parent      a set of major elements we do not hold, in an element system we do hold
    new system      a set of elements we do not hold at all
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_mp_candidates as M  # noqa: E402
import build_scratchnet_data as B  # noqa: E402
import scratchnet_common as C  # noqa: E402

CLASSES = ("held", "near twin", "known parent", "new parent", "new system")


def held_compositions() -> tuple[dict[str, dict], set[str]]:
    ds = C.load_dataset()
    held = {row["composition"]: B.parse_key(row["composition"]) for row in ds.rows}
    table = len(held)
    nims = json.loads((ROOT / "data/processed/children-knowledge/table.json").read_text(encoding="utf-8"))["keys"]
    for key in nims:
        held.setdefault(key, B.parse_key(key))
    print(f"held: {table} table compositions + {len(held) - table} NIMS-only = {len(held)}", file=sys.stderr)
    return held, set(ds.table.symbols)


def classify(held: dict[str, dict], new: dict[str, dict]) -> dict[str, str]:
    from scipy.spatial import cKDTree

    elements = sorted({e for comp in list(held.values()) + list(new.values()) for e in comp})
    column = {e: i for i, e in enumerate(elements)}

    def dense(comps):
        out = np.zeros((len(comps), len(elements)))
        for r, comp in enumerate(comps):
            for e, v in comp.items():
                out[r, column[e]] = v
        return out

    held_majors = {B.major_group(c) for c in held.values()}
    held_systems = {frozenset(c) for c in held.values()}
    tree = cKDTree(dense(list(held.values())))
    keys = [k for k in new if k not in held]
    distance = tree.query(dense([new[k] for k in keys]), k=1, p=np.inf)[0] if keys else np.array([])
    out = {k: "held" for k in new if k in held}
    for key, d in zip(keys, distance):
        if d <= B.NEAR_TOLERANCE + 1e-9:
            out[key] = "near twin"
        elif B.major_group(new[key]) in held_majors:
            out[key] = "known parent"
        elif frozenset(new[key]) in held_systems:
            out[key] = "new parent"
        else:
            out[key] = "new system"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--column", default="formula")
    ap.add_argument("--year", default=None)
    ap.add_argument("--sep", default=",")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None, help="a column holding 1 (superconductor) or 0 (not); splits the report")
    args = ap.parse_args()
    held, valid = held_compositions()
    csv.field_size_limit(10 ** 9)
    with open(args.file, newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.DictReader(handle, delimiter=args.sep))
    counts, new, years, records, labels = Counter(), {}, {}, Counter(), {}
    for row in rows:
        text = (row.get(args.column) or "").strip()
        if not text:
            counts["no formula"] += 1
            continue
        fractions = M.fractions(text)
        if not fractions or not set(fractions) <= valid or len(fractions) < 1:
            counts["formula not usable"] += 1
            continue
        counts["usable"] += 1
        key = B.composition_key(fractions)
        new[key] = B.parse_key(key)
        records[key] += 1
        if args.label and (row.get(args.label) or "").strip() in ("0", "1", "0.0", "1.0"):
            labels[key] = max(labels.get(key, 0), int(float(row[args.label])))    # any superconducting report wins
        if args.year and (row.get(args.year) or "").strip()[:4].isdigit():
            years[key] = max(years.get(key, 0), int(row[args.year].strip()[:4]))
    label = classify(held, new)
    by_class = Counter(label.values())
    family = {k: M.family(set(new[k])) for k in new}
    report = {"file": args.file, "records": len(rows), "record_counts": dict(counts), "unique_compositions": len(new),
              "by_class": {c: by_class.get(c, 0) for c in CLASSES},
              "share": {c: round(by_class.get(c, 0) / max(len(new), 1), 4) for c in CLASSES},
              "records_by_class": {c: int(sum(records[k] for k in new if label[k] == c)) for c in CLASSES},
              "by_family": {f: {c: sum(1 for k in new if family[k] == f and label[k] == c) for c in CLASSES}
                            for f in sorted(set(family.values()))}}
    if args.label:
        report["by_class_and_label"] = {
            name: {c: sum(1 for k in new if label[k] == c and labels.get(k) == value) for c in CLASSES}
            for name, value in (("superconductor", 1), ("not superconductor", 0), ("no label", None))}
    fresh = [k for k in new if label[k] in ("new parent", "new system")]
    report["new_chemistry_single_element"] = sum(1 for k in fresh if len(new[k]) == 1)
    if years:
        report["new_chemistry_by_year"] = dict(sorted(Counter(
            ("2021+" if years[k] >= 2021 else "2016-2020" if years[k] >= 2016 else "to 2015") for k in fresh if k in years).items()))
    report["examples_new_system"] = [C.formula_text(new[k]) for k in sorted(k for k in new if label[k] == "new system")[:25]]
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
