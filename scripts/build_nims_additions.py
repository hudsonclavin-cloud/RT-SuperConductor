#!/usr/bin/env python3
"""The NIMS compositions a learning cycle may add to the ranking of record's training (cycle 4).

    python scripts/build_nims_additions.py

Plan and guards: ``docs/learning-cycles.md`` ("Cycle 4"). The training table is not rebuilt, because adding rows to
it can merge groups and move existing rows between the development, locked and train-only splits. Instead every NIMS
composition absent from the table is tied to the table's groups the way the table ties its own rows (major elements
and near twins, chained through other added rows), and is kept only where that is safe:

* tied to a locked group: left out;
* tied to groups in more than one fold or split: left out;
* tied to one place: it takes that fold (-1 for a train-only group);
* tied to nothing: a new group, with a fold from a hash of the group's name.

Writes ``data/processed/learning-cycles/nims-additions.csv.gz`` (not committed),
``docs/learning-cycles/nims-additions.json`` (counts and checksum) and
``docs/learning-cycles/nims-disagreements.csv`` (shared compositions whose labels differ; not changed here).
"""
from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict

import build_scratchnet_data as B
import scratchnet_common as C
from scratchnet_common import ROOT

OUT = ROOT / "data/processed/learning-cycles/nims-additions.csv.gz"
REPORT = ROOT / "docs/learning-cycles/nims-additions.json"
DISAGREEMENTS = ROOT / "docs/learning-cycles/nims-disagreements.csv"
FIELDS = ["composition", "label", "tc_k", "major_group", "group", "tied_groups", "fold", "placed"]


def place(table: list[dict], new: list[str]) -> list[dict]:
    """Where each new composition falls among the table's groups. ``table`` rows need composition, major_group,
    group, split and fold. Returns one dict per new composition with ``placed`` and, where kept, ``fold``."""
    fractions = [C.fractions_of(row["composition"]) for row in table] + [C.fractions_of(key) for key in new]
    group_of_major = {row["major_group"]: row["group"] for row in table}
    where = {row["group"]: (row["split"], int(row["fold"])) for row in table}
    parent: dict = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    major = [B.major_group(C.fractions_of(key)) for key in new]
    for i, name in enumerate(major):
        union(("new", i), ("group", group_of_major[name]) if name in group_of_major else ("major", name))
    n_table = len(table)
    for a, b in B.near_pairs(fractions, B.NEAR_TOLERANCE):
        if b < n_table and a < n_table:
            continue
        nodes = [("group", table[i]["group"]) if i < n_table else ("new", i - n_table) for i in (a, b)]
        union(*nodes)
    members: dict = defaultdict(lambda: {"new": [], "group": []})
    for node in list(parent):
        if node[0] in ("new", "group"):
            members[find(node)][node[0]].append(node[1])
    out: list[dict | None] = [None] * len(new)
    for component in members.values():
        if not component["new"]:
            continue
        groups = sorted(component["group"])
        places = {where[g] for g in groups}
        if not groups:
            sizes = Counter(major[i] for i in component["new"])
            name = min(sizes, key=lambda m: (-sizes[m], m))
            placed, fold, groups = "new group", (B.stable_hash("scratchnet-split:" + name) // 100) % B.FOLDS, [name]
        elif any(split == "lockbox" for split, _ in places):
            placed, fold = "left out: tied to a locked group", None
        elif len(places) > 1:
            placed, fold = "left out: tied to more than one fold", None
        else:
            split, fold = next(iter(places))
            placed = "joins a development group" if split == "dev" else "joins a train-only group"
        for i in component["new"]:
            out[i] = {"composition": new[i], "major_group": major[i], "group": groups[0],
                      "tied_groups": ";".join(groups), "fold": fold, "placed": placed}
    return out  # type: ignore[return-value]


def nims_labels() -> dict[str, tuple[int, float | None]]:
    """composition -> (label, Tc): 1 if any NIMS record reports superconductivity; Tc is the median distinct value."""
    import children_knowledge as G

    table = json.loads((G.WORK / "table.json").read_text(encoding="utf-8"))
    out = {}
    for key in table["keys"]:
        records = table["records"][key]
        positive = sorted({round(float(tc), 4) for _, label, tc in records if label and tc and tc > 0})
        out[key] = (int(any(label for _, label, _ in records)), statistics.median(positive) if positive else None)
    return out


def main() -> int:
    ds = C.load_dataset()
    labels = nims_labels()
    have = {row["composition"]: row for row in ds.rows}
    new = [key for key in labels if key not in have]
    placed = place(ds.rows, new)
    rows = []
    for entry in placed:
        if entry["fold"] is None:
            continue
        label, tc = labels[entry["composition"]]
        rows.append({**entry, "label": label, "tc_k": "" if tc is None else tc})
    rows.sort(key=lambda row: row["composition"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    C.write_table(OUT, rows, FIELDS)
    # Shared compositions whose labels differ. Locked rows' labels are not read.
    differ = [{"composition": key, "split": row["split"], "record_label": row["label"], "nims_label": labels[key][0]}
              for key, row in sorted(have.items())
              if key in labels and row["split"] != "lockbox" and row["label"] != ""
              and int(float(row["label"])) != labels[key][0]]
    C.write_table(DISAGREEMENTS, differ, ["composition", "split", "record_label", "nims_label"])
    shared = Counter(have[key]["split"] for key in labels if key in have)
    report = {"nims_compositions": len(labels), "already_in_the_table": dict(shared),
              "absent_from_the_table": len(new),
              "placed": dict(Counter(entry["placed"] for entry in placed)),
              "added_rows": len(rows), "added_labels": dict(Counter(str(row["label"]) for row in rows)),
              "added_by_fold": {str(k): v for k, v in sorted(Counter(row["fold"] for row in rows).items())},
              "label_disagreements_listed": len(differ), "table_sha256": ds.sha256,
              "additions_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}
    REPORT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
