#!/usr/bin/env python3
"""The NEMAD compositions cycle 7 may add to the ranking of record's training (plan: docs/learning-cycles.md, "Cycle 7").

    python scripts/build_nemad_additions.py

Takes the usable subset of the NEMAD superconductor table (docs/nemad-superconductor.md): records flagged
experimental, at normal or unstated pressure. Keeps the compositions that are new to the project, meaning not in the
table (locked rows included), not in the NIMS file, and not a near twin of either. Places them by cycle 4's rule
(``build_nims_additions.place``), with cycle 4's added NIMS rows counted as part of the table so that ties chain
through them too:
* tied to a locked group: left out;
* tied to groups in more than one fold or split: left out;
* tied to one place: it takes that fold (-1 for a train-only group);
* tied to nothing: a new group, with a fold from a hash of the group's name.
A composition is a superconductor if any usable record reports it so; its Tc is the median of the distinct reported
values above zero.

Writes ``data/processed/learning-cycles/cycle7-additions.csv.gz`` (not committed): cycle 4's NIMS rows unchanged
(source "nims"), then the placed NEMAD rows (source "nemad"). Counts and checksums go to
``docs/learning-cycles/cycle7-additions.json``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_mp_candidates as M  # noqa: E402
import build_nims_additions as BN  # noqa: E402
import build_scratchnet_data as B  # noqa: E402
import novelty_check as N  # noqa: E402
import scratchnet_common as C  # noqa: E402

NEMAD = ROOT / "data/processed/nemad/superconductor.csv"
NIMS_ADDITIONS = ROOT / "data/processed/learning-cycles/nims-additions.csv.gz"
OUT = ROOT / "data/processed/learning-cycles/cycle7-additions.csv.gz"
REPORT = ROOT / "docs/learning-cycles/cycle7-additions.json"
FIELDS = BN.FIELDS + ["source"]


def usable(row: dict) -> bool:
    """The filter the paper check supported: flagged experimental, not at a known high pressure."""
    return row["experimental_flag"] == "True" and row["ambient"] != "False"


def nemad_labels(rows: list[dict]) -> dict[str, tuple[int, float | None]]:
    """composition -> (label, Tc) from the usable records; compositions with no label are dropped."""
    sc, nsc, tcs = Counter(), Counter(), defaultdict(set)
    for row in rows:
        if not usable(row):
            continue
        if row["label"] == "1":
            sc[row["composition"]] += 1
            if row["tc_k"] and float(row["tc_k"]) > 0:
                tcs[row["composition"]].add(round(float(row["tc_k"]), 4))
        elif row["label"] == "0":
            nsc[row["composition"]] += 1
    out = {}
    for key in set(sc) | set(nsc):
        positive = sorted(tcs.get(key, ()))
        out[key] = (1, statistics.median(positive) if positive else None) if sc[key] else (0, None)
    return out


def select_new(labels: dict, held: dict[str, dict], valid: set[str]) -> tuple[list[str], Counter]:
    """Compositions new to the project: elements we model, at most C.MAX_ELEMENTS of them, and not held or a near twin."""
    candidates = {k: B.parse_key(k) for k in labels
                  if set(B.parse_key(k)) <= valid and len(B.parse_key(k)) <= C.MAX_ELEMENTS}
    where = N.classify(held, candidates)
    keep = sorted(k for k, c in where.items() if c in ("known parent", "new parent", "new system"))
    return keep, Counter(where.values())


def main() -> int:
    ds = C.load_dataset()
    held, valid = N.held_compositions()
    with open(NEMAD, newline="", encoding="utf-8") as handle:
        labels = nemad_labels(list(csv.DictReader(handle)))
    new, classes = select_new(labels, held, valid)
    nims_rows = C.read_table(NIMS_ADDITIONS)
    have = {row["composition"] for row in ds.rows} | {row["composition"] for row in nims_rows}
    if have & set(new):
        raise SystemExit("a selected NEMAD composition is already in the table or the NIMS additions")
    pseudo = [{"composition": r["composition"], "major_group": r["major_group"], "group": r["group"],
               "split": "train_only" if int(r["fold"]) < 0 else "dev", "fold": r["fold"]} for r in nims_rows]
    placed = BN.place(ds.rows + pseudo, new)
    rows = [{**{k: r[k] for k in BN.FIELDS}, "source": "nims"} for r in nims_rows]
    added = []
    for entry in placed:
        if entry["fold"] is None:
            continue
        label, tc = labels[entry["composition"]]
        added.append({**entry, "label": label, "tc_k": "" if tc is None else tc, "source": "nemad"})
    added.sort(key=lambda row: row["composition"])
    rows += added
    OUT.parent.mkdir(parents=True, exist_ok=True)
    C.write_table(OUT, rows, FIELDS)
    family = Counter(M.family(set(C.fractions_of(r["composition"]))) for r in added)
    report = {"nemad_compositions_with_a_label_in_the_usable_subset": len(labels),
              "novelty_classes": dict(classes), "new_compositions": len(new),
              "placed": dict(Counter(entry["placed"] for entry in placed)),
              "nemad_rows_added": len(added), "nemad_labels": dict(Counter(str(r["label"]) for r in added)),
              "nemad_by_family": dict(family),
              "nemad_by_fold": {str(k): v for k, v in sorted(Counter(r["fold"] for r in added).items())},
              "nims_rows_kept": len(nims_rows), "table_sha256": ds.sha256,
              "nims_additions_sha256": hashlib.sha256(NIMS_ADDITIONS.read_bytes()).hexdigest(),
              "additions_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}
    REPORT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
