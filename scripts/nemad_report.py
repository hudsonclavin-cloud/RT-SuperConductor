#!/usr/bin/env python3
"""How much of the NEMAD superconductor table is new to this project, and how far its labels agree with ours.

    python scripts/nemad_report.py        # needs data/processed/nemad/superconductor.csv (scripts/nemad_download.py)

Writes docs/nemad-superconductor-check.json. Locked rows count as held (so they are never counted as new) but their
labels are never read: label agreement uses only development and train-only rows and the NIMS file.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_mp_candidates as M  # noqa: E402
import build_scratchnet_data as B  # noqa: E402
import novelty_check as N  # noqa: E402
import scratchnet_common as C  # noqa: E402

TABLE = ROOT / "data/processed/nemad/superconductor.csv"
OUT = ROOT / "docs/nemad-superconductor-check.json"


def readable_labels(ds) -> dict[str, int]:
    """Our labels that may be read: development and train-only rows, then NIMS for compositions not locked."""
    out = {}
    for row, split, label in zip(ds.rows, ds.split, ds.arrays.label):
        if split != "lockbox" and np.isfinite(label):
            out[row["composition"]] = int(label)
    locked = {row["composition"] for row, split in zip(ds.rows, ds.split) if split == "lockbox"}
    nims = json.loads((ROOT / "data/processed/children-knowledge/table.json").read_text(encoding="utf-8"))
    for key in nims["keys"]:
        if key not in out and key not in locked:
            out[key] = int(any(label for _, label, _ in nims["records"][key]))
    return out


def per_composition(rows: list[dict]) -> dict[str, dict]:
    """One entry per composition: its label (any superconducting report wins), report counts, best Tc, flags."""
    out: dict[str, dict] = defaultdict(lambda: {"sc": 0, "nsc": 0, "tc": [], "ambient": 0, "experimental": 0, "dois": set()})
    for row in rows:
        entry = out[row["composition"]]
        if row["label"] == "1":
            entry["sc"] += 1
        elif row["label"] == "0":
            entry["nsc"] += 1
        if row["tc_k"]:
            entry["tc"].append(float(row["tc_k"]))
        entry["ambient"] += row["ambient"] == "True"
        entry["experimental"] += row["experimental_flag"] == "True"
        if row["doi"]:
            entry["dois"].add(row["doi"])
    for entry in out.values():
        entry["label"] = 1 if entry["sc"] else (0 if entry["nsc"] else None)
        entry["mixed"] = bool(entry["sc"] and entry["nsc"])
    return dict(out)


def main() -> int:
    with open(TABLE, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    comps = per_composition(rows)
    held, valid = N.held_compositions()
    usable = {k: B.parse_key(k) for k in comps if set(B.parse_key(k)) <= valid}
    where = N.classify(held, usable)
    ds = C.load_dataset()
    ours = readable_labels(ds)

    def table(keys):
        by = Counter(where[k] for k in keys)
        return {c: by.get(c, 0) for c in N.CLASSES}

    by_label = {name: [k for k in usable if comps[k]["label"] == value]
                for name, value in (("superconductor", 1), ("not superconductor", 0), ("no label", None))}
    agree = Counter()
    disagreements = []
    for key in usable:
        if key in ours and comps[key]["label"] is not None:
            same = ours[key] == comps[key]["label"]
            agree[("ours_sc" if ours[key] else "ours_null", "same" if same else "differs")] += 1
            if not same and len(disagreements) < 40:
                disagreements.append({"formula": C.formula_text(usable[key]), "ours": ours[key], "nemad": comps[key]["label"],
                                      "nemad_reports_sc": comps[key]["sc"], "nemad_reports_not": comps[key]["nsc"],
                                      "dois": sorted(comps[key]["dois"])[:3]})
    fresh = [k for k in usable if where[k] in ("known parent", "new parent", "new system")]
    family = {k: M.family(set(usable[k])) for k in usable}
    report = {
        "rows": len(rows), "compositions": len(comps), "compositions_with_elements_we_model": len(usable),
        "by_class": table(usable),
        "by_class_and_label": {name: table(keys) for name, keys in by_label.items()},
        "new_by_family_and_label": {f: {name: sum(1 for k in fresh if family[k] == f and comps[k]["label"] == value)
                                        for name, value in (("superconductor", 1), ("not superconductor", 0))}
                                    for f in sorted({family[k] for k in fresh})},
        "new_with_both_labels_reported": sum(1 for k in fresh if comps[k]["mixed"]),
        "new_flags": {"at_least_one_ambient_report": sum(1 for k in fresh if comps[k]["ambient"]),
                      "at_least_one_experimental_flag": sum(1 for k in fresh if comps[k]["experimental"]),
                      "with_a_tc": sum(1 for k in fresh if comps[k]["tc"])},
        "new_superconductor_tc_k": {q: float(np.percentile([max(comps[k]["tc"]) for k in fresh if comps[k]["label"] == 1 and comps[k]["tc"]], p))
                                    for q, p in (("p10", 10), ("median", 50), ("p90", 90))},
        "label_agreement_where_we_hold_a_readable_label": {f"{a}/{b}": n for (a, b), n in sorted(agree.items())},
        "disagreement_examples": disagreements,
        "examples_new_superconductors": [C.formula_text(usable[k]) for k in sorted(fresh) if comps[k]["label"] == 1][:30],
        "examples_new_nulls": [C.formula_text(usable[k]) for k in sorted(fresh) if comps[k]["label"] == 0][:30],
    }
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not k.startswith(("examples", "disagreement"))}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
