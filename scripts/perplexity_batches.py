#!/usr/bin/env python3
"""Input files for the owner's Perplexity Computer tasks (prompts: docs/perplexity/README.md).

    python scripts/perplexity_batches.py

Writes two blinded CSVs under docs/perplexity/. Neither carries a label, a Tc or a model score, so the readers
cannot be steered, and neither holds a locked row:
* ``structures.csv``: compositions in the table with no 3DSC structure, limited to ordered non-cuprate compounds
  (whole-number ratios with at most 20 atoms per formula unit), in batches of 100 in a fixed random order;
* ``label-reader-01.csv``: the label audit's strongest leads (two or more signals, and the closest probable
  duplicates), for an independent first reading under the owner's correction rule.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_mp_candidates as M  # noqa: E402
import build_scratchnet_data as B  # noqa: E402
import scratchnet_common as C  # noqa: E402

OUT = ROOT / "docs/perplexity"
AUDIT = ROOT / "data/processed/label-audit/label-audit.json"
BATCH = 100


def whole_number_scale(fractions: dict[str, float], largest: int = 20) -> int | None:
    """The smallest n with every n * fraction within 0.02 * n of a whole number of at least 1."""
    for n in range(1, largest + 1):
        if all(abs(v * n - round(v * n)) < 0.02 * n and round(v * n) >= 1 for v in fractions.values()):
            return n
    return None


def readable(fractions: dict[str, float]) -> str:
    """A formula as people write it: scaled so the most counts are whole numbers ('Ce0.18CuNd1.82O4')."""
    best, best_n = None, 1
    for n in range(1, 21):
        whole = sum(abs(v * n - round(v * n)) < 0.03 and round(v * n) >= 1 for v in fractions.values())
        if best is None or whole > best:
            best, best_n = whole, n
    parts = []
    for element, v in sorted(fractions.items()):
        count = v * best_n
        if abs(count - 1) < 0.03:
            text = ""
        elif count >= 1 and abs(count - round(count)) < 0.03:
            text = str(int(round(count)))
        else:                                            # small dopant amounts keep three decimals: 'Mn0.004'
            text = (f"{count:.3f}" if count < 0.1 else f"{count:.2f}").rstrip("0").rstrip(".")
        parts.append(element + text)
    return "".join(parts)


def structured_keys() -> set[str]:
    out = set()
    with open(M.THREEDSC, encoding="utf-8") as handle:
        handle.readline()
        for row in csv.DictReader(handle):
            fractions = M.fractions(row["formula_sc"]) if row["formula_sc"] else None
            if fractions:
                out.add(B.composition_key(fractions))
    return out


def structures(ds, have: set[str]) -> list[dict]:
    rows = []
    for row, split, family, label in zip(ds.rows, ds.split, ds.family, ds.arrays.label):
        if split == "lockbox" or not np.isfinite(label) or family == "Cuprates" or row["composition"] in have:
            continue
        fractions = C.fractions_of(row["composition"])
        n = whole_number_scale(fractions)
        if n is None:
            continue
        formula = "".join(e + ("" if round(v * n) == 1 else str(int(round(v * n)))) for e, v in sorted(fractions.items()))
        rows.append({"formula": formula, "composition_key": row["composition"]})
    rows.sort(key=lambda r: r["composition_key"])
    random.Random(0).shuffle(rows)
    return [{"batch": i // BATCH + 1, "id": f"S{i + 1:05d}", **r} for i, r in enumerate(rows)]


def leads(report: dict) -> list[dict]:
    keys = []
    for kind in ("possible false negatives", "possible false positives"):
        keys += [x["composition"] for x in report[kind] if x["signals"] >= 2]
    for d in sorted(report["probable duplicates"], key=lambda d: d["distance"])[:15]:
        keys += [d["first"], d["second"]]
    seen, rows = set(), []
    for key in keys:
        if key not in seen:
            seen.add(key)
            rows.append({"composition_key": key, "formula": readable(C.fractions_of(key))})
    rows.sort(key=lambda r: r["composition_key"])
    random.Random(1).shuffle(rows)                      # mixed order: the reader cannot tell which list a row came from
    return [{"id": f"L{i + 1:03d}", **r} for i, r in enumerate(rows)]


def write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ds = C.load_dataset()
    OUT.mkdir(parents=True, exist_ok=True)
    s = structures(ds, structured_keys())
    write(OUT / "structures.csv", s)
    locked = {row["composition"] for row, split in zip(ds.rows, ds.split) if split == "lockbox"}
    ell = leads(json.loads(AUDIT.read_text(encoding="utf-8")))
    if locked & {r["composition_key"] for r in ell} or locked & {r["composition_key"] for r in s}:
        raise SystemExit("a locked composition reached an output file; refusing")
    write(OUT / "label-reader-01.csv", ell)
    print(json.dumps({"structures": len(s), "structure_batches": s[-1]["batch"], "label_reader_rows": len(ell)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
