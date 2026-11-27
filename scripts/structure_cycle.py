#!/usr/bin/env python3
"""Do crystal-structure features improve the ranking? Scored once (plan: docs/structure-cycle.md).

    python scripts/structure_cycle.py       # refuses if docs/structure-cycle.json exists

Models A (cycle 6's 155 formula features) and B (the same plus space-group number, crystal system and volume per atom,
missing where there is no structure) are trained in the same run with cycle 6's recipe: boosting settings, seed offset 0,
the 851 NIMS added rows, the same folds and early-stopping groups. Locked rows are neither trained on nor scored.
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import high_tc_target as H  # noqa: E402
import run_learning_cycle as L  # noqa: E402
import scratchnet_common as C  # noqa: E402
from scratchnet_common import T  # noqa: E402

OUT = ROOT / "docs/structure-cycle.json"
FEATURES = ROOT / "data/processed/structures/features.csv.gz"
SYSTEMS = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"]


def structure_columns(keys: list[str]) -> np.ndarray:
    """Nine columns per composition: space-group number, seven crystal-system indicators, volume per atom; nan if none."""
    table = {r["composition"]: r for r in csv.DictReader(gzip.open(FEATURES, "rt", newline="", encoding="utf-8"))}
    out = np.full((len(keys), 9), np.nan)
    for i, key in enumerate(keys):
        r = table.get(key)
        if r and r["sg_number"]:
            out[i, 0] = float(r["sg_number"])
            out[i, 1:8] = [1.0 if r["crystal_system"] == s else 0.0 for s in SYSTEMS]
            out[i, 8] = float(r["volume_per_atom"])
    return out


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: this test is scored once")
    ds = C.load_dataset()
    formulas = [C.formula_text(C.fractions_of(row["composition"])) for row in ds.rows]
    X, _ = C.formula_features(formulas, cache=C.WORK / "formula-features.npz")
    S_table = structure_columns([row["composition"] for row in ds.rows])
    settings = L.boosting_settings()
    results = {}
    for name, X_use in (("A", X), ("B", np.hstack([X, S_table]))):
        extra = L.load_additions(H.ADDITIONS, ds)
        if name == "B":
            extra["X"] = np.hstack([extra["X"], np.full((len(extra["X"]), 9), np.nan)])
        extra["score"] = False
        dev, pred, _, info = L.cross_validate(ds, X_use, settings, seed_offset=0, extra=extra)
        results[name] = (dev, pred, info)
    dev = results["A"][0]
    if not np.array_equal(dev, results["B"][0]):
        raise SystemExit("the two runs scored different rows")
    a, b = results["A"][1], results["B"][1]
    has = ~np.isnan(S_table[dev, 0])
    labelled = np.isfinite(ds.arrays.label[dev])
    non_cuprate = ds.family[dev] != "Cuprates"
    e1_rows, e2_rows = non_cuprate & labelled & has, non_cuprate & labelled
    e1 = C.paired_differences(ds, dev, b, a, e1_rows, resamples=1000)["auc"]
    e2 = C.paired_differences(ds, dev, b, a, e2_rows, resamples=1000)["auc"]
    y10 = H.high_tc_label(ds.arrays.label, ds.arrays.tc_k)
    ds10 = __import__("dataclasses").replace(ds, arrays=__import__("dataclasses").replace(ds.arrays, label=y10))
    e3_rows = non_cuprate & np.isfinite(y10[dev]) & has
    e3 = C.paired_differences(ds10, dev, (b[0] * b[1], b[1]), (a[0] * a[1], a[1]), e3_rows, resamples=1000)["auc"]

    def auc(rows, pred, labels):
        return float(T.roc_auc(labels[dev][rows], pred[0][rows]))
    adopt = bool(e1["median"] >= 0.005 and e1["low"] > 0 and e2["median"] >= -0.005)
    report = {"plan": "docs/structure-cycle.md", "data_sha256": ds.sha256, "features": str(FEATURES.relative_to(ROOT)),
              "rows": {"E1 (non-cuprate, labelled, with a structure)": int(e1_rows.sum()),
                       "E2 (non-cuprate, labelled)": int(e2_rows.sum()), "E3 (Tc >= 10 K, with a structure)": int(e3_rows.sum())},
              "auc": {"E1": {"A": auc(e1_rows, a, ds.arrays.label), "B": auc(e1_rows, b, ds.arrays.label)},
                      "E2": {"A": auc(e2_rows, a, ds.arrays.label), "B": auc(e2_rows, b, ds.arrays.label)},
                      "E3 (expected Tc, Tc >= 10 K)": {"A": float(T.roc_auc(y10[dev][e3_rows], (a[0] * a[1])[e3_rows])),
                                                         "B": float(T.roc_auc(y10[dev][e3_rows], (b[0] * b[1])[e3_rows]))}},
              "B_minus_A": {"E1": e1, "E2": e2, "E3": e3},
              "cross_validation": {"A": results["A"][2], "B": results["B"][2]},
              "reading": {"adopt_structure_features": adopt,
                          "what_follows": "cycle 8 with structure features" if adopt else "no change"}}
    OUT.write_text(json.dumps(L.rounded(report), indent=1) + "\n", encoding="utf-8")
    print(json.dumps(L.rounded({k: report[k] for k in ("rows", "auc", "B_minus_A", "reading")}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
