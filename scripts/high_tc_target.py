#!/usr/bin/env python3
"""A "Tc of at least 10 K" target for the ranking, scored once on the development rows (plan: docs/high-tc-target.md).

    python scripts/high_tc_target.py        # refuses if docs/high-tc-target.json exists

The new model is cycle 6's recipe (boosting settings, seed offset 0, the 851 NIMS added rows, the same folds and early-
stopping groups) with one change: the classifier's label is 1 for a superconductor with Tc >= 10 K and 0 otherwise
(nulls and lower-Tc superconductors); a superconductor with no Tc is left out. The incumbent is the ranking of record's
out-of-fold blend of cycle 6 and TabICLv2 (the committed out-of-fold files), as a probability and as expected Tc.
Locked rows are neither trained on nor scored.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_learning_cycle as L  # noqa: E402
import scratchnet_common as C  # noqa: E402
from scratchnet_common import T  # noqa: E402

OUT = ROOT / "docs/high-tc-target.json"
ADDITIONS = ROOT / "data/processed/learning-cycles/nims-additions.csv.gz"
CYCLE6_OOF = ROOT / "data/benchmarks/learning-cycles/cycle6-oof.npz"
TABICL_OOF = ROOT / "data/benchmarks/local-gpu/fm-tabicl-20261123/oof.npz"
THRESHOLD, SECOND = 10.0, 20.0
RULE = ("New model minus the incumbent's better score (probability or expected Tc), non-cuprate AUC for Tc >= 10 K: "
        "median gain >= +0.01 with the 95% interval above zero -> a second, high-Tc ranking for the room-temperature "
        "track. Anything else -> the incumbent's better score serves. The ranking of record is unchanged either way.")


def high_tc_label(label: np.ndarray, tc: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """1 for a superconductor at or above the threshold, 0 for a null or a lower-Tc superconductor, nan if unknown."""
    out = np.where(np.isfinite(label), 0.0, np.nan)
    sc = label == 1
    out[sc & (tc >= threshold)] = 1.0
    out[sc & ~np.isfinite(tc)] = np.nan
    return out


def oof_by_key(path: Path) -> dict[str, tuple[float, float]]:
    with np.load(path, allow_pickle=False) as d:
        return {str(k): (float(p), float(t)) for k, p, t in zip(d["compositions"], d["p"], d["tc"])}


def incumbent(keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    record, tabicl = oof_by_key(CYCLE6_OOF), oof_by_key(TABICL_OOF)
    missing = [k for k in keys if k not in record or k not in tabicl]
    if missing:
        raise SystemExit(f"{len(missing)} development rows have no out-of-fold prediction, e.g. {missing[:3]}")
    r = (np.array([record[k][0] for k in keys]), np.array([record[k][1] for k in keys]))
    t = (np.array([tabicl[k][0] for k in keys]), np.array([tabicl[k][1] for k in keys]))
    return C.blend(t, r)


def scores_at(ds, dev, keep, scores: dict[str, tuple[np.ndarray, np.ndarray]], threshold: float) -> dict:
    y = high_tc_label(ds.arrays.label, ds.arrays.tc_k, threshold)[dev]
    mask = keep & np.isfinite(y)
    return {"rows": int(mask.sum()), "positives": int((y[mask] == 1).sum()),
            "auc": {name: float(T.roc_auc(y[mask], s[0][mask])) for name, s in scores.items()}}


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: this test is scored once")
    ds = C.load_dataset()
    y10 = high_tc_label(ds.arrays.label, ds.arrays.tc_k)
    ds10 = dataclasses.replace(ds, arrays=dataclasses.replace(ds.arrays, label=y10))
    formulas = [C.formula_text(C.fractions_of(row["composition"])) for row in ds.rows]
    X, _ = C.formula_features(formulas, cache=C.WORK / "formula-features.npz")
    extra = L.load_additions(ADDITIONS, ds)
    extra["label"] = high_tc_label(extra["label"], extra["tc_k"])
    extra["score"] = False
    dev, new, _, info = L.cross_validate(ds10, X, L.boosting_settings(), seed_offset=0, extra=extra)
    keys = [ds.rows[i]["composition"] for i in dev]
    p_inc, tc_inc = incumbent(keys)
    candidates = {"new (trained on Tc >= 10 K)": new, "incumbent p": (p_inc, tc_inc),
                  "incumbent expected Tc": (p_inc * tc_inc, tc_inc)}
    keep = (ds.family[dev] != "Cuprates") & np.isfinite(y10[dev])
    main10 = scores_at(ds, dev, keep, candidates, THRESHOLD)
    better = max(("incumbent p", "incumbent expected Tc"), key=lambda n: main10["auc"][n])
    paired = C.paired_differences(ds10, dev, new, candidates[better], keep, resamples=1000)["auc"]
    adopt = bool(paired["median"] >= 0.01 and paired["low"] > 0)
    report = {"plan": "docs/high-tc-target.md", "data_sha256": ds.sha256, "added_rows": int(len(extra["label"])),
              "cross_validation": info, "rule": RULE,
              "not cuprates, Tc >= 10 K": main10,
              "not cuprates, Tc >= 20 K (second reading; models as trained for 10 K)":
                  scores_at(ds, dev, (ds.family[dev] != "Cuprates"), candidates, SECOND),
              "all development rows, Tc >= 10 K": scores_at(ds, dev, np.ones(len(dev), bool), candidates, THRESHOLD),
              "incumbent_better_score": better, "new_minus_better_incumbent": paired,
              "reading": {"adopt_high_tc_ranking": adopt,
                          "what_follows": "a second, high-Tc ranking for the room-temperature track" if adopt
                          else "the incumbent's better score serves the high-Tc question"},
              "disclosure": "The table differs from cycle 6's by one corrected train-only cuprate row (YBa2Cu3O6.32)."}
    OUT.write_text(json.dumps(L.rounded(report), indent=1) + "\n", encoding="utf-8")
    print(json.dumps(L.rounded({k: report[k] for k in ("not cuprates, Tc >= 10 K", "incumbent_better_score",
                                                        "new_minus_better_incumbent", "reading")}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
