#!/usr/bin/env python3
"""Does better negative evidence help? Current -> audited -> audited plus new.

Every condition trains on the same positives but a different set of negatives,
and is scored on one frozen set of test rows, so the only thing that changes
is which Tc = 0 labels the model learns from:

  current          the negatives the project used before evidence tiers: every
                   Tc = 0 from the 2018 SuperCon export / 3DSC (any tier)
  audited          current minus tier D (near matches, wide-gap insulators)
  audited_plus_new audited plus SuperCon-MTG negatives in tiers B/C
  plus_hosono      audited_plus_new plus Hosono's unsuccessful-search zeros (H)

Folds: StratifiedGroupKFold over element systems of all labelled rows, 5 x 3
shuffles. Test rows are the frozen "current" labelled set; AUC is reported on
all of them and on the audited subset (tier D test rows removed), since tier D
labels are as doubtful in testing as in training.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "scripts", ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from benchmark_sc_models import _fit, model_factories, provenance  # noqa: E402

CONDITIONS = ["current", "audited", "audited_plus_new", "plus_hosono"]
LEGACY_SOURCES = {"supercon_stanev2018", "3dsc_mp"}


def _rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [r for r in csv.DictReader(handle)
                if r["known_status"] in ("superconductor", "tested_non_superconductor")]


def condition_mask(rows: list[dict[str, str]], condition: str) -> np.ndarray:
    keep = []
    for r in rows:
        if r["known_status"] == "superconductor":
            keep.append(True)
            continue
        tier = r.get("negative_evidence_tier", "")
        legacy = bool(set(r.get("known_sources", "").split()) & LEGACY_SOURCES)
        if condition == "current":
            keep.append(legacy)
        elif condition == "audited":
            keep.append(legacy and tier != "D_suspect")
        elif condition == "audited_plus_new":
            keep.append(tier in {"A_referenced_null", "B_corroborated_zero", "C_database_zero"})
        elif condition == "plus_hosono":
            keep.append(tier in {"A_referenced_null", "B_corroborated_zero", "C_database_zero",
                                 "H_unsuccessful_search"})
        else:
            raise ValueError(condition)
    return np.asarray(keep)


def run(candidates: Path, out_json: Path, *, models: list[str], repeats: int = 3) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    from train_mp_sc_classifier import feature_matrix

    rows = _rows(candidates)
    with gzip.open(candidates, "rt", encoding="utf-8") as handle:
        systems = sorted({r["crystal_system"] for r in csv.DictReader(handle)})
    X, _ = feature_matrix(rows, systems)
    y = np.asarray([int(r["known_status"] == "superconductor") for r in rows])
    groups = np.asarray(["-".join(sorted(r["elements"].split())) for r in rows])
    masks = {c: condition_mask(rows, c) for c in CONDITIONS}
    test_all = masks["current"]
    test_audited = masks["audited"]
    aux = np.asarray([float(r["known_tc_max_k"]) if r["known_tc_max_k"] else np.nan for r in rows])

    results: dict[str, Any] = {m: {c: {"all": [], "audited_subset": []} for c in CONDITIONS} for m in models}
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=3000 + repeat)
        preds = {m: {c: np.full(len(rows), np.nan) for c in CONDITIONS} for m in models}
        for train, test in splitter.split(X, y, groups):
            test_rows = test[test_all[test]]
            for condition in CONDITIONS:
                idx = train[masks[condition][train]]
                for m in models:
                    model = model_factories(7 + repeat)[m]()
                    _fit(model, m, X[idx], y[idx], groups[idx], aux[idx])
                    preds[m][condition][test_rows] = model.predict_proba(X[test_rows])[:, 1]
        for m in models:
            for condition in CONDITIONS:
                p = preds[m][condition]
                results[m][condition]["all"].append(float(roc_auc_score(y[test_all], p[test_all])))
                results[m][condition]["audited_subset"].append(
                    float(roc_auc_score(y[test_audited], p[test_audited])))
        print(f"repeat {repeat}: " + "; ".join(
            f"{m} " + " ".join(f"{c}={results[m][c]['all'][-1]:.3f}" for c in CONDITIONS) for m in models), flush=True)

    summary: dict[str, Any] = {}
    for m in models:
        base = {k: np.asarray(v) for k, v in results[m]["current"].items()}
        summary[m] = {}
        for c in CONDITIONS:
            summary[m][c] = {}
            for k, v in results[m][c].items():
                diff = np.asarray(v) - base[k]
                summary[m][c][k] = {"auc_mean": float(np.mean(v)), "auc_per_repeat": v,
                                    "paired_gain_vs_current": float(diff.mean()), "wins": int((diff > 0).sum())}
    report = {
        "labelled_rows": len(rows), "positives": int(y.sum()),
        "negatives_per_condition": {c: int((masks[c] & (y == 0)).sum()) for c in CONDITIONS},
        "test_rows": {"all": int(test_all.sum()), "audited_subset": int(test_audited.sum())},
        "split": f"StratifiedGroupKFold over element systems, 5 folds x {repeats} shuffles",
        "results": summary,
        "provenance": provenance(X, candidates),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, default=ROOT / "data/candidates/mp-candidates.csv.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/sc-negative-tiers.json")
    parser.add_argument("--models", nargs="*", default=["gradient_boosting", "random_forest"])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    report = run(args.candidates, args.output, models=args.models, repeats=args.repeats)
    print(json.dumps({"negatives_per_condition": report["negatives_per_condition"],
                      "test_rows": report["test_rows"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
