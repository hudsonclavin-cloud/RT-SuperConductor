#!/usr/bin/env python3
"""Do more labels help? Train with extra labels, test on the same core materials.

Core labels: the 2,244 labelled materials in the stable snapshot
(data/candidates/mp-candidates.csv.gz). Extra labels: materials from a wider
Materials Project window (energy above hull <= 0.1 eV/atom, <= 40 sites)
that also match a SuperCon/3DSC report (scripts/fetch_mp_snapshot.py
--hull-max 0.1 --max-sites 40, then build_mp_candidates.py).

Outer folds are StratifiedGroupKFold over element systems of the union, so a
core test material never shares an element set with any training row, core or
extra. Every condition is scored on exactly the same core test rows:

  core_only           train on core labels
  core_plus_extra     train on core + extra labels
  core_plus_extra_w   same, each composition's polymorphs share a total
                      weight of 1 (one report is not counted once per
                      polymorph)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "scripts", ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from benchmark_sc_models import LABELS, _fit, model_factories, provenance  # noqa: E402

CONDITIONS = ["core_only", "core_plus_extra", "core_plus_extra_w"]


def _labelled(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [r for r in csv.DictReader(handle) if r["known_status"] in LABELS]


def run(core_path: Path, wide_path: Path, out_json: Path, *, models: list[str], repeats: int = 3) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    from build_mp_candidates import reduced
    from train_mp_sc_classifier import feature_matrix

    core = _labelled(core_path)
    core_ids = {r["mp_id"] for r in core}
    extra = [r for r in _labelled(wide_path) if r["mp_id"] not in core_ids]
    rows = core + extra
    systems = sorted({r["crystal_system"] for r in rows})
    X, _ = feature_matrix(rows, systems)
    y = np.asarray([LABELS[r["known_status"]] for r in rows])
    groups = np.asarray(["-".join(sorted(r["elements"].split())) for r in rows])
    is_core = np.asarray([i < len(core) for i in range(len(rows))])
    compositions = [reduced(r["formula"]) for r in rows]
    counts = Counter(compositions)
    weights = np.asarray([1.0 / counts[c] for c in compositions])
    aux = np.asarray([float(r["known_tc_max_k"]) if r["known_tc_max_k"] else np.nan for r in rows])

    results: dict[str, Any] = {m: {c: [] for c in CONDITIONS} for m in models}
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2000 + repeat)
        preds = {m: {c: np.full(len(rows), np.nan) for c in CONDITIONS} for m in models}
        for train, test in splitter.split(X, y, groups):
            test_core = test[is_core[test]]
            for condition in CONDITIONS:
                idx = train[is_core[train]] if condition == "core_only" else train
                w = weights[idx] if condition.endswith("_w") else None
                for m in models:
                    model = model_factories(7 + repeat)[m]()
                    _fit(model, m, X[idx], y[idx], groups[idx], aux[idx], weight=w)
                    preds[m][condition][test_core] = model.predict_proba(X[test_core])[:, 1]
        for m in models:
            for condition in CONDITIONS:
                results[m][condition].append(float(roc_auc_score(y[is_core], preds[m][condition][is_core])))
        print(f"repeat {repeat}: " + "; ".join(f"{m} " + " ".join(f"{c}={results[m][c][-1]:.3f}" for c in CONDITIONS)
                                               for m in models), flush=True)
    summary = {}
    for m in models:
        base = np.asarray(results[m]["core_only"])
        summary[m] = {c: {"auc_mean": float(np.mean(v)), "auc_per_repeat": v,
                          "paired_gain_vs_core_only": float(np.mean(np.asarray(v) - base)),
                          "wins_vs_core_only": int(np.sum(np.asarray(v) > base))}
                      for c, v in results[m].items()}
    report = {
        "core_labels": len(core), "extra_labels": len(extra),
        "extra_new_compositions": len({reduced(r["formula"]) for r in extra}
                                      - {reduced(r["formula"]) for r in core}),
        "test_rows": "core labels only, identical across conditions",
        "split": f"StratifiedGroupKFold over element systems of core+extra, 5 folds x {repeats} shuffles",
        "results": summary,
        "provenance": provenance(X, core_path),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--core", type=Path, default=ROOT / "data/candidates/mp-candidates.csv.gz")
    parser.add_argument("--wide", type=Path, default=ROOT / "data/candidates/mp-wide-labelled.csv.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/sc-label-expansion.json")
    parser.add_argument("--models", nargs="*", default=["gradient_boosting", "random_forest", "torch_mlp"])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    run(args.core, args.wide, args.output, models=args.models, repeats=args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
