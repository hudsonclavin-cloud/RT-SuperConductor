#!/usr/bin/env python3
"""Would more training rows of the same kind help, and by how much? Cycle 6's learner on fractions of its own rows.

    python scripts/learning_curve.py                 # writes docs/learning-curve.json (about 15 minutes on 4 cores)

For each fraction, each development fold's training rows are thinned at random (keeping the share of superconductors),
cycle 6's boosting classifier is fitted with its own early-stopping rows, and the held-out fold is scored. The folds'
predictions are pooled into one non-cuprate AUC, as in every comparison in this project. At the full fraction the
result must equal cycle 6's own development AUC, which checks the set-up. The slope per doubling of the rows is the
number that says what more of the same data is worth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import foundation_gpu_train as F
import local_gpu as G
import scratchnet_common as C
from scratchnet_common import ROOT, T

FRACTIONS = (0.125, 0.25, 0.5, 0.75, 1.0)
OUT = ROOT / "docs/learning-curve.json"


def not_cuprate(keys) -> np.ndarray:
    return np.array([C.family_of(set(C.fractions_of(str(k)))) != "Cuprates" for k in keys])


def fit_classifier(Xc, yc, Xv, yv, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(**F.HGB_COMMON, max_leaf_nodes=F.HGB_CLASSIFIER_LEAVES, random_state=seed)
    return model.fit(Xc, yc, X_val=Xv, y_val=yv)


def curve(bundle, fractions=FRACTIONS, seeds: int = 3, keep=None) -> dict:
    """Pooled held-out AUC at each fraction of the training rows; ``keep`` limits scoring to some rows (by bundle index)."""
    folds = [int(f) for f in bundle["folds"]]
    label = bundle["label"]
    out = {}
    for fraction in fractions:
        runs = []
        for repeat in range(1 if fraction == 1.0 else seeds):
            scores, truths, rows = [], [], 0
            for fold in folds:
                full = F.fold_arrays(bundle, fold, "hgb", cap=0, seed=0)
                cap = 0 if fraction == 1.0 else max(20, int(round(fraction * full["counts"]["classifier_rows"])))
                data = F.fold_arrays(bundle, fold, "hgb", cap=cap, seed=1000 * repeat + fold)
                model = fit_classifier(data["Xc"], data["yc"], data["Xc_val"], data["yc_val"], seed=fold)
                test = data["test"]
                if keep is not None:
                    test = test[keep[test]]
                test = test[np.isfinite(label[test])]
                scores.append(model.predict_proba(bundle["X"][test])[:, 1])
                truths.append(label[test])
                rows += data["counts"]["classifier_rows"]
            auc = T.roc_auc(np.concatenate(truths), np.concatenate(scores))
            runs.append({"auc": float(auc), "mean_training_rows": rows / len(folds)})
        aucs = [r["auc"] for r in runs]
        out[str(fraction)] = {"training_rows": round(float(np.mean([r["mean_training_rows"] for r in runs])), 1),
                              "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                              "aucs": aucs}
    return out


def slope_per_doubling(points: dict) -> float:
    """Least-squares slope of AUC against log2(training rows)."""
    x = np.log2([p["training_rows"] for p in points.values()])
    y = np.array([p["auc_mean"] for p in points.values()])
    return float(np.polyfit(x, y, 1)[0])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundle", type=Path, default=G.WORK / "bundle.npz")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    with np.load(args.bundle, allow_pickle=False) as saved:
        bundle = {k: saved[k] for k in saved.files}
    keep = not_cuprate(bundle["keys"])
    points = curve(bundle, seeds=args.seeds, keep=keep)
    upper = {k: v for k, v in points.items() if float(k) >= 0.25}
    report = {"what": "cycle 6's boosting classifier on fractions of its own training rows; pooled non-cuprate AUC "
                      "on the held-out development folds",
              "fractions": points,
              "slope_per_doubling_all_points": slope_per_doubling(points),
              "slope_per_doubling_from_a_quarter": slope_per_doubling(upper),
              "last_doubling": points["1.0"]["auc_mean"] - points["0.5"]["auc_mean"]}
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
