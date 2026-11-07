#!/usr/bin/env python3
"""Controlled benchmark: which model best separates superconductors?

Every model sees the same labelled Materials Project materials, the same
features and the same folds:

* labels: ``known_status`` superconductor (1) vs tested_non_superconductor (0)
  from data/candidates/mp-candidates.csv.gz
* folds: StratifiedGroupKFold over element systems, 5 folds, repeated with
  3 shuffles, so no test material shares its element set with training
* metrics: ROC AUC, average precision, and the superconductor share among
  each fold's top-scored 10%

Models: logistic regression (linear baseline), random forest (the current
classifier), extra trees, histogram gradient boosting, a multilayer
perceptron (a small neural network), and a soft-vote ensemble of the
forest, boosting and neural network. Only scikit-learn is used.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "scripts", ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

LABELS = {"superconductor": 1, "tested_non_superconductor": 0}


def labelled_rows(candidates: Path) -> list[dict[str, str]]:
    """Positives plus negatives whose evidence tier permits training (see build_mp_candidates)."""
    from build_mp_candidates import training_label

    with gzip.open(candidates, "rt", encoding="utf-8") as handle:
        return [r for r in csv.DictReader(handle) if training_label(r) is not None]


def model_factories(seed: int) -> dict[str, Callable[[], Any]]:
    from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                                  RandomForestClassifier, VotingClassifier)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def logistic():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.5, max_iter=4000))

    def forest():
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1))

    def extra_trees():
        return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(
            n_estimators=600, min_samples_leaf=2, max_features=0.3, class_weight="balanced",
            random_state=seed, n_jobs=-1))

    def boosting():
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                              min_samples_leaf=20, l2_regularization=1.0,
                                              early_stopping=True, validation_fraction=0.15,
                                              random_state=seed)

    def neural_net():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(256, 64), alpha=1e-2, learning_rate_init=1e-3, batch_size=64,
            max_iter=400, early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
            random_state=seed))

    def ensemble():
        return VotingClassifier([("rf", forest()), ("hgb", boosting()), ("mlp", neural_net())], voting="soft")

    factories = {"logistic_regression": logistic, "random_forest": forest, "extra_trees": extra_trees,
                 "gradient_boosting": boosting, "neural_network_mlp": neural_net, "ensemble_rf_hgb_mlp": ensemble}
    from torch_models import torch_available, torch_factories

    if torch_available():
        factories.update(torch_factories(seed))
    return factories


BLENDS = [("gradient_boosting", "torch_mlp"), ("gradient_boosting", "torch_mlp_multitask_tc"),
          ("gradient_boosting", "torch_mlp_tuned"), ("gradient_boosting", "random_forest")]


def _fit(model: Any, name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, aux: np.ndarray | None,
         weight: np.ndarray | None = None) -> Any:
    """Torch models also receive element-system groups (inner validation) and the Tc auxiliary target."""
    if name.startswith("torch_"):
        return model.fit(X, y, groups=groups, aux_target=aux, sample_weight=weight)
    if weight is not None:
        last = model.steps[-1][0] + "__sample_weight" if hasattr(model, "steps") else "sample_weight"
        return model.fit(X, y, **{last: weight})
    return model.fit(X, y)


def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, repeats: int = 3, folds: int = 5,
             models: list[str] | None = None, ids: list[str] | None = None,
             oof_rows: list[dict[str, Any]] | None = None, aux: np.ndarray | None = None) -> dict[str, Any]:
    """Outer StratifiedGroupKFold; preprocessing and early stopping use training folds only."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    results: dict[str, Any] = {}
    names = models or list(model_factories(0))
    oof_by_model: dict[str, list[np.ndarray]] = {}
    for name in names:
        aucs, aps, briers, tops, seconds = [], [], [], [], 0.0
        oof_by_model[name] = []
        for repeat in range(repeats):
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=1000 + repeat)
            pred = np.zeros(len(y))
            for fold, (train, test) in enumerate(splitter.split(X, y, groups)):
                model = model_factories(7 + repeat)[name]()
                start = time.perf_counter()
                _fit(model, name, X[train], y[train], groups[train], None if aux is None else aux[train])
                seconds += time.perf_counter() - start
                pred[test] = model.predict_proba(X[test])[:, 1]
                k = max(1, len(test) // 10)
                tops.append(float(y[test][np.argsort(-pred[test])[:k]].mean()))
                if oof_rows is not None:
                    oof_rows.extend({"mp_id": ids[i] if ids else i, "model": name, "repeat": repeat,
                                     "fold": fold, "label": int(y[i]), "score": round(float(pred[i]), 6)}
                                    for i in test)
            oof_by_model[name].append(pred.copy())
            aucs.append(roc_auc_score(y, pred))
            aps.append(average_precision_score(y, pred))
            briers.append(brier_score_loss(y, pred))
        results[name] = {
            "auc_per_repeat": [float(a) for a in aucs],
            "auc_mean": float(np.mean(aucs)), "auc_sd_across_repeats": float(np.std(aucs)),
            "average_precision_mean": float(np.mean(aps)), "brier_mean": float(np.mean(briers)),
            "top_decile_superconductor_share_mean": float(np.mean(tops)),
            "fit_seconds_total": round(seconds, 1),
        }
    for a, b in BLENDS:
        if a in oof_by_model and b in oof_by_model:
            aucs = [roc_auc_score(y, (pa + pb) / 2) for pa, pb in zip(oof_by_model[a], oof_by_model[b])]
            results[f"blend_{a}+{b}"] = {
                "auc_per_repeat": [float(v) for v in aucs], "auc_mean": float(np.mean(aucs)),
                "auc_sd_across_repeats": float(np.std(aucs)),
                "average_precision_mean": float(np.mean([average_precision_score(y, (pa + pb) / 2) for pa, pb in
                                                         zip(oof_by_model[a], oof_by_model[b])])),
                "brier_mean": float(np.mean([brier_score_loss(y, (pa + pb) / 2) for pa, pb in
                                             zip(oof_by_model[a], oof_by_model[b])])),
                "top_decile_superconductor_share_mean": float("nan"), "fit_seconds_total": 0.0,
                "note": "average of the two models' out-of-fold probabilities on identical folds",
            }
    if "random_forest" in results:
        reference = np.asarray(results["random_forest"]["auc_per_repeat"])
        for name, result in results.items():
            diff = np.asarray(result["auc_per_repeat"]) - reference
            result["paired_auc_gain_vs_random_forest"] = {
                "mean": float(diff.mean()), "min": float(diff.min()), "max": float(diff.max()),
                "wins": int((diff > 0).sum()), "of": int(len(diff))}
    return results


def provenance(X: np.ndarray, candidates: Path) -> dict[str, Any]:
    import hashlib
    import platform
    import subprocess

    import sklearn

    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - not a git checkout
        commit = None
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = None
    return {"git_commit": commit, "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "torch": torch_version,
            "numpy": np.__version__, "cpu_count": __import__("os").cpu_count(),
            "candidates_sha256": hashlib.sha256(candidates.read_bytes()).hexdigest(),
            "feature_matrix_sha256": hashlib.sha256(np.nan_to_num(X, nan=-9.99e9).tobytes()).hexdigest(),
            "seeds": {"outer_split": "1000 + repeat", "models": "7 + repeat"}}


def run(candidates: Path, out_json: Path, *, extra_features: Path | None = None, repeats: int = 3,
        models: list[str] | None = None, oof_out: Path | None = None) -> dict[str, Any]:
    from train_mp_sc_classifier import feature_matrix

    rows = labelled_rows(candidates)
    with gzip.open(candidates, "rt", encoding="utf-8") as handle:
        systems = sorted({r["crystal_system"] for r in csv.DictReader(handle)})
    X, names = feature_matrix(rows, systems)
    if extra_features is not None:
        from structure_features import load_feature_table

        X_extra, extra_names = load_feature_table(extra_features, [r["mp_id"] for r in rows])
        X, names = np.hstack([X, X_extra]), names + extra_names
    y = np.asarray([LABELS[r["known_status"]] for r in rows])
    oof: list[dict[str, Any]] = []
    aux = np.asarray([float(r["known_tc_max_k"]) if r["known_tc_max_k"] else np.nan for r in rows])
    groups = np.asarray(["-".join(sorted(r["elements"].split())) for r in rows])
    report = {
        "task": "superconductor (reported Tc > 0) vs tested non-superconductor, Materials Project materials",
        "rows": len(rows), "positives": int(y.sum()), "element_system_groups": int(len(set(groups))),
        "features": len(names),
        "split": f"StratifiedGroupKFold by element system, 5 folds x {repeats} shuffles",
        "models": evaluate(X, y, groups, repeats=repeats, models=models,
                           ids=[r["mp_id"] for r in rows], oof_rows=oof, aux=aux),
        "label_policy": ("positives: reported Tc > 0 (SuperCon/3DSC); negatives: explicit Tc = 0 reports in 3DSC. "
                         "Materials with no report are never used as negatives. One row per MP material."),
        "inner_validation": ("gradient boosting and the MLP early-stop on a random 15% of the training fold; "
                             "the outer test fold is never used for fitting, scaling, imputation or stopping"),
        "provenance": provenance(X, candidates),
    }
    best = max(report["models"], key=lambda m: report["models"][m]["auc_mean"])
    report["best_by_auc"] = best
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if oof_out is not None:
        import io

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["mp_id", "model", "repeat", "fold", "label", "score"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(oof)
        oof_out.parent.mkdir(parents=True, exist_ok=True)
        with oof_out.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
            gz.write(buffer.getvalue().encode("utf-8"))
        report["out_of_fold_predictions"] = str(oof_out)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, default=ROOT / "data/candidates/mp-candidates.csv.gz")
    parser.add_argument("--extra-features", type=Path, default=None,
                        help="per-material feature table (mp_id + numeric columns), e.g. structure features")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/sc-model-benchmark.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--oof", type=Path, default=ROOT / "data/candidates/sc-benchmark-oof.csv.gz",
                        help="where to save every out-of-fold prediction")
    args = parser.parse_args()
    report = run(args.candidates, args.output, extra_features=args.extra_features, repeats=args.repeats,
                 models=args.models, oof_out=args.oof)
    for name, m in sorted(report["models"].items(), key=lambda kv: -kv[1]["auc_mean"]):
        print(f"{name:22} AUC {m['auc_mean']:.3f} ± {m['auc_sd_across_repeats']:.3f}  "
              f"AP {m['average_precision_mean']:.3f}  Brier {m['brier_mean']:.3f}  "
              f"top10% {m['top_decile_superconductor_share_mean']:.2f}  "
              f"vs RF {m.get('paired_auc_gain_vs_random_forest', {}).get('mean', 0):+.3f}  "
              f"fit {m['fit_seconds_total']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
