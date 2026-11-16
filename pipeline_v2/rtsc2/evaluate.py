"""Benchmark on the repository's standard folds.

Splits are StratifiedGroupKFold by element system, five folds, seeds
``1000 + repeat``: exactly what ``scripts/benchmark_sc_models.py`` uses. The
out-of-fold scores the legacy benchmark saved can therefore be compared row
for row, and paired differences are reported per shuffle.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import ROOT, Dataset
from .models import registry

LEGACY_OOF = ROOT / "data/candidates/sc-benchmark-oof.csv.gz"


def calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins."""
    order = np.argsort(p)
    return float(sum(len(chunk) / len(y) * abs(y[chunk].mean() - p[chunk].mean())
                     for chunk in np.array_split(order, bins) if len(chunk)))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    top = np.argsort(-p)[: max(1, len(p) // 10)]
    return {"auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "calibration_error": calibration_error(y, p),
            "top_decile_share": float(y[top].mean())}


def folds(dataset: Dataset, repeat: int, n_splits: int = 5):
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=1000 + repeat)
    return list(splitter.split(dataset.X, dataset.label.astype(int), dataset.groups))


def out_of_fold(dataset: Dataset, name: str, repeat: int, n_splits: int = 5) -> tuple[np.ndarray, float]:
    """Out-of-fold probability for every labelled row, and the fit time in seconds."""
    y = dataset.label.astype(int)
    scores, seconds = np.zeros(len(y)), 0.0
    for train, test in folds(dataset, repeat, n_splits):
        model = registry(seed=7 + repeat)[name]()
        start = time.perf_counter()
        model.fit(dataset.X[train], y[train], groups=dataset.groups[train], tc=dataset.tc[train],
                  tokens=(dataset.el_index[train], dataset.el_fraction[train]))
        seconds += time.perf_counter() - start
        scores[test] = model.predict(dataset.X[test], tokens=(dataset.el_index[test], dataset.el_fraction[test]))[0]
    return scores, seconds


def legacy_scores(dataset: Dataset, path: Path = LEGACY_OOF) -> dict[str, list[np.ndarray]]:
    """Legacy out-of-fold scores aligned to ``dataset`` rows: model -> one array per repeat."""
    if not path.is_file():
        return {}
    table = pd.read_csv(path)
    aligned: dict[str, list[np.ndarray]] = {}
    for (model, repeat), part in table.groupby(["model", "repeat"]):
        series = part.set_index("mp_id")["score"].reindex(dataset.ids)
        if series.isna().any():
            return {}  # saved for a different label set; not comparable
        aligned.setdefault(model, []).append(series.to_numpy())
    return aligned


def benchmark(dataset: Dataset, names: list[str], repeats: int = 3) -> dict[str, Any]:
    y = dataset.label.astype(int)
    report: dict[str, Any] = {"rows": int(len(y)), "positives": int(y.sum()),
                              "element_systems": int(len(set(dataset.groups))), "features": dataset.X.shape[1],
                              "split": f"StratifiedGroupKFold by element system, 5 folds x {repeats} shuffles",
                              "models": {}}
    scores: dict[str, list[np.ndarray]] = {}
    for name in names:
        per_repeat, seconds = [], 0.0
        scores[name] = []
        for repeat in range(repeats):
            p, spent = out_of_fold(dataset, name, repeat)
            scores[name].append(p)
            per_repeat.append(metrics(y, p))
            seconds += spent
        report["models"][name] = _summary(per_repeat) | {"fit_seconds_total": round(seconds, 1), "source": "rtsc2"}
    for name, per in legacy_scores(dataset).items():
        scores[f"legacy:{name}"] = per[:repeats]
        report["models"][f"legacy:{name}"] = _summary([metrics(y, p) for p in per[:repeats]]) | {
            "source": "scores saved by scripts/benchmark_sc_models.py on the same folds"}
    reference = "legacy:gradient_boosting" if "legacy:gradient_boosting" in scores else names[0]
    base = np.asarray(report["models"][reference]["auc_per_repeat"])
    for name, entry in report["models"].items():
        diff = np.asarray(entry["auc_per_repeat"]) - base[: len(entry["auc_per_repeat"])]
        entry["paired_auc_gain"] = {"against": reference, "mean": float(diff.mean()), "min": float(diff.min()),
                                    "max": float(diff.max()), "wins": int((diff > 0).sum()), "of": int(len(diff))}
    report["scores"] = scores
    return report


def _summary(per_repeat: list[dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {"auc_per_repeat": [m["auc"] for m in per_repeat]}
    for key in per_repeat[0]:
        out[f"{key}_mean"] = float(np.mean([m[key] for m in per_repeat]))
    out["auc_sd_across_repeats"] = float(np.std(out["auc_per_repeat"]))
    return out
