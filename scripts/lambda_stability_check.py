#!/usr/bin/env python3
"""Does training the lambda model only on dynamically stable JARVIS rows help?

JARVIS-SuperconDB flags 431 of the 1,057 rows the lambda / omega_log model trains on as dynamically unstable,
and their lambda reaches 18, which is unphysical. This script trains the model's random forests twice on the
same grouped folds - once on every row, as `scripts/train_physics_targets.py` does, and once on the stable
rows only - and scores both on the same held-out **stable** rows. It reads the local characteristics table,
so it needs no download.

    python scripts/lambda_stability_check.py --out docs/lambda-stability-check.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

import train_physics_targets as P

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data/training/physics-target-characteristics.csv"
VARIANTS = ("all rows", "stable rows only")


def feature_names(row: dict) -> list[str]:
    """The production model's inputs: composition features, the two Lewis proxies and the bond features."""
    names = [key for key in row if key.startswith("magpie__") or key.startswith("shell__")]
    return names + ["lewis_dot_proxy_avg_outer_electrons", "lewis_dot_proxy_total_electrons_per_reduced_formula"] \
        + P.BOND_FEATURE_NAMES


def compare(rows: list[dict], n_splits: int = 5, trees: int = 180, resamples: int = 1000, seed: int = 0) -> dict:
    names = feature_names(rows[0])
    X = P._make_matrix(rows, names)
    lam = np.array([float(r["lambda"]) for r in rows])
    wlog = np.array([float(r["omega_log_k"]) for r in rows])
    tc = np.array([float(r["allen_dynes_tc_mu0p1_k"]) for r in rows])
    stable = np.array([r["source_stability"] == "stable" for r in rows])
    groups = np.array([P._element_system(r["formula"]) for r in rows])
    pred = {v: {"lambda": np.full(len(rows), np.nan), "omega_log": np.full(len(rows), np.nan)} for v in VARIANTS}
    for fold, (train, test) in enumerate(GroupKFold(n_splits=n_splits).split(X, tc, groups), 1):
        for variant, keep in zip(VARIANTS, (np.ones(len(rows), bool), stable)):
            fit = train[keep[train]]
            imputer = SimpleImputer(strategy="median", add_indicator=True)
            x_fit, x_test = imputer.fit_transform(X[fit]), imputer.transform(X[test])
            for target, values, offset in (("lambda", lam, 3026), ("omega_log", wlog, 4026)):
                model = RandomForestRegressor(n_estimators=trees, min_samples_leaf=2, random_state=offset + fold, n_jobs=1)
                pred[variant][target][test] = np.exp(model.fit(x_fit, np.log(values[fit])).predict(x_test))
    tc_pred = {v: np.array([P.allen_dynes_tc_k(a, b) for a, b in zip(pred[v]["lambda"], pred[v]["omega_log"])])
               for v in VARIANTS}
    s = stable
    report: dict = {"rows": len(rows), "stable_rows": int(s.sum()), "scored_on": "held-out stable rows only",
                    "split": f"{n_splits}-fold GroupKFold by element system, the same folds for both",
                    "largest_lambda_unstable": float(lam[~s].max()) if (~s).any() else None,
                    "largest_lambda_stable": float(lam[s].max()),
                    "lambda_error_of_always_guessing_the_stable_median": float(np.mean(np.abs(lam[s] - np.median(lam[s]))))}
    for variant in VARIANTS:
        report[variant] = {
            "lambda_mae": float(np.mean(np.abs(pred[variant]["lambda"][s] - lam[s]))),
            "omega_log_mae_k": float(np.mean(np.abs(pred[variant]["omega_log"][s] - wlog[s]))),
            "tc_mae_k": float(np.mean(np.abs(tc_pred[variant][s] - tc[s]))),
            "lambda_spearman": float(spearmanr(pred[variant]["lambda"][s], lam[s]).correlation),
            "tc_spearman": float(spearmanr(tc_pred[variant][s], tc[s]).correlation)}
    names_g = sorted(set(groups[s].tolist()))
    members = [np.flatnonzero(s & (groups == g)) for g in names_g]
    rng = np.random.default_rng(seed)
    draws = {"lambda_mae": [], "tc_mae_k": [], "tc_spearman": []}
    for _ in range(resamples):
        pick = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        a, b = "stable rows only", "all rows"
        draws["lambda_mae"].append(np.mean(np.abs(pred[a]["lambda"][pick] - lam[pick]))
                                   - np.mean(np.abs(pred[b]["lambda"][pick] - lam[pick])))
        draws["tc_mae_k"].append(np.mean(np.abs(tc_pred[a][pick] - tc[pick])) - np.mean(np.abs(tc_pred[b][pick] - tc[pick])))
        draws["tc_spearman"].append(spearmanr(tc_pred[a][pick], tc[pick]).correlation
                                    - spearmanr(tc_pred[b][pick], tc[pick]).correlation)
    report["stable_only_minus_all"] = {
        key: dict(zip(("low", "median", "high"), (float(v) for v in np.percentile(values, [2.5, 50, 97.5]))))
        for key, values in draws.items()}
    report["stable_only_minus_all"]["resampled"] = f"{len(names_g)} element systems, {resamples} times"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with open(args.table, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    report = compare(rows)
    report["table"] = args.table.name
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
