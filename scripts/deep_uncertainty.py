#!/usr/bin/env python3
"""Audit held-out forest spread and independently calibrate expected absolute error.

OOF errors are used only for descriptive plots and extrapolation checks. The
calibrator uses a separately fitted model and three disjoint group partitions;
it never trains on pooled OOF errors from models that saw its test labels.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from deep_ablation import FAMILIES, load
from train_physics_targets import BOND_FEATURE_NAMES, _make_matrix, allen_dynes_tc_k
from train_pressure_aware import elemental_system_signature

RF_BOUND_PROOF = (
    "Each regression-tree leaf is a nonnegative weighted mean of training labels. "
    "A forest is a nonnegative mean of those leaf predictions, so its output stays "
    "within the training-label range. This does not apply to a neural head or to "
    "a nonlinear Allen-Dynes transformation of separately predicted targets."
)
CALIBRATION_NOTE = (
    "Approximately 60/20/20 by group (actual row/group counts are reported): fit the model only on fit groups, fit "
    "isotonic expected absolute error only on calibration groups, and score once "
    "on untouched test groups. Shared reference MP IDs are purged from fit versus "
    "calibration/test and calibration versus test. This estimates error magnitude, "
    "not interval coverage or a probability of superconductivity. It is a separate "
    "model-training-budget experiment, not a calibration claim for the final "
    "full-data production model."
)


def bins(sd, errors):
    sd, errors = np.asarray(sd, dtype=float), np.asarray(errors, dtype=float)
    if sd.shape != errors.shape or sd.ndim != 1 or not len(sd):
        raise ValueError("Spread and error must be equally sized, nonempty vectors")
    if not np.isfinite(sd).all() or not np.isfinite(errors).all() or (sd < 0).any() or (errors < 0).any():
        raise ValueError("Spread/error diagnostics must be finite and nonnegative")
    order = np.argsort(sd, kind="stable")
    return [
        {"n": len(ids), "mean_sd": float(sd[ids].mean()), "observed_mae": float(errors[ids].mean())}
        for ids in np.array_split(order, min(8, len(sd))) if len(ids)
    ]


def independent_partitions(groups, reference_ids=None, seed=2026):
    """Return disjoint group partitions, with reference-structure embargoes."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(set(groups)) < 3:
        raise ValueError("At least three distinct groups are required")
    idx = np.arange(len(groups))
    fit_cal, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(idx, groups=groups))
    fit_local, cal_local = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed + 1).split(fit_cal, groups=groups[fit_cal]))
    fit, cal = fit_cal[fit_local], fit_cal[cal_local]
    before_fit, before_cal = len(fit), len(cal)
    if reference_ids is not None:
        references = np.asarray(reference_ids, dtype=object)
        if len(references) != len(groups):
            raise ValueError("Reference IDs must align with rows")
        held_test = {references[i] for i in test if references[i] not in (None, "")}
        held_cal_test = held_test | {references[i] for i in cal if references[i] not in (None, "")}
        fit = np.array([i for i in fit if references[i] in (None, "") or references[i] not in held_cal_test], dtype=int)
        cal = np.array([i for i in cal if references[i] in (None, "") or references[i] not in held_test], dtype=int)
    meta = {
        "input_rows": len(groups), "input_groups": len(set(groups)),
        "fit_rows": len(fit), "calibration_rows": len(cal), "test_rows": len(test),
        "fit_groups": len(set(groups[fit])), "calibration_groups": len(set(groups[cal])), "test_groups": len(set(groups[test])),
        "fit_rows_purged_shared_reference": before_fit - len(fit),
        "calibration_rows_purged_shared_reference": before_cal - len(cal),
        "fit_indices_sha256": hashlib.sha256(fit.tobytes()).hexdigest(),
        "calibration_indices_sha256": hashlib.sha256(cal.tobytes()).hexdigest(),
        "test_indices_sha256": hashlib.sha256(test.tobytes()).hexdigest(),
        "seed": seed,
    }
    return fit, cal, test, meta


def physics_point_and_sd(lambda_trees, log_omega_trees):
    """Point Tc uses mean lambda and geometric-mean omega; tree pairing is heuristic."""
    lt, wt = np.asarray(lambda_trees), np.asarray(log_omega_trees)
    if lt.shape != wt.shape or lt.ndim != 2:
        raise ValueError("Target-tree arrays must have equal tree-by-row shapes")
    point = np.array([allen_dynes_tc_k(a, b) for a, b in zip(lt.mean(axis=0), np.exp(wt.mean(axis=0)))])
    tree_tc = np.array([[allen_dynes_tc_k(a, b) for a, b in zip(lam, np.exp(wlog))] for lam, wlog in zip(lt, wt)])
    return point, tree_tc.std(axis=0)


def independent_calibration(X, y, groups, reference_ids=None, epc_targets=None, trees=None):
    """Fit fresh forests, then a fresh calibrator, and evaluate untouched groups."""
    try:
        fit, cal, test, result = independent_partitions(groups, reference_ids)
    except ValueError as error:
        return {"status": "blocked_partition", "reason": str(error), "note": CALIBRATION_NOTE}
    result.update(note=CALIBRATION_NOTE, status="blocked_insufficient_partition_rows")
    if min(len(fit), len(cal), len(test)) < 5:
        result["reason"] = "Fewer than five rows remain in at least one partition after embargo"
        return result
    y = np.asarray(y, dtype=float)
    physics = epc_targets is not None
    imputer = SimpleImputer(strategy="median" if physics else "mean", add_indicator=physics, keep_empty_features=not physics)
    xt = imputer.fit_transform(X[fit])
    predict_idx = np.r_[cal, test]
    xv = imputer.transform(X[predict_idx])
    if physics:
        lam, omega = (np.asarray(v, dtype=float) for v in epc_targets)
        target_arrays = [(lam, 3026), (np.log(omega), 4026)]
        tree_arrays = []
        for target, seed in target_arrays:
            model = RandomForestRegressor(n_estimators=trees or 180, min_samples_leaf=2, random_state=seed, n_jobs=2).fit(xt, target[fit])
            tree_arrays.append(np.array([tree.predict(xv) for tree in model.estimators_]))
        pred, sd = physics_point_and_sd(*tree_arrays)
        result["model"] = "Separate RFs for lambda/log(omega), 180 trees and leaf size 2 by default"
    else:
        model = RandomForestRegressor(n_estimators=trees or 150, random_state=4201, n_jobs=2).fit(xt, y[fit])
        tree_predictions = np.array([tree.predict(xv) for tree in model.estimators_])
        pred, sd = tree_predictions.mean(axis=0), tree_predictions.std(axis=0)
        result["model"] = "RF Tc regression, 150 trees and default leaf size by default"
    cut = len(cal)
    cal_error, test_error = np.abs(y[cal] - pred[:cut]), np.abs(y[test] - pred[cut:])
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(sd[:cut], cal_error)
    expected_error = calibrator.predict(sd[cut:])
    result.update(
        status="completed", model_fit_training_max_k=float(max(y[fit])),
        prediction_test_mae_k=float(test_error.mean()),
        sd_as_error_test_mae_k=float(mean_absolute_error(test_error, sd[cut:])),
        calibrated_expected_error_test_mae_k=float(mean_absolute_error(test_error, expected_error)),
        constant_calibration_mean_error_test_mae_k=float(mean_absolute_error(test_error, np.repeat(cal_error.mean(), len(test)))),
        predicted_expected_error_test_mean_k=float(expected_error.mean()),
        test_observed_mean_absolute_error_k=float(test_error.mean()),
        calibration_observed_mean_absolute_error_k=float(cal_error.mean()),
        calibration_bins=bins(sd[:cut], cal_error), test_bins=bins(sd[cut:], test_error),
        test_predictions_above_fit_training_max=int(np.sum(pred[cut:] > max(y[fit]) + 1e-8)),
    )
    return result


def validate_oof(rows, y, expected_indices):
    """Guard the legacy index-only OOF files against row loss or target mismatch."""
    idx = [row["index"] for row in rows]
    if len(set(idx)) != len(idx) or set(idx) != set(expected_indices):
        raise ValueError("OOF predictions do not cover the expected rows exactly once")
    for row in rows:
        if not np.isclose(row["actual"], y[row["index"]], rtol=0, atol=1e-8):
            raise ValueError("OOF row ordering or actual target differs from current input")
        if not np.isfinite([row["actual"], row["predicted"], row["sd"], row["training_max"]]).all() or row["sd"] < 0:
            raise ValueError("OOF diagnostic contains invalid numbers")
    return rows


def plot_bins(curve, path, label, xlabel="Mean tree-spread SD (K)"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([row["mean_sd"] for row in curve], [row["observed_mae"] for row in curve], "o-", label=label)
    extent = max(max(row["mean_sd"] for row in curve), max(row["observed_mae"] for row in curve), 1)
    ax.plot([0, extent], [0, extent], "--", color="grey", label="MAE = SD reference")
    ax.set(xlabel=xlabel, ylabel="Mean held-out absolute error (K)")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def physics_audit(rows, trees=180):
    if any(str(row.get("is_generated_candidate", "")).strip().lower() in ("true", "1", "1.0", "yes") or str(row.get("jid", "")).startswith("candidate-") for row in rows):
        raise ValueError("Generated candidates cannot serve as reference EPC labels")
    names = [key for key in rows[0] if key.startswith(("magpie__", "shell__"))] + ["lewis_dot_proxy_avg_outer_electrons", "lewis_dot_proxy_total_electrons_per_reduced_formula"] + BOND_FEATURE_NAMES
    X = _make_matrix(rows, names)
    lam = np.array([float(row["lambda"]) for row in rows])
    omega = np.array([float(row["omega_log_k"]) for row in rows])
    if not np.isfinite(lam).all() or not np.isfinite(omega).all() or (lam <= 0).any() or (omega <= 0).any():
        raise ValueError("Physics audit needs valid reference EPC targets, not invented defaults")
    tc = np.array([allen_dynes_tc_k(a, b) for a, b in zip(lam, omega)])
    groups = np.array([elemental_system_signature(row["formula"]) for row in rows])
    prediction, spread = np.zeros(len(rows)), np.zeros(len(rows))
    exceedances, folds = [], []
    for fold, (fit, test) in enumerate(GroupKFold(5).split(X, tc, groups), 1):
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        xt, xv = imputer.fit_transform(X[fit]), imputer.transform(X[test])
        arrays = []
        for target, seed in [(lam, 3026), (np.log(omega), 4026)]:
            model = RandomForestRegressor(n_estimators=trees, min_samples_leaf=2, random_state=seed + fold, n_jobs=2).fit(xt, target[fit])
            arrays.append(np.array([tree.predict(xv) for tree in model.estimators_]))
        pt, sd = physics_point_and_sd(*arrays)
        prediction[test], spread[test] = pt, sd
        maximum = float(max(tc[fit]))
        folds.append({"fold": fold, "training_max_k": maximum, "test_rows": len(test), "exceeded_count": int(sum(pt > maximum + 1e-8))})
        for index, value in zip(test, pt):
            if value > maximum + 1e-8:
                row = rows[index]
                exceedances.append({"jid": row["jid"], "formula": row["formula"], "predicted_tc_k": float(value), "reference_derived_tc_k": float(tc[index]), "source_reported_dft_tc_k": float(row["source_tc_k"]) if row.get("source_tc_k") else None, "training_max_k": maximum, "actual_derived_tc_exceeds_training_max": bool(tc[index] > maximum), "absolute_error_k": float(abs(value - tc[index]))})
        print("physics fold", fold, "above training maximum", folds[-1]["exceeded_count"], flush=True)
    error = abs(tc - prediction)
    return {
        "rows": len(rows), "groups": len(set(groups)), "feature_count": len(names), "mu_star": .1,
        "tc_mae_k": float(error.mean()), "bins": bins(spread, error),
        "count": len(exceedances), "fraction": len(exceedances) / len(rows),
        "folds": folds, "exceedances": exceedances,
        "real_above_training_max_in_exceedances": sum(row["actual_derived_tc_exceeds_training_max"] for row in exceedances),
        "point_estimator": "Allen-Dynes(mean lambda, geometric-mean omega_log)",
        "spread_note": "SD across independently fitted tree pairs is a heuristic, not a joint posterior",
        "label_note": "JARVIS labels are DFT EPC calculations, not experimental high-Tc evidence; target Tc is recomputed at mu*=0.1 and can differ from the source-reported Tc",
        "independent_calibration": independent_calibration(X, tc, groups, epc_targets=(lam, omega), trees=trees),
    }


def run(input_path, features, predictions, baseline_predictions, jarvis, output, plot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records, X, names, extra = load(input_path, features)
    full = np.column_stack([X] + [values for values, _ in extra.values()])
    y = np.array([record.tc_kelvin for record in records])
    families = np.array([record.chemical_family for record in records])
    systems = np.array([elemental_system_signature(record.formula) for record in records])
    mp_ids = np.array([record.mp_id for record in records], dtype=object)
    global_rows = [row for row in json.loads(predictions.read_text()) if row["scope"] == "cross-family"]
    validate_oof(global_rows, y, range(len(y)))
    curve = bins([row["sd"] for row in global_rows], [abs(row["actual"] - row["predicted"]) for row in global_rows])
    plot_bins(curve, plot, "All-feature cross-family OOF")
    report = {
        "rows": len(records), "global_oof_bins": curve, "calibration_note": CALIBRATION_NOTE,
        "condition_limit": "3DSC measurement conditions are not individually verified here; these pressure-agnostic errors do not establish ambient-pressure performance",
        "rf_bound_proof": RF_BOUND_PROOF,
        "family_training_maxima_k": {family: float(max(y[families == family])) for family in sorted(set(families))},
        "rf_oof_predictions_above_fold_training_max": sum(row["predicted"] > row["training_max"] + 1e-8 for row in global_rows),
        "independent_calibration": {}, "within_family_baseline_oof_bins": {},
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (input_path, features, predictions, baseline_predictions, jarvis)},
        "software": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "matplotlib")},
        "units": {"tc": "K", "tree_spread": "K", "absolute_error": "K", "lambda": "dimensionless", "omega_log": "K"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    report["independent_calibration"]["global_all_cross_family"] = independent_calibration(full, y, families, mp_ids)
    print("independent global calibration", report["independent_calibration"]["global_all_cross_family"], flush=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    baseline_rows = json.loads(baseline_predictions.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for family, ax in zip(FAMILIES, axes.flat):
        idx = np.where(families == family)[0]
        rows = [row for row in baseline_rows if row["scope"] == family]
        if len(idx) < 100:
            ax.set_title(family + " — blocked (<100 rows)")
            continue
        validate_oof(rows, y, idx)
        family_curve = bins([row["sd"] for row in rows], [abs(row["actual"] - row["predicted"]) for row in rows])
        report["within_family_baseline_oof_bins"][family] = family_curve
        report["independent_calibration"][family] = independent_calibration(X[idx], y[idx], systems[idx], mp_ids[idx])
        ax.plot([r["mean_sd"] for r in family_curve], [r["observed_mae"] for r in family_curve], "o-")
        extent = max(max(r["mean_sd"] for r in family_curve), max(r["observed_mae"] for r in family_curve), 1)
        ax.plot([0, extent], [0, extent], "--", color="grey")
        ax.set(title=family, xlabel="Mean tree SD (K)", ylabel="OOF MAE (K)")
        print("independent calibration", family, report["independent_calibration"][family], flush=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fig.tight_layout()
    fig.savefig(plot.with_stem(plot.stem + "-within-families"), dpi=160)
    plt.close(fig)
    with jarvis.open() as handle:
        physics_rows = list(csv.DictReader(handle))
    report["imputation_protocol"] = "Mean for empirical deep baseline; median plus missingness indicators for EPC, matching the published-dataset benchmark pipeline"
    report["physics_extrapolation"] = physics_audit(physics_rows)
    plot_bins(report["physics_extrapolation"]["bins"], plot.with_stem(plot.stem + "-physics"), "DFT-EPC OOF error", "Paired-tree physics Tc SD (K)")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/training/harmonized_supercon.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/structure_features_full.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("data/processed/deep-ablation-35/all-predictions.json"))
    parser.add_argument("--baseline-predictions", type=Path, default=Path("data/processed/deep-ablation-35/baseline-predictions.json"))
    parser.add_argument("--jarvis", type=Path, default=Path("data/processed/physics-target/jarvis-supercond-characteristics.csv"))
    parser.add_argument("--output", type=Path, default=Path("docs/deep-uncertainty.json"))
    parser.add_argument("--plot", type=Path, default=Path("docs/uncertainty-calibration.png"))
    args = parser.parse_args()
    run(args.input, args.features, args.predictions, args.baseline_predictions, args.jarvis, args.output, args.plot)
