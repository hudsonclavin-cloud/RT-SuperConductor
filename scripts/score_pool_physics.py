#!/usr/bin/env python3
"""Score untested-pool candidates with the lambda/omega_log physics models.

This is the step that lets a new Hazel result change what gets calculated
next. It predicts the electron-phonon coupling lambda and omega_log for each
candidate, turns them into a Tc with the same McMillan/Allen-Dynes form EPW
prints, and ranks candidates for the EPW route by Tc minus tree spread.

Models come from one of two places:

* ``--model``: the joblib written by ``train_physics_targets.py`` (CI, where
  JARVIS and Materials Project data are available).
* otherwise: a formula-only model fitted here from the committed JARVIS table
  ``data/training/physics-target-characteristics.csv`` plus any reviewed Hazel
  records in ``data/training/hazel-epw-records.json``. That works offline.

Candidates are scored from formula features (Magpie, shell and Lewis-dot
proxies), computed exactly as the trainer computes them. Predictions are
screening hypotheses: random forests cannot exceed their training targets, the
tree spread is not a calibrated interval, and EPW applies only to
phonon-mediated (conventional) superconductors.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "scripts", ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from train_physics_targets import (  # noqa: E402
    _make_matrix, allen_dynes_tc_k, build_characteristics, load_extra_records,
)
from rt_superconductor.descriptors import descriptor_names, formula_descriptor_record  # noqa: E402
from chemistry_policy import radioactive_elements  # noqa: E402

DEFAULT_TABLE = ROOT / "data/training/physics-target-characteristics.csv"
DEFAULT_HAZEL = ROOT / "data/training/hazel-epw-records.json"
ELIGIBLE_LIST = ROOT / "docs/epw-eligible-shortlist.csv"
POOL_SHORTLIST = ROOT / "docs/untested-pool-shortlist.csv"
CONVENTIONAL_FAMILIES = {"Other"}
MAGNETIC_ELEMENTS = {"Fe", "Co", "Ni", "Mn", "Cr"}
LEWIS_FEATURES = ["lewis_dot_proxy_avg_outer_electrons", "lewis_dot_proxy_total_electrons_per_reduced_formula"]
SCORED_OUTPUT_COLUMNS = (
    "epw_rank", "mp_id", "formula", "chemical_family", "num_sites", "physics_lambda",
    "physics_lambda_tree_sd", "physics_omega_log_k", "physics_omega_log_tree_sd_k",
    "physics_tc_k", "physics_tc_tree_sd_k", "physics_conservative_tc_k", "rf_family_tc_k",
    "consensus_tc_k", "nearest_known_superconductor", "nearest_known_tc_k", "epw_eligible",
    "epw_ineligible_reasons", "magnetism_source", "mu_star", "known_status", "known_tc_max_k",
    "theoretical_only", "dft_verification",
)


def formula_feature_names(header: list[str]) -> list[str]:
    """The formula-only feature set, in the trainer's order, present in ``header``."""
    names = [h for h in header if h.startswith("magpie__")]
    names += [f"shell__{n}" for n in descriptor_names() if f"shell__{n}" in header]
    names += [n for n in LEWIS_FEATURES if n in header]
    return names


def formula_feature_rows(formulas: list[str]) -> list[dict[str, Any]]:
    """Magpie, shell and Lewis-dot features computed the same way as the trainer."""
    import pandas as pd
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition

    frame = pd.DataFrame({"composition": [Composition(f) for f in formulas]})
    magpie = ElementProperty.from_preset("magpie").featurize_dataframe(
        frame, col_id="composition", ignore_errors=False, pbar=False)
    magpie_names = [str(n) for n in magpie.columns if n != "composition"]
    rows = []
    for formula, (_, magpie_row) in zip(formulas, magpie.iterrows()):
        reduced = _flat_reduced_formula(formula)
        descriptor = formula_descriptor_record(reduced)
        amounts = Composition(reduced).get_el_amt_dict()
        lewis = {token.split(":", 1)[0]: float(token.split(":", 1)[1])
                 for token in str(descriptor["lewis_dot_proxy_outer_electrons"]).split(";") if ":" in token}
        total = sum(float(v) for v in amounts.values())
        lewis_avg = sum(float(amounts[el]) * lewis[el] for el in amounts) / total
        row: dict[str, Any] = {"formula": reduced}
        row.update({f"magpie__{n}": float(magpie_row[n]) for n in magpie_names})
        row.update({f"shell__{k}": v for k, v in descriptor.items() if k in descriptor_names() and v is not None})
        row["lewis_dot_proxy_avg_outer_electrons"] = lewis_avg
        row["lewis_dot_proxy_total_electrons_per_reduced_formula"] = lewis_avg * total
        rows.append(row)
    return rows


def _flat_reduced_formula(formula: str) -> str:
    """Reduced formula, flattened exactly as train_physics_targets does for grouped formulas."""
    from pymatgen.core import Composition

    composition = Composition(formula)
    reduced_text = composition.reduced_formula
    if not any(mark in reduced_text for mark in "()[]"):
        return reduced_text
    return "".join(
        f"{element}{'' if abs(float(amount) - 1.0) < 1e-8 else format(float(amount), '.8g')}"
        for element, amount in sorted(composition.reduced_composition.get_el_amt_dict().items())
    )


def _display(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(path)


def _coerce(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value in (None, ""):
            out[key] = None
            continue
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def _fit_forests(X: np.ndarray, lam: np.ndarray, wlog: np.ndarray, seed: int, n_trees: int) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median", add_indicator=True)
    X_fit = imputer.fit_transform(X)
    lam_model = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    wlog_model = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=2, random_state=seed + 1, n_jobs=-1)
    lam_model.fit(X_fit, np.log(lam))
    wlog_model.fit(X_fit, np.log(wlog))
    return {"log_lambda": lam_model, "log_omega_log_k": wlog_model, "imputer": imputer}


def _group_cv(X: np.ndarray, lam: np.ndarray, wlog: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """5-fold GroupKFold by elemental system: is this model better than guessing the median?"""
    from scipy.stats import spearmanr
    from sklearn.model_selection import GroupKFold

    tc = np.asarray([allen_dynes_tc_k(a, b) or 0.0 for a, b in zip(lam, wlog)])
    pred_lam, pred_wlog = np.zeros_like(lam), np.zeros_like(wlog)
    for fold, (train, test) in enumerate(GroupKFold(n_splits=5).split(X, lam, groups)):
        m = _fit_forests(X[train], lam[train], wlog[train], seed=100 + 2 * fold, n_trees=180)
        X_test = m["imputer"].transform(X[test])
        pred_lam[test] = np.exp(m["log_lambda"].predict(X_test))
        pred_wlog[test] = np.exp(m["log_omega_log_k"].predict(X_test))
    pred_tc = np.asarray([allen_dynes_tc_k(a, b) or 0.0 for a, b in zip(pred_lam, pred_wlog)])
    out = {}
    for name, truth, pred in (("lambda", lam, pred_lam), ("omega_log_k", wlog, pred_wlog), ("tc_k", tc, pred_tc)):
        out[name] = {"mae": float(np.mean(np.abs(truth - pred))),
                     "median_guess_mae": float(np.mean(np.abs(truth - np.median(truth)))),
                     "spearman": float(spearmanr(truth, pred)[0])}
    out["split"] = "5-fold GroupKFold by elemental system"
    return out


def fit_formula_model(table: Path, hazel_records: Path | None, *, evaluate: bool = True
                      ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit log-lambda and log-omega_log forests on formula features.

    Training rows: the committed JARVIS table plus reviewed Hazel records.
    lambda is fitted on a log scale. On the JARVIS table this halves the
    held-out Tc error compared with fitting lambda directly (5.9 K -> 2.7 K),
    because a tail of very large lambda values (up to 18) otherwise dominates.
    """
    from train_physics_targets import _element_system

    with table.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [_coerce(r) for r in reader]
    feature_names = formula_feature_names(header)
    local = load_extra_records(hazel_records)
    if local:
        local_rows, _, _ = build_characteristics(local, None)
        rows.extend(local_rows)
    rows = [r for r in rows if (r.get("lambda") or 0) > 0 and (r.get("omega_log_k") or 0) > 0]
    X = _make_matrix(rows, feature_names)
    lam = np.asarray([float(r["lambda"]) for r in rows])
    wlog = np.asarray([float(r["omega_log_k"]) for r in rows])
    models = _fit_forests(X, lam, wlog, seed=42, n_trees=300)
    models.update({"feature_names": feature_names, "mu_star": 0.1})
    meta: dict[str, Any] = {
        "feature_mode": "formula_only", "targets": "log(lambda), log(omega_log)",
        "training_rows": len(rows), "feature_count": len(feature_names),
        "training_table": _display(table),
        "training_table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "hazel_records_used": [str(r["jid"]) for r in local],
    }
    if evaluate:
        groups = np.asarray([_element_system(str(r["formula"])) for r in rows])
        meta["held_out_quality"] = _group_cv(X, lam, wlog, groups)
    return models, meta


def _magnetic_flag(row: dict[str, str], formula: str) -> tuple[bool, str]:
    value = str(row.get("is_magnetic", "")).strip().lower()
    if value in {"true", "false"}:
        return value == "true", "materials_project_is_magnetic"
    from pymatgen.core import Composition

    elements = {str(el) for el in Composition(formula).elements}
    return bool(elements & MAGNETIC_ELEMENTS), "magnetic_element_heuristic"


def _not_metallic(row: dict[str, str]) -> bool:
    """True only when the row carries MP metallicity and it says not metallic."""
    is_metal = str(row.get("is_metal", "")).strip().lower()
    if is_metal not in {"true", "false"}:
        return False
    if is_metal == "true":
        return False
    try:
        return float(row.get("band_gap_ev", "")) > 0
    except (TypeError, ValueError):
        return True


def _open_text(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode.replace("t", ""), encoding="utf-8", newline="")


def external_check(rows: list[dict[str, Any]], training_table: Path | None) -> dict[str, Any] | None:
    """Score the physics model against measured outcomes it never trained on.

    Uses rows whose formula is a known superconductor (reported Tc > 0) or a
    tested non-superconductor (3DSC Tc = 0 controls). Formulas present in the
    training table are excluded so the check is out-of-sample.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    from build_mp_candidates import reduced

    trained: set[str] = set()
    if training_table is not None and training_table.is_file():
        with training_table.open(newline="", encoding="utf-8-sig") as handle:
            trained = {reduced(r["formula"]) or "" for r in csv.DictReader(handle)}
    sc = [r for r in rows if r.get("known_status") == "superconductor" and reduced(r["formula"]) not in trained]
    non = [r for r in rows if r.get("known_status") == "tested_non_superconductor" and reduced(r["formula"]) not in trained]
    if len(sc) < 10 or len(non) < 10:
        return None
    labels = [1] * len(sc) + [0] * len(non)
    scores = [r["physics_tc_k"] or 0.0 for r in sc + non]
    measured = [float(r["known_tc_max_k"]) for r in sc]
    predicted = [r["physics_tc_k"] or 0.0 for r in sc]
    return {
        "superconductors_out_of_sample": len(sc), "non_superconductors_out_of_sample": len(non),
        "auc_superconductor_vs_tested_non_superconductor": float(roc_auc_score(labels, scores)),
        "spearman_predicted_vs_measured_tc_among_superconductors": float(spearmanr(predicted, measured)[0]),
        "median_measured_tc_k": float(np.median(measured)),
        "median_predicted_tc_superconductors_k": float(np.median(predicted)),
        "median_predicted_tc_non_superconductors_k": float(np.median([r["physics_tc_k"] or 0.0 for r in non])),
        "note": "AUC 0.5 = no better than chance; 1.0 = perfect separation. Measured Tc is the maximum reported value.",
    }


def _consensus(physics_tc: float | None, rf_tc: str | float | None) -> float | None:
    """The lower of the two models' Tc: high only when both models agree it is high."""
    try:
        rf = float(rf_tc)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return min(physics_tc or 0.0, rf)


def score(candidates: list[dict[str, str]], models: dict[str, Any]) -> list[dict[str, Any]]:
    formulas = [c["formula"] for c in candidates]
    features = formula_feature_rows(formulas)
    X = models["imputer"].transform(_make_matrix(features, models["feature_names"]))
    if "log_lambda" in models:
        lam_trees = np.exp(np.asarray([t.predict(X) for t in models["log_lambda"].estimators_]))
    else:  # trainer joblib fits lambda directly
        lam_trees = np.asarray([t.predict(X) for t in models["lambda"].estimators_])
    wlog_trees = np.exp(np.asarray([t.predict(X) for t in models["log_omega_log_k"].estimators_]))
    mu_star = float(models.get("mu_star", 0.1))
    output = []
    for idx, cand in enumerate(candidates):
        lam_v, wlog_v = lam_trees[:, idx], wlog_trees[:, idx]
        tc_v = np.asarray([allen_dynes_tc_k(l, w, mu_star) for l, w in zip(lam_v, wlog_v)], dtype=float)
        lam = float(np.mean(lam_v))
        wlog = float(np.exp(np.mean(np.log(wlog_v))))
        tc = allen_dynes_tc_k(lam, wlog, mu_star)
        tc_sd = float(np.std(tc_v))
        magnetic, magnetic_source = _magnetic_flag(cand, cand["formula"])
        family = cand.get("chemical_family", "")
        sites = cand.get("num_sites", "")
        too_big = sites not in ("", None) and float(sites) > 20
        reasons = []
        if family not in CONVENTIONAL_FAMILIES:
            reasons.append(f"family_{family or 'unknown'}_not_phonon_mediated_route")
        if magnetic:
            reasons.append(f"magnetic_{magnetic_source}")
        if too_big:
            reasons.append("more_than_20_sites")
        if _not_metallic(cand):
            reasons.append("not_metallic_in_materials_project")
        radioactive = radioactive_elements(cand["formula"])
        if radioactive:
            reasons.append("radioactive_" + "_".join(radioactive))
        if cand.get("known_status") == "superconductor":
            reasons.append("already_known_superconductor")
        if cand.get("known_status") == "tested_non_superconductor":
            reasons.append("already_tested_non_superconductor")
        output.append({
            "mp_id": cand.get("mp_id", ""), "formula": cand["formula"], "chemical_family": family,
            "num_sites": sites,
            "physics_lambda": lam, "physics_lambda_tree_sd": float(np.std(lam_v)),
            "physics_omega_log_k": wlog, "physics_omega_log_tree_sd_k": float(np.std(wlog_v)),
            "physics_tc_k": tc, "physics_tc_tree_sd_k": tc_sd,
            "physics_conservative_tc_k": tc - tc_sd,
            "rf_family_tc_k": cand.get("predicted_tc_kelvin", ""),
            "consensus_tc_k": _consensus(tc, cand.get("predicted_tc_kelvin", "")),
            "nearest_known_superconductor": cand.get("nearest_known_superconductor_formula", ""),
            "nearest_known_tc_k": cand.get("nearest_known_superconductor_tc_kelvin", ""),
            "epw_eligible": not reasons, "epw_ineligible_reasons": ";".join(reasons),
            "magnetism_source": magnetic_source, "mu_star": mu_star,
            "known_status": cand.get("known_status", ""), "known_tc_max_k": cand.get("known_tc_max_k", ""),
            "theoretical_only": cand.get("theoretical", ""),
            "dft_verification": "pending",
        })
    # Rank by agreement: the lower of the physics-model Tc and the measured-Tc
    # random forest's Tc. Tc minus tree spread is kept as a column, but with
    # spreads about as large as the predictions it mostly rewards near-zero Tc.
    def key(r: dict[str, Any]) -> tuple[float, float]:
        # Without a second model's Tc (e.g. the full MP snapshot), rank by the
        # physics model alone.
        consensus = r["consensus_tc_k"] if r["consensus_tc_k"] is not None else r["physics_tc_k"]
        return (-(consensus or 0.0), -(r["physics_tc_k"] or 0.0))

    eligible = sorted((r for r in output if r["epw_eligible"]), key=key)
    others = sorted((r for r in output if not r["epw_eligible"]), key=key)
    for rank, row in enumerate(eligible, 1):
        row["epw_rank"] = rank
    for row in others:
        row["epw_rank"] = ""
    return eligible + others


def run(candidates_path: Path, output: Path, summary: Path, *, model_path: Path | None,
        table: Path, hazel_records: Path | None, top: int | None = None) -> dict[str, Any]:
    with _open_text(candidates_path) as handle:
        candidates = [r for r in csv.DictReader(handle) if r.get("formula")]
    if candidates:
        if model_path is not None:
            import joblib

            models = joblib.load(model_path)
            meta = {"feature_mode": "trainer_joblib", "model_path": _display(model_path)}
        else:
            models, meta = fit_formula_model(table, hazel_records)
        rows = score(candidates, models)
    else:
        # An empty strict-gate shortlist is a valid result. Do not fit a model
        # or call sklearn with a zero-row feature matrix just to publish it.
        rows = []
        meta = {"feature_mode": "not_run_empty_candidate_pool"}
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epw_rank"] + [k for k in rows[0] if k != "epw_rank"] if rows else list(SCORED_OUTPUT_COLUMNS)
    written = rows[:top] if top else rows
    if output.suffix == ".gz":
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(written)
        with output.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
            gz.write(buffer.getvalue().encode("utf-8"))
    else:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(written)
    eligible = [r for r in rows if r["epw_eligible"]]
    reasons: dict[str, int] = {}
    for r in rows:
        for reason in filter(None, r["epw_ineligible_reasons"].split(";")):
            reasons[reason] = reasons.get(reason, 0) + 1
    report = {
        "status": "ready" if candidates else "empty_candidate_pool",
        "candidates_path": _display(candidates_path),
        "candidates_sha256": hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
        "candidates_scored": len(rows), "epw_eligible": len(eligible), "rows_written": len(written),
        "ineligible_reason_counts": dict(sorted(reasons.items())),
        "external_check": external_check(rows, table if model_path is None else None),
        "ranking": "EPW-eligible rows by consensus_tc_k = min(physics-model Tc, measured-Tc random forest Tc)",
        "top_eligible": [{k: r[k] for k in ("epw_rank", "mp_id", "formula", "consensus_tc_k", "physics_lambda",
                                              "physics_omega_log_k", "physics_tc_k",
                                              "physics_tc_tree_sd_k", "rf_family_tc_k")}
                         for r in eligible[:20]],
        "model": {k: v for k, v in meta.items()},
        "limits": [
            "Formula-only features cannot tell polymorphs apart; structure features need MP structures.",
            "Random forests stay inside the JARVIS training range of lambda and omega_log.",
            "Tree spread is a diagnostic, not a calibrated interval.",
            "EPW checks only phonon-mediated superconductivity; cuprate and iron-based rows are scored but not eligible.",
        ],
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_candidates = ELIGIBLE_LIST if ELIGIBLE_LIST.is_file() else POOL_SHORTLIST
    parser.add_argument("--candidates", type=Path, default=default_candidates)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/epw-next-candidates.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs/epw-next-candidates.json")
    parser.add_argument("--model", type=Path, default=None, help="trainer joblib; omit to fit a formula-only model")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--hazel-records", type=Path, default=DEFAULT_HAZEL)
    parser.add_argument("--top", type=int, default=None, help="write only the first N rows (eligible first)")
    args = parser.parse_args()
    report = run(args.candidates, args.output, args.summary, model_path=args.model,
                 table=args.table, hazel_records=args.hazel_records, top=args.top)
    print(json.dumps({k: report[k] for k in ("candidates_scored", "epw_eligible", "top_eligible")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
