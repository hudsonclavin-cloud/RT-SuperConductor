#!/usr/bin/env python3
"""Learn "superconductor or not" from labelled Materials Project materials.

Input: data/candidates/mp-candidates.csv.gz (build_mp_candidates.py). About
2,200 of the ~30,000 snapshot materials match a SuperCon/3DSC report: a
superconductor (reported Tc > 0) or a tested non-superconductor (Tc = 0). A
classifier (gradient boosting, chosen by docs/sc-model-benchmark.json) learns to tell them apart from formula features
(Magpie, shell, Lewis-dot) plus Materials Project's own computed properties
(band gap, metallicity, magnetism, density, formation energy, symmetry). It
then scores every untested material.

Quality is measured by 5-fold GroupKFold over element systems, so a test
fold never shares an element set with training. Labels carry a selection
bias: "tested non-superconductor" means someone reported a negative result,
so the model partly learns which compounds people chose to test.

Outputs a full scored table (gz) and the EPW-eligible ranking: untested,
Other family, metallic and non-magnetic in MP, at most 20 sites. Each row
also gets the lambda/omega_log physics-model Tc for reference.
"""
from __future__ import annotations

import argparse
import csv
import gzip
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

from score_pool_physics import DEFAULT_TABLE, fit_formula_model, formula_feature_rows, score  # noqa: E402

NUMERIC = ["band_gap_ev", "density_g_cm3", "formation_energy_ev_per_atom", "energy_above_hull_ev_per_atom",
           "num_sites", "nelements", "spacegroup_number", "total_magnetization_mub_per_formula_unit", "volume_a3"]
BOOLEAN = ["is_metal", "is_magnetic", "theoretical"]


def _num(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _bool(value: str) -> float:
    return {"True": 1.0, "False": 0.0}.get(value, float("nan"))


def mp_features(rows: list[dict[str, str]], systems: list[str]) -> tuple[np.ndarray, list[str]]:
    names = NUMERIC + ["volume_per_site_a3"] + BOOLEAN + [f"crystal_system_{s}" for s in systems]
    data = []
    for r in rows:
        values = [_num(r[k]) for k in NUMERIC]
        sites = _num(r["num_sites"])
        values.append(_num(r["volume_a3"]) / sites if sites and sites > 0 else float("nan"))
        values += [_bool(r[k]) for k in BOOLEAN]
        values += [1.0 if r["crystal_system"] == s else 0.0 for s in systems]
        data.append(values)
    return np.asarray(data, dtype=float), names


def feature_matrix(rows: list[dict[str, str]], systems: list[str]) -> tuple[np.ndarray, list[str]]:
    formula = formula_feature_rows([r["formula"] for r in rows])
    formula_names = [k for k in formula[0] if k != "formula"]
    X_formula = np.asarray([[f.get(k) if f.get(k) is not None else np.nan for k in formula_names]
                            for f in formula], dtype=float)
    X_mp, mp_names = mp_features(rows, systems)
    return np.hstack([X_formula, X_mp]), formula_names + mp_names


DEFAULT_MODEL = "gradient_boosting"  # best AUC in docs/sc-model-benchmark.json


def _model(name: str, seed: int):
    from benchmark_sc_models import model_factories

    return model_factories(seed)[name]()


def group_cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model_name: str = DEFAULT_MODEL
                 ) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    pred = np.zeros(len(y))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1000)
    for fold, (train, test) in enumerate(splitter.split(X, y, groups)):
        model = _model(model_name, 10 + fold).fit(X[train], y[train])
        pred[test] = model.predict_proba(X[test])[:, 1]
    top = np.argsort(-pred)[: max(1, len(pred) // 10)]
    return {"auc": float(roc_auc_score(y, pred)), "model": model_name,
            "split": "5-fold StratifiedGroupKFold by element system",
            "superconductor_share_overall": float(y.mean()),
            "superconductor_share_in_top_decile": float(y[top].mean())}


def run(candidates: Path, scores_out: Path, ranking_out: Path, summary_out: Path,
        *, top: int = 200, evaluate: bool = True, physics_table: Path = DEFAULT_TABLE,
        model_name: str = DEFAULT_MODEL, extra_features: Path | None = None) -> dict[str, Any]:
    from sklearn.inspection import permutation_importance

    with gzip.open(candidates, "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    systems = sorted({r["crystal_system"] for r in rows})
    X, names = feature_matrix(rows, systems)
    if extra_features is not None:
        from structure_features import load_feature_table

        X_extra, extra_names = load_feature_table(extra_features, [r["mp_id"] for r in rows])
        X, names = np.hstack([X, X_extra]), names + extra_names
    labelled = np.asarray([r["known_status"] in ("superconductor", "tested_non_superconductor") for r in rows])
    y = np.asarray([r["known_status"] == "superconductor" for r in rows], dtype=int)[labelled]
    groups = np.asarray(["-".join(sorted(r["elements"].split())) for r in rows])[labelled]
    report: dict[str, Any] = {"materials": len(rows), "labelled": int(labelled.sum()),
                              "labelled_superconductors": int(y.sum()),
                              "labelled_non_superconductors": int(len(y) - y.sum()),
                              "feature_count": len(names)}
    if evaluate:
        report["held_out_quality"] = group_cv_auc(X[labelled], y, groups, model_name)
    model = _model(model_name, 42).fit(X[labelled], y)
    probability = model.predict_proba(X)[:, 1]
    importance = permutation_importance(model, X[labelled], y, scoring="roc_auc", n_repeats=3,
                                        random_state=0, n_jobs=1).importances_mean
    report["model"] = model_name
    report["extra_features"] = str(extra_features) if extra_features else None

    physics_models, _ = fit_formula_model(physics_table, None, evaluate=False)
    physics = {r["mp_id"]: r for r in score(rows, physics_models)}
    out = []
    for r, p in zip(rows, probability):
        ph = physics[r["mp_id"]]
        out.append({
            "mp_id": r["mp_id"], "formula": r["formula"], "chemical_family": r["chemical_family"],
            "num_sites": r["num_sites"], "spacegroup_symbol": r["spacegroup_symbol"],
            "known_status": r["known_status"], "known_tc_max_k": r["known_tc_max_k"],
            "p_superconductor": float(p),
            "physics_tc_k": ph["physics_tc_k"], "physics_tc_tree_sd_k": ph["physics_tc_tree_sd_k"],
            "is_metal": r["is_metal"], "is_magnetic": r["is_magnetic"], "theoretical_only": r["theoretical"],
            "energy_above_hull_ev_per_atom": r["energy_above_hull_ev_per_atom"],
            "epw_eligible": ph["epw_eligible"], "epw_ineligible_reasons": ph["epw_ineligible_reasons"],
        })
    eligible = sorted((o for o in out if o["epw_eligible"]), key=lambda o: -o["p_superconductor"])
    for rank, row in enumerate(eligible, 1):
        row["rank"] = rank
    for row in out:
        row.setdefault("rank", "")
    fields = ["rank"] + [k for k in out[0] if k != "rank"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(sorted(out, key=lambda o: str(o["mp_id"])))
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    with scores_out.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        gz.write(buffer.getvalue().encode("utf-8"))
    with ranking_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(eligible[:top])
    report.update({
        "epw_eligible": len(eligible),
        "top_eligible": [{k: row[k] for k in ("rank", "mp_id", "formula", "p_superconductor", "physics_tc_k",
                                               "theoretical_only")} for row in eligible[:25]],
        "top_eligible_physics_tc_ge_10k": [
            {k: row[k] for k in ("rank", "mp_id", "formula", "p_superconductor", "physics_tc_k", "theoretical_only")}
            for row in eligible if (row["physics_tc_k"] or 0) >= 10.0][:25],
        "importances_top15_note": "in-sample permutation importance (drop in AUC when a feature is shuffled)",
        "importances_top15": sorted(
            ({"feature": n, "importance": float(v)} for n, v in zip(names, importance)),
            key=lambda d: -d["importance"])[:15],
        "limits": [
            "p_superconductor ranks likelihood of any reported superconductivity, not Tc.",
            "Negative labels are reported non-superconductors, a biased sample of what people chose to test.",
            "Formula plus MP summary features; no structure-derived bonding or phonon information.",
            "Untested means no SuperCon/3DSC formula match; a literature check is still required.",
        ],
    })
    summary_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, default=ROOT / "data/candidates/mp-candidates.csv.gz")
    parser.add_argument("--scores", type=Path, default=ROOT / "data/candidates/mp-sc-classifier-scores.csv.gz")
    parser.add_argument("--ranking", type=Path, default=ROOT / "docs/mp-sc-classifier-top200.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs/mp-sc-classifier.json")
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="a model name from benchmark_sc_models.py")
    parser.add_argument("--extra-features", type=Path, default=None)
    args = parser.parse_args()
    report = run(args.candidates, args.scores, args.ranking, args.summary, top=args.top,
                 model_name=args.model, extra_features=args.extra_features)
    print(json.dumps({k: report[k] for k in ("materials", "labelled", "held_out_quality", "epw_eligible",
                                              "top_eligible")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
