#!/usr/bin/env python3
"""Which inputs help? Cross feature sets with prediction heads on identical folds.

Each feature set is evaluated with a random forest, gradient boosting and the
MLP neural network, using benchmark_sc_models.evaluate (same folds, same
seeds, fold-local preprocessing). Comparing along a row shows the effect of
the model; comparing down a column shows the effect of the inputs.

Feature sets:
  formula            Magpie, shell and Lewis-dot formula features
  formula+mp         + Materials Project properties (current classifier)
  formula+mp+struct  + symmetry, lattice, local geometry, connectivity
  formula+mp+struct+stiff  + elastic moduli and Debye temperature
  mp+struct+stiff    no formula features (is structure alone informative?)
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

from benchmark_sc_models import LABELS, evaluate, labelled_rows, provenance  # noqa: E402
from structure_features import load_feature_table  # noqa: E402

HEADS = ["random_forest", "gradient_boosting", "neural_network_mlp"]


def feature_blocks(rows: list[dict[str, str]], candidates: Path, structure_table: Path
                   ) -> dict[str, tuple[np.ndarray, list[str]]]:
    from score_pool_physics import formula_feature_rows
    from train_mp_sc_classifier import mp_features

    with gzip.open(candidates, "rt", encoding="utf-8") as handle:
        systems = sorted({r["crystal_system"] for r in csv.DictReader(handle)})
    formula = formula_feature_rows([r["formula"] for r in rows])
    formula_names = [k for k in formula[0] if k != "formula"]
    X_formula = np.asarray([[f.get(k) if f.get(k) is not None else np.nan for k in formula_names]
                            for f in formula], dtype=float)
    X_mp, mp_names = mp_features(rows, systems)
    X_all, all_names = load_feature_table(structure_table, [r["mp_id"] for r in rows])
    stiff = [i for i, n in enumerate(all_names) if n.startswith("stiff_")]
    struct = [i for i, n in enumerate(all_names) if not n.startswith("stiff_")]
    return {
        "formula": (X_formula, formula_names),
        "mp": (X_mp, mp_names),
        "struct": (X_all[:, struct], [all_names[i] for i in struct]),
        "stiff": (X_all[:, stiff], [all_names[i] for i in stiff]),
    }


FEATURE_SETS = {
    "formula": ["formula"],
    "formula+mp": ["formula", "mp"],
    "formula+mp+struct": ["formula", "mp", "struct"],
    "formula+mp+struct+stiff": ["formula", "mp", "struct", "stiff"],
    "mp+struct+stiff": ["mp", "struct", "stiff"],
}


def run(candidates: Path, structure_table: Path, out_json: Path, *, repeats: int = 3,
        heads: list[str] | None = None, sets: list[str] | None = None) -> dict[str, Any]:
    rows = labelled_rows(candidates)
    y = np.asarray([LABELS[r["known_status"]] for r in rows])
    groups = np.asarray(["-".join(sorted(r["elements"].split())) for r in rows])
    blocks = feature_blocks(rows, candidates, structure_table)
    report: dict[str, Any] = {
        "rows": len(rows), "positives": int(y.sum()),
        "split": f"StratifiedGroupKFold by element system, 5 folds x {repeats} shuffles (same as the benchmark)",
        "block_sizes": {k: len(v[1]) for k, v in blocks.items()}, "results": {},
    }
    for name in sets or list(FEATURE_SETS):
        X = np.hstack([blocks[b][0] for b in FEATURE_SETS[name]])
        result = evaluate(X, y, groups, repeats=repeats, models=heads or HEADS)
        report["results"][name] = {head: {k: result[head][k] for k in (
            "auc_mean", "auc_sd_across_repeats", "auc_per_repeat", "average_precision_mean", "brier_mean",
            "top_decile_superconductor_share_mean")} for head in result}
        report["results"][name]["features"] = int(X.shape[1])
        print(name, {h: round(result[h]["auc_mean"], 3) for h in result}, flush=True)
    best = max(((s, h) for s, r in report["results"].items() for h in r if h != "features"),
               key=lambda sh: report["results"][sh[0]][sh[1]]["auc_mean"])
    report["best"] = {"feature_set": best[0], "model": best[1],
                      "auc_mean": report["results"][best[0]][best[1]]["auc_mean"]}
    if "formula+mp" in report["results"]:
        base = report["results"]["formula+mp"]
        report["paired_gain_vs_formula_mp"] = {
            s: {h: {"mean": float(np.mean(np.subtract(r[h]["auc_per_repeat"], base[h]["auc_per_repeat"]))),
                    "wins": int(np.sum(np.subtract(r[h]["auc_per_repeat"], base[h]["auc_per_repeat"]) > 0))}
                for h in r if h != "features" and h in base}
            for s, r in report["results"].items()}
    report["provenance"] = provenance(np.hstack([b[0] for b in blocks.values()]), candidates)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, default=ROOT / "data/candidates/mp-candidates.csv.gz")
    parser.add_argument("--structure-table", type=Path, default=ROOT / "data/candidates/mp-structure-features.csv.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/sc-feature-ablation.json")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    report = run(args.candidates, args.structure_table, args.output, repeats=args.repeats)
    print(json.dumps(report["best"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
