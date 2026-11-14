#!/usr/bin/env python3
"""Score materials from outside the training table with the saved boosting and network ensembles.

The input is a CSV with a ``formula`` column, an ``outcome`` column (``superconductor`` or
``not_superconductor``; anything else is carried through but not scored) and, where known, ``tc_k``.
Every other column is copied to the output unchanged.

For each row the script records whether that composition is already in the training table (and with what
label), how far it is from the nearest composition the models were trained on, both models' answers, and -
if a ranked candidate file is given - where the material would have stood in that ranking.

The summary separates the rows the models have never seen ("clean": no table row within ``--clean-distance``)
from the rest, because a material that is already in the table is not a test. A row counts as "in the table"
when its composition, rounded the way the table rounds it, is one of the table's rows.

    python scripts/score_external_cases.py --cases data/benchmarks/prospective-cases/cases.csv \
        --name round2 --out-rows docs/scratchnet-prospective-rows.csv --out docs/scratchnet-prospective.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

import run_scratchnet as R
import scratchnet_common as C
from rt_superconductor.scratchnet.data import composition_key

OUTCOMES = {"superconductor": 1.0, "not_superconductor": 0.0}
SCORE_FIELDS = ["scored", "in_table", "table_label", "table_split", "distance_to_nearest_table_row",
                "distance_to_nearest_training", "nearest_known", "nearest_known_tc_k", "clean",
                "boosting_p", "boosting_tc_k", "boosting_expected_tc_k", "network_p", "network_tc_k",
                "network_expected_tc_k", "network_view", "pool_percentile_by_expected_tc", "pool_percentile_by_p"]


def read_cases(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "formula" not in rows[0] or "outcome" not in rows[0]:
        raise SystemExit(f"{path} needs the columns 'formula' and 'outcome'")
    return rows


def auc_with_interval(score: np.ndarray, label: np.ndarray, resamples: int = 2000, seed: int = 0,
                      groups: np.ndarray | None = None) -> dict:
    """Area under the ROC curve with a percentile interval from resampling.

    With ``groups`` (for example the source paper of each case) whole groups are resampled, because cases
    from one paper share a method and a laboratory and are not independent.
    """
    positives, negatives = int((label == 1).sum()), int((label == 0).sum())
    out = {"superconductors": positives, "not_superconductors": negatives, "auc": None, "low": None, "high": None,
           "resampled": "whole groups" if groups is not None else "cases"}
    if positives == 0 or negatives == 0:
        return out
    out["auc"] = float(C.T.roc_auc(label, score))
    rng = np.random.default_rng(seed)
    if groups is None:
        members = [np.array([i]) for i in range(len(label))]
    else:
        names = sorted(set(groups.tolist()))
        members = [np.flatnonzero(groups == name) for name in names]
    draws = []
    for _ in range(resamples):
        pick = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        if 0 < label[pick].sum() < len(pick):
            draws.append(C.T.roc_auc(label[pick], score[pick]))
    if len(draws) >= resamples // 2:
        out["low"], out["high"] = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    return out


def percentile_in(pool: np.ndarray, value: float) -> float:
    """Share of the pool that scores strictly higher, in percent (0 = above everything in the pool)."""
    return float(100.0 * np.mean(pool > value))


def summarise(rows: list[dict], which: str, group_column: str | None = None) -> dict:
    """Scores for one subset of the scored rows that have a known outcome."""
    label = np.array([OUTCOMES[r["outcome"]] for r in rows])
    groups = np.array([r[group_column] for r in rows]) if group_column and rows else None
    out: dict = {"subset": which, "rows": len(rows)}
    if groups is not None:
        out["groups"] = len(set(groups.tolist()))
    for model in ("boosting", "network"):
        p = np.array([r[f"{model}_p"] for r in rows], dtype=float)
        expected = np.array([r[f"{model}_expected_tc_k"] for r in rows], dtype=float)
        out[model] = {"by_probability": auc_with_interval(p, label, groups=groups) if len(rows) else None,
                      "by_expected_tc": auc_with_interval(expected, label, groups=groups) if len(rows) else None,
                      "median_p_superconductors": float(np.median(p[label == 1])) if (label == 1).any() else None,
                      "median_p_not_superconductors": float(np.median(p[label == 0])) if (label == 0).any() else None}
        known = [(float(r["tc_k"]), r[f"{model}_tc_k"]) for r in rows
                 if r["outcome"] == "superconductor" and str(r.get("tc_k", "")).strip() not in ("", "nan")]
        if known:
            true, predicted = (np.array(v, dtype=float) for v in zip(*known))
            out[model]["tc_rows"] = len(known)
            out[model]["tc_median_log_error"] = float(np.median(np.abs(np.log1p(predicted) - np.log1p(true))))
            out[model]["tc_median_predicted_over_measured"] = float(np.median(predicted / np.maximum(true, 1e-9)))
    views = [r["network_view"] for r in rows]
    out["network_view_superconductors"] = {v: sum(view == v and lab == 1 for view, lab in zip(views, label))
                                           for v in ("agrees", "lukewarm", "rejects")}
    out["network_view_not_superconductors"] = {v: sum(view == v and lab == 0 for view, lab in zip(views, label))
                                               for v in ("agrees", "lukewarm", "rejects")}
    return out


def score_cases(cases: list[dict], name: str, pool: Path | None = None, clean_distance: float = 0.02,
                group_column: str | None = None) -> tuple[list[dict], dict]:
    ds = C.load_dataset()
    known_elements = set(ds.table.symbols)
    parsed: list[tuple[int, dict[str, float]]] = []
    out_rows = [dict(row) for row in cases]
    for number, row in enumerate(out_rows):
        row.update({field: "" for field in SCORE_FIELDS})
        try:
            fractions = C.D_parse(row["formula"])
        except ValueError:
            row["scored"] = "no: formula not readable"
            continue
        if not set(fractions) <= known_elements:
            row["scored"] = "no: element outside the table"
        elif len(fractions) > C.MAX_ELEMENTS:
            row["scored"] = "no: too many elements"
        else:
            row["scored"] = "yes"
            parsed.append((number, fractions))
    summary: dict = {"name": name, "data_sha256": ds.sha256, "rows": len(cases), "scored": len(parsed),
                     "clean_means": f"no row of the training table within {clean_distance:g} (half the summed "
                                    f"difference of the atomic fractions)"}
    if not parsed:
        return out_rows, summary
    compositions = [fractions for _, fractions in parsed]
    scored = R.ensemble_predict(ds, name, compositions)
    network, boosting = scored["network"], scored["boosting"]
    dense = C.dense_fractions(compositions, ds.table)
    table_distance, _ = C.nearest_distance(ds.dense, dense)                      # any row, labelled or not
    by_key = {row["composition"]: row for row in ds.rows}                         # the table's own identity of a row
    labelled = np.flatnonzero(np.isfinite(ds.arrays.label))
    _, nearest = C.nearest_distance(ds.dense[labelled], dense)
    nearest = labelled[nearest]
    trained_on = np.flatnonzero((ds.split != "lockbox") & np.isfinite(ds.arrays.label))
    training_distance, _ = C.nearest_distance(ds.dense[trained_on], dense)
    pool_expected = pool_p = None
    if pool is not None:
        with gzip.open(pool, "rt", encoding="utf-8", newline="") as handle:
            ranked = list(csv.DictReader(handle))
        orders = {r.get("ranked_by") or "unknown" for r in ranked}
        if orders != {"boosting"}:
            raise SystemExit(f"{pool} is ordered by {', '.join(sorted(orders))}; the cases are placed by boosting's "
                             f"figures, so --pool must be a ranking made with --order boosting")
        pool_expected = np.array([float(r["expected_tc_k"]) for r in ranked])
        pool_p = np.array([float(r["p_superconductor"]) for r in ranked])
        summary["pool"] = {"file": pool.name, "rows": len(ranked), "ranked_by": "boosting"}
    for position, (number, fractions) in enumerate(parsed):
        row = out_rows[number]
        twin, same = ds.rows[nearest[position]], by_key.get(composition_key(fractions))
        in_table = same is not None
        b_p, b_tc = float(boosting[0][position]), float(boosting[1][position])
        n_p, n_tc = float(network[0][position]), float(network[1][position])
        row.update({
            "in_table": "yes" if in_table else "no",
            "table_label": same["label"] if in_table else "", "table_split": same["split"] if in_table else "",
            "distance_to_nearest_table_row": round(float(table_distance[position]), 3),
            "distance_to_nearest_training": round(float(training_distance[position]), 3),
            "nearest_known": twin["composition"], "nearest_known_tc_k": twin["tc_k"] if twin["label"] == "1" else "0",
            "clean": "yes" if table_distance[position] >= clean_distance else "no",
            "boosting_p": round(b_p, 3), "boosting_tc_k": round(b_tc, 2), "boosting_expected_tc_k": round(b_p * b_tc, 2),
            "network_p": round(n_p, 3), "network_tc_k": round(n_tc, 2), "network_expected_tc_k": round(n_p * n_tc, 2),
            "network_view": R.network_view(round(n_p * n_tc, 2), round(b_p * b_tc, 2)),
        })
        if pool_expected is not None:
            row["pool_percentile_by_expected_tc"] = round(percentile_in(pool_expected, b_p * b_tc), 1)
            row["pool_percentile_by_p"] = round(percentile_in(pool_p, b_p), 1)
    decided = [r for r in out_rows if r["scored"] == "yes" and r["outcome"] in OUTCOMES]
    summary["with_known_outcome"] = len(decided)
    if group_column and cases and group_column not in cases[0]:
        raise SystemExit(f"no column {group_column!r} in the cases")
    summary["all"] = summarise(decided, "every scored row with a known outcome", group_column)
    summary["clean"] = summarise([r for r in decided if r["clean"] == "yes"], "rows the models have never seen",
                                 group_column)
    summary["in_table"] = {
        "rows": sum(r["in_table"] == "yes" for r in decided),
        "table_label_disagrees_with_outcome": [
            r["formula"] for r in decided if r["in_table"] == "yes" and r["table_label"] != ""
            and float(r["table_label"]) != OUTCOMES[r["outcome"]]]}
    if pool_expected is not None:
        found = [r for r in decided if r["clean"] == "yes" and r["outcome"] == "superconductor"]
        for key in ("pool_percentile_by_expected_tc", "pool_percentile_by_p"):
            values = [float(r[key]) for r in found]
            summary[f"clean_superconductors_median_{key}"] = float(np.median(values)) if values else None
            summary[f"clean_superconductors_in_top_10_percent_{key.removeprefix('pool_percentile_')}"] = \
                sum(v < 10.0 for v in values)
    return out_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--name", default="round2")
    parser.add_argument("--pool", type=Path, default=None, help="a candidate file ranked with --order boosting (csv.gz) to place the cases in")
    parser.add_argument("--clean-distance", type=float, default=0.02)
    parser.add_argument("--group-column", default=None,
                        help="resample whole groups of cases named by this column (for example the source paper)")
    parser.add_argument("--out-rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = read_cases(args.cases)
    rows, summary = score_cases(cases, args.name, args.pool, args.clean_distance, args.group_column)
    fields = list(cases[0].keys()) + [f for f in SCORE_FIELDS if f not in cases[0]]
    C.write_table(args.out_rows, rows, fields)
    summary["cases_file"] = args.cases.name
    args.out.write_text(json.dumps(R.rounded(summary), indent=1) + "\n", encoding="utf-8")
    print(json.dumps(R.rounded(summary), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
