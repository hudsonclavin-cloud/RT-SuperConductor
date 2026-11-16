#!/usr/bin/env python3
"""A steadier ranking of the made metals: bagged boosting, with each candidate's rank as a range.

The method and the decision rule were fixed in ``docs/stable-ranking.md`` before any bagged model was trained.

For each development fold, ``--bags`` bags resample the *groups* of the fold's fit rows with replacement (all rows
of a drawn group, repeated as drawn) and fit round 2's boosting models; the fold's validation rows stop each model
early. The ranking is made from the mean probability over all fold x bag models, through the same
``made_metals_order`` as ``run_learning_cycle.py``. Each model also ranks the made metals alone, and a candidate's
range is the 10th to 90th percentile of its ranks over the models.

    python scripts/run_bagged_ranking.py --name stab-bag-base --cv
    python scripts/run_bagged_ranking.py --name stab-bag-1 --previous stab-bag-base --placebo-flips 4 --placebo-seed 41

``--cv`` adds the grouped five-fold cross-validation (each test fold predicted by the mean of its bags). Locked rows
are neither trained on nor scored. Writes ``docs/stable-ranking/<name>.json`` and the top-50 table.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import scratchnet_common as C
from scratchnet_common import ROOT
from run_scratchnet import fold_rows, rounded
import run_learning_cycle as L

OUT = ROOT / "docs/stable-ranking"
WORK = L.WORK


def bag_seed(bag: int, fold: int) -> int:
    return 1000 * bag + fold


def bag_rows(groups: np.ndarray, fit: np.ndarray, seed: int) -> np.ndarray:
    """``fit`` rows with their groups resampled with replacement: every row of a drawn group, once per draw."""
    names, inverse = np.unique(groups[fit], return_inverse=True)
    drawn = np.random.default_rng(seed).integers(0, len(names), size=len(names))
    repeats = np.bincount(drawn, minlength=len(names))[inverse]
    return np.repeat(fit, repeats)


def member_ranks(p: np.ndarray, tc: np.ndarray) -> np.ndarray:
    """Ranks (1 = best) by rounded probability, ties by probability x Tc: the order ``made_metals_order`` makes."""
    order = np.lexsort((-(p * tc), -np.round(p, 3)))
    ranks = np.empty(len(p), int)
    ranks[order] = np.arange(1, len(p) + 1)
    return ranks


def cross_validate_bagged(ds, X, settings, bags):
    dev = np.flatnonzero(ds.split == "dev")
    position = {int(row): i for i, row in enumerate(dev)}
    p, tc, distance = np.full(len(dev), np.nan), np.full(len(dev), np.nan), np.full(len(dev), np.nan)
    for fold in sorted(set(ds.fold[dev].tolist())):
        fit, valid, test = fold_rows(ds, fold)
        where = np.array([position[int(row)] for row in test])
        ps, tcs = [], []
        for bag in range(bags):
            p_b, tc_b, _ = C.boosting_predict(X, ds.arrays, bag_rows(ds.arrays.group, fit, bag_seed(bag, fold)),
                                              valid, X[test], seed=bag_seed(bag, fold), **settings)
            ps.append(p_b)
            tcs.append(tc_b)
        p[where], tc[where] = np.mean(ps, axis=0), C.log_mean_tc(tcs)
        distance[where] = C.nearest_distance(ds.dense[np.concatenate([fit, valid])], ds.dense[test])[0]
        print(f"cv fold {fold}: {bags} bags on {len(fit)} fit rows", flush=True)
    return dev, (p, tc), distance


def final_members(ds, X, settings, bags, X_pool):
    """(p, Tc) on the candidate pool from every fold x bag model."""
    dev = ds.split == "dev"
    members = []
    for fold in sorted(set(ds.fold[dev].tolist())):
        valid = dev & (ds.fold == fold)
        fit = np.flatnonzero(((ds.split == "train_only") | dev) & ~valid)
        assert not (ds.split[fit] == "lockbox").any()
        for bag in range(bags):
            model, _ = C.boosting_fit(X, ds.arrays, bag_rows(ds.arrays.group, fit, bag_seed(bag, fold)),
                                      np.flatnonzero(valid), seed=bag_seed(bag, fold), **settings)
            members.append(C.boosting_apply(model, X_pool))
        print(f"final fold {fold}: {bags} bags", flush=True)
    return members


def rank_ranges(rows, order, members):
    """For the ensemble's top 200: the 10th, 50th and 90th percentile of the member ranks."""
    index = {r["formula"]: i for i, r in enumerate(rows)}
    eligible = np.array([index[r["formula"]] for r in order])
    ranks = np.array([member_ranks(p[eligible], tc[eligible]) for p, tc in members])
    for j, record in enumerate(order[:200]):
        low, mid, high = np.percentile(ranks[:, j], [10, 50, 90])
        record.update(rank_p10=int(round(low)), rank_median=int(round(mid)), rank_p90=int(round(high)))
    return order


def run(name, previous, bags, cv, placebo_flips, placebo_seed):
    started = time.time()
    ds = C.load_dataset()
    settings = L.boosting_settings()
    state = L.table_state(ds)
    report = {"name": name, "previous": previous, "bags": bags, "bag_unit": "development-fold groups, with replacement",
              "data_sha256": ds.sha256, "boosting_settings": settings, "locked_rows": "neither trained on nor scored"}
    if placebo_flips:
        report["placebo"] = L.placebo(ds, placebo_flips, placebo_seed)
    prev_order = None
    if previous:
        prev_state, _, prev_order = L.load_previous(previous)
        comparison = L.compare_tables(prev_state, state)
        L.check_unchanged_layout(comparison)
        report["table_change"] = comparison
    formulas = [C.formula_text(C.fractions_of(row["composition"])) for row in ds.rows]
    X, _ = C.formula_features(formulas, cache=C.WORK / "formula-features.npz")
    keys, prediction = [], (np.array([]), np.array([]))
    if cv:
        dev, prediction, distance = cross_validate_bagged(ds, X, settings, bags)
        keys = [ds.rows[i]["composition"] for i in dev]
        label, tc = ds.arrays.label[dev], ds.arrays.tc_k[dev]
        report["cross_validation"] = {"development_rows": int(len(dev)),
                                      "scores": L.score_with_labels(ds, dev, distance, prediction, label, tc)}
    rows, compositions = L.candidate_pool(L.CANDIDATES, set(ds.table.symbols))
    X_pool, _ = C.formula_features([C.formula_text(c) for c in compositions], cache=C.WORK / "candidate-features.npz")
    members = final_members(ds, X, settings, bags, X_pool)
    p_pool = np.mean([p for p, _ in members], axis=0)
    tc_pool = C.log_mean_tc([t for _, t in members])
    order = rank_ranges(rows, L.made_metals_order(rows, compositions, p_pool, tc_pool), members)
    top = order[:50]
    report["made_metals"] = {
        "untested_made_metals": len(order), "models": len(members),
        "top_10": [{k: r[k] for k in ("rank", "formula", "p_superconductor", "tc_if_superconductor_k",
                                      "rank_p10", "rank_p90")} for r in order[:10]],
        "top_50_range_inside_top_100": int(sum(r["rank_p90"] <= 100 for r in top)),
        "top_50_median_range_width": float(np.median([r["rank_p90"] - r["rank_p10"] for r in top]))}
    ranks = {L.composition_key(r["formula"]): r for r in order}
    report["made_metals"]["watched"] = {
        f: {k: ranks[L.composition_key(f)].get(k) for k in ("rank", "rank_p10", "rank_p90")}
        if L.composition_key(f) in ranks else None for f in L.WATCH}
    if previous:
        report["made_metals"]["against_previous"] = L.compare_orders(prev_order, order)
    L.save_state(name, state, keys, prediction, order)
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["rank", "formula", "mp_id", "element_system", "p_superconductor", "tc_if_superconductor_k",
              "expected_tc_k", "rank_p10", "rank_median", "rank_p90"]
    C.write_table(OUT / f"{name}-made-metals-top50.csv", top, fields)
    report["minutes"] = round((time.time() - started) / 60.0, 1)
    (OUT / f"{name}.json").write_text(json.dumps(rounded(report), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT / (name + '.json')}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--previous", help="the bagged run to compare with (omit for the unperturbed baseline)")
    parser.add_argument("--bags", type=int, default=8)
    parser.add_argument("--cv", action="store_true", help="also cross-validate (the baseline needs this for its AUC)")
    parser.add_argument("--placebo-flips", type=int, default=0)
    parser.add_argument("--placebo-seed", type=int, default=0)
    args = parser.parse_args()
    run(args.name, args.previous, args.bags, args.cv, args.placebo_flips, args.placebo_seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
