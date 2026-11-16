#!/usr/bin/env python3
"""One learning cycle for the ranking of record (gradient boosting on formula features).

A learning cycle needs new evidence (``docs/active-learning.md``). Here the evidence is label corrections
recorded in ``data/training/negative_evidence.csv`` under the owner's rule of 2026-11-16
(``docs/learning-cycles.md``). Rebuild the two derived tables from the corrected file first:

    python scripts/build_mp_candidates.py
    python scripts/build_scratchnet_data.py --skip-elements

then run, for example:

    python scripts/run_learning_cycle.py --name cycle1 --previous cycle0

What it does, in order:

1. Compares the composition table with the previous cycle's: no material may change split or fold, and no
   material may appear or disappear (a correction is recorded against an existing composition). Every label or Tc
   that changed is listed.
2. Grouped five-fold cross-validation of boosting on the development rows, with the settings chosen in round 2
   (``docs/scratchnet-round2.json``). The locked rows are neither trained on nor scored.
3. Scores the previous cycle's saved out-of-fold predictions against today's labels too, so a change in the scores
   splits into "the answer key changed" (previous model, both answer keys) and "the model changed" (both models,
   today's key).
4. Trains the final boosting models (one per development fold, held back for early stopping; locked rows excluded)
   and ranks the made metals: calculated metals, reported from experiment, no copper, no copper-oxygen or
   iron-pnictogen chemistry, untested by formula, ordered by the chance of superconducting.
5. Compares that order with the previous cycle's.

Writes ``docs/learning-cycles/<name>.json`` and ``docs/learning-cycles/<name>-made-metals-top50.csv``.
Predictions, the table state and the models go under ``data/processed/learning-cycles/`` (not committed).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import scratchnet_common as C
from scratchnet_common import ROOT, T
from run_scratchnet import fold_rows, made_metal, rounded

OUT = ROOT / "docs/learning-cycles"
WORK = ROOT / "data/processed/learning-cycles"
SELECTION = ROOT / "docs/scratchnet-round2.json"
CANDIDATES = ROOT / "data/candidates/mp-candidates.csv.gz"
# Compounds whose place is reported every cycle: the current experimental-partner candidates and their relatives.
WATCH = ("Zr6Ga2Ni", "Zr6Ga2Co", "Hf6Ga2Co", "Ti3SnH", "ZrSbRu", "TiAlPt", "Y2B3C2", "ZrS", "NbCoSb", "Ti2Ga",
         "PtPb3Bi")


def boosting_settings(selection: Path = SELECTION) -> dict:
    chosen = json.loads(selection.read_text(encoding="utf-8"))["selection"]["boosting"]
    return {"classifier_settings": chosen["classifier"], "regressor_settings": chosen["regressor"]}


# ---------------------------------------------------------------------------
# 1. The table against the previous cycle's.

def table_state(ds: C.Dataset) -> dict[str, dict]:
    """composition -> what a cycle may and may not change about it."""
    return {row["composition"]: {"label": row["label"], "tc_k": row["tc_k"], "split": row["split"],
                                 "fold": row["fold"], "group": row["group"]} for row in ds.rows}


def compare_tables(previous: dict[str, dict], now: dict[str, dict]) -> dict:
    """Moved, added and removed compositions (all must be empty) and every changed label or Tc."""
    moved = sorted(key for key in previous.keys() & now.keys()
                   if (previous[key]["split"], previous[key]["fold"]) != (now[key]["split"], now[key]["fold"]))
    changed = []
    for key in sorted(previous.keys() & now.keys()):
        before, after = previous[key], now[key]
        if (before["label"], before["tc_k"]) != (after["label"], after["tc_k"]):
            changed.append({"composition": key, "split": after["split"], "fold": int(after["fold"]),
                            "label_before": before["label"], "label_after": after["label"],
                            "tc_k_before": before["tc_k"], "tc_k_after": after["tc_k"]})
    return {"moved_between_splits": moved, "added": sorted(now.keys() - previous.keys()),
            "removed": sorted(previous.keys() - now.keys()), "changed": changed}


def check_unchanged_layout(comparison: dict) -> None:
    problems = {name: comparison[name] for name in ("moved_between_splits", "added", "removed") if comparison[name]}
    if problems:
        raise SystemExit("the corrections changed the table's layout, so this cycle would not be comparable with the "
                         "previous one:\n" + json.dumps({k: v[:20] for k, v in problems.items()}, indent=1) +
                         "\nRecord each correction against an existing composition (within 0.0015 on every element).")


# ---------------------------------------------------------------------------
# 2. Cross-validation.

def cross_validate(ds: C.Dataset, X: np.ndarray, settings: dict, seed_offset: int = 0, extra: dict | None = None
                   ) -> tuple[np.ndarray, tuple, np.ndarray, list]:
    """Out-of-fold boosting predictions on the development rows; the same folds and seeds as run_scratchnet.py cv.

    ``extra`` is a block of added training rows (``load_additions`` or ``duplicate_rows``). They are fitted, never
    scored as development rows. Where ``extra["score"]`` is set, each added row is also predicted by the round that
    held its fold out, with and without the other added rows in training (written into ``extra``).
    """
    dev = np.flatnonzero(ds.split == "dev")
    position = {int(row): i for i, row in enumerate(dev)}
    p, tc, distance = np.full(len(dev), np.nan), np.full(len(dev), np.nan), np.full(len(dev), np.nan)
    info = []
    if extra is not None:
        X_all, arrays_all = extended(ds, X, extra)
        extra["p_with"], extra["p_without"] = np.full(len(extra["label"]), np.nan), np.full(len(extra["label"]), np.nan)
        extra["p_dev_without"] = np.full(len(dev), np.nan)
    for fold in sorted(set(ds.fold[dev].tolist())):
        fit, valid, test = fold_rows(ds, fold)
        where = np.array([position[int(row)] for row in test])
        if extra is None:
            p_fold, tc_fold, fold_info = C.boosting_predict(X, ds.arrays, fit, valid, X[test],
                                                            seed=fold + seed_offset, **settings)
        else:
            use = added_rows_for(extra, fold, set(ds.arrays.group[valid].tolist()))
            models, fold_info = C.boosting_fit(X_all, arrays_all, np.concatenate([fit, len(ds.rows) + use]), valid,
                                               seed=fold + seed_offset, **settings)
            p_fold, tc_fold = C.boosting_apply(models, X[test])
            fold_info["added_rows_fitted"] = int(len(use))
            held = np.flatnonzero(extra["fold"] == fold)
            if extra["score"]:
                plain, _ = C.boosting_fit(X, ds.arrays, fit, valid, seed=fold + seed_offset, **settings)
                extra["p_dev_without"][where] = C.boosting_apply(plain, X[test])[0]
                if len(held):
                    extra["p_with"][held] = C.boosting_apply(models, extra["X"][held])[0]
                    extra["p_without"][held] = C.boosting_apply(plain, extra["X"][held])[0]
        p[where], tc[where] = p_fold, tc_fold
        distance[where] = C.nearest_distance(ds.dense[np.concatenate([fit, valid])], ds.dense[test])[0]
        info.append({"fold": fold, **fold_info})
        print(f"fold {fold}: {len(fit)} fit, {len(valid)} validation, {len(test)} test rows; {fold_info}", flush=True)
    return dev, (p, tc), distance, info


def aligned(saved: dict[str, tuple[float, float]], keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """A previous cycle's (probability, Tc) predictions put in the order of ``keys``."""
    p = np.array([saved[key][0] if key in saved else np.nan for key in keys])
    tc = np.array([saved[key][1] if key in saved else np.nan for key in keys])
    return p, tc


def score_with_labels(ds: C.Dataset, rows: np.ndarray, distance: np.ndarray, prediction: tuple,
                      label: np.ndarray, tc: np.ndarray) -> dict:
    """The usual scores for ``prediction`` on all rows and on non-cuprates, against the given answer key."""
    out = {}
    for slice_name, mask in C.slices(ds, rows, distance).items():
        if slice_name in ("all", "not cuprates"):
            out[slice_name] = T.score(label[mask], tc[mask], prediction[0][mask], prediction[1][mask])
    return out


# ---------------------------------------------------------------------------
# 4. Final models and the made-metals order.

def final_models(ds: C.Dataset, X: np.ndarray, settings: dict, model_dir: Path, seed_offset: int = 0,
                 extra: dict | None = None) -> tuple[list, list]:
    """One model per development fold held back for early stopping; locked rows are never fitted."""
    import joblib

    dev = ds.split == "dev"
    models, info = [], []
    model_dir.mkdir(parents=True, exist_ok=True)
    X_all, arrays_all = (X, ds.arrays) if extra is None else extended(ds, X, extra)
    for fold in sorted(set(ds.fold[dev].tolist())):
        valid = dev & (ds.fold == fold)
        fit = ((ds.split == "train_only") | dev) & ~valid
        assert not (fit & (ds.split == "lockbox")).any()
        fit_rows = np.flatnonzero(fit)
        if extra is not None:
            use = added_rows_for(extra, fold, set(ds.arrays.group[valid].tolist()))
            fit_rows = np.concatenate([fit_rows, len(ds.rows) + use])
        model, fold_info = C.boosting_fit(X_all, arrays_all, fit_rows, np.flatnonzero(valid),
                                          seed=fold + seed_offset, **settings)
        joblib.dump(model, model_dir / f"boosting-fold{fold}.joblib")
        models.append(model)
        info.append({"validation_fold": fold, "fit_rows": int(len(fit_rows)), **fold_info})
    return models, info


# ---------------------------------------------------------------------------
# 4b. A block of added training rows (cycle 4): never scored as development rows, never tied to a locked group.

def extended(ds: C.Dataset, X: np.ndarray, extra: dict) -> tuple[np.ndarray, SimpleNamespace]:
    """The table's rows followed by the added rows: features, and the labels and Tc that boosting reads."""
    return (np.vstack([X, extra["X"]]),
            SimpleNamespace(label=np.concatenate([ds.arrays.label, extra["label"]]),
                            tc_k=np.concatenate([ds.arrays.tc_k, extra["tc_k"]])))


def added_rows_for(extra: dict, held_fold: int, validation_groups: set[str]) -> np.ndarray:
    """Added rows a round may fit: not of the held-out fold, and not tied to a group in its early-stopping rows."""
    return np.array([i for i, (fold, tied) in enumerate(zip(extra["fold"], extra["tied"]))
                     if fold != held_fold and not (tied & validation_groups)], dtype=int)


def load_additions(path: Path, ds: C.Dataset) -> dict:
    """The added rows written by ``build_nims_additions.py``, refused if any is in the table or tied to a locked group."""
    import build_mp_candidates as M

    rows = C.read_table(path)
    have = {row["composition"] for row in ds.rows}
    locked = {row["group"] for row in ds.rows if row["split"] == "lockbox"}
    tied = [set(row["tied_groups"].split(";")) for row in rows]
    clash = [row["composition"] for row, groups in zip(rows, tied) if row["composition"] in have or groups & locked]
    if clash:
        raise SystemExit(f"{len(clash)} added rows are in the table or tied to a locked group, e.g. {clash[:5]}")
    fractions = [C.fractions_of(row["composition"]) for row in rows]
    X, _ = C.formula_features([C.formula_text(f) for f in fractions])
    return {"source": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "keys": [row["composition"] for row in rows], "X": X,
            "label": np.array([float(row["label"]) for row in rows]),
            "tc_k": np.array([float(row["tc_k"]) if row["tc_k"] != "" else np.nan for row in rows]),
            "fold": np.array([int(row["fold"]) for row in rows]), "tied": tied,
            "group": np.array([row["group"] for row in rows]),
            "cuprate": np.array([M.family(set(f)) == "Cuprates" for f in fractions]), "score": True}


def duplicate_rows(ds: C.Dataset, X: np.ndarray, count: int, seed: int) -> dict:
    """A placebo block: ``count`` random labelled development rows a second time. Same size, no new information."""
    pool = np.flatnonzero((ds.split == "dev") & np.isfinite(ds.arrays.label))
    chosen = np.sort(np.random.default_rng(seed).choice(pool, size=count, replace=False))
    return {"source": f"placebo: {count} development rows duplicated, seed {seed}", "sha256": None,
            "keys": [ds.rows[i]["composition"] for i in chosen], "X": X[chosen],
            "label": ds.arrays.label[chosen].copy(), "tc_k": ds.arrays.tc_k[chosen].copy(),
            "fold": ds.fold[chosen].astype(int), "tied": [{str(g)} for g in ds.arrays.group[chosen]],
            "group": ds.arrays.group[chosen], "cuprate": ds.family[chosen] == "Cuprates", "score": False}


def score_added_rows(extra: dict, resamples: int = 1000) -> dict:
    """E1: AUC on the added rows, each predicted by the round that held its fold out, with and without the others."""
    y, with_, without = extra["label"], extra["p_with"], extra["p_without"]
    out = {}
    for name, keep in (("all", np.ones(len(y), bool)), ("not cuprates", ~extra["cuprate"])):
        yk, a, b, groups = y[keep], with_[keep], without[keep], extra["group"][keep]
        gap = C.group_bootstrap(groups, lambda s: T.roc_auc(yk[s], a[s]) - T.roc_auc(yk[s], b[s]), resamples)
        out[name] = {"rows": int(keep.sum()), "superconductors": int((yk == 1).sum()), "nulls": int((yk == 0).sum()),
                     "auc_with_added_rows": T.roc_auc(yk, a), "auc_without": T.roc_auc(yk, b),
                     "difference": T.roc_auc(yk, a) - T.roc_auc(yk, b), "difference_interval_95": gap}
    return out


def candidate_pool(path: Path, known_elements: set[str]) -> tuple[list[dict], list[dict[str, float]]]:
    """One row per composition (the lowest-energy structure), as in run_scratchnet.py rank."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        pool = list(csv.DictReader(handle))
    best: dict[str, tuple[float, dict, dict[str, float]]] = {}
    for row in pool:
        try:
            fractions = C.D_parse(row["formula"])
        except ValueError:
            continue
        if not set(fractions) <= known_elements or len(fractions) > C.MAX_ELEMENTS:
            continue
        key = C.formula_text(fractions)
        try:
            hull = float(row["energy_above_hull_ev_per_atom"])
        except ValueError:
            hull = float("inf")
        if key not in best or hull < best[key][0] or (row.get("known_match", "") and not best[key][1].get("known_match", "")):
            best[key] = (hull, row, fractions)
    keys = sorted(best)
    return [best[key][1] for key in keys], [best[key][2] for key in keys]


def made_metals_order(rows: list[dict], compositions: list[dict[str, float]], p: np.ndarray,
                      tc: np.ndarray) -> list[dict]:
    """Untested made metals by the chance of superconducting (rounded as printed), ties by probability x Tc."""
    expected = p * tc
    records = []
    for i in np.argsort(-expected, kind="mergesort"):
        row = rows[i]
        if row.get("known_match", ""):
            continue
        record = {"mp_id": row["mp_id"], "formula": row["formula"], "family": C.family_of(set(compositions[i])),
                  "element_system": "-".join(sorted(compositions[i])), "is_metal": row.get("is_metal", ""),
                  "theoretical": row.get("theoretical", ""), "p_superconductor": round(float(p[i]), 3),
                  "tc_if_superconductor_k": round(float(tc[i]), 2), "expected_tc_k": round(float(expected[i]), 2)}
        if made_metal(record):
            records.append(record)
    ordered = sorted(records, key=lambda r: -r["p_superconductor"])
    for rank, record in enumerate(ordered, start=1):
        record["rank"] = rank
    return ordered


def composition_key(formula: str) -> str:
    """One spelling per composition (PtPb3Bi and BiPtPb3 are the same compound)."""
    return C.formula_text(C.D_parse(formula))


def compare_orders(previous: list[dict], now: list[dict], watch: tuple[str, ...] = WATCH) -> dict:
    """How the made-metals order moved: overlap of the heads, rank correlation, largest moves, watched compounds."""
    before = {composition_key(r["formula"]): r["rank"] for r in previous}
    after = {composition_key(r["formula"]): r["rank"] for r in now}
    common = sorted(before.keys() & after.keys())
    overlap = {f"top_{n}": len({composition_key(r["formula"]) for r in previous[:n]} &
                               {composition_key(r["formula"]) for r in now[:n]}) for n in (10, 50, 200)}
    rho = float(T.spearman(np.array([before[f] for f in common], float), np.array([after[f] for f in common], float))) \
        if len(common) > 2 else float("nan")
    head = [f for f in common if min(before[f], after[f]) <= 200]
    moves = sorted(head, key=lambda f: -abs(before[f] - after[f]))[:15]
    return {"compounds_in_both": len(common), "entered": sorted(after.keys() - before.keys())[:50],
            "left": sorted(before.keys() - after.keys())[:50], "head_overlap": overlap,
            "rank_spearman": rho,
            "largest_moves_touching_top_200": [{"formula": f, "before": before[f], "after": after[f]} for f in moves],
            "watched": {f: {"before": before.get(composition_key(f)), "after": after.get(composition_key(f))}
                        for f in watch}}


# ---------------------------------------------------------------------------

def placebo(ds: C.Dataset, flips: int, seed: int) -> list[str]:
    """Flip ``flips`` random development non-cuprate negatives to superconductors at 0.9 K, in memory only.

    A control with no evidence behind it: the size of change one real cycle makes, put in random places. The files
    on disk are not touched, and the cycle's saved table state is the unchanged one.
    """
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero((ds.split == "dev") & (ds.arrays.label == 0) & (ds.family != "Cuprates"))
    chosen = rng.choice(pool, size=flips, replace=False)
    ds.arrays.label[chosen] = 1.0
    ds.arrays.tc_k[chosen] = 0.9
    return [ds.rows[i]["composition"] for i in chosen]


def save_state(name: str, state: dict, keys: list[str], prediction: tuple, order: list[dict]) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with gzip.open(WORK / f"{name}-state.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(state, handle)
    np.savez_compressed(WORK / f"{name}-oof.npz", compositions=np.array(keys), p=prediction[0], tc=prediction[1])
    with gzip.open(WORK / f"{name}-made-metals.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(order, handle)


def load_previous(name: str) -> tuple[dict, dict, list[dict]]:
    with gzip.open(WORK / f"{name}-state.json.gz", "rt", encoding="utf-8") as handle:
        state = json.load(handle)
    with np.load(WORK / f"{name}-oof.npz") as saved:
        oof = {str(k): (float(p), float(t)) for k, p, t in zip(saved["compositions"], saved["p"], saved["tc"])}
    with gzip.open(WORK / f"{name}-made-metals.json.gz", "rt", encoding="utf-8") as handle:
        order = json.load(handle)
    return state, oof, order


def run(name: str, previous: str | None, seed_offset: int = 0, placebo_flips: int = 0, placebo_seed: int = 0,
        additions: Path | None = None, duplicates: int = 0) -> dict:
    import joblib  # noqa: F401  (fail early if missing)

    started = time.time()
    ds = C.load_dataset()
    settings = boosting_settings()
    state = table_state(ds)
    report: dict = {"name": name, "previous": previous, "seed_offset": seed_offset, "data_sha256": ds.sha256,
                    "boosting_settings": settings,
                    "locked_rows": "neither trained on nor scored"}
    if placebo_flips:
        report["placebo"] = placebo(ds, placebo_flips, placebo_seed)
    prev_state = prev_oof = prev_order = None
    if previous:
        prev_state, prev_oof, prev_order = load_previous(previous)
        comparison = compare_tables(prev_state, state)
        check_unchanged_layout(comparison)
        report["table_change"] = comparison

    formulas = [C.formula_text(C.fractions_of(row["composition"])) for row in ds.rows]
    X, _ = C.formula_features(formulas, cache=C.WORK / "formula-features.npz")
    extra = None
    if additions and duplicates:
        raise SystemExit("--additions and --duplicate-rows are separate runs")
    if additions:
        extra = load_additions(additions, ds)
    elif duplicates:
        extra = duplicate_rows(ds, X, duplicates, placebo_seed)
    dev, prediction, distance, cv_info = cross_validate(ds, X, settings, seed_offset, extra)
    keys = [ds.rows[i]["composition"] for i in dev]
    label, tc = ds.arrays.label[dev], ds.arrays.tc_k[dev]
    not_cuprate = ds.family[dev] != "Cuprates"
    report["cross_validation"] = {
        "development_rows": int(len(dev)), "labelled_rows": int(np.isfinite(label).sum()),
        "superconductors": int((label == 1).sum()), "folds": cv_info,
        "scores": score_with_labels(ds, dev, distance, prediction, label, tc),
        "intervals": {"all": C.intervals(ds, dev, prediction),
                      "not cuprates": C.intervals(ds, dev, prediction, not_cuprate)}}
    if previous:
        before = aligned(prev_oof, keys)
        old_label = np.array([float(prev_state[k]["label"]) if prev_state[k]["label"] != "" else np.nan for k in keys])
        old_tc = np.array([float(prev_state[k]["tc_k"]) if prev_state[k]["tc_k"] != "" else np.nan for k in keys])
        changed = {c["composition"] for c in report["table_change"]["changed"]}
        report["cross_validation"]["previous_model"] = {
            "scored_against_previous_labels": score_with_labels(ds, dev, distance, before, old_label, old_tc),
            "scored_against_todays_labels": score_with_labels(ds, dev, distance, before, label, tc),
            "this_model_minus_previous_todays_labels": {
                "all": C.paired_differences(ds, dev, prediction, before),
                "not cuprates": C.paired_differences(ds, dev, prediction, before, not_cuprate)},
            "corrected_development_rows": [
                {"composition": k, "label_now": state[k]["label"], "p_previous_model": round(prev_oof[k][0], 4),
                 "p_this_model": round(float(prediction[0][i]), 4)}
                for i, k in enumerate(keys) if k in changed]}

    if extra is not None:
        report["added_rows"] = {"source": extra["source"], "sha256": extra["sha256"], "rows": len(extra["keys"]),
                                "superconductors": int((extra["label"] == 1).sum()),
                                "nulls": int((extra["label"] == 0).sum()),
                                "scored_as_development_rows": False}
        if extra["score"]:
            report["added_rows"]["E1_held_out_added_rows"] = score_added_rows(extra)
            if previous:                                  # the no-additions models must be the previous cycle's
                report["added_rows"]["largest_gap_to_previous_cycle_without_added_rows"] = float(
                    np.nanmax(np.abs(extra["p_dev_without"] - aligned(prev_oof, keys)[0])))
    models, final_info = final_models(ds, X, settings, WORK / "models" / name, seed_offset, extra)
    rows, compositions = candidate_pool(CANDIDATES, set(ds.table.symbols))
    X_pool, _ = C.formula_features([C.formula_text(c) for c in compositions], cache=C.WORK / "candidate-features.npz")
    applied = [C.boosting_apply(model, X_pool) for model in models]
    p_pool = np.mean([p for p, _ in applied], axis=0)
    tc_pool = C.log_mean_tc([t for _, t in applied])
    order = made_metals_order(rows, compositions, p_pool, tc_pool)
    report["final_models"] = final_info
    report["made_metals"] = {"untested_made_metals": len(order), "top_10": [
        {k: r[k] for k in ("rank", "formula", "p_superconductor", "tc_if_superconductor_k")} for r in order[:10]]}
    if previous:
        report["made_metals"]["against_previous"] = compare_orders(prev_order, order)
    else:
        ranks = {composition_key(r["formula"]): r["rank"] for r in order}
        report["made_metals"]["watched"] = {f: ranks.get(composition_key(f)) for f in WATCH}
    save_state(name, state, keys, prediction, order)
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["rank", "formula", "mp_id", "element_system", "p_superconductor", "tc_if_superconductor_k",
              "expected_tc_k"]
    C.write_table(OUT / f"{name}-made-metals-top50.csv", order[:50], fields)
    report["minutes"] = round((time.time() - started) / 60.0, 1)
    out = OUT / f"{name}.json"
    out.write_text(json.dumps(rounded(report), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True, help="this cycle, e.g. cycle1")
    parser.add_argument("--previous", help="the cycle to compare with, e.g. cycle0 (omit for the baseline)")
    parser.add_argument("--placebo-flips", type=int, default=0,
                        help="a control: in memory only, mark this many random development non-cuprate negatives as "
                             "superconductors at 0.9 K, to see how much any change of that size moves the results")
    parser.add_argument("--placebo-seed", type=int, default=0)
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="added to every model's random seed; a run on an unchanged table with another offset "
                             "measures how much of a change is training noise")
    parser.add_argument("--additions", type=Path,
                        help="a block of added training rows from build_nims_additions.py (cycle 4); the table itself "
                             "is unchanged and the added rows are never scored as development rows")
    parser.add_argument("--duplicate-rows", type=int, default=0,
                        help="a control for --additions: add this many random development rows a second time "
                             "(--placebo-seed picks them)")
    args = parser.parse_args()
    additions = args.additions.resolve() if args.additions else None
    run(args.name, args.previous, args.seed_offset, args.placebo_flips, args.placebo_seed, additions,
        args.duplicate_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
