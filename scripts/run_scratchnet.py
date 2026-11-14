#!/usr/bin/env python3
"""Train and test the from-scratch composition network.

``cv``    Grouped five-fold test on the development groups. For each fold the
          network, gradient boosting on formula features and a
          nearest-neighbour lookup are trained on the same rows and scored on
          the same held-out groups. The locked test set is not read.

``final`` Train the final ensemble on every development group and score it once
          on the locked test set. Refuses to run a second time. With
          ``--rehearse`` it treats development fold 0 as the test set instead,
          to check the code path without touching the locked set.

``rank``  Score the Materials Project candidates with the final ensemble.

Results go to ``docs/scratchnet-<name>.json``; out-of-fold predictions and
model files go under ``data/processed/scratchnet/`` and are not committed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

import scratchnet_common as C
from scratchnet_common import ROOT, T, Config

# Named network variants: model settings, training settings that differ from the defaults, and two switches:
# "formula" feeds the whole-formula features to the network, "aux" adds Materials Project rows with
# calculated properties as auxiliary targets.
ATTENTION = dict(width=64, blocks=2, heads=4, pool_heads=2, trunk=(128, 64), dropout=0.1)
TUNED = dict(weight_decay=0.02, ema_decay=0.995, clip_norm=1.0, init_class_bias=True)
VARIANTS: dict[str, dict] = {
    "attention": {"model": ATTENTION},                               # round 1
    "deepsets": {"model": {**ATTENTION, "blocks": 0}},               # round 1, no attention
    "tuned": {"model": ATTENTION, "train": TUNED},                   # round 2: more regularization, weight averaging
    "multitask": {"model": ATTENTION, "train": TUNED, "aux": True},
    "hybrid": {"model": ATTENTION, "train": TUNED, "formula": True},
    "hybrid-multitask": {"model": ATTENTION, "train": TUNED, "formula": True, "aux": True},
}
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
SELECTION_METRIC = "expected_tc_spearman"       # how well probability x Tc orders the materials by their true Tc
SELECTION_SLICE = "not cuprates"
TRAIN_DEFAULTS = dict(epochs=80, batch_size=128, lr=2e-3, weight_decay=1e-4, warmup_epochs=3, patience=12,
                      min_epochs=10, tc_weight=1.0)
_STATE: dict = {}


def model_config(ds: C.Dataset, variant: str) -> Config:
    settings = VARIANTS[variant]
    return Config(n_elements=len(ds.table.symbols), n_features=ds.table.features.shape[1],
                  n_global=_STATE["X"].shape[1] if settings.get("formula") else 0,
                  n_aux=len(C.AUX_TARGETS) if settings.get("aux") else 0, **settings["model"])


def prepare(variants: list[str]) -> C.Dataset:
    """Load everything the chosen variants need into ``_STATE`` (shared with worker processes by fork)."""
    ds = C.load_dataset()
    formulas = [C.formula_text(C.fractions_of(row["composition"])) for row in ds.rows]
    X, names = C.formula_features(formulas, cache=C.WORK / "formula-features.npz")
    uses_aux = [v for v in variants if VARIANTS[v].get("aux")]
    aux = C.load_aux(ds.table, CANDIDATES, with_features=any(VARIANTS[v].get("formula") for v in uses_aux)) \
        if uses_aux else None
    _STATE.update(ds=ds, X=X, feature_names=names, aux=aux)
    return ds


def fit_network(variant: str, seed: int, fit: np.ndarray, valid: np.ndarray, test: np.ndarray,
                save: Path | None = None) -> tuple[tuple[np.ndarray, np.ndarray], dict]:
    ds = _STATE["ds"]
    (prediction,), info = C.network_fit_predict(
        ds, fit, valid, [test], model_config(ds, variant), train_config(variant, seed), X=_STATE["X"],
        aux=_STATE["aux"] if VARIANTS[variant].get("aux") else None, save=save)
    return prediction, info


def train_config(variant: str, seed: int) -> T.TrainConfig:
    return T.TrainConfig(**{**TRAIN_DEFAULTS, **VARIANTS[variant].get("train", {}), "seed": seed})


def fold_rows(ds: C.Dataset, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fit, validation for early stopping, test) row numbers for one development fold."""
    test = np.flatnonzero((ds.split == "dev") & (ds.fold == fold))
    train = np.flatnonzero((ds.split == "train_only") | ((ds.split == "dev") & (ds.fold != fold)))
    fit, valid = T.inner_split(ds.arrays.group[train], percent=12, seed=fold)
    return train[fit], train[valid], test


def _network_task(task: tuple[int, str, int]) -> dict:
    fold, variant, seed = task
    ds = _STATE["ds"]
    fit, valid, test = fold_rows(ds, fold)
    started = time.time()
    prediction, info = fit_network(variant, seed, fit, valid, test)
    info.update(fold=fold, variant=variant, seed=seed, seconds=round(time.time() - started, 1))
    print(f"  network {variant} fold {fold} seed {seed}: best epoch {info['best_epoch']} of {info['epochs_run']}, "
          f"{info['seconds']:.0f} s", flush=True)
    return {"task": task, "prediction": prediction, "info": info}


def _boosting_task(task: tuple[int, int, str]) -> dict:
    """One gradient-boosting model (classifier or regressor) for one grid setting and one development fold."""
    setting, fold, part = task
    ds = _STATE["ds"]
    fit, valid, test = fold_rows(ds, fold)
    key = "classifier_settings" if part == "classifier" else "regressor_settings"
    models, info = C.boosting_fit(_STATE["X"], ds.arrays, fit, valid, seed=fold, parts=part,
                                  **{key: C.BOOSTING_GRID[setting]})
    p, tc = C.boosting_apply(models, _STATE["X"][test])
    return {"task": task, "prediction": p if part == "classifier" else tc, "info": info}


def tune_boosting(ds: C.Dataset, dev: np.ndarray, position: dict, folds: list[int], pool) -> dict:
    """Try every grid setting on the development folds; keep the best classifier and the best regressor."""
    tasks = [(setting, fold, part) for setting in range(len(C.BOOSTING_GRID)) for fold in folds
             for part in ("classifier", "regressor")]
    results = pool.map(_boosting_task, tasks, chunksize=1)
    label, tc = ds.arrays.label[dev], ds.arrays.tc_k[dev]
    labelled, positive = np.isfinite(label), (label == 1) & np.isfinite(tc)
    table = []
    for setting in range(len(C.BOOSTING_GRID)):
        p_all, tc_all = np.full(len(dev), np.nan), np.full(len(dev), np.nan)
        for result in results:
            s_, fold, part = result["task"]
            if s_ == setting:
                where = np.array([position[int(row)] for row in fold_rows(ds, fold)[2]])
                (p_all if part == "classifier" else tc_all)[where] = result["prediction"]
        table.append({"settings": C.BOOSTING_GRID[setting], "auc": T.roc_auc(label[labelled], p_all[labelled]),
                      "tc_log_mae": float(np.mean(np.abs(np.log1p(tc_all[positive]) - np.log1p(tc[positive]))))})
    best_c = max(range(len(table)), key=lambda i: table[i]["auc"])
    best_r = min(range(len(table)), key=lambda i: table[i]["tc_log_mae"])
    return {"grid": table, "classifier": C.BOOSTING_GRID[best_c], "regressor": C.BOOSTING_GRID[best_r]}


def choose(ds: C.Dataset, dev: np.ndarray, predictions: dict, variants: list[str]) -> dict:
    """Pick the network variant and the blend weight on development rows, by one metric fixed in advance."""
    keep = ds.family[dev] != "Cuprates"
    label, tc = ds.arrays.label[dev][keep], ds.arrays.tc_k[dev][keep]

    def metric(prediction: tuple[np.ndarray, np.ndarray]) -> float:
        return T.score(label, tc, prediction[0][keep], prediction[1][keep])[SELECTION_METRIC]

    table = {variant: {f"{w:g}": metric(C.blend(predictions[f"network {variant}"], predictions["boosting"], w))
                       for w in BLEND_WEIGHTS} for variant in variants}
    variant = max(variants, key=lambda v: table[v]["1"])                  # best network on its own
    weight = max(BLEND_WEIGHTS, key=lambda w: table[variant][f"{w:g}"])   # then the best mix with boosting
    return {"metric": f"{SELECTION_METRIC} on development rows, {SELECTION_SLICE}",
            "rule": "variant = best network alone (weight 1); blend weight = best for that variant; "
                    "weight 0 means boosting alone ranks better and the network is only a second opinion",
            "by_variant_and_weight_on_network": table, "variant": variant, "blend_weight": weight}


def rounded(value, digits: int = 4):
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item, digits) for item in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else round(float(value), digits)
    if isinstance(value, np.integer):
        return int(value)
    return value


def run_cv(name: str, variants: list[str], seeds: list[int], jobs: int, out_dir: Path, tune: bool = False) -> dict:
    started = time.time()
    ds = prepare(variants)
    X, feature_names = _STATE["X"], _STATE["feature_names"]
    dev = np.flatnonzero(ds.split == "dev")
    position = {int(row): i for i, row in enumerate(dev)}
    folds = sorted(set(ds.fold[dev].tolist()))

    def empty() -> tuple[np.ndarray, np.ndarray]:
        return np.full(len(dev), np.nan), np.full(len(dev), np.nan)

    predictions = {"boosting": empty(), "neighbour": empty()}
    distance = np.full(len(dev), np.nan)
    boosting_info = []
    pool = multiprocessing.get_context("fork").Pool(jobs)
    tuned = tune_boosting(ds, dev, position, folds, pool) if tune else None
    chosen = {"classifier_settings": tuned["classifier"], "regressor_settings": tuned["regressor"]} if tune else {}
    for fold in folds:
        fit, valid, test = fold_rows(ds, fold)
        where = np.array([position[int(row)] for row in test])
        train = np.concatenate([fit, valid])
        p, tc, info = C.boosting_predict(X, ds.arrays, fit, valid, X[test], seed=fold, **chosen)
        predictions["boosting"][0][where], predictions["boosting"][1][where] = p, tc
        boosting_info.append({"fold": fold, **info})
        p, tc = C.neighbour_predict(ds, train, ds.dense[test])
        predictions["neighbour"][0][where], predictions["neighbour"][1][where] = p, tc
        distance[where] = C.nearest_distance(ds.dense[train], ds.dense[test])[0]
        print(f"fold {fold}: {len(fit)} fit, {len(valid)} validation, {len(test)} test rows; boosting {info}", flush=True)

    tasks = [(fold, variant, seed) for variant in variants for seed in seeds for fold in folds]
    results = pool.map(_network_task, tasks, chunksize=1)
    pool.close()
    network_info = [result["info"] for result in results]
    for variant in variants:
        per_seed = []
        for seed in seeds:
            p_all, tc_all = empty()
            for result in results:
                fold, v, s = result["task"]
                if v == variant and s == seed:
                    where = np.array([position[int(row)] for row in fold_rows(ds, fold)[2]])
                    p_all[where], tc_all[where] = result["prediction"]
            per_seed.append((p_all, tc_all))
            predictions[f"network {variant} seed {seed}"] = (p_all, tc_all)
        predictions[f"network {variant}"] = (np.mean([p for p, _ in per_seed], axis=0),
                                             C.log_mean_tc([tc for _, tc in per_seed]))
        predictions[f"blend {variant} + boosting"] = C.blend(predictions[f"network {variant}"], predictions["boosting"])

    scores = C.score_table(ds, dev, distance, predictions)
    fold_auc = {}
    for model, (p, _) in predictions.items():
        fold_auc[model] = [T.roc_auc(ds.arrays.label[dev][m], p[m]) for m in
                           ((ds.fold[dev] == fold) & np.isfinite(ds.arrays.label[dev]) for fold in folds)]
    not_cuprate = ds.family[dev] != "Cuprates"
    differences = {}
    for variant in variants:
        for first in (f"network {variant}", f"blend {variant} + boosting"):
            differences[f"{first} minus boosting"] = {
                "all": C.paired_differences(ds, dev, predictions[first], predictions["boosting"]),
                "not cuprates": C.paired_differences(ds, dev, predictions[first], predictions["boosting"], not_cuprate)}
    C.WORK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(C.WORK / f"{name}-oof.npz", rows=dev, distance=distance,
                        **{f"p::{model}": p for model, (p, _) in predictions.items()},
                        **{f"tc::{model}": tc for model, (_, tc) in predictions.items()})
    report = {
        "name": name, "data_sha256": ds.sha256, "development_rows": int(len(dev)),
        "labelled_rows": int(np.isfinite(ds.arrays.label[dev]).sum()), "folds": folds, "seeds": seeds,
        "formula_features": len(feature_names),
        "variants": {variant: {"model": VARIANTS[variant]["model"],
                               "train": {**TRAIN_DEFAULTS, **VARIANTS[variant].get("train", {})},
                               "formula_features_as_input": bool(VARIANTS[variant].get("formula")),
                               "materials_project_auxiliary_rows": len(_STATE["aux"].arrays)
                               if VARIANTS[variant].get("aux") else 0,
                               "parameters": next(i["parameters"] for i in network_info if i["variant"] == variant)}
                     for variant in variants},
        "selection": {**choose(ds, dev, predictions, variants),
                      "boosting": {"classifier": chosen.get("classifier_settings", C.BOOSTING_DEFAULT),
                                   "regressor": chosen.get("regressor_settings", C.BOOSTING_DEFAULT)}},
        "boosting_grid": tuned["grid"] if tune else None,
        "distance_bands_rows": {band: int(mask.sum()) for band, mask in C.slices(ds, dev, distance).items()},
        "scores": scores, "auc_by_fold": fold_auc, "differences": differences,
        "boosting": boosting_info,
        "network_runs": [{k: v for k, v in info.items() if k != "tc_scale"} for info in network_info],
        "minutes": round((time.time() - started) / 60.0, 1),
    }
    out = out_dir / f"scratchnet-{name}.json"
    out.write_text(json.dumps(rounded(report), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return report


LOCKBOX_REPORT = ROOT / "docs/scratchnet-lockbox.json"
CANDIDATES = ROOT / "data/candidates/mp-candidates.csv.gz"


def final_rows(ds: C.Dataset, valid_fold: int, rehearse: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fit, validation for early stopping, test). The test set is the locked one unless rehearsing."""
    dev = ds.split == "dev"
    test = dev & (ds.fold == 0) if rehearse else ds.split == "lockbox"
    valid = dev & (ds.fold == valid_fold)
    fit = ((ds.split == "train_only") | dev) & ~valid & ~test
    return np.flatnonzero(fit), np.flatnonzero(valid), np.flatnonzero(test)


def _final_task(task: tuple[str, int, int]) -> dict:
    variant, seed, fold = task
    ds, rehearse, model_dir = _STATE["ds"], _STATE["rehearse"], _STATE["model_dir"]
    fit, valid, test = final_rows(ds, fold, rehearse)
    started = time.time()
    prediction, info = fit_network(variant, seed, fit, valid, test,
                                   save=model_dir / f"network-{variant}-fold{fold}-seed{seed}.npz")
    info.update(fold=fold, variant=variant, seed=seed, seconds=round(time.time() - started, 1))
    print(f"  network {variant} validation fold {fold} seed {seed}: best epoch {info['best_epoch']} of "
          f"{info['epochs_run']}, {info['seconds']:.0f} s", flush=True)
    return {"task": task, "prediction": prediction, "info": info}


def run_final(name: str, variants: list[str], seeds: list[int], jobs: int, blend_weight: float, rehearse: bool,
              out_dir: Path, boosting: dict | None = None) -> dict:
    import joblib

    boosting = boosting or {"classifier": C.BOOSTING_DEFAULT, "regressor": C.BOOSTING_DEFAULT}

    report_path = out_dir / ("scratchnet-rehearsal.json" if rehearse else LOCKBOX_REPORT.name)
    if not rehearse and LOCKBOX_REPORT.exists():
        raise SystemExit(f"{LOCKBOX_REPORT} exists: the locked test set has already been used once.\n"
                         "Scoring it again after seeing that result would make it an ordinary validation set.")
    started = time.time()
    ds = prepare(variants)
    model_dir = C.WORK / "models" / (name + ("-rehearsal" if rehearse else ""))
    model_dir.mkdir(parents=True, exist_ok=True)
    _STATE.update(rehearse=rehearse, model_dir=model_dir)
    X = _STATE["X"]
    folds = [fold for fold in sorted(set(ds.fold[ds.split == "dev"].tolist())) if not (rehearse and fold == 0)]
    test = final_rows(ds, folds[0], rehearse)[2]
    train_all = np.flatnonzero(((ds.split == "train_only") | (ds.split == "dev")) & ~np.isin(np.arange(len(ds.rows)), test))
    distance = C.nearest_distance(ds.dense[train_all], ds.dense[test])[0]

    boosted, boosting_info = [], []
    for fold in folds:
        fit, valid, _ = final_rows(ds, fold, rehearse)
        models, info = C.boosting_fit(X, ds.arrays, fit, valid, seed=fold, classifier_settings=boosting["classifier"],
                                      regressor_settings=boosting["regressor"])
        joblib.dump(models, model_dir / f"boosting-fold{fold}.joblib")
        boosted.append(C.boosting_apply(models, X[test]))
        boosting_info.append({"validation_fold": fold, **info})
    predictions = {"boosting": (np.mean([p for p, _ in boosted], axis=0), C.log_mean_tc([tc for _, tc in boosted])),
                   "neighbour": C.neighbour_predict(ds, train_all, ds.dense[test])}
    tasks = [(variant, seed, fold) for variant in variants for seed in seeds for fold in folds]
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        results = pool.map(_final_task, tasks, chunksize=1)
    spread = {}
    for variant in variants:
        members = [result["prediction"] for result in results if result["task"][0] == variant]
        predictions[f"network {variant}"] = (np.mean([p for p, _ in members], axis=0),
                                             C.log_mean_tc([tc for _, tc in members]))
        spread[variant] = np.std([p for p, _ in members], axis=0)
    primary = f"network {variants[0]}"
    predictions["blend"] = C.blend(predictions[primary], predictions["boosting"], blend_weight)

    not_cuprate = ds.family[test] != "Cuprates"
    report = {
        "name": name, "rehearsal": rehearse, "data_sha256": ds.sha256,
        "test_set": "development fold 0 (rehearsal)" if rehearse else "locked test set, scored once",
        "test_rows": int(len(test)), "test_groups": int(len(set(ds.arrays.group[test].tolist()))),
        "training_rows": int(len(train_all)), "validation_folds": folds, "seeds": seeds,
        "ensemble": {variant: len(seeds) * len(folds) for variant in variants},
        "blend": {"network": primary, "weight_on_network": blend_weight},
        "boosting_settings": boosting,
        "variants": {variant: {"model": VARIANTS[variant]["model"],
                               "train": {**TRAIN_DEFAULTS, **VARIANTS[variant].get("train", {})}}
                     for variant in variants},
        "distance_bands_rows": {band: int(mask.sum()) for band, mask in C.slices(ds, test, distance).items()},
        "scores": C.score_table(ds, test, distance, predictions),
        "intervals": {model: {"all": C.intervals(ds, test, prediction),
                              "not cuprates": C.intervals(ds, test, prediction, not_cuprate)}
                      for model, prediction in predictions.items()},
        "differences": {f"{first} minus boosting": {
            "all": C.paired_differences(ds, test, predictions[first], predictions["boosting"]),
            "not cuprates": C.paired_differences(ds, test, predictions[first], predictions["boosting"], not_cuprate)}
            for first in (primary, "blend")},
        "boosting": boosting_info,
        "network_runs": [{k: v for k, v in result["info"].items() if k != "tc_scale"} for result in results],
        "minutes": round((time.time() - started) / 60.0, 1),
    }
    np.savez_compressed(C.WORK / f"{name}-{'rehearsal' if rehearse else 'lockbox'}.npz", rows=test, distance=distance,
                        spread=spread[variants[0]],
                        **{f"p::{model}": p for model, (p, _) in predictions.items()},
                        **{f"tc::{model}": tc for model, (_, tc) in predictions.items()})
    (model_dir / "ensemble.json").write_text(json.dumps(
        {"variants": variants, "seeds": seeds, "folds": folds, "blend_weight": blend_weight,
         "data_sha256": ds.sha256}, indent=1) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(rounded(report), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    return report


def run_rank(name: str, candidates: Path, top: int, out_dir: Path) -> dict:
    import joblib

    ds = C.load_dataset()
    model_dir = C.WORK / "models" / name
    ensemble = json.loads((model_dir / "ensemble.json").read_text(encoding="utf-8"))
    if ensemble["data_sha256"] != ds.sha256:
        raise SystemExit("the saved models were trained on a different composition table")
    with gzip.open(candidates, "rt", encoding="utf-8", newline="") as handle:
        pool = list(csv.DictReader(handle))
    usable, compositions, skipped = [], [], 0
    known_elements = set(ds.table.symbols)
    for row in pool:
        try:
            fractions = C.D_parse(row["formula"])
        except ValueError:
            skipped += 1
            continue
        if not set(fractions) <= known_elements or len(fractions) > C.MAX_ELEMENTS:
            skipped += 1
            continue
        usable.append(row)
        compositions.append(fractions)
    index, fraction, mask = C.encode(compositions, ds.table, C.MAX_ELEMENTS)
    blank = np.full(len(usable), np.nan)
    arrays = T.Arrays(index, fraction, mask, blank, blank, np.array([""] * len(usable)))
    variant = ensemble["variants"][0]
    X, _ = C.formula_features([C.formula_text(c) for c in compositions], cache=C.WORK / "candidate-features.npz")
    members = []
    for path in sorted(model_dir.glob(f"network-{variant}-fold*-seed*.npz")):
        model, scale, scaler = C.load_network(path, ds.table)
        members.append(T.predict(model, C.with_extra(arrays, X, scaler) if scaler else arrays, scale))
    network = (np.mean([p for p, _ in members], axis=0), C.log_mean_tc([tc for _, tc in members]))
    spread = np.std([p for p, _ in members], axis=0)
    boosted = [C.boosting_apply(joblib.load(path), X) for path in sorted(model_dir.glob("boosting-fold*.joblib"))]
    boosting = (np.mean([p for p, _ in boosted], axis=0), C.log_mean_tc([tc for _, tc in boosted]))
    blended = C.blend(network, boosting, ensemble["blend_weight"])
    dense = C.dense_fractions(compositions, ds.table)
    labelled = np.flatnonzero(np.isfinite(ds.arrays.label))
    distance, nearest = C.nearest_distance(ds.dense[labelled], dense)
    nearest = labelled[nearest]
    expected = blended[0] * blended[1]
    families = np.array([C.family_of(set(c)) for c in compositions])
    untested = np.array([row.get("known_match", "") == "" for row in usable])
    order = [i for i in np.argsort(-expected, kind="mergesort") if untested[i]]
    fields = ["rank", "mp_id", "formula", "family", "expected_tc_k", "p_superconductor", "tc_if_superconductor_k",
              "network_p", "network_tc_k", "boosting_p", "boosting_tc_k", "network_disagreement",
              "distance_to_nearest_known", "nearest_known", "nearest_known_tc_k", "is_metal", "band_gap_ev",
              "energy_above_hull_ev_per_atom", "theoretical"]

    def record(rank: int, i: int) -> dict:
        row, twin = usable[i], ds.rows[nearest[i]]
        return {"rank": rank, "mp_id": row["mp_id"], "formula": row["formula"], "family": families[i],
                "expected_tc_k": round(float(expected[i]), 2), "p_superconductor": round(float(blended[0][i]), 3),
                "tc_if_superconductor_k": round(float(blended[1][i]), 2), "network_p": round(float(network[0][i]), 3),
                "network_tc_k": round(float(network[1][i]), 2), "boosting_p": round(float(boosting[0][i]), 3),
                "boosting_tc_k": round(float(boosting[1][i]), 2), "network_disagreement": round(float(spread[i]), 3),
                "distance_to_nearest_known": round(float(distance[i]), 3), "nearest_known": twin["composition"],
                "nearest_known_tc_k": twin["tc_k"] if twin["label"] == "1" else "0",
                "is_metal": row.get("is_metal", ""), "band_gap_ev": row.get("band_gap_ev", ""),
                "energy_above_hull_ev_per_atom": row.get("energy_above_hull_ev_per_atom", ""),
                "theoretical": row.get("theoretical", "")}

    ranked = [record(rank, i) for rank, i in enumerate(order, start=1)]
    C.write_table(C.WORK / f"{name}-candidates.csv.gz", ranked, fields)
    C.write_table(out_dir / "scratchnet-candidates-top.csv", ranked[:top], fields)
    far_first = [r for r in ranked if r["family"] != "Cuprates"][:top]
    C.write_table(out_dir / "scratchnet-candidates-top-not-cuprates.csv", far_first, fields)
    head = ranked[:top]
    agree = float(C.T.spearman(network[0][untested] * network[1][untested], boosting[0][untested] * boosting[1][untested]))
    summary = {
        "name": name, "candidates_in_pool": len(pool), "scored": len(usable), "skipped_unreadable_or_unknown_element": skipped,
        "untested_scored": int(untested.sum()), "network_members": len(members), "boosting_members": len(boosted),
        "network_vs_boosting_rank_agreement": agree,
        "top": top, "top_families": dict(sorted({f: sum(r["family"] == f for r in head) for f in set(families)}.items())),
        "top_median_distance_to_nearest_known": float(np.median([r["distance_to_nearest_known"] for r in head])),
        "top_within_0.1_of_a_known_composition": sum(r["distance_to_nearest_known"] < 0.1 for r in head),
        "top_beyond_0.3_of_any_known_composition": sum(r["distance_to_nearest_known"] >= 0.3 for r in head),
        "pool_median_distance_to_nearest_known": float(np.median(distance[untested])),
        "pool_share_beyond_0.3": float(np.mean(distance[untested] >= 0.3)),
        "highest_expected_tc_k": head[0]["expected_tc_k"] if head else None,
    }
    out = out_dir / "scratchnet-candidates.json"
    out.write_text(json.dumps(rounded(summary), indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rounded(summary), indent=1))
    return summary


def print_summary(report: dict) -> None:
    for slice_name in ("all", "not cuprates"):
        print(f"\n{slice_name}:")
        print(f"  {'model':34s} {'AUC':>6s} {'Tc MAE K':>9s} {'log MAE':>8s} {'Tc rank':>8s} {'top-10%':>8s}")
        for model, s in report["scores"][slice_name].items():
            print(f"  {model:34s} {s['auc']:6.3f} {s['tc_mae_k']:9.2f} {s['tc_log_mae']:8.3f} "
                  f"{s['tc_spearman']:8.3f} {s.get('top_decile_recall', float('nan')):8.3f}")
    for pair, by_slice in report["differences"].items():
        for slice_name, gaps in by_slice.items():
            a, t = gaps["auc"], gaps["tc_log_mae"]
            print(f"{pair} [{slice_name}]: AUC {a['median']:+.4f} ({a['low']:+.4f} to {a['high']:+.4f}); "
                  f"Tc log error {t['median']:+.4f} ({t['low']:+.4f} to {t['high']:+.4f})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    cv = sub.add_parser("cv", help="grouped five-fold test on the development groups")
    cv.add_argument("--name", required=True, help="label for the output files, e.g. round1")
    cv.add_argument("--variants", default="attention,deepsets")
    cv.add_argument("--seeds", default="0,1")
    cv.add_argument("--jobs", type=int, default=2)
    cv.add_argument("--tune-boosting", action="store_true", help="also try the gradient-boosting settings grid")
    cv.add_argument("--out-dir", type=Path, default=ROOT / "docs")
    final = sub.add_parser("final", help="train the final ensemble and score the locked test set once")
    final.add_argument("--name", required=True)
    final.add_argument("--selection", type=Path, required=True,
                       help="the cv report whose selection block fixes the variant, blend weight and boosting settings")
    final.add_argument("--seeds", default="0,1,2")
    final.add_argument("--jobs", type=int, default=2)
    final.add_argument("--rehearse", action="store_true", help="use development fold 0 in place of the locked set")
    final.add_argument("--out-dir", type=Path, default=ROOT / "docs")
    rank = sub.add_parser("rank", help="score the Materials Project candidates with the final ensemble")
    rank.add_argument("--name", required=True, help="name used for the final run")
    rank.add_argument("--candidates", type=Path, default=CANDIDATES)
    rank.add_argument("--top", type=int, default=200)
    rank.add_argument("--out-dir", type=Path, default=ROOT / "docs")
    args = parser.parse_args()
    if args.command == "rank":
        run_rank(args.name, args.candidates, args.top, args.out_dir)
        return 0
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.command == "final":
        selection = json.loads(args.selection.read_text(encoding="utf-8"))["selection"]
        report = run_final(args.name, [selection["variant"]], seeds, args.jobs, float(selection["blend_weight"]),
                           args.rehearse, args.out_dir, boosting=selection["boosting"])
    else:
        variants = args.variants.split(",")
        unknown = [v for v in variants if v not in VARIANTS]
        if unknown:
            raise SystemExit(f"unknown variants {unknown}; known: {sorted(VARIANTS)}")
        report = run_cv(args.name, variants, seeds, args.jobs, args.out_dir, tune=args.tune_boosting)
        print("selection:", json.dumps(rounded(report["selection"]), indent=1))
    print_summary(rounded(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
