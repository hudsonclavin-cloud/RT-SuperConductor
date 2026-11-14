"""Shared pieces for the from-scratch network's training rounds.

Loading the composition table, building the comparison models (gradient
boosting on formula features and a nearest-neighbour lookup), running every
model on the same grouped folds, and scoring with intervals.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "scripts", ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from rt_superconductor.scratchnet import train as T  # noqa: E402
from rt_superconductor.scratchnet.data import (  # noqa: E402
    ElementTable, encode, parse_formula as D_parse, parse_key, read_table, write_table)
from rt_superconductor.scratchnet.model import CompositionNet, Config  # noqa: E402

DATA = ROOT / "data/processed/scratchnet/compositions.csv.gz"
ELEMENTS = ROOT / "data/training/element-features.csv"
DATA_REPORT = ROOT / "docs/scratchnet-data.json"
WORK = ROOT / "data/processed/scratchnet"
MAX_ELEMENTS = 8
# These two change with how a formula is written (MgB2 against Mg0.5B), not with the material.
NOTATION_DEPENDENT = {"shell__descriptor_total_atoms", "lewis_dot_proxy_total_electrons_per_reduced_formula"}
DISTANCE_BANDS = (("near", 0.0, 0.1), ("middle", 0.1, 0.3), ("far", 0.3, 1.01))


@dataclass
class Dataset:
    rows: list[dict]
    table: ElementTable
    arrays: T.Arrays
    split: np.ndarray
    fold: np.ndarray
    family: np.ndarray
    dense: np.ndarray               # (rows, elements) atomic fractions, for distances
    sha256: str


def family_of(elements: set[str]) -> str:
    """The project's coarse chemical family for a set of elements."""
    from build_mp_candidates import family

    return family(elements)


def fractions_of(key: str) -> dict[str, float]:
    raw = parse_key(key)
    total = sum(raw.values())
    return {element: value / total for element, value in raw.items()}


def dense_fractions(compositions: list[dict[str, float]], table: ElementTable) -> np.ndarray:
    lookup = table.index
    dense = np.zeros((len(compositions), len(table.symbols) + 1))
    for row, composition in enumerate(compositions):
        for element, value in composition.items():
            dense[row, lookup[element]] = value
    return dense


def load_dataset(path: Path = DATA, elements: Path = ELEMENTS, report: Path | None = DATA_REPORT) -> Dataset:
    """Read the composition table. With ``report``, refuse a table whose checksum differs from the recorded one."""
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if report is not None:
        recorded = json.loads(Path(report).read_text(encoding="utf-8"))["output_sha256"]
        if recorded != sha:
            raise SystemExit(f"{path} does not match the checksum in {report}; the splits may have moved.\n"
                             "Rebuild with scripts/build_scratchnet_data.py and review the change.")
    rows = read_table(path)
    table = ElementTable.load(elements)
    compositions = [fractions_of(row["composition"]) for row in rows]
    index, fraction, mask = encode(compositions, table, MAX_ELEMENTS)
    label = np.array([float(row["label"]) if row["label"] != "" else np.nan for row in rows])
    tc = np.array([float(row["tc_k"]) if row["tc_k"] != "" else np.nan for row in rows])
    arrays = T.Arrays(index, fraction, mask, label, tc, np.array([row["group"] for row in rows]))
    return Dataset(rows, table, arrays, np.array([row["split"] for row in rows]),
                   np.array([int(row["fold"]) for row in rows]), np.array([row["family"] for row in rows]),
                   dense_fractions(compositions, table), sha)


def formula_text(fractions: dict[str, float]) -> str:
    return "".join(f"{element}{value:.3f}" for element, value in sorted(fractions.items()) if round(value, 3) > 0)


def formula_features(formulas: list[str], cache: Path | None = None) -> tuple[np.ndarray, list[str]]:
    """The project's Magpie and shell-filling formula features, minus the two that depend on notation."""
    digest = hashlib.sha256("\n".join(formulas).encode()).hexdigest()
    if cache is not None and cache.is_file():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["digest"]) == digest:
                return saved["X"], [str(name) for name in saved["names"]]
    from score_pool_physics import formula_feature_rows

    records = formula_feature_rows(formulas)
    names = [name for name in records[0] if name != "formula" and name not in NOTATION_DEPENDENT]
    X = np.array([[float(record.get(name, np.nan)) for name in names] for record in records])
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, names=np.array(names), digest=np.array(digest))
    return X, names


# ---------------------------------------------------------------------------
# The comparison models. Each returns (probability, Tc in kelvin) for the rows asked for.

def log_mean_tc(tc_list: list[np.ndarray]) -> np.ndarray:
    """Average several Tc predictions on the log(1 + Tc) scale."""
    return np.expm1(np.mean([np.log1p(tc) for tc in tc_list], axis=0))


BOOSTING_DEFAULT = dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0)
BOOSTING_GRID = [BOOSTING_DEFAULT,
                 {**BOOSTING_DEFAULT, "max_leaf_nodes": 15}, {**BOOSTING_DEFAULT, "max_leaf_nodes": 63},
                 {**BOOSTING_DEFAULT, "learning_rate": 0.02}, {**BOOSTING_DEFAULT, "min_samples_leaf": 5},
                 {**BOOSTING_DEFAULT, "max_leaf_nodes": 15, "l2_regularization": 10.0},
                 {**BOOSTING_DEFAULT, "max_leaf_nodes": 7, "learning_rate": 0.1}]


def boosting_fit(X: np.ndarray, ds_arrays: T.Arrays, fit_rows: np.ndarray, valid_rows: np.ndarray, seed: int = 0,
                 classifier_settings: dict | None = None, regressor_settings: dict | None = None,
                 parts: str = "both") -> tuple[tuple, dict]:
    """Gradient-boosted trees: a classifier on labelled rows and a log-Tc regressor on superconductors.

    Both stop adding trees when the score on ``valid_rows`` (whole held-back groups) stops improving.
    ``parts`` can be "classifier" or "regressor" to fit only one of them (the other is returned as None).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    label, tc = ds_arrays.label, ds_arrays.tc_k
    common = dict(max_iter=1500, early_stopping=True, n_iter_no_change=40, random_state=seed)

    def usable(rows: np.ndarray, superconductors_only: bool) -> np.ndarray:
        keep = (label[rows] == 1) & np.isfinite(tc[rows]) if superconductors_only else np.isfinite(label[rows])
        return rows[keep]

    classifier = regressor = None
    info = {}
    needs = [flag for flag, part in ((False, "classifier"), (True, "regressor")) if parts in ("both", part)]
    if any(not len(usable(valid_rows, flag)) for flag in needs):
        raise ValueError("the validation rows hold nothing this model can be scored on; early stopping needs some")
    if parts in ("both", "classifier"):
        fit_c, valid_c = usable(fit_rows, False), usable(valid_rows, False)
        classifier = HistGradientBoostingClassifier(**common, **(classifier_settings or BOOSTING_DEFAULT))
        classifier.fit(X[fit_c], label[fit_c], X_val=X[valid_c], y_val=label[valid_c])
        info["classifier_trees"] = int(classifier.n_iter_)
    if parts in ("both", "regressor"):
        fit_r, valid_r = usable(fit_rows, True), usable(valid_rows, True)
        regressor = HistGradientBoostingRegressor(loss="absolute_error", **common,
                                                  **(regressor_settings or BOOSTING_DEFAULT))
        regressor.fit(X[fit_r], np.log1p(tc[fit_r]), X_val=X[valid_r], y_val=np.log1p(tc[valid_r]))
        info["regressor_trees"] = int(regressor.n_iter_)
    return (classifier, regressor), info


def boosting_apply(models: tuple, X_predict: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    classifier, regressor = models
    return (None if classifier is None else classifier.predict_proba(X_predict)[:, 1],
            None if regressor is None else np.clip(np.expm1(regressor.predict(X_predict)), 0.0, None))


def boosting_predict(X: np.ndarray, ds_arrays: T.Arrays, fit_rows: np.ndarray, valid_rows: np.ndarray,
                     X_predict: np.ndarray, seed: int = 0, **settings) -> tuple[np.ndarray, np.ndarray, dict]:
    models, info = boosting_fit(X, ds_arrays, fit_rows, valid_rows, seed, **settings)
    return (*boosting_apply(models, X_predict), info)


def nearest_distance(dense_train: np.ndarray, dense_query: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """(distance, index) of the k nearest training compositions.

    Distance is the share of atoms that would have to be swapped to turn one
    composition into the other (half the summed absolute difference of fractions).
    """
    from scipy.spatial import cKDTree

    distance, index = cKDTree(dense_train).query(dense_query, k=k, p=1)
    return distance / 2.0, index


def neighbour_predict(ds: Dataset, train_rows: np.ndarray, dense_query: np.ndarray, k: int = 5
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Lookup baseline: share of superconductors among the k nearest labelled training compositions,
    and the Tc of the single nearest superconducting one."""
    label, tc = ds.arrays.label, ds.arrays.tc_k
    labelled = train_rows[np.isfinite(label[train_rows])]
    _, index = nearest_distance(ds.dense[labelled], dense_query, k=k)
    probability = label[labelled][index.reshape(len(dense_query), -1)].mean(axis=1)
    positive = train_rows[(label[train_rows] == 1) & np.isfinite(tc[train_rows])]
    _, nearest = nearest_distance(ds.dense[positive], dense_query, k=1)
    return probability, tc[positive][nearest]


AUX_TARGETS = ["formation_energy_ev_per_atom", "log1p_band_gap_ev", "is_metal", "log_volume_per_atom_a3",
               "log1p_abs_magnetization_mub_per_atom"]


@dataclass
class AuxData:
    """Materials Project compositions with computed properties, used as auxiliary training targets."""
    arrays: T.Arrays                # label and Tc are all nan; ``aux`` holds the scaled targets
    formulas: list[str]
    X: np.ndarray | None = None     # formula features, when a model with a formula branch needs them


def load_aux(table: ElementTable, candidates: Path, with_features: bool = False) -> AuxData:
    """One row per distinct Materials Project composition, taking the lowest-energy structure's properties.

    These are calculated properties of the material, not superconductivity labels, so using them
    for compositions that sit in a test fold does not leak the answer.
    """
    import csv
    import gzip

    def number(text: str) -> float:
        try:
            return float(text)
        except ValueError:
            return float("nan")

    best: dict[str, tuple[float, dict[str, float], list[float]]] = {}
    known = set(table.symbols)
    with gzip.open(candidates, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                fractions = D_parse(row["formula"])
            except ValueError:
                continue
            if not set(fractions) <= known or len(fractions) > MAX_ELEMENTS:
                continue
            sites = number(row["num_sites"])
            hull = number(row["energy_above_hull_ev_per_atom"])
            metal = {"True": 1.0, "False": 0.0}.get(row["is_metal"], float("nan"))
            targets = [number(row["formation_energy_ev_per_atom"]), np.log1p(max(number(row["band_gap_ev"]), 0.0)),
                       metal, np.log(number(row["volume_a3"]) / sites),
                       np.log1p(abs(number(row["total_magnetization_mub"])) / sites)]
            key = formula_text(fractions)
            if key not in best or (hull == hull and hull < best[key][0]):
                best[key] = (hull if hull == hull else float("inf"), fractions, targets)
    keys = sorted(best)
    compositions = [best[key][1] for key in keys]
    raw = np.array([best[key][2] for key in keys], dtype=np.float64)
    raw[~np.isfinite(raw)] = np.nan
    scaled = (raw - np.nanmean(raw, axis=0)) / np.nanstd(raw, axis=0)
    index, fraction, mask = encode(compositions, table, MAX_ELEMENTS)
    blank = np.full(len(keys), np.nan)
    arrays = T.Arrays(index, fraction, mask, blank, blank.copy(), np.array(["materials-project"] * len(keys)),
                      aux=scaled)
    X = formula_features(keys, cache=WORK / "aux-features.npz")[0] if with_features else None
    return AuxData(arrays, keys, X)


def feature_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, std = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
    return mean, np.where(std > 0, std, 1.0)


def scale_features(X: np.ndarray, scaler: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Standardize with the training rows' mean and spread; unknown values become 0 (the mean); extremes are capped."""
    return np.clip(np.nan_to_num((X - scaler[0]) / scaler[1], nan=0.0), -6.0, 6.0).astype(np.float32)


def with_extra(arrays: T.Arrays, X: np.ndarray, scaler: tuple[np.ndarray, np.ndarray]) -> T.Arrays:
    return T.Arrays(arrays.index, arrays.fraction, arrays.mask, arrays.label, arrays.tc_k, arrays.group,
                    extra=scale_features(X, scaler), aux=arrays.aux)


def network_fit_predict(ds: Dataset, fit_rows: np.ndarray, valid_rows: np.ndarray, predict_rows: list[np.ndarray],
                        model_config: Config, train_config: T.TrainConfig, X: np.ndarray | None = None,
                        aux: AuxData | None = None, save: Path | None = None
                        ) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """Train one network and predict each set of rows in ``predict_rows``.

    ``X`` (formula features for every row of the table) is needed when the model has a formula branch;
    ``aux`` adds Materials Project rows with auxiliary targets to the training set.
    """
    scaler = feature_scaler(X[fit_rows]) if model_config.n_global else None

    def rows(which: np.ndarray) -> T.Arrays:
        part = ds.arrays.take(which)
        return with_extra(part, X[which], scaler) if scaler else part

    fit_data = rows(fit_rows)
    scale = T.TcScale.fit(fit_data.tc_k[fit_data.label == 1])
    if model_config.n_aux:
        extra_rows = with_extra(aux.arrays, aux.X, scaler) if scaler else aux.arrays
        fit_data = T.Arrays.join([fit_data, extra_rows])
    model = CompositionNet(model_config, ds.table.features, seed=train_config.seed)
    info = T.fit(model, fit_data, train_config, valid=rows(valid_rows), scale=scale)
    info.pop("history")
    if model_config.use_embedding:
        # An element with no training row still holds its random starting vector. Give it no learned
        # offset instead, so such a material is judged on its physical properties alone.
        untrained = np.setdiff1d(np.arange(1, model_config.n_elements + 1), np.unique(fit_data.index))
        model.layers["embedding"].params["E"][untrained] = 0.0
        info["elements_never_trained"] = [ds.table.symbols[i - 1] for i in untrained]
    info["parameters"] = model.parameter_count()
    if save is not None:
        extra = {"tc_scale": info["tc_scale"], "symbols": ds.table.symbols,
                 "feature_names": ds.table.feature_names, "data_sha256": ds.sha256}
        if scaler:
            extra["formula_scaler"] = [scaler[0].tolist(), scaler[1].tolist()]
        model.save(save, extra=extra)
    return [T.predict(model, rows(which), scale) for which in predict_rows], info


def load_network(path: Path, table: ElementTable) -> tuple[CompositionNet, T.TcScale, tuple | None]:
    """(model, Tc scale, formula-feature scaler or None) from a saved member."""
    model, extra = CompositionNet.load(path, table.features)
    if extra["symbols"] != table.symbols or extra["feature_names"] != table.feature_names:
        raise SystemExit(f"{path} was trained with a different element table")
    scaler = extra.get("formula_scaler")
    return model, T.TcScale(**extra["tc_scale"]), (np.array(scaler[0]), np.array(scaler[1])) if scaler else None


# ---------------------------------------------------------------------------
# Scoring.

def slices(ds: Dataset, rows: np.ndarray, distance: np.ndarray) -> dict[str, np.ndarray]:
    """Named subsets of ``rows`` (boolean masks over them) that every score is reported for."""
    family = ds.family[rows]
    out = {"all": np.ones(len(rows), bool), "not cuprates": family != "Cuprates"}
    for name in ("Cuprates", "Ferrites", "Oxides", "Other"):
        out[name] = family == name
    for name, low, high in DISTANCE_BANDS:
        out[f"{name} (distance {low:g} to {min(high, 1.0):g})"] = (distance >= low) & (distance < high)
    return out


def score_table(ds: Dataset, rows: np.ndarray, distance: np.ndarray,
                predictions: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    label, tc = ds.arrays.label[rows], ds.arrays.tc_k[rows]
    out: dict = {}
    for slice_name, keep in slices(ds, rows, distance).items():
        out[slice_name] = {name: T.score(label[keep], tc[keep], p[keep], t[keep])
                           for name, (p, t) in predictions.items()}
    return out


def group_bootstrap(groups: np.ndarray, statistic, resamples: int = 300, seed: int = 0) -> dict:
    """Resample whole groups with replacement; returns the statistic's 2.5%, 50% and 97.5% points."""
    names, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(names))]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(names), len(names))
        value = statistic(np.concatenate([members[i] for i in chosen]))
        if np.isfinite(value):
            values.append(value)
    low, mid, high = np.percentile(values, [2.5, 50, 97.5])
    return {"low": float(low), "median": float(mid), "high": float(high), "resamples": len(values)}


def paired_differences(ds: Dataset, rows: np.ndarray, first: tuple[np.ndarray, np.ndarray],
                       second: tuple[np.ndarray, np.ndarray], keep: np.ndarray | None = None,
                       resamples: int = 300) -> dict:
    """first minus second, with group-bootstrap intervals, for AUC and for Tc error on the log scale.

    For AUC a positive number favours ``first``; for the Tc error a negative number does.
    """
    keep = np.ones(len(rows), bool) if keep is None else keep
    label, tc, groups = ds.arrays.label[rows][keep], ds.arrays.tc_k[rows][keep], ds.arrays.group[rows][keep]
    p1, t1, p2, t2 = first[0][keep], first[1][keep], second[0][keep], second[1][keep]
    labelled = np.isfinite(label)
    positive = (label == 1) & np.isfinite(tc)

    def auc_gap(sample: np.ndarray) -> float:
        sample = sample[labelled[sample]]
        return T.roc_auc(label[sample], p1[sample]) - T.roc_auc(label[sample], p2[sample])

    def tc_gap(sample: np.ndarray) -> float:
        sample = sample[positive[sample]]
        if not len(sample):
            return float("nan")
        truth = np.log1p(tc[sample])
        return float(np.mean(np.abs(np.log1p(t1[sample]) - truth)) - np.mean(np.abs(np.log1p(t2[sample]) - truth)))

    return {"auc": group_bootstrap(groups, auc_gap, resamples), "tc_log_mae": group_bootstrap(groups, tc_gap, resamples)}


def intervals(ds: Dataset, rows: np.ndarray, prediction: tuple[np.ndarray, np.ndarray],
              keep: np.ndarray | None = None, resamples: int = 300) -> dict:
    """Group-bootstrap intervals for one model's AUC and Tc error on the log scale."""
    keep = np.ones(len(rows), bool) if keep is None else keep
    label, tc, groups = ds.arrays.label[rows][keep], ds.arrays.tc_k[rows][keep], ds.arrays.group[rows][keep]
    p, t = prediction[0][keep], prediction[1][keep]
    labelled, positive = np.isfinite(label), (label == 1) & np.isfinite(tc)

    def auc(sample: np.ndarray) -> float:
        sample = sample[labelled[sample]]
        return T.roc_auc(label[sample], p[sample])

    def tc_error(sample: np.ndarray) -> float:
        sample = sample[positive[sample]]
        return float(np.mean(np.abs(np.log1p(t[sample]) - np.log1p(tc[sample])))) if len(sample) else float("nan")

    return {"auc": group_bootstrap(groups, auc, resamples), "tc_log_mae": group_bootstrap(groups, tc_error, resamples)}


def blend(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray],
          weight: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Weighted average of two models: probabilities directly, Tc on the log(1 + Tc) scale."""
    probability = weight * first[0] + (1.0 - weight) * second[0]
    tc = np.expm1(weight * np.log1p(first[1]) + (1.0 - weight) * np.log1p(second[1]))
    return probability, tc
