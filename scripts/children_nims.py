#!/usr/bin/env python3
"""Generations 3 to 5 of the children experiment, on the dated NIMS SuperCon data (``docs/children-experiment.md``).

    python scripts/children_nims.py data      # download, parse, and cut the two time slices
    python scripts/children_nims.py evolve    # generations 3, 4 and 5: five lineages, two mutants each
    python scripts/children_nims.py final     # once: the five generation 5 members and the baseline on 2016-2020

Plain code on the CPU. The data file (CC BY 4.0, doi:10.48505/nims.4487) goes to ``data/raw/nims-supercon/``, which
git ignores. The ``physics`` lineage also needs Bercx et al.'s tables (``python scripts/bercx_published.py download``).

Choices the plan left to the code, fixed before any model was trained on this data:
- A record's Tc is the recommended value, else the mid-point, R = 0, R = 100% or susceptibility value. A record with
  no Tc but a "lowest temperature (not superconducting)" value, or a Tc of 0, is a measured null.
- Training labels are as known at the cutoff: only records published up to the training year count.
- Groups (for early stopping and bagging) are element systems. Boosting stops early on 12% of them.
- A mutant's random seed is sha256("<lineage>:<generation>:<mutant>"). On a tie the parent survives.
- Final intervals resample whole element systems of the final slice, 1,000 times, seed 0.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import children as K
import scratchnet_common as C
from scratchnet_common import ROOT

URL = "https://mdr.nims.go.jp/filesets/2347b413-9c15-43b7-90e6-41fe9243b1a5/download"
RAW = ROOT / "data/raw/nims-supercon/20240322_MDR_OAndM.txt"
RAW_SHA256 = "f599ef0040c18521e386f758ee826fe269f67ef369c1a007656035d03ccdfdf6"
WORK = ROOT / "data/processed/children-nims"
DOCS = ROOT / "docs/children"
TC_COLUMNS = ("Tc (of this sample) recommended", "transition temperature (mid point)",
              "transition temperature (R = 0)", "transition temperature (R = 100%)",
              "Tc from susceptibility measurement")
NULL_COLUMN = "lowest temperature for measurement (not superconducting)"
SLICES = {"selection": {"train_until": 2010, "first": 2011, "last": 2015},
          "final": {"train_until": 2015, "first": 2016, "last": 2020}}
NEAR = 0.02
KINDS = ("boost", "bag", "knn", "mlp", "physics")
BOOST = {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 1.0}
START = {"boost": {**BOOST, "max_leaf_nodes": 63},          # generation 2's promoted setting
         "bag": {**BOOST, "bags": 8},
         "knn": {"k": 15, "weights": "distance", "p": 2},
         "mlp": {"layers": (128, 64), "alpha": 1e-4, "learning_rate_init": 1e-3},
         "physics": dict(BOOST)}
BASELINE = ("boost", dict(BOOST))
BOOST_LADDER = {"max_leaf_nodes": [7, 15, 31, 63, 127], "learning_rate": [0.02, 0.05, 0.1],
                "min_samples_leaf": [5, 10, 20, 40], "l2_regularization": [0.0, 1.0, 10.0]}
LADDERS = {"boost": BOOST_LADDER, "physics": BOOST_LADDER, "bag": {**BOOST_LADDER, "bags": [4, 8, 16]},
           "knn": {"k": [3, 5, 10, 15, 30, 60], "weights": ["uniform", "distance"], "p": [1, 2]},
           "mlp": {"layers": [(64,), (128, 64), (256, 128), (256, 128, 64)], "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                   "learning_rate_init": [3e-4, 1e-3, 3e-3]}}
GENERATIONS = (3, 4, 5)


# ------------------------------------------------------------------------------------------------ data

def number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def record_label(tcs: list[float | None], null_temperature: float | None) -> int | None:
    """1 for a Tc above 0, 0 for a measured null, None when the record says neither."""
    values = [t for t in tcs if t is not None]
    if values and max(values) > 0:
        return 1
    if null_temperature is not None or values:
        return 0
    return None


def label_at(records: list[tuple[int, int]], year: int) -> int | None:
    seen = [label for y, label in records if y <= year]
    return int(any(seen)) if seen else None


def cut_slice(compositions: dict, first_year: dict, train_until: int, first: int, last: int) -> dict:
    """Training keys (first seen by ``train_until``) and test keys: first seen in the window, in a new element system."""
    by_system = defaultdict(list)
    train = [k for k in compositions if first_year[k] <= train_until]
    for key in train:
        by_system[frozenset(compositions[key])].append(compositions[key])
    window = [k for k in compositions if first <= first_year[k] <= last]
    clean = [k for k in window if not any(K.half_gap(compositions[k], other) <= NEAR
                                          for other in by_system.get(frozenset(compositions[k]), []))]
    new_system = [k for k in clean if frozenset(compositions[k]) not in by_system]
    return {"train": sorted(train), "window": len(window), "not_near": len(clean), "test": sorted(new_system)}


def parse(path: Path, valid: set[str]) -> tuple[dict, dict, Counter]:
    import build_mp_candidates as M
    import build_scratchnet_data as B

    csv.field_size_limit(10 ** 9)
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    header = rows[0]
    col = {name: header.index(name) for name in (*TC_COLUMNS, NULL_COLUMN, "chemical formula",
                                                 "publication year of reference", "maximum pressure applied")}
    records, compositions, counts = defaultdict(list), {}, Counter()
    for row in rows[2:]:                                    # row 1 repeats the header as short names
        year = row[col["publication year of reference"]].strip()
        if not (year.isdigit() and 1900 <= int(year) <= 2024):
            counts["no_year"] += 1
            continue
        label = record_label([number(row[col[name]]) for name in TC_COLUMNS], number(row[col[NULL_COLUMN]]))
        if label is None:
            counts["no_tc_and_no_null"] += 1
            continue
        if row[col["maximum pressure applied"]].strip():
            counts["under_pressure"] += 1
            continue
        try:
            fractions = M.fractions(row[col["chemical formula"]].strip())
        except Exception:  # noqa: BLE001 - formulas with variables (O7-Z) are not parseable
            fractions = None
        if not fractions or not set(fractions) <= valid:
            counts["formula_not_usable"] += 1
            continue
        counts["used"] += 1
        key = B.composition_key(fractions)
        compositions[key] = B.parse_key(key)
        records[key].append((int(year), label))
    return dict(records), compositions, counts


def cmd_data() -> dict:
    import build_mp_candidates as M

    if not RAW.is_file():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, RAW)
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    if digest != RAW_SHA256:
        raise SystemExit(f"NIMS file has sha256 {digest}, expected {RAW_SHA256}")
    valid = set(C.load_dataset().table.symbols)
    records, compositions, counts = parse(RAW, valid)
    first_year = {k: min(y for y, _ in v) for k, v in records.items()}
    table = {"records": {k: v for k, v in records.items()}, "first_year": first_year, "slices": {}}
    summary = {"sha256": digest, "record_counts": dict(counts), "compositions": len(compositions), "slices": {}}
    for name, spec in SLICES.items():
        cut = cut_slice(compositions, first_year, **spec)
        table["slices"][name] = {"train": cut["train"], "test": cut["test"]}
        labels = [int(any(l for _, l in records[k])) for k in cut["test"]]
        families = Counter(M.family(set(compositions[k])) for k in cut["test"])
        summary["slices"][name] = {**spec, "training_compositions": len(cut["train"]),
                                   "first_seen_in_window": cut["window"], "not_within_0.02": cut["not_near"],
                                   "test_new_element_system": len(cut["test"]),
                                   "test_superconductors": int(sum(labels)),
                                   "test_not_superconductors": int(len(labels) - sum(labels)),
                                   "test_element_systems": len({frozenset(compositions[k]) for k in cut["test"]}),
                                   "test_families": dict(families)}
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "table.json").write_text(json.dumps(table), encoding="utf-8")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "nims-data.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return summary


# ------------------------------------------------------------------------------------------------ models

def load(slice_name: str) -> dict:
    """Features, labels and groups for one slice's training and test compositions."""
    import build_mp_candidates as M

    table = json.loads((WORK / "table.json").read_text(encoding="utf-8"))
    spec, keys = SLICES[slice_name], table["slices"][slice_name]
    text = lambda key: C.formula_text(C.fractions_of(key))  # noqa: E731
    system = lambda key: "-".join(sorted(C.fractions_of(key)))  # noqa: E731
    X, _ = C.formula_features([text(k) for k in keys["train"]], cache=WORK / f"{slice_name}-train-features.npz")
    T, _ = C.formula_features([text(k) for k in keys["test"]], cache=WORK / f"{slice_name}-test-features.npz")
    y = np.array([label_at([tuple(r) for r in table["records"][k]], spec["train_until"]) for k in keys["train"]])
    y_test = np.array([int(any(label for _, label in table["records"][k])) for k in keys["test"]])
    physics = WORK / f"{slice_name}-physics.npz"
    if not physics.is_file():
        np.savez_compressed(physics, train=K.physics_feature(keys["train"]), test=K.physics_feature(keys["test"]))
    with np.load(physics) as saved:
        p_train, p_test = saved["train"], saved["test"]
    return {"X": X, "T": T, "y": y, "y_test": y_test, "p_train": p_train, "p_test": p_test,
            "groups": np.array([system(k) for k in keys["train"]]),
            "test_groups": np.array([system(k) for k in keys["test"]]),
            "test_family": np.array([M.family(set(C.fractions_of(k))) for k in keys["test"]]),
            "test_keys": keys["test"]}


def fit_boost(X, y, groups, settings: dict, seed: int = 0):
    rows = np.arange(len(y))
    fit, valid = C.T.inner_split(groups, percent=12, seed=0)
    boost = {k: settings[k] for k in BOOST}
    model, _ = C.boosting_fit(X, K._Labels(y), rows[fit], rows[valid], seed=seed, classifier_settings=boost,
                              parts="classifier")
    return model[0]


def predict(kind: str, settings: dict, data: dict) -> np.ndarray:
    """Scores for the slice's test compositions from a model of this kind and these settings."""
    X, T, y, groups = data["X"], data["T"], data["y"], data["groups"]
    if kind == "physics":
        X, T = np.column_stack([X, data["p_train"]]), np.column_stack([T, data["p_test"]])
    if kind in ("boost", "physics"):
        return fit_boost(X, y, groups, settings).predict_proba(T)[:, 1]
    if kind == "bag":
        scores = []
        for bag in range(settings["bags"]):
            chosen = K.bag_rows(groups, 1000 + bag)
            scores.append(fit_boost(X[chosen], y[chosen], groups[chosen], settings, seed=bag).predict_proba(T)[:, 1])
        return np.mean(scores, axis=0)
    Xs, Ts = K.standardise(X, T)
    if kind == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=settings["k"], weights=settings["weights"], p=settings["p"])
        return model.fit(Xs, y).predict_proba(Ts)[:, 1]
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=tuple(settings["layers"]), alpha=settings["alpha"],
                              learning_rate_init=settings["learning_rate_init"], early_stopping=True,
                              validation_fraction=0.12, max_iter=300, random_state=0)
        return model.fit(Xs, y).predict_proba(Ts)[:, 1]
    raise ValueError(kind)


def mutate(kind: str, settings: dict, generation: int, mutant: int) -> dict:
    """One setting, chosen at random, moved one step along its ladder (inward when it is at an end)."""
    seed = int(hashlib.sha256(f"{kind}:{generation}:{mutant}".encode()).hexdigest(), 16) % 2 ** 32
    rng = np.random.default_rng(seed)
    ladder = LADDERS[kind]
    name = sorted(ladder)[int(rng.integers(0, len(ladder)))]
    steps = [tuple(s) if isinstance(s, (list, tuple)) else s for s in ladder[name]]
    current = tuple(settings[name]) if isinstance(settings[name], (list, tuple)) else settings[name]
    at = steps.index(current)
    move = 1 if rng.integers(0, 2) else -1
    if not 0 <= at + move < len(steps):
        move = -move
    return {**settings, name: steps[at + move]}


def auc(y: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, scores))


def plain(settings: dict) -> dict:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in settings.items()}


def cmd_evolve() -> dict:
    data = load("selection")
    y = data["y_test"]
    cache: dict[str, float] = {}

    def score(kind: str, settings: dict) -> float:
        key = kind + json.dumps(plain(settings), sort_keys=True)
        if key not in cache:
            cache[key] = round(auc(y, predict(kind, settings, data)), 4)
        return cache[key]

    report = {"slice": SLICES["selection"], "test_compositions": int(len(y)), "superconductors": int(y.sum()),
              "baseline": {"kind": BASELINE[0], "settings": BASELINE[1], "auc": score(*BASELINE)}, "lineages": {}}
    for kind in KINDS:
        member = dict(START[kind])
        history = [{"generation": 2, "settings": plain(member), "auc": score(kind, member)}]
        for generation in GENERATIONS:
            mutants = [mutate(kind, member, generation, m) for m in (1, 2)]
            scores = [score(kind, member)] + [score(kind, m) for m in mutants]
            best = int(np.argmax(scores))                    # argmax takes the first on a tie: the parent
            entry = {"generation": generation, "parent_auc": scores[0],
                     "mutants": [{"settings": plain(m), "auc": s} for m, s in zip(mutants, scores[1:])],
                     "survivor": "parent" if best == 0 else f"mutant {best}"}
            member = member if best == 0 else mutants[best - 1]
            entry.update(settings=plain(member), auc=scores[best])
            history.append(entry)
            print(kind, generation, entry["survivor"], entry["auc"], plain(member), flush=True)
        report["lineages"][kind] = {"history": history, "final_member": plain(member)}
    (DOCS / "nims-evolve.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


def system_bootstrap(y, groups, scores: dict, baseline: str, resamples: int = 1000, seed: int = 0) -> dict:
    names, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(names))]
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in scores}
    for _ in range(resamples):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(names), len(names))])
        if len(set(y[idx].tolist())) < 2:
            continue
        for name, s in scores.items():
            draws[name].append(auc(y[idx], s[idx]))
    out = {}
    for name in scores:
        a, b = np.array(draws[name]), np.array(draws[baseline])
        q = lambda v, lo, hi: [round(float(x), 4) for x in np.percentile(v, [lo, hi])]  # noqa: E731
        out[name] = {"auc_95": q(a, 2.5, 97.5)}
        if name != baseline:
            out[name].update(minus_baseline=round(float(auc(y, scores[name]) - auc(y, scores[baseline])), 4),
                             minus_baseline_95=q(a - b, 2.5, 97.5), minus_baseline_99=q(a - b, 0.5, 99.5))
    return out


def cmd_final() -> dict:
    out = DOCS / "nims-final.json"
    if out.is_file():
        raise SystemExit(f"{out} exists: the final slice is scored once.")
    evolved = json.loads((DOCS / "nims-evolve.json").read_text(encoding="utf-8"))
    data = load("final")
    y, groups = data["y_test"], data["test_groups"]
    scores = {"baseline": predict(*BASELINE, data)}
    for kind in KINDS:
        settings = evolved["lineages"][kind]["final_member"]
        scores[kind] = predict(kind, settings, data)
        print(kind, round(auc(y, scores[kind]), 4), flush=True)
    intervals = system_bootstrap(y, groups, scores, "baseline")
    report = {"slice": SLICES["final"], "test_compositions": int(len(y)), "superconductors": int(y.sum()),
              "element_systems": int(len(set(groups))), "models": {}}
    for name, s in scores.items():
        entry = {"auc": round(auc(y, s), 4), **intervals[name], "by_family": {}}
        for family in sorted(set(data["test_family"])):
            mask = data["test_family"] == family
            if 0 < y[mask].sum() < mask.sum():
                entry["by_family"][family] = {"compositions": int(mask.sum()), "superconductors": int(y[mask].sum()),
                                              "auc": round(auc(y[mask], s[mask]), 4)}
        if name != "baseline":
            entry["settings"] = evolved["lineages"][name]["final_member"]
            entry["beats_baseline"] = bool(intervals[name]["minus_baseline_99"][0] > 0)
        report["models"][name] = entry
    np.savez_compressed(WORK / "final-scores.npz", y=y, **scores)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["data", "evolve", "final"])
    {"data": cmd_data, "evolve": cmd_evolve, "final": cmd_final}[parser.parse_args().step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
