#!/usr/bin/env python3
"""Generations 6 and 7 of the children experiment: children that differ in what they know.

    python scripts/children_knowledge.py data          # parse NIMS with Tc, cut the folds, build every knowledge block
    python scripts/children_knowledge.py gen6          # five children, one change each, on the four selection folds
    python scripts/children_knowledge.py gen7          # breed generation 6's changes; name a champion or none
    python scripts/children_knowledge.py second-look   # champion (or best) and baseline on the spent 2016-2020 slice
    python scripts/children_knowledge.py pool          # second opinion: the champion scores rung 4's shortlist 2 survivors

The plan, rules and predictions are in ``docs/children-experiment.md`` ("Generations 6 and 7"). The learner is fixed:
the ranking of record's boosting settings. Each child adds knowledge, not tuning.

Choices the plan left to the code, fixed before any model was trained:
- A composition's training Tc is the highest Tc among its records published up to the training year; 0 for a null.
- ``tc-size`` fits a squared-error boosting regressor to log(1 + Tc) on every training row.
- ``family`` needs both classes among the family's training rows as well as the plan's 200 rows.
- "Average of the two models' ranks" is over the whole test set of the fold.
- A bag draws element systems with replacement (seed 1000 + bag); a bagged child averages its bags' scores.
- An element with no listed superconducting Tc counts as 0 K. Other missing element properties are left out of the
  formula's averages; a formula with none of them gets a missing value, which the trees handle.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import children as K
import children_nims as N
import scratchnet_common as C
from scratchnet_common import ROOT

WORK = ROOT / "data/processed/children-knowledge"
DOCS = ROOT / "docs/children"
FOLDS = {"A": {"train_until": 1995, "first": 1996, "last": 2000}, "B": {"train_until": 2000, "first": 2001, "last": 2005},
         "C": {"train_until": 2005, "first": 2006, "last": 2010}, "D": {"train_until": 2010, "first": 2011, "last": 2015}}
SECOND_LOOK = {"train_until": 2015, "first": 2016, "last": 2020}
BOOST = dict(N.BOOST)
CHANGES = ("elements", "proxy", "alloy", "tc-size", "family")
BLOCKS = ("elements", "proxy", "alloy")            # changes that add input columns
FAMILIES = ("Cuprates", "Ferrites", "Oxides", "Other")
PRIMARY_FAMILY = "Other"
EXPERT_MINIMUM = 200
BAGS = 8
MATTHIAS_PEAKS = (4.7, 6.5)
ELEMENT_PROPERTIES = ("superconduction_temperature", "bulk_modulus", "velocity_of_sound", "molar_volume",
                      "electrical_resistivity", "thermal_conductivity", "coefficient_of_linear_thermal_expansion")
STATS = ("mean", "min", "max", "range", "spread")


# ------------------------------------------------------------------------------------------------ knowledge blocks

def valence_electrons(symbol: str) -> float:
    """Electrons outside closed shells, as Matthias counted them: the group number, less 10 in the p block."""
    from pymatgen.core import Element

    element = Element(symbol)
    if element.is_lanthanoid or element.is_actinoid:
        return 3.0
    return float(element.group - 10 if element.group > 12 else element.group)


def debye_estimate(sound_velocity_m_s: float, molar_volume_cm3: float) -> float:
    """Debye temperature (K) from the sound velocity and the number density of atoms."""
    h, k_b, avogadro = 6.62607015e-34, 1.380649e-23, 6.02214076e23
    density = avogadro / (molar_volume_cm3 * 1e-6)                      # atoms per cubic metre
    return h / k_b * sound_velocity_m_s * (3.0 * density / (4.0 * math.pi)) ** (1.0 / 3.0)


def element_table() -> dict[str, dict[str, float]]:
    import warnings

    from pymatgen.core import Element

    table = {}
    for element in Element:
        row = {}
        for name in ELEMENT_PROPERTIES:
            with warnings.catch_warnings():        # pymatgen warns for every property an element lacks
                warnings.simplefilter("ignore")
                value = getattr(element, name, None)
            row[name] = float(value) if value is not None else math.nan
        if math.isnan(row["superconduction_temperature"]):
            row["superconduction_temperature"] = 0.0
        row["electrical_resistivity"] = (math.log10(row["electrical_resistivity"])
                                         if row["electrical_resistivity"] > 0 else math.nan)
        row["debye_estimate"] = debye_estimate(row["velocity_of_sound"], row["molar_volume"])
        row["inverse_root_mass"] = float(element.atomic_mass) ** -0.5
        row["valence_electrons"] = valence_electrons(element.symbol)
        table[element.symbol] = row
    return table


def element_names() -> list[str]:
    properties = (*ELEMENT_PROPERTIES, "debye_estimate")
    return ([f"{p}:{s}" for p in properties for s in STATS]
            + ["inverse_root_mass:mean", "valence_electrons:mean"]
            + [f"valence_electrons:distance_from_{peak}" for peak in MATTHIAS_PEAKS])


def element_features(fractions: dict[str, float], table: dict | None = None) -> list[float]:
    """Summaries over the formula of measured element properties, plus the isotope term and Matthias's electron count."""
    table = table or element_table()
    out = []
    for name in (*ELEMENT_PROPERTIES, "debye_estimate"):
        pairs = [(table[s][name], f) for s, f in fractions.items() if not math.isnan(table[s][name])]
        if not pairs:
            out += [math.nan] * len(STATS)
            continue
        values, weights = np.array([v for v, _ in pairs]), np.array([w for _, w in pairs])
        weights = weights / weights.sum()
        mean = float(values @ weights)
        out += [mean, float(values.min()), float(values.max()), float(values.max() - values.min()),
                float(np.abs(values - mean) @ weights)]
    total = sum(fractions.values())
    out.append(sum(table[s]["inverse_root_mass"] * f for s, f in fractions.items()) / total)
    electrons = sum(table[s]["valence_electrons"] * f for s, f in fractions.items()) / total
    out += [electrons, *(abs(electrons - peak) for peak in MATTHIAS_PEAKS)]
    return out


def alloy_features(fractions_list: list[dict[str, float]]) -> tuple[np.ndarray, list[str]]:
    """Miedema's formation enthalpies and Yang's solid-solution parameters, from matminer."""
    import warnings

    from matminer.featurizers.composition.alloy import Miedema, YangSolidSolution
    from pymatgen.core import Composition

    featurizers = [Miedema(), YangSolidSolution()]
    names = [label for f in featurizers for label in f.feature_labels()]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fractions in fractions_list:
            composition, row = Composition(fractions), []
            for featurizer in featurizers:
                try:
                    row += [float(v) for v in featurizer.featurize(composition)]
                except Exception:  # noqa: BLE001 - elements outside Miedema's tables
                    row += [math.nan] * len(featurizer.feature_labels())
            rows.append(row)
    return np.array(rows, dtype=float), names


# ------------------------------------------------------------------------------------------------ data

def parse_tc(path: Path, valid: set[str]) -> tuple[dict, dict]:
    """As ``children_nims.parse``, but each record also keeps its Tc: (year, label, Tc or 0)."""
    import build_mp_candidates as M
    import build_scratchnet_data as B

    csv.field_size_limit(10 ** 9)
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    header = rows[0]
    col = {name: header.index(name) for name in (*N.TC_COLUMNS, N.NULL_COLUMN, "chemical formula",
                                                 "publication year of reference", "maximum pressure applied")}
    records, compositions = defaultdict(list), {}
    for row in rows[2:]:
        year = row[col["publication year of reference"]].strip()
        if not (year.isdigit() and 1900 <= int(year) <= 2024):
            continue
        tcs = [N.number(row[col[name]]) for name in N.TC_COLUMNS]
        label = N.record_label(tcs, N.number(row[col[N.NULL_COLUMN]]))
        if label is None or row[col["maximum pressure applied"]].strip():
            continue
        try:
            fractions = M.fractions(row[col["chemical formula"]].strip())
        except Exception:  # noqa: BLE001
            fractions = None
        if not fractions or not set(fractions) <= valid:
            continue
        key = B.composition_key(fractions)
        compositions[key] = B.parse_key(key)
        records[key].append((int(year), label, record_tc(tcs) if label else 0.0))
    return dict(records), compositions


def record_tc(tcs: list[float | None]) -> float:
    """The record's Tc by the order of the columns: recommended first, then mid-point, and so on."""
    return next((float(t) for t in tcs if t is not None and t > 0), 0.0)


def known_at(records: list, year: int) -> tuple[int, float] | None:
    """Label and highest Tc from the records published up to ``year``; None if there are none."""
    seen = [(label, tc) for y, label, tc in records if y <= year]
    if not seen:
        return None
    return int(any(label for label, _ in seen)), max(tc for _, tc in seen)


def cmd_data() -> dict:
    import build_mp_candidates as M

    valid = set(C.load_dataset().table.symbols)
    records, compositions = parse_tc(N.RAW, valid)
    keys = sorted(compositions)
    first_year = {k: min(y for y, _, _ in records[k]) for k in keys}
    table = {"keys": keys, "records": {k: records[k] for k in keys}, "folds": {}}
    summary = {"compositions": len(keys), "folds": {}}
    for name, spec in {**FOLDS, "second-look": SECOND_LOOK}.items():
        cut = N.cut_slice(compositions, first_year, **spec)
        table["folds"][name] = {"train": cut["train"], "test": cut["test"]}
        family = Counter(M.family(set(compositions[k])) for k in cut["test"])
        summary["folds"][name] = {**spec, "training_compositions": len(cut["train"]), "test_compositions": len(cut["test"]),
                                  "test_families": dict(family)}
    fractions = [C.fractions_of(k) for k in keys]
    X, names = C.formula_features([C.formula_text(f) for f in fractions], cache=WORK / "formula-features.npz")
    elements_table = element_table()
    elements = np.array([element_features(f, elements_table) for f in fractions], dtype=float)
    alloy, alloy_names = alloy_features(fractions)
    proxy = K.physics_feature(keys)
    families = np.array([M.family(set(f)) for f in fractions])
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "table.json").write_text(json.dumps(table), encoding="utf-8")
    np.savez_compressed(WORK / "blocks.npz", X=X, elements=elements, alloy=alloy, proxy=proxy[:, None],
                        family=np.array([[float(f == name) for name in FAMILIES] for f in families]),
                        family_name=families, system=np.array(["-".join(sorted(f)) for f in fractions]))
    summary["columns"] = {"formula": len(names), "elements": elements.shape[1], "alloy": alloy.shape[1], "proxy": 1,
                          "family": len(FAMILIES)}
    summary["coverage"] = {"elements_any_value": round(float(np.isfinite(elements).any(axis=1).mean()), 4),
                           "alloy_miedema": round(float(np.isfinite(alloy[:, 0]).mean()), 4),
                           "alloy_yang": round(float(np.isfinite(alloy[:, -1]).mean()), 4),
                           "proxy": round(float(np.isfinite(proxy).mean()), 4)}
    summary["element_columns"], summary["alloy_columns"] = element_names(), alloy_names
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "knowledge-data.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_columns")}, indent=1))
    return summary


def load(fold: str) -> dict:
    table = json.loads((WORK / "table.json").read_text(encoding="utf-8"))
    spec = {**FOLDS, "second-look": SECOND_LOOK}[fold]
    index = {k: i for i, k in enumerate(table["keys"])}
    train = np.array([index[k] for k in table["folds"][fold]["train"]])
    test = np.array([index[k] for k in table["folds"][fold]["test"]])
    known = [known_at(table["records"][table["keys"][i]], spec["train_until"]) for i in train]
    with np.load(WORK / "blocks.npz", allow_pickle=False) as saved:
        blocks = {name: saved[name] for name in saved.files}
    y_test = np.array([int(any(label for _, label, _ in table["records"][table["keys"][i]])) for i in test])
    return {"blocks": blocks, "train": train, "test": test, "y": np.array([k[0] for k in known]),
            "tc": np.array([k[1] for k in known]), "y_test": y_test,
            "groups": blocks["system"][train], "test_groups": blocks["system"][test],
            "family": blocks["family_name"][train], "test_family": blocks["family_name"][test]}


# ------------------------------------------------------------------------------------------------ children

def columns(changes: tuple[str, ...], blocks: dict, rows: np.ndarray) -> np.ndarray:
    parts = [blocks["X"][rows]] + [blocks[name][rows] for name in BLOCKS if name in changes]
    if "family" in changes:
        parts.append(blocks["family"][rows])
    return np.column_stack(parts)


def ranks(scores: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(scores) / len(scores)


def fit_regressor(X, target, groups, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    rows = np.arange(len(target))
    fit, valid = C.T.inner_split(groups, percent=12, seed=0)
    model = HistGradientBoostingRegressor(max_iter=1500, early_stopping=True, n_iter_no_change=40, random_state=seed,
                                          **BOOST)
    return model.fit(X[rows[fit]], target[rows[fit]], X_val=X[rows[valid]], y_val=target[rows[valid]])


def score_once(changes: tuple[str, ...], data: dict, chosen: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """One fit of a child (on ``chosen`` training rows, default all): a score for every test composition."""
    chosen = np.arange(len(data["train"])) if chosen is None else chosen
    X = columns(changes, data["blocks"], data["train"][chosen])
    T = columns(changes, data["blocks"], data["test"])
    y, groups = data["y"][chosen], data["groups"][chosen]
    parts = [N.fit_boost(X, y, groups, BOOST, seed=seed).predict_proba(T)[:, 1]]
    if "tc-size" in changes:
        parts.append(fit_regressor(X, np.log1p(data["tc"][chosen]), groups, seed=seed).predict(T))
    if "family" in changes:
        expert = parts[0].copy()
        family = data["family"][chosen]
        for name in sorted(set(data["test_family"])):
            mine = np.flatnonzero(family == name)
            if len(mine) < EXPERT_MINIMUM or len(set(y[mine].tolist())) < 2:
                continue
            try:
                model = N.fit_boost(X[mine], y[mine], groups[mine], BOOST, seed=seed)
            except ValueError:          # the held-back groups of a small family can lack a class
                continue
            at = data["test_family"] == name
            expert[at] = model.predict_proba(T[at])[:, 1]
        parts.append(expert)
    return parts[0] if len(parts) == 1 else np.mean([ranks(p) for p in parts], axis=0)


def score_child(changes: tuple[str, ...], data: dict, bags: int = 0) -> np.ndarray:
    if not bags:
        return score_once(changes, data)
    return np.mean([ranks(score_once(changes, data, K.bag_rows(data["groups"], 1000 + bag), seed=bag))
                    for bag in range(bags)], axis=0)


def breed(order: list[str], above_baseline: set[str]) -> dict[str, dict]:
    """Generation 7's children from generation 6's changes, best first (the plan's table)."""
    winners = [name for name in order if name in above_baseline]
    if not winners:
        return {"all": {"changes": CHANGES, "bags": 0}, "baseline-bagged": {"changes": (), "bags": BAGS}}
    return {"best": {"changes": tuple(winners[:1]), "bags": 0},
            "best-two": {"changes": tuple(winners[:2]), "bags": 0},
            "best-three": {"changes": tuple(winners[:3]), "bags": 0},
            "best-two-bagged": {"changes": tuple(winners[:2]), "bags": BAGS},
            "all": {"changes": CHANGES, "bags": 0}}


# ------------------------------------------------------------------------------------------------ scoring

def endpoint(y: np.ndarray, scores: np.ndarray, family: np.ndarray, which: str | None) -> float:
    mask = np.ones(len(y), dtype=bool) if which is None else family == which
    if len(set(y[mask].tolist())) < 2:
        return math.nan
    return N.auc(y[mask], scores[mask])


def fold_bootstrap(folds: dict, names: list[str], resamples: int = 1000, seed: int = 0) -> dict:
    """Mean-over-folds endpoints under resampling of element systems within each fold; gains over the baseline."""
    rng = np.random.default_rng(seed)
    draws = {which: {name: [] for name in names} for which in ("primary", "all")}
    members = {}
    for fold, f in folds.items():
        _, inverse = np.unique(f["groups"], return_inverse=True)
        members[fold] = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]
    for _ in range(resamples):
        picks = {fold: np.concatenate([m[g] for g in rng.integers(0, len(m), len(m))]) for fold, m in members.items()}
        for which, family in (("primary", PRIMARY_FAMILY), ("all", None)):
            row = {name: np.mean([endpoint(f["y"][picks[fold]], f["scores"][name][picks[fold]],
                                           f["family"][picks[fold]], family) for fold, f in folds.items()])
                   for name in names}
            if not any(math.isnan(v) for v in row.values()):
                for name, value in row.items():
                    draws[which][name].append(value)
    out = {}
    for name in names:
        out[name] = {}
        for which in ("primary", "all"):
            if name == "baseline":
                continue
            gain = np.array(draws[which][name]) - np.array(draws[which]["baseline"])
            out[name][f"{which}_gain_95"] = [round(float(x), 4) for x in np.percentile(gain, [2.5, 97.5])]
            out[name][f"{which}_gain_99"] = [round(float(x), 4) for x in np.percentile(gain, [0.5, 99.5])]
    return out


def run_generation(children: dict[str, dict], tag: str) -> dict:
    folds, report = {}, {"learner": BOOST, "children": {}, "folds": {}}
    for fold in FOLDS:
        data = load(fold)
        scores = {"baseline": score_child((), data)}
        for name, spec in children.items():
            scores[name] = score_child(tuple(spec["changes"]), data, spec.get("bags", 0))
            print(tag, fold, name, round(endpoint(data["y_test"], scores[name], data["test_family"], PRIMARY_FAMILY), 4),
                  flush=True)
        folds[fold] = {"y": data["y_test"], "groups": data["test_groups"], "family": data["test_family"],
                       "scores": scores}
        np.savez_compressed(WORK / f"{tag}-{fold}-scores.npz", y=data["y_test"], **scores)
        report["folds"][fold] = {name: {"primary": round(endpoint(data["y_test"], s, data["test_family"], PRIMARY_FAMILY), 4),
                                        "all": round(endpoint(data["y_test"], s, data["test_family"], None), 4)}
                                 for name, s in scores.items()}
    names = ["baseline", *children]
    intervals = fold_bootstrap(folds, names)
    for name in names:
        entry = {"primary": round(float(np.mean([report["folds"][f][name]["primary"] for f in FOLDS])), 4),
                 "all": round(float(np.mean([report["folds"][f][name]["all"] for f in FOLDS])), 4)}
        if name != "baseline":
            entry.update(changes=list(children[name]["changes"]), bags=children[name].get("bags", 0), **intervals[name])
        report["children"][name] = entry
    base = report["children"]["baseline"]
    for name in children:
        entry = report["children"][name]
        entry["primary_gain"] = round(entry["primary"] - base["primary"], 4)
        entry["all_gain"] = round(entry["all"] - base["all"], 4)
    return report


def cmd_gen6() -> dict:
    report = run_generation({name: {"changes": (name,)} for name in CHANGES}, "gen6")
    order = sorted(CHANGES, key=lambda n: -report["children"][n]["primary"])
    report["order"] = order
    report["above_baseline"] = [n for n in order if report["children"][n]["primary_gain"] > 0]
    (DOCS / "knowledge-gen6.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["children"], indent=1))
    return report


def cmd_gen7() -> dict:
    gen6 = json.loads((DOCS / "knowledge-gen6.json").read_text(encoding="utf-8"))
    children = breed(gen6["order"], set(gen6["above_baseline"]))
    report = run_generation(children, "gen7")
    best = max(children, key=lambda n: report["children"][n]["primary"])
    report["best"] = best
    report["champion"] = best if report["children"][best]["primary_gain_99"][0] > 0 else None
    (DOCS / "knowledge-gen7.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report["children"], indent=1), "\nchampion:", report["champion"])
    return report


def cmd_second_look() -> dict:
    out = DOCS / "knowledge-second-look.json"
    if out.is_file():
        raise SystemExit(f"{out} exists: the second look is taken once.")
    gen7 = json.loads((DOCS / "knowledge-gen7.json").read_text(encoding="utf-8"))
    name = gen7["champion"] or gen7["best"]
    spec = gen7["children"][name]
    data = load("second-look")
    scores = {"baseline": score_child((), data), name: score_child(tuple(spec["changes"]), data, spec["bags"])}
    fold = {"second-look": {"y": data["y_test"], "groups": data["test_groups"], "family": data["test_family"],
                            "scores": scores}}
    report = {"not_blind": "the 2016-2020 slice was the final test of generations 3 to 5; this decides nothing",
              "child": name, "is_champion": gen7["champion"] is not None, "changes": spec["changes"],
              "bags": spec["bags"], "test_compositions": int(len(data["y_test"])),
              "scores": {n: {"primary": round(endpoint(data["y_test"], s, data["test_family"], PRIMARY_FAMILY), 4),
                             "all": round(endpoint(data["y_test"], s, data["test_family"], None), 4)}
                         for n, s in scores.items()},
              "gain_intervals": fold_bootstrap(fold, ["baseline", name])[name]}
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


def cmd_pool() -> dict:
    """A second opinion for rung 4: the champion, trained on every NIMS composition, scores shortlist 2's survivors.

    Added after generation 7 (not in the plan). It ranks nothing by itself: ``rung4_shortlist2.py`` still orders the
    shortlist with the ranking of record, and this is a column beside it.
    """
    import build_mp_candidates as M
    import build_scratchnet_data as B
    import rung4 as R

    gen7 = json.loads((DOCS / "knowledge-gen7.json").read_text(encoding="utf-8"))
    spec = gen7["children"][gen7["champion"] or gen7["best"]]
    with open(ROOT / "data/benchmarks/rung4/shortlist2-filtered.csv", newline="", encoding="utf-8") as handle:
        pool = list(csv.DictReader(handle))
    fractions = [R.fractions(r["formula"]) for r in pool]
    keys = [B.composition_key(f) for f in fractions]
    table = json.loads((WORK / "table.json").read_text(encoding="utf-8"))
    with np.load(WORK / "blocks.npz", allow_pickle=False) as saved:
        blocks = {name: saved[name] for name in saved.files}
    X, _ = C.formula_features([C.formula_text(f) for f in fractions])
    elements_table = element_table()
    families = np.array([M.family(set(f)) for f in fractions])
    extra = {"X": X, "elements": np.array([element_features(f, elements_table) for f in fractions], dtype=float),
             "alloy": alloy_features(fractions)[0], "proxy": K.physics_feature(keys)[:, None],
             "family": np.array([[float(f == name) for name in FAMILIES] for f in families]),
             "family_name": families, "system": np.array(["-".join(sorted(f)) for f in fractions])}
    joined = {name: np.concatenate([blocks[name], extra[name]]) for name in blocks}
    count = len(table["keys"])
    known = [known_at(table["records"][k], 2024) for k in table["keys"]]
    train, test = np.arange(count), count + np.arange(len(pool))
    data = {"blocks": joined, "train": train, "test": test, "y": np.array([k[0] for k in known]),
            "tc": np.array([k[1] for k in known]), "groups": joined["system"][train],
            "family": joined["family_name"][train], "test_family": joined["family_name"][test]}
    champion = score_child(tuple(spec["changes"]), data, spec["bags"])
    baseline = score_child((), data)
    measured = set(table["keys"])
    out = ROOT / "data/benchmarks/rung4/shortlist2-second-opinion.csv"
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mp_id", "formula", "champion_percentile", "nims_baseline_percentile", "composition_in_nims"])
        for row, key, c, b in zip(pool, keys, ranks(champion), ranks(baseline)):
            writer.writerow([row["mp_id"], row["formula"], round(100 * float(c), 1), round(100 * float(b), 1),
                             key in measured])
    summary = {"child": gen7["champion"] or gen7["best"], "changes": spec["changes"], "scored": len(pool),
               "training_compositions": count, "pool_compositions_already_in_nims": int(sum(k in measured for k in keys)),
               "rank_correlation_with_nims_baseline": round(float(np.corrcoef(ranks(champion), ranks(baseline))[0, 1]), 3)}
    (DOCS / "knowledge-pool.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("data", "gen6", "gen7", "second-look", "pool"))
    {"data": cmd_data, "gen6": cmd_gen6, "gen7": cmd_gen7, "second-look": cmd_second_look,
     "pool": cmd_pool}[parser.parse_args().command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
