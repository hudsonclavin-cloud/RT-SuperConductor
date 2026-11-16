#!/usr/bin/env python3
"""The five-children experiment (``docs/children-experiment.md``): two generations, judged on later discoveries.

    python scripts/children.py data     # pool, hits and the 2018-era training table
    python scripts/children.py gen1     # five children that differ in kind, scored on the selection slice
    python scripts/children.py gen2     # offspring of the generation 1 winner, and the promotion rule
    python scripts/children.py final    # once: champion against the baseline on the 2016-2021 hits

Everything is plain code on the CPU. ``data`` needs SuperCon2 (downloaded to ``data/raw/supercon2/``, which git
ignores) and, for the ``physics`` child, Bercx et al.'s tables (``python scripts/bercx_published.py download``).
``final`` needs ``data/benchmarks/children/final-hits-check.csv`` (one reader's check of each final-slice hit) and
refuses to run a second time.

Choices the plan left to the code, fixed before any child was scored:
- Boosting children stop early on 12% of the training groups (``inner_split``, seed 0), like the ranking of record.
- ``knn`` and ``mlp`` use features standardised on the training rows; missing values become the training median.
- ``physics``: five formula-only boosting classifiers on Bercx et al.'s Table B (target: in Table C), each trained
  without one fifth of the element systems; a composition is scored by the model that never saw its system.
- A hit's percentile is the share of the non-hit pool it outranks (ties count half). AUC is the mean percentile.
- The generation 2 settings changed for each kind of winner are in ``VARIANTS``.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

import scratchnet_common as C
from scratchnet_common import ROOT

SUPERCON2_URL = "https://raw.githubusercontent.com/lfoppiano/supercon/main/data/supercon2_v22.12.03.csv"
SUPERCON2 = ROOT / "data/raw/supercon2/supercon2_v22.12.03.csv"
SUPERCON2_SHA256 = "13a18935213c570bb06d1f7ba4aa209abe535b37eaae3c8dafa382b4bb8efc52"
CANDIDATES = ROOT / "data/candidates/mp-candidates.csv.gz"
BENCH = ROOT / "data/benchmarks/children"
WORK = ROOT / "data/processed/children"
DOCS = ROOT / "docs/children"
CHECK = BENCH / "final-hits-check.csv"
NEAR, EXACT = 0.02, 0.0015
GEN1 = ("boost", "bag", "knn", "mlp", "physics")
VARIANTS = {
    "boost": ({"max_leaf_nodes": 15}, {"max_leaf_nodes": 63}),
    "bag": ({"max_leaf_nodes": 15}, {"bags": 16}),
    "knn": ({"k": 5}, {"k": 50}),
    "mlp": ({"layers": (64,)}, {"layers": (256, 128, 64)}),
    "physics": ({"max_leaf_nodes": 15}, {"max_leaf_nodes": 63}),
}


# ------------------------------------------------------------------------------------------------ data

def parse_tc(text: str) -> float | None:
    """'2.8 K' -> 2.8; ranges, bounds and anything else -> None."""
    match = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)\s*K\s*$", text or "")
    return float(match.group(1)) if match else None


def half_gap(a: dict, b: dict) -> float:
    return 0.5 * sum(abs(a[e] - b[e]) for e in a)


def slice_of(year: int) -> str:
    return "selection" if year < 2016 else "final"


def hit_status(records: list[dict], fractions: dict) -> str:
    """'hit', or why a pool composition with SuperCon2 records is left out of both hits and negatives."""
    if fractions.get("H", 0.0) >= 0.5:
        return "excluded_hydrogen_rich"
    if all(r["pressure"] for r in records):
        return "excluded_pressure_only"
    return "hit"


def supercon2_records(path: Path, valid: set[str]) -> dict:
    import build_mp_candidates as M

    out = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            year = row["year"].strip()
            if not (year.isdigit() and 1986 <= int(year) <= 2022):
                continue
            tc = parse_tc(row["criticalTemperature"])
            if tc is None or not 0 < tc <= 300:
                continue
            try:
                fractions = M.fractions(row["formula"].strip())
            except Exception:  # noqa: BLE001 - extracted formulas are often not formulas
                fractions = None
            if not fractions or not set(fractions) <= valid:
                continue
            out[frozenset(fractions)].append({"fractions": fractions, "year": int(year), "tc": tc,
                                              "pressure": bool(row["appliedPressure"].strip()),
                                              "doi": row["doi"].strip(), "title": row["title"].strip()})
    return out


def old_era_labels() -> tuple[dict, dict]:
    """(label by composition key, compositions by element set) from the 2018 export and 3DSC only."""
    import build_mp_candidates as M
    import build_scratchnet_data as B

    index = M.known_index(M.THREEDSC, M.STANEV, None, None, M.QUARANTINE_QUEUE)
    label: dict[str, int] = {}
    by_system = defaultdict(list)
    for reports in index.values():
        for fractions, tc, _ in reports:
            key = B.composition_key(fractions)
            label[key] = max(label.get(key, 0), int(tc > 0))
            by_system[frozenset(fractions)].append(fractions)
    return label, by_system


def cmd_data() -> dict:
    import build_mp_candidates as M
    import build_scratchnet_data as B

    if not SUPERCON2.is_file():
        SUPERCON2.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SUPERCON2_URL, SUPERCON2)
    digest = hashlib.sha256(SUPERCON2.read_bytes()).hexdigest()
    if digest != SUPERCON2_SHA256:
        raise SystemExit(f"SuperCon2 file has sha256 {digest}, expected {SUPERCON2_SHA256}")
    ds = C.load_dataset()
    valid = set(ds.table.symbols)
    records = supercon2_records(SUPERCON2, valid)
    label, by_system = old_era_labels()

    pool, hits, seen = [], [], set()
    with gzip.open(CANDIDATES, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            fractions = M.fractions(row["formula"])
            if not fractions or not set(fractions) <= valid:
                continue
            key = B.composition_key(fractions)
            if key in seen:
                continue
            seen.add(key)
            if any(half_gap(fractions, other) <= NEAR for other in by_system.get(frozenset(fractions), [])):
                continue
            matched = [r for r in records.get(frozenset(fractions), [])
                       if max(abs(fractions[e] - r["fractions"][e]) for e in fractions) <= EXACT]
            status = "negative"
            if matched:
                status = hit_status(matched, fractions)
                first = min(r["year"] for r in matched)
                earliest = min(matched, key=lambda r: r["year"])
                hits.append({"mp_id": row["mp_id"], "formula": row["formula"], "composition": key,
                             "first_year": first, "slice": slice_of(first), "status": status,
                             "records": len(matched), "papers": len({r["doi"] for r in matched}),
                             "tc_max_k": max(r["tc"] for r in matched), "mp_theoretical": row["theoretical"],
                             "first_doi": earliest["doi"], "first_title": earliest["title"]})
            pool.append({"mp_id": row["mp_id"], "formula": row["formula"], "composition": key, "status": status,
                         "slice": slice_of(min(r["year"] for r in matched)) if matched else ""})

    train = []
    for i, row in enumerate(ds.rows):
        if ds.split[i] == "lockbox" or row["composition"] not in label:
            continue
        train.append({"composition": row["composition"], "label": label[row["composition"]],
                      "group": str(ds.arrays.group[i])})

    BENCH.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    fields = ["mp_id", "formula", "composition", "first_year", "slice", "status", "records", "papers", "tc_max_k",
              "mp_theoretical", "first_doi", "first_title"]
    with open(BENCH / "hits.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(hits, key=lambda h: (h["slice"], h["first_year"], h["formula"])))
    with gzip.open(WORK / "pool.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(pool, handle)
    with gzip.open(WORK / "train.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(train, handle)
    counts = defaultdict(int)
    for hit in hits:
        counts[f"{hit['slice']}:{hit['status']}"] += 1
    summary = {"supercon2_sha256": digest, "table_sha256": ds.sha256,
               "candidate_table_sha256": hashlib.sha256(CANDIDATES.read_bytes()).hexdigest(),
               "pool_compositions": len(pool), "pool_with_a_supercon2_record": len(hits),
               "by_slice_and_status": dict(sorted(counts.items())),
               "training_rows": len(train), "training_superconductors": int(sum(t["label"] for t in train))}
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "data.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return summary


# ------------------------------------------------------------------------------------------------ children

class _Labels:
    def __init__(self, label: np.ndarray):
        self.label = label.astype(float)
        self.tc_k = np.full(len(label), np.nan)


def load_tables() -> dict:
    with gzip.open(WORK / "pool.json.gz", "rt", encoding="utf-8") as handle:
        pool = json.load(handle)
    with gzip.open(WORK / "train.json.gz", "rt", encoding="utf-8") as handle:
        train = json.load(handle)
    text = lambda key: C.formula_text(C.fractions_of(key))  # noqa: E731
    X, names = C.formula_features([text(t["composition"]) for t in train], cache=WORK / "train-features.npz")
    P, _ = C.formula_features([text(p["composition"]) for p in pool], cache=WORK / "pool-features.npz")
    return {"pool": pool, "train": train, "X": X, "P": P, "names": names,
            "y": np.array([t["label"] for t in train]), "groups": np.array([t["group"] for t in train])}


def classifier_settings(overrides: dict) -> dict:
    import run_learning_cycle as L

    settings = dict(L.boosting_settings()["classifier_settings"])
    settings.update({k: v for k, v in overrides.items() if k in settings})
    return settings


def fit_boost(X, y, groups, rows, seed, overrides):
    fit, valid = C.T.inner_split(groups[rows], percent=12, seed=0)
    model, _ = C.boosting_fit(X, _Labels(y), rows[fit], rows[valid], seed=seed,
                              classifier_settings=classifier_settings(overrides), parts="classifier")
    return model[0]


def bag_rows(groups: np.ndarray, seed: int) -> np.ndarray:
    names, inverse = np.unique(groups, return_inverse=True)
    drawn = np.random.default_rng(seed).integers(0, len(names), size=len(names))
    return np.repeat(np.arange(len(groups)), np.bincount(drawn, minlength=len(names))[inverse])


def standardise(X: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(X, axis=0)
    X, P = np.where(np.isfinite(X), X, median), np.where(np.isfinite(P), P, median)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, (P - mean) / std


def physics_feature(compositions: list[str]) -> np.ndarray:
    """Cross-fitted score of 'coarse calculated Tc above 1 K' from Bercx et al.'s Table B, formula features only."""
    import rung4 as R

    rows = R.bercx_rows()
    Xb, _ = C.formula_features([C.formula_text(r["fractions"]) for r in rows])
    yb = np.array([r["target"] for r in rows])
    gb = np.array([r["system"] for r in rows])
    fold_b = np.array([R.system_fold(s) for s in gb])
    fractions = [C.fractions_of(key) for key in compositions]
    Xc, _ = C.formula_features([C.formula_text(f) for f in fractions])
    fold_c = np.array([R.system_fold("-".join(sorted(f))) for f in fractions])
    out = np.full(len(compositions), np.nan)
    for fold in range(5):
        model = fit_boost(Xb, yb, gb, np.flatnonzero(fold_b != fold), seed=fold, overrides={})
        if (fold_c == fold).any():
            out[fold_c == fold] = model.predict_proba(Xc[fold_c == fold])[:, 1]
    return out


def score_child(kind: str, tables: dict, overrides: dict | None = None) -> np.ndarray:
    """The child's score for every pool composition (higher = more likely a superconductor)."""
    overrides = overrides or {}
    X, P, y, groups = tables["X"], tables["P"], tables["y"], tables["groups"]
    rows = np.arange(len(y))
    if kind == "boost":
        return fit_boost(X, y, groups, rows, 0, overrides).predict_proba(P)[:, 1]
    if kind == "bag":
        scores = []
        for bag in range(overrides.get("bags", 8)):
            chosen = bag_rows(groups, 1000 + bag)
            scores.append(fit_boost(X[chosen], y[chosen], groups[chosen], np.arange(len(chosen)), bag,
                                    overrides).predict_proba(P)[:, 1])
        return np.mean(scores, axis=0)
    if kind == "physics":
        if "physics" not in tables:
            tables["physics"] = (physics_feature([t["composition"] for t in tables["train"]]),
                                 physics_feature([p["composition"] for p in tables["pool"]]))
        X2, P2 = np.column_stack([X, tables["physics"][0]]), np.column_stack([P, tables["physics"][1]])
        return fit_boost(X2, y, groups, rows, 0, overrides).predict_proba(P2)[:, 1]
    Xs, Ps = standardise(X, P)
    if kind == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=overrides.get("k", 15), weights="distance").fit(Xs, y)
        return model.predict_proba(Ps)[:, 1]
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=overrides.get("layers", (128, 64)), early_stopping=True,
                              validation_fraction=0.12, max_iter=300, random_state=0).fit(Xs, y)
        return model.predict_proba(Ps)[:, 1]
    raise ValueError(kind)


# ------------------------------------------------------------------------------------------------ scoring

def percentiles(scores: np.ndarray, hit_rows: np.ndarray, negative_rows: np.ndarray) -> np.ndarray:
    """For each hit: the share of negatives it outranks, ties counting half."""
    negatives = np.sort(scores[negative_rows])
    below = np.searchsorted(negatives, scores[hit_rows], side="left")
    ties = np.searchsorted(negatives, scores[hit_rows], side="right") - below
    return (below + 0.5 * ties) / len(negatives)


def summarise(scores: np.ndarray, hit_rows: np.ndarray, negative_rows: np.ndarray) -> dict:
    pct = percentiles(scores, hit_rows, negative_rows)
    return {"hits": int(len(hit_rows)), "auc": round(float(pct.mean()), 4),
            "hits_in_top_1_percent": int((pct >= 0.99).sum()), "hits_in_top_5_percent": int((pct >= 0.95).sum()),
            "median_percentile": round(float(np.median(pct)), 4)}


def paired_interval(a: np.ndarray, b: np.ndarray, resamples: int = 1000, seed: int = 0) -> dict:
    """Bootstrap over hits of mean(a) - mean(b), for per-hit percentiles ``a`` and ``b``."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(resamples, len(a)))
    diff = (a[idx] - b[idx]).mean(axis=1)
    low, high = np.percentile(diff, [2.5, 97.5])
    return {"difference": round(float(a.mean() - b.mean()), 4), "low": round(float(low), 4),
            "high": round(float(high), 4), "resamples": resamples}


def slice_rows(pool: list[dict], which: str, confirmed: set[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    hits = np.array([i for i, p in enumerate(pool) if p["status"] == "hit" and p["slice"] == which
                     and (confirmed is None or p["mp_id"] in confirmed)], int)
    negatives = np.array([i for i, p in enumerate(pool) if p["status"] == "negative"], int)
    return hits, negatives


def rank_average(scores: list[np.ndarray]) -> np.ndarray:
    from scipy.stats import rankdata
    return np.mean([rankdata(s) / len(s) for s in scores], axis=0)


def save_scores(name: str, scores: np.ndarray) -> None:
    np.save(WORK / f"scores-{name}.npy", scores)


def load_scores(name: str) -> np.ndarray:
    return np.load(WORK / f"scores-{name}.npy")


def cmd_gen1() -> dict:
    tables = load_tables()
    hits, negatives = slice_rows(tables["pool"], "selection")
    report = {"slice": "selection (pre-2016 SuperCon2 hits absent from the 2018 export)", "children": {}}
    for kind in GEN1:
        scores = score_child(kind, tables)
        save_scores(f"gen1-{kind}", scores)
        report["children"][kind] = summarise(scores, hits, negatives)
        print(kind, report["children"][kind], flush=True)
    order = sorted(GEN1, key=lambda k: -report["children"][k]["auc"])
    report["order"] = order
    report["winner"], report["runner_up"] = order[0], order[1]
    base = percentiles(load_scores("gen1-boost"), hits, negatives)
    report["against_boost"] = {k: paired_interval(percentiles(load_scores(f"gen1-{k}"), hits, negatives), base)
                               for k in GEN1 if k != "boost"}
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "gen1.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("order", "winner", "runner_up")}, indent=1))
    return report


def cmd_gen2() -> dict:
    gen1 = json.loads((DOCS / "gen1.json").read_text(encoding="utf-8"))
    tables = load_tables()
    hits, negatives = slice_rows(tables["pool"], "selection")
    winner, order = gen1["winner"], gen1["order"]
    a, b = VARIANTS[winner]
    children = {"elite": load_scores(f"gen1-{winner}"),
                "variant_a": score_child(winner, tables, a), "variant_b": score_child(winner, tables, b),
                "cross2": rank_average([load_scores(f"gen1-{k}") for k in order[:2]]),
                "cross3": rank_average([load_scores(f"gen1-{k}") for k in order[:3]])}
    report = {"parent": winner, "variant_a": {k: list(v) if isinstance(v, tuple) else v for k, v in a.items()},
              "variant_b": {k: list(v) if isinstance(v, tuple) else v for k, v in b.items()},
              "cross2": order[:2], "cross3": order[:3], "children": {}}
    parent = percentiles(children["elite"], hits, negatives)
    for name, scores in children.items():
        save_scores(f"gen2-{name}", scores)
        report["children"][name] = summarise(scores, hits, negatives)
        if name != "elite":
            report["children"][name]["against_parent"] = paired_interval(percentiles(scores, hits, negatives), parent)
    best = max((n for n in children if n != "elite"), key=lambda n: report["children"][n]["auc"])
    promoted = report["children"][best]["against_parent"]["low"] > 0
    report["best_offspring"], report["promoted"] = best, bool(promoted)
    report["champion"] = f"gen2-{best}" if promoted else f"gen1-{winner}"
    (DOCS / "gen2.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


def confirmed_hits() -> tuple[set[str], dict]:
    with open(CHECK, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts = defaultdict(int)
    for row in rows:
        counts[row["confirmed"].strip().lower()] += 1
    return {r["mp_id"] for r in rows if r["confirmed"].strip().lower() == "yes"}, dict(counts)


def cmd_final() -> dict:
    out = DOCS / "final.json"
    if out.is_file():
        raise SystemExit(f"{out} exists: the final slice is scored once.")
    gen2 = json.loads((DOCS / "gen2.json").read_text(encoding="utf-8"))
    tables = load_tables()
    confirmed, check_counts = confirmed_hits()
    champion, baseline = load_scores(gen2["champion"]), load_scores("gen1-boost")
    report = {"champion": gen2["champion"], "baseline": "gen1-boost", "reader_check": check_counts}
    for label, keep in (("confirmed_hits", confirmed), ("all_raw_hits", None)):
        hits, negatives = slice_rows(tables["pool"], "final", keep)
        entry = {"champion": summarise(champion, hits, negatives), "baseline": summarise(baseline, hits, negatives)}
        if gen2["champion"] != "gen1-boost":
            entry["champion_minus_baseline"] = paired_interval(percentiles(champion, hits, negatives),
                                                               percentiles(baseline, hits, negatives))
        entry["baseline_auc_minus_chance"] = paired_interval(percentiles(baseline, hits, negatives),
                                                             np.full(len(hits), 0.5))
        report[label] = entry
    hits, negatives = slice_rows(tables["pool"], "final", confirmed)
    pct = percentiles(baseline, hits, negatives)
    report["confirmed_hits_by_baseline_percentile"] = sorted(
        ({"formula": tables["pool"][i]["formula"], "percentile": round(float(p), 4)} for i, p in zip(hits, pct)),
        key=lambda r: -r["percentile"])
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["data", "gen1", "gen2", "final"])
    {"data": cmd_data, "gen1": cmd_gen1, "gen2": cmd_gen2, "final": cmd_final}[parser.parse_args().step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
