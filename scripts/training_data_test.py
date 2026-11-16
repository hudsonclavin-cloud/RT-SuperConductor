#!/usr/bin/env python3
"""Which training data: the ranking of record's table, the NIMS file, or both? (``docs/training-data-test.md``)

    python scripts/training_data_test.py

One learner, three training sets, five folds by element system. A held-out fold's element systems are removed from
every training source. Each model is scored on the held-out rows of each table, by that table's own labels. The
record table's locked test rows are never read into a model or a score.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np

import children_knowledge as G
import children_nims as N
import rung4 as R
import scratchnet_common as C
from scratchnet_common import ROOT

OUT = ROOT / "docs/training-data-test.json"
FOLDS = 5
MODELS = ("record", "nims", "both")
PRIMARY_FAMILY = "Other"
HOME_TOLERANCE = 0.01


def fold_of(system: str) -> int:
    return int(hashlib.sha256(f"data-test:{system}".encode()).hexdigest(), 16) % FOLDS


def system_of(key: str) -> str:
    return "-".join(sorted(C.fractions_of(key)))


def record_rows() -> list[dict]:
    """The record table's development and train-only rows with a label. Locked test rows are left out."""
    ds = C.load_dataset()
    rows = []
    for row, split, label, family in zip(ds.rows, ds.split, ds.arrays.label, ds.family):
        if split in ("dev", "train_only") and np.isfinite(label):
            rows.append({"key": row["composition"], "label": int(label), "family": str(family)})
    return rows


def nims_rows() -> list[dict]:
    import build_mp_candidates as M

    table = json.loads((G.WORK / "table.json").read_text(encoding="utf-8"))
    return [{"key": key, "label": int(any(label for _, label, _ in table["records"][key])),
             "family": M.family(set(C.fractions_of(key)))} for key in table["keys"]]


def cross_fit(sources: dict[str, list[dict]], features: dict[str, np.ndarray]) -> dict:
    """For every row of every table: each model's score from the fold that never saw the row's element system."""
    arrays = {}
    for name, rows in sources.items():
        systems = np.array([system_of(r["key"]) for r in rows])
        arrays[name] = {"X": np.array([features[r["key"]] for r in rows]), "y": np.array([r["label"] for r in rows]),
                        "system": systems, "fold": np.array([fold_of(s) for s in systems]),
                        "family": np.array([r["family"] for r in rows])}
    scores = {domain: {model: np.full(len(a["y"]), np.nan) for model in MODELS} for domain, a in arrays.items()}
    for fold in range(FOLDS):
        train = {name: np.flatnonzero(a["fold"] != fold) for name, a in arrays.items()}
        parts = {"record": ["record"], "nims": ["nims"], "both": ["record", "nims"]}
        for model, names in parts.items():
            X = np.concatenate([arrays[n]["X"][train[n]] for n in names])
            y = np.concatenate([arrays[n]["y"][train[n]] for n in names])
            groups = np.concatenate([arrays[n]["system"][train[n]] for n in names])
            fitted = N.fit_boost(X, y, groups, G.BOOST, seed=fold)
            for domain, a in arrays.items():
                held = a["fold"] == fold
                scores[domain][model][held] = fitted.predict_proba(a["X"][held])[:, 1]
            print(f"fold {fold} {model}: {len(y)} training rows", flush=True)
    return {"arrays": arrays, "scores": scores}


def block(y: np.ndarray, groups: np.ndarray, scores: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    out = {"rows": int(mask.sum()), "superconductors": int(y[mask].sum()), "nulls": int(mask.sum() - y[mask].sum()),
           "auc": {model: round(N.auc(y[mask], scores[model][mask]), 4) for model in MODELS}}
    for other in ("record", "nims"):
        boot = R.group_bootstrap(y[mask], [scores[other][mask], scores["both"][mask]], groups[mask])
        wide = group_interval(y[mask], scores[other][mask], scores["both"][mask], groups[mask], (0.5, 99.5))
        out[f"both_minus_{other}"] = {"difference": round(out["auc"]["both"] - out["auc"][other], 4),
                                      "interval_95": [boot[2]["low"], boot[2]["high"]], "interval_99": wide}
    return out


def group_interval(y, a, b, groups, percentiles, resamples: int = 1000, seed: int = 0) -> list[float]:
    """Interval for AUC(b) - AUC(a), resampling whole element systems."""
    names, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(names))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(names), len(names))])
        if len(set(y[idx].tolist())) == 2:
            draws.append(N.auc(y[idx], b[idx]) - N.auc(y[idx], a[idx]))
    return [round(float(x), 4) for x in np.percentile(draws, percentiles)]


def disagreement(record: list[dict], nims: list[dict]) -> dict:
    a = {r["key"]: r["label"] for r in record}
    b = {r["key"]: r["label"] for r in nims}
    shared = sorted(set(a) & set(b))
    differ = [k for k in shared if a[k] != b[k]]
    kinds = Counter("record superconductor, NIMS null" if a[k] else "record negative, NIMS superconductor" for k in differ)
    return {"shared_compositions": len(shared), "labels_differ": len(differ),
            "fraction": round(len(differ) / len(shared), 4) if shared else None, "kinds": dict(kinds)}


def main() -> int:
    sources = {"record": record_rows(), "nims": nims_rows()}
    keys = sorted({r["key"] for rows in sources.values() for r in rows})
    X, _ = C.formula_features([C.formula_text(C.fractions_of(k)) for k in keys],
                              cache=ROOT / "data/processed/training-data-test-features.npz")
    fitted = cross_fit(sources, dict(zip(keys, X)))
    report = {"learner": G.BOOST, "folds": FOLDS,
              "training_rows": {name: len(rows) for name, rows in sources.items()},
              "labels": {name: dict(Counter(r["label"] for r in rows)) for name, rows in sources.items()},
              "shared_compositions": disagreement(sources["record"], sources["nims"]), "scored_on": {}}
    for domain, a in fitted["arrays"].items():
        intermetallic = a["family"] == PRIMARY_FAMILY
        report["scored_on"][f"{domain}_rows"] = {
            "intermetallics": block(a["y"], a["system"], fitted["scores"][domain], intermetallic),
            "all_families": block(a["y"], a["system"], fitted["scores"][domain], np.ones(len(a["y"]), dtype=bool))}
    abroad = report["scored_on"]["nims_rows"]["intermetallics"]["both_minus_record"]
    home = report["scored_on"]["record_rows"]["intermetallics"]["both_minus_record"]
    helps, holds = abroad["interval_99"][0] > 0, home["difference"] >= -HOME_TOLERANCE
    report["verdict"] = {"both_helps_on_nims_rows": bool(helps), "both_holds_on_record_rows": bool(holds),
                         "reading": ("candidate change to the ranking of record: propose it to the owner" if helps and holds
                                     else "the two tables mean different things by a negative; no change" if helps
                                     else "the blind-test difference was chance; drop it")}
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
