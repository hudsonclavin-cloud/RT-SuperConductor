#!/usr/bin/env python3
"""Rung 4, steps 1 and 2 of ``docs/rung4-plan.md`` (new compositions in known families).

    python scripts/rung4.py pool     # step 1: freeze the never-made pool
    python scripts/rung4.py proxy    # step 2: the physics proxy and tests T1 and T2

Step 1 reads the committed candidate table only. Step 2 also needs Bercx et al.'s Tables B and C
(``python scripts/bercx_published.py download`` puts them in ``data/raw/bercx-2025/``, which git ignores) and
cycle 3's saved out-of-fold predictions (``data/processed/learning-cycles/cycle3-oof.npz``).

Implementation choices not spelled out in the plan, fixed here before step 2 was run:
- A Bercx entry is keyed by (pymatgen reduced formula, space group), as in ``bercx_published.py``. Duplicate keys
  are one row; its target is 1 when the key is also in Table C.
- Folds for T1: whole element systems, by a hash of the system name.
- Proxy inputs: the ranking of record's formula features plus the space group number and the crystal system.
  Classifier only, round 2's settings; 12% of the training groups held back for early stopping.
- T2 rows: one per (reduced formula, space group) among experimentally reported metals in the candidate table with
  an exact database match; label 1 when any report gives Tc above 0. Rows whose key is in Table B are left out.
- "Adding the proxy to the ranking of record": on the T2 rows that are development rows of the composition table,
  the mean of the two normalised ranks (cycle 3's out-of-fold probability and the proxy), equal weights, no tuning.
  Paired group bootstrap (element systems, 1,000 resamples, seed 0) of the AUC difference.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

import scratchnet_common as C
from scratchnet_common import ROOT

CANDIDATES = ROOT / "data/candidates/mp-candidates.csv.gz"
BERCX = ROOT / "data/raw/bercx-2025"
OUT = ROOT / "data/benchmarks/rung4"
DOCS = ROOT / "docs/rung4"
OOF = ROOT / "data/processed/learning-cycles/cycle3-oof.npz"
MIN_KNOWN = 3
CRYSTAL_SYSTEMS = ((2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"), (142, "tetragonal"),
                   (167, "trigonal"), (194, "hexagonal"), (230, "cubic"))


def fractions(formula: str) -> dict[str, float] | None:
    try:
        return C.D_parse(formula)
    except ValueError:
        return None


def prototype(formula: str, spacegroup: str) -> tuple | None:
    """Sorted atomic fractions (3 decimals) and the space group: the plan's crude 'known family'."""
    frac = fractions(formula)
    if not frac:
        return None
    values = sorted(frac.values(), reverse=True)
    total = sum(values)
    return tuple(round(v / total, 3) for v in values), str(spacegroup).strip()


def prototype_label(key: tuple) -> str:
    return "-".join(f"{v:.3f}" for v in key[0]) + f"|sg{key[1]}"


def crystal_system(spacegroup: int) -> int:
    for code, (upper, _) in enumerate(CRYSTAL_SYSTEMS):
        if spacegroup <= upper:
            return code
    raise ValueError(f"space group {spacegroup} is not 1 to 230")


def known_tc(row: dict) -> float:
    try:
        return float(row.get("known_tc_max_k") or 0.0)
    except ValueError:
        return 0.0


def hull(row: dict) -> float:
    try:
        return float(row["energy_above_hull_ev_per_atom"])
    except (KeyError, ValueError):
        return float("inf")


def superconductor_prototypes(rows: list[dict], minimum: int = MIN_KNOWN) -> Counter:
    """Prototypes with at least ``minimum`` experimentally reported, exactly matched known superconductors."""
    counts: Counter = Counter()
    for row in rows:
        if row["theoretical"] == "False" and row["known_match"] == "exact" and known_tc(row) > 0:
            key = prototype(row["formula"], row["spacegroup_number"])
            if key:
                counts[key] += 1
    return Counter({key: n for key, n in counts.items() if n >= minimum})


def pool_steps(rows: list[dict], families: Counter) -> tuple[list[dict], dict]:
    """The plan's filters in its order, with the count after each."""
    steps = {}
    keep = [r for r in rows if r["theoretical"] == "True" and r["is_metal"] == "True" and not r["known_match"]
            and hull(r) <= 0.025]
    steps["never_made_untested_metals"] = len(keep)
    keep = [r for r in keep if prototype(r["formula"], r["spacegroup_number"]) in families]
    steps["in_known_superconductor_prototype"] = len(keep)
    keep = [r for r in keep if r["is_magnetic"] == "False"]
    steps["not_magnetic"] = len(keep)
    keep = [r for r in keep if "Cu" not in (fractions(r["formula"]) or {})]
    steps["no_copper"] = len(keep)
    steps["on_hull"] = sum(hull(r) <= 1e-9 for r in keep)
    return keep, steps


def load_candidates() -> list[dict]:
    with gzip.open(CANDIDATES, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def cmd_pool() -> dict:
    rows = load_candidates()
    families = superconductor_prototypes(rows)
    pool, steps = pool_steps(rows, families)
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["mp_id", "formula", "element_system", "prototype", "known_superconductors_in_prototype",
              "spacegroup_number", "energy_above_hull_ev_per_atom", "on_hull", "chemical_family"]
    with open(OUT / "pool.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(pool, key=lambda r: r["mp_id"]):
            key = prototype(row["formula"], row["spacegroup_number"])
            writer.writerow({"mp_id": row["mp_id"], "formula": row["formula"],
                             "element_system": "-".join(sorted(fractions(row["formula"]))),
                             "prototype": prototype_label(key), "known_superconductors_in_prototype": families[key],
                             "spacegroup_number": row["spacegroup_number"],
                             "energy_above_hull_ev_per_atom": row["energy_above_hull_ev_per_atom"],
                             "on_hull": hull(row) <= 1e-9, "chemical_family": row.get("chemical_family", "")})
    summary = {"candidate_table_sha256": hashlib.sha256(CANDIDATES.read_bytes()).hexdigest(),
               "known_superconductor_prototypes": len(families),
               "known_superconductors_in_them": int(sum(families.values())), "steps": steps,
               "pool_prototypes": len({prototype(r["formula"], r["spacegroup_number"]) for r in pool}),
               "pool_element_systems": len({"-".join(sorted(fractions(r["formula"]))) for r in pool})}
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "pool-summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return summary


# ------------------------------------------------------------------------------------------------ step 2

def bercx_rows() -> list[dict]:
    from bercx_published import composition_key

    def keys(name: str) -> dict:
        out = {}
        with open(BERCX / name, newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                key = (composition_key(r["Formula"].strip()), r[" Spacegroup"].strip())
                out.setdefault(key, r["Formula"].strip())
        return out

    table_b, table_c = keys("Table_B.csv"), keys("Table_C.csv")
    rows = []
    for (formula, sg), raw in sorted(table_b.items()):
        frac = fractions(formula)
        if frac:
            rows.append({"key": (formula, sg), "fractions": frac, "spacegroup": int(sg),
                         "system": "-".join(sorted(frac)), "target": int((formula, sg) in table_c)})
    return rows


def features(rows: list[dict]) -> np.ndarray:
    X, _ = C.formula_features([C.formula_text(r["fractions"]) for r in rows])
    extra = np.array([[r["spacegroup"], crystal_system(r["spacegroup"])] for r in rows], float)
    return np.hstack([X, extra])


class _Labels:
    def __init__(self, label: np.ndarray):
        self.label = label.astype(float)
        self.tc_k = np.full(len(label), np.nan)


def system_fold(system: str, folds: int = 5) -> int:
    return int(hashlib.sha256(f"rung4-proxy:{system}".encode()).hexdigest(), 16) % folds


def fit_proxy(X: np.ndarray, y: np.ndarray, groups: np.ndarray, rows: np.ndarray, seed: int, settings: dict):
    fit, valid = C.T.inner_split(groups[rows], percent=12, seed=seed)
    model, _ = C.boosting_fit(X, _Labels(y), rows[fit], rows[valid], seed=seed,
                              classifier_settings=settings.get("classifier_settings"), parts="classifier")
    return model


def auc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score))


def group_bootstrap(y, scores: list[np.ndarray], groups, resamples=1000, seed=0) -> list[dict]:
    """AUC intervals for each score, and for the difference of the last two, resampling whole groups."""
    names, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(names))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(names), len(names))])
        if len(set(y[idx].tolist())) < 2:
            continue
        draws.append([auc(y[idx], s[idx]) for s in scores])
    draws = np.array(draws)

    def summary(values):
        low, mid, high = np.percentile(values, [2.5, 50, 97.5])
        return {"low": round(float(low), 4), "median": round(float(mid), 4), "high": round(float(high), 4),
                "resamples": int(len(values))}
    out = [summary(draws[:, i]) for i in range(len(scores))]
    if len(scores) >= 2:
        out.append(summary(draws[:, -1] - draws[:, -2]))
    return out


def experimental_rows(candidates: list[dict], table_b: set) -> list[dict]:
    from bercx_published import composition_key

    rows, seen = [], set()
    for r in candidates:
        if r["theoretical"] != "False" or r["is_metal"] != "True" or r["known_match"] != "exact":
            continue
        frac = fractions(r["formula"])
        if not frac:
            continue
        key = (composition_key(r["formula"]), r["spacegroup_number"].strip())
        if key in seen:
            continue
        seen.add(key)
        rows.append({"key": key, "formula": r["formula"], "fractions": frac, "spacegroup": int(key[1]),
                     "system": "-".join(sorted(frac)), "label": int(known_tc(r) > 0),
                     "in_table_b": key in table_b, "formula_in_table_b": key[0] in {k[0] for k in table_b}})
    return rows


def cmd_proxy() -> dict:
    import run_learning_cycle as L
    import build_mp_candidates as M
    import build_scratchnet_data as B
    from scipy.stats import rankdata

    settings = L.boosting_settings()
    train = bercx_rows()
    X, y = features(train), np.array([r["target"] for r in train])
    groups = np.array([r["system"] for r in train])
    folds = np.array([system_fold(s) for s in groups])
    oof = np.full(len(train), np.nan)
    for fold in range(5):
        rows = np.flatnonzero(folds != fold)
        model = fit_proxy(X, y, groups, rows, seed=fold, settings=settings)
        oof[folds == fold] = C.boosting_apply(model, X[folds == fold])[0]
    t1 = {"rows": len(train), "positives": int(y.sum()), "element_systems": int(len(set(groups))),
          "auc": round(auc(y, oof), 4), "interval": group_bootstrap(y, [oof], groups)[0], "pass_mark": 0.70}
    t1["passes"] = t1["auc"] >= 0.70

    final = fit_proxy(X, y, groups, np.arange(len(train)), seed=0, settings=settings)
    test = experimental_rows(load_candidates(), {r["key"] for r in train})
    primary = [r for r in test if not r["in_table_b"]]
    Xt = features(primary)
    proxy = C.boosting_apply(final, Xt)[0]
    yt = np.array([r["label"] for r in primary])
    gt = np.array([r["system"] for r in primary])
    t2 = {"rows": len(primary), "superconductors": int(yt.sum()), "element_systems": int(len(set(gt))),
          "auc": round(auc(yt, proxy), 4), "interval": group_bootstrap(yt, [proxy], gt)[0], "pass_mark": 0.60}
    t2["passes"] = t2["auc"] >= 0.60
    strict = np.array([not r["formula_in_table_b"] for r in primary])
    t2["exploratory_formula_not_in_table_b"] = {
        "rows": int(strict.sum()), "superconductors": int(yt[strict].sum()),
        "auc": round(auc(yt[strict], proxy[strict]), 4)}

    with np.load(OOF) as saved:
        ranking = dict(zip((str(k) for k in saved["compositions"]), saved["p"]))
    keyed = [ranking.get(B.composition_key(M.fractions(r["formula"]))) for r in primary]
    have = np.array([k is not None and np.isfinite(k) for k in keyed])
    p_rank = np.array([k for k in keyed if k is not None and np.isfinite(k)], float)
    yp, gp, proxy_p = yt[have], gt[have], proxy[have]
    combined = (rankdata(p_rank) + rankdata(proxy_p)) / (2 * len(p_rank))
    boot = group_bootstrap(yp, [proxy_p, p_rank, combined], gp)
    added = {"rows": int(have.sum()), "superconductors": int(yp.sum()),
             "auc_proxy": round(auc(yp, proxy_p), 4), "auc_ranking_of_record": round(auc(yp, p_rank), 4),
             "auc_combined": round(auc(yp, combined), 4),
             "combined_minus_ranking_interval": boot[3], "gain_counts": boot[3]["low"] > 0}
    report = {"t1_learns_the_calculation": t1, "t2_carries_to_experiments": t2,
              "adding_the_proxy_to_the_ranking_of_record": added,
              "boosting_settings": settings,
              "note": "The proxy never sees an experimental label. Bercx et al.'s target is a coarse calculated "
                      "Allen-Dynes Tc above 1 K (Table C) among their Table B metals."}
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "proxy.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["pool", "proxy"])
    args = parser.parse_args()
    {"pool": cmd_pool, "proxy": cmd_proxy}[args.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
