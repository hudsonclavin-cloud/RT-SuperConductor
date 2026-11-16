#!/usr/bin/env python3
"""Generations 8, 9 and 10 of the children experiment: evolution on selection halves, one lockbox run.

    python scripts/children_evolve.py gen8       # single genes added to the baseline; single genes removed from the old champion
    python scripts/children_evolve.py gen9       # from the three fittest: one more gene each, and each pair merged
    python scripts/children_evolve.py gen10      # the same step again
    python scripts/children_evolve.py lockbox    # once: champion, baseline and old champion on the lockbox halves

Plan, rules and predictions: ``docs/children-experiment.md`` ("Generations 8, 9 and 10"). A child is a set of genes on
the fixed learner. Every element system of a fold's test set belongs to one half by a hash of its name; fitness reads
the selection half only, and nothing but the ``lockbox`` command reads the other.

Choices the plan left to the code, fixed before any run:
- ``recent`` weights a training row by exp(-(cut-off year - its first year) / 15); ``balance`` gives each class half
  the total weight; with both, the weights multiply.
- When a merge would hold both proxies, the fitter parent's proxy is kept.
- Ties in fitness go to the child with fewer genes, then by name.
- A scored child is cached by its genes, so a stopped generation continues where it left off.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

import children as K
import children_knowledge as G
import children_nims as N
import physics_proxy2 as P
import scratchnet_common as C
from scratchnet_common import ROOT

WORK = ROOT / "data/processed/children-evolve"
DOCS = ROOT / "docs/children"
FOLDS = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "second-look"}
GENES = ("elements", "proxy", "proxy2", "alloy", "tc-size", "family", "bag", "balance", "recent", "leaves63")
OLD_CHAMPION = frozenset({"elements", "proxy", "alloy", "tc-size", "family"})
BASELINE = frozenset()
RECENCY_YEARS = 15.0
BAGS = 8
KEEP = 3


def half(system: str) -> str:
    return "lockbox" if int(hashlib.sha256(f"gen8-lockbox:{system}".encode()).hexdigest(), 16) % 2 else "selection"


def name_of(genes: frozenset) -> str:
    return "+".join(g for g in GENES if g in genes) or "baseline"


def valid(genes: frozenset) -> bool:
    return genes <= set(GENES) and not {"proxy", "proxy2"} <= genes


def add_gene(genes: frozenset, gene: str) -> frozenset:
    """The set with one more gene; adding one proxy replaces the other."""
    other = {"proxy": "proxy2", "proxy2": "proxy"}.get(gene)
    return frozenset((genes - {other}) | {gene})


def merge(fitter: frozenset, other: frozenset) -> frozenset:
    union = fitter | other
    if {"proxy", "proxy2"} <= union:
        union = union - ({"proxy", "proxy2"} - fitter) if fitter & {"proxy", "proxy2"} else union - {"proxy"}
    return frozenset(union)


def generation_8() -> list[frozenset]:
    children = [add_gene(BASELINE, g) for g in GENES]
    children += [OLD_CHAMPION - {g} for g in sorted(OLD_CHAMPION)]
    children.append(add_gene(OLD_CHAMPION, "proxy2"))              # the proxy plan's second P3 child
    return unique(children)


def next_generation(ranked: list[frozenset]) -> list[frozenset]:
    """From the fittest ``KEEP`` (best first): each with one more gene, and each pair merged."""
    top = ranked[:KEEP]
    children = [add_gene(parent, g) for parent in top for g in GENES if g not in parent]
    children += [merge(top[i], top[j]) for i in range(len(top)) for j in range(i + 1, len(top))]
    return unique(children)


def unique(children: list[frozenset]) -> list[frozenset]:
    seen, out = set(), []
    for child in children:
        if valid(child) and child not in seen:
            seen.add(child)
            out.append(child)
    return out


def rank(fitness: dict[frozenset, float]) -> list[frozenset]:
    return sorted(fitness, key=lambda genes: (-fitness[genes], len(genes), name_of(genes)))


# ------------------------------------------------------------------------------------------------ scoring a child

def load(fold: str) -> dict:
    data = G.load(FOLDS[fold])
    table = json.loads((G.WORK / "table.json").read_text(encoding="utf-8"))
    cutoff = {**G.FOLDS, "second-look": G.SECOND_LOOK}[FOLDS[fold]]["train_until"]
    first = np.array([min(year for year, _, _ in table["records"][table["keys"][i]]) for i in data["train"]])
    data["age"] = np.clip(cutoff - first, 0, None).astype(float)
    data["half"] = np.array([half(s) for s in data["test_groups"]])
    if (P.WORK / "table.csv").is_file():
        data["proxy2"] = P.nims_features()
    return data


def weights_of(genes: frozenset, y: np.ndarray, age: np.ndarray) -> np.ndarray | None:
    if not genes & {"balance", "recent"}:
        return None
    w = np.ones(len(y))
    if "recent" in genes:
        w *= np.exp(-age / RECENCY_YEARS)
    if "balance" in genes:
        total = w.sum()
        for label in (0, 1):
            mask = y == label
            if mask.any():
                w[mask] *= 0.5 * total / w[mask].sum()
    return w * len(w) / w.sum()


def fit(kind: str, X, target, groups, settings: dict, weights, seed: int):
    """Boosting with the project's early stopping (12% of element systems held back), and optional row weights."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    chosen, valid_rows = C.T.inner_split(groups, percent=12, seed=0)
    model = (HistGradientBoostingClassifier if kind == "classifier" else HistGradientBoostingRegressor)(
        max_iter=1500, early_stopping=True, n_iter_no_change=40, random_state=seed, **settings)
    extra = {} if weights is None else {"sample_weight": weights[chosen], "sample_weight_val": weights[valid_rows]}
    return model.fit(X[chosen], target[chosen], X_val=X[valid_rows], y_val=target[valid_rows], **extra)


def score_once(genes: frozenset, data: dict, chosen: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    chosen = np.arange(len(data["train"])) if chosen is None else chosen
    blocks, changes = data["blocks"], set(genes) & {"elements", "proxy", "alloy", "family"}
    if "proxy2" in genes:
        blocks, changes = {**blocks, "proxy": data["proxy2"]}, changes | {"proxy"}
    X = G.columns(tuple(changes), blocks, data["train"][chosen])
    T = G.columns(tuple(changes), blocks, data["test"])
    y, groups = data["y"][chosen], data["groups"][chosen]
    weights = weights_of(genes, y, data["age"][chosen])
    settings = {**G.BOOST, "max_leaf_nodes": 63} if "leaves63" in genes else dict(G.BOOST)
    parts = [fit("classifier", X, y, groups, settings, weights, seed).predict_proba(T)[:, 1]]
    if "tc-size" in genes:
        parts.append(fit("regressor", X, np.log1p(data["tc"][chosen]), groups, settings, weights, seed).predict(T))
    if "family" in genes:
        expert, family = parts[0].copy(), data["family"][chosen]
        for name in sorted(set(data["test_family"])):
            mine = np.flatnonzero(family == name)
            if len(mine) < G.EXPERT_MINIMUM or len(set(y[mine].tolist())) < 2:
                continue
            try:
                model = fit("classifier", X[mine], y[mine], groups[mine], settings,
                            None if weights is None else weights[mine], seed)
            except ValueError:
                continue
            at = data["test_family"] == name
            expert[at] = model.predict_proba(T[at])[:, 1]
        parts.append(expert)
    return parts[0] if len(parts) == 1 else np.mean([G.ranks(p) for p in parts], axis=0)


def score_child(genes: frozenset, data: dict) -> np.ndarray:
    core = genes - {"bag"}
    if "bag" not in genes:
        return score_once(core, data)
    return np.mean([G.ranks(score_once(core, data, K.bag_rows(data["groups"], 1000 + bag), seed=bag))
                    for bag in range(BAGS)], axis=0)


def scores_for(genes: frozenset, fold: str, data: dict) -> np.ndarray:
    """A child's scores for every test row of a fold, from the cache or freshly fitted."""
    cache = cache_file(genes, fold)
    if cache.is_file():
        return np.load(cache)
    scores = score_child(genes, data)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, scores)
    return scores


def cache_file(genes: frozenset, fold: str):
    return WORK / "scores" / f"{fold}-{hashlib.sha256(name_of(genes).encode()).hexdigest()[:16]}.npy"


def _warm_fold(fold: str, names: list[str]) -> int:
    """One worker process: fit and cache these children's scores for one fold."""
    data = load(fold)
    for name in names:
        scores_for(frozenset(name.split("+")) if name != "baseline" else BASELINE, fold, data)
        print(f"  cached {fold} {name}", flush=True)
    return len(names)


def warm(children: list[frozenset], jobs: int) -> None:
    """Fit the children whose scores are not cached yet, one process per fold. The scores are the same as when
    fitted one after another; this only uses the processors better."""
    from joblib import Parallel, delayed

    todo = {fold: [name_of(c) for c in children if not cache_file(c, fold).is_file()] for fold in FOLDS}
    todo = {fold: names for fold, names in todo.items() if names}
    if todo and jobs > 1:
        Parallel(n_jobs=min(jobs, len(todo)))(delayed(_warm_fold)(fold, names) for fold, names in todo.items())


def auc_on(data: dict, scores: np.ndarray, which: str) -> float:
    mask = (data["half"] == which) & (data["test_family"] == G.PRIMARY_FAMILY)
    return N.auc(data["y_test"][mask], scores[mask])


# ------------------------------------------------------------------------------------------------ generations

EXTRA = -1    # scored, but not a child the plan's rule breeds (an interrupted run bred from its own generation)


def genes_of(name: str) -> frozenset:
    return BASELINE if name == "baseline" else frozenset(name.split("+"))


def eligible(state: dict, upto: int | None = None) -> dict[frozenset, float]:
    """Fitness of the children the rule bred, born in generation ``upto`` or earlier (any generation if None)."""
    born = state["born"]
    return {genes_of(name): value for name, value in state["fitness"].items()
            if born[name] != EXTRA and (upto is None or born[name] <= upto)}


def relabel(born: dict[str, int], number: int, proper: set[str]) -> list[str]:
    """Mark as EXTRA every child recorded for this generation that the rule does not breed for it."""
    strays = sorted(name for name, generation in born.items() if generation == number and name not in proper)
    for name in strays:
        born[name] = EXTRA
    return strays


def read_state() -> dict:
    path = WORK / "state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"fitness": {}, "generation": {}}


def gain_interval(datas: dict, child: frozenset, against: frozenset, which: str) -> dict:
    folds = {}
    for fold, data in datas.items():
        mask = data["half"] == which
        folds[fold] = {"y": data["y_test"][mask], "groups": data["test_groups"][mask],
                       "family": data["test_family"][mask],
                       "scores": {"baseline": scores_for(against, fold, data)[mask],
                                  "child": scores_for(child, fold, data)[mask]}}
    out = G.fold_bootstrap(folds, ["baseline", "child"])["child"]
    return {"gain_95": out["primary_gain_95"], "gain_99": out["primary_gain_99"]}


def run_generation(number: int) -> dict:
    state = read_state()
    fitness = {genes_of(k): v for k, v in state["fitness"].items()}
    # Which generation first scored a child. A state written before this field existed holds generation 8 only.
    born = state.setdefault("born", {name: 8 for name in state["fitness"]})
    if number == 8:
        children = [BASELINE, OLD_CHAMPION, *generation_8()]
    else:
        if str(number - 1) not in state["generation"]:
            raise SystemExit(f"generation {number - 1} has not been run")
        # Parents are the fittest of EARLIER generations only. Breeding from every scored child let an interrupted
        # generation, when resumed, breed from its own children (found and fixed 2026-11-16).
        children = next_generation(rank(eligible(state, number - 1)))
    strays = relabel(born, number, {name_of(c) for c in unique(children)})
    datas = {fold: load(fold) for fold in FOLDS}
    waiting = [c for c in children if "proxy2" in c] if "proxy2" not in datas["A"] else []
    children = [c for c in children if c not in waiting]      # scored now; the rest when the proxy's table exists
    if number == 8:                                           # this script's baseline must be the earlier one
        for genes, changes in ((BASELINE, ()), (OLD_CHAMPION, G.CHANGES)):
            if not np.allclose(G.score_child(tuple(changes), datas["A"]), scores_for(genes, "A", datas["A"])):
                raise SystemExit(f"this script's {name_of(genes)} differs from generation 7's on fold A")
    warm([c for c in unique(children) if c not in fitness], int(os.environ.get("EVOLVE_JOBS", "1")))
    for child in unique(children):
        if child in fitness:
            if born[name_of(child)] == EXTRA:                 # fitted earlier outside the rule; the rule breeds it now
                born[name_of(child)] = number
            continue
        values = {fold: auc_on(data, scores_for(child, fold, data), "selection") for fold, data in datas.items()}
        fitness[child] = float(np.mean(list(values.values())))
        print(f"gen{number} {name_of(child)} {fitness[child]:.4f}", flush=True)
        state["fitness"][name_of(child)] = fitness[child]
        born[name_of(child)] = number
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "state.json").write_text(json.dumps(state), encoding="utf-8")
    if waiting:
        raise SystemExit(f"{len(waiting)} children wait for the second proxy's table (physics_proxy2.py data); "
                         "run this generation again when it exists")
    counted = eligible(state, number)
    ranked = rank(counted)
    base = fitness[BASELINE]
    per_fold = {name_of(c): {fold: round(auc_on(data, scores_for(c, fold, data), "selection"), 4)
                             for fold, data in datas.items()}
                for c in fitness if born[name_of(c)] == number}      # from the cached scores; nothing is refitted
    report = {"generation": number, "selection_half_only": True, "baseline_fitness": round(base, 4),
              "children_scored_this_generation": len(per_fold), "children_scored_so_far": len(counted),
              "scored_outside_the_rule_and_not_counted": sorted(n for n, g in born.items() if g == EXTRA),
              "relabelled_in_this_run": strays,
              "this_generation": [{"child": name, "fitness": round(state["fitness"][name], 4),
                                   "gain": round(state["fitness"][name] - base, 4), "by_fold": folds}
                                  for name, folds in sorted(per_fold.items(), key=lambda kv: -state["fitness"][kv[0]])],
              "fittest_so_far": [{"child": name_of(c), "fitness": round(fitness[c], 4), "gain": round(fitness[c] - base, 4),
                                  **(gain_interval(datas, c, BASELINE, "selection") if c != BASELINE else {})}
                                 for c in ranked[:KEEP]]}
    if number == 8:                                           # the second proxy's third check (P3), selection halves
        swapped = add_gene(OLD_CHAMPION, "proxy2")
        alone = gain_interval(datas, frozenset({"proxy2"}), BASELINE, "selection")
        report["proxy2_p3"] = {
            "proxy2_minus_baseline": {"gain": round(fitness[frozenset({"proxy2"})] - base, 4), **alone},
            "proxy_minus_baseline": {"gain": round(fitness[frozenset({"proxy"})] - base, 4),
                                     **gain_interval(datas, frozenset({"proxy"}), BASELINE, "selection")},
            "old_champion_with_proxy2_minus_old_champion": {
                "gain": round(fitness[swapped] - fitness[OLD_CHAMPION], 4),
                **gain_interval(datas, swapped, OLD_CHAMPION, "selection")},
            "passes": bool(alone["gain_99"][0] > 0)}
    state["generation"][str(number)] = [name_of(c) for c in ranked[:KEEP]]
    (WORK / "state.json").write_text(json.dumps(state), encoding="utf-8")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / f"evolve-gen{number}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "this_generation"}, indent=1))
    return report


def cmd_lockbox() -> dict:
    out = DOCS / "evolve-lockbox.json"
    if out.is_file():
        raise SystemExit(f"{out} exists: the lockbox is opened once.")
    state = read_state()
    if "10" not in state["generation"]:
        raise SystemExit("generation 10 has not been run")
    fitness = eligible(state)
    champion = rank(fitness)[0]
    datas = {fold: load(fold) for fold in FOLDS}
    entrants = {"baseline": BASELINE, "champion": champion, "old_champion": OLD_CHAMPION}
    report = {"champion": name_of(champion), "champion_selection_fitness": round(fitness[champion], 4),
              "baseline_selection_fitness": round(fitness[BASELINE], 4),
              "selection_gain": round(fitness[champion] - fitness[BASELINE], 4), "children_scored": len(fitness),
              "lockbox": {}}
    for label, genes in entrants.items():
        values = {fold: auc_on(data, scores_for(genes, fold, data), "lockbox") for fold, data in datas.items()}
        entry = {"genes": name_of(genes), "auc": round(float(np.mean(list(values.values()))), 4),
                 "by_fold": {f: round(v, 4) for f, v in values.items()}}
        if label != "baseline":
            entry["gain"] = round(entry["auc"] - report["lockbox"]["baseline"]["auc"], 4)
            entry.update(gain_interval(datas, genes, BASELINE, "lockbox"))
        report["lockbox"][label] = entry
    champion_entry = report["lockbox"]["champion"]
    report["reading"] = ("confirmed on held-out data" if champion_entry["gain_95"][0] > 0 else
                         "not confirmed; the baseline stands" if champion_entry["gain"] > 0 else
                         "selection found noise again; the baseline stands")
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("gen8", "gen9", "gen10", "lockbox"))
    command = parser.parse_args().command
    if command == "lockbox":
        cmd_lockbox()
    else:
        run_generation(int(command[3:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
