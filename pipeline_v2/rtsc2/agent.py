"""The selection agent, and a replay harness that scores it.

The agent's job each round: given a model's opinion of every untested
material, choose the next batch to send for checking. The legacy loop picks by
a fixed formula and never measures whether that choice rule is any good.

``replay`` measures it. Labels are hidden, the agent starts from a small
seed set, picks a batch, the hidden labels of that batch are revealed, the
model is refitted, and the round repeats. The count of hits found per query
is then compared with random picking. Policies:

    random     baseline
    greedy     highest predicted probability
    ucb        mean + kappa * spread across ensemble members
    thompson   rank by one randomly drawn ensemble member per pick
    diverse    greedy, but at most ``per_system`` picks per element system
               per batch, so one chemistry cannot fill the batch

A replay can only use materials whose outcome is already known, so it shows
which policy searches *this labelled set* best. It does not prove the same
ordering holds on the untested pool, where hits are far rarer.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .data import Dataset
from .models import TreeBag

POLICIES = ("random", "greedy", "ucb", "thompson", "diverse")


def _walk(order: np.ndarray, groups: np.ndarray, batch: int, cap: int | None, fill: bool) -> np.ndarray:
    """Take rows in ``order``, allowing at most ``cap`` per element system."""
    if cap is None:
        return order[:batch]
    picked, used = [], {}
    for position in order:
        if used.get(groups[position], 0) < cap:
            picked.append(int(position))
            used[groups[position]] = used.get(groups[position], 0) + 1
            if len(picked) == batch:
                break
    if fill and len(picked) < batch:  # too few systems: top up with the best remaining rows
        taken = set(picked)
        picked += [int(p) for p in order if int(p) not in taken][: batch - len(picked)]
    return np.asarray(picked, dtype=int)


def choose(policy: str, mean: np.ndarray, members: np.ndarray, groups: np.ndarray, batch: int,
           rng: np.random.Generator, *, kappa: float = 1.0, per_system: int = 2,
           max_per_system: int | None = None) -> np.ndarray:
    """Positions (into the pool arrays) of the next batch.

    ``max_per_system`` is a hard cap applied to every policy; the batch comes back short
    when the pool has too few element systems. ``per_system`` is the softer cap that
    defines the ``diverse`` policy, which tops up from the best remaining rows.
    """
    batch = min(batch, len(mean))
    hard = max_per_system
    if policy == "random":
        return _walk(rng.permutation(len(mean)), groups, batch, hard, False)
    if policy == "greedy":
        return _walk(np.argsort(-mean, kind="stable"), groups, batch, hard, False)
    if policy == "ucb":
        return _walk(np.argsort(-(mean + kappa * members.std(axis=0)), kind="stable"), groups, batch, hard, False)
    if policy == "thompson":
        picked: list[int] = []
        blocked = np.zeros(len(mean), dtype=bool)
        used: dict[str, int] = {}
        for _ in range(batch):
            if blocked.all():
                break
            best = int(np.argmax(np.where(blocked, -np.inf, members[rng.integers(len(members))])))
            picked.append(best)
            blocked[best] = True
            used[groups[best]] = used.get(groups[best], 0) + 1
            if hard is not None and used[groups[best]] >= hard:
                blocked |= groups == groups[best]
        return np.asarray(picked, dtype=int)
    if policy == "diverse":
        cap = per_system if hard is None else min(per_system, hard)
        return _walk(np.argsort(-mean, kind="stable"), groups, batch, cap, hard is None)
    raise ValueError(f"unknown policy {policy!r}; choose from {POLICIES}")


def _fast_model(seed: int) -> TreeBag:
    return TreeBag(members=4, seed=seed, max_iter=80, early_stopping=False)


def _seed_set(target: np.ndarray, groups: np.ndarray, initial: int, strict: bool,
              rng: np.random.Generator) -> np.ndarray:
    """Starting labelled rows. Strict mode draws whole element systems, so no system is split."""
    for _ in range(100):  # the seed set must contain both outcomes
        known = np.zeros(len(target), dtype=bool)
        if strict:
            unique, inverse = np.unique(groups, return_inverse=True)
            sizes = np.bincount(inverse)
            order = rng.permutation(len(unique))
            enough = int(np.searchsorted(np.cumsum(sizes[order]), initial)) + 1
            known = np.isin(inverse, order[:enough])
        else:
            known[rng.choice(len(target), initial, replace=False)] = True
        if 0 < target[known].sum() < known.sum():
            return known
    raise ValueError("could not draw a seed set containing both outcomes")


def replay(dataset: Dataset, target: np.ndarray, policy: str, *, seed: int = 0, initial: int = 100,
           batch: int = 25, rounds: int = 20, strict: bool = False,
           model: Callable[[int], Any] = _fast_model) -> dict[str, Any]:
    """One simulated campaign. ``target`` is the 0/1 outcome the agent is hunting.

    ``strict=True`` is the discovery test. The seed set is whole element systems, the agent
    may query each remaining element system once, and the other materials of a queried
    system leave the pool without being revealed. Every query is therefore a first look at
    an element system the model has never seen, and a hit cannot be a polymorph or a
    different stoichiometry of something already known.
    """
    rng = np.random.default_rng(seed)
    target = np.asarray(target).astype(int)
    known = _seed_set(target, dataset.groups, initial, strict, rng)
    seed_rows = int(known.sum())
    retired = np.zeros(len(target), dtype=bool)  # strict mode: siblings of queried materials
    found, novel = [], []
    for round_number in range(rounds):
        pool = np.where(~known & ~retired)[0]
        if not len(pool):
            break
        if policy == "random":
            mean, members = np.zeros(len(pool)), np.zeros((1, len(pool)))
        else:
            fitted = model(seed * 1000 + round_number).fit(
                dataset.X[known], target[known], groups=dataset.groups[known],
                tokens=(dataset.el_index[known], dataset.el_fraction[known]))
            mean, members = fitted.predict(dataset.X[pool], tokens=(dataset.el_index[pool], dataset.el_fraction[pool]))
        picks = pool[choose(policy, mean, members, dataset.groups[pool], batch, rng,
                            max_per_system=1 if strict else None)]
        if not len(picks):
            break
        # A hit is "new-system" when no hit was yet known in its element system: finding a
        # second polymorph of a known superconductor is a hit but not a discovery.
        seen = set(dataset.groups[known & (target == 1)])
        novel.append(len(set(dataset.groups[picks][target[picks] == 1]) - seen))
        known[picks] = True
        if strict:
            retired |= np.isin(dataset.groups, dataset.groups[picks]) & ~known
        found.append(int(target[picks].sum()))
    return {"policy": policy, "seed": seed, "strict": strict, "hits_per_round": found,
            "seed_rows": seed_rows, "queries": int(known.sum()) - seed_rows,
            "hits": int(sum(found)), "new_system_hits": int(sum(novel))}


def compare_policies(dataset: Dataset, target: np.ndarray, *, policies: tuple[str, ...] = POLICIES,
                     seeds: int = 5, n_jobs: int = 4, **settings: Any) -> dict[str, Any]:
    """Every policy on the same seeds; the same seed gives the same starting set."""
    from joblib import Parallel, delayed

    runs = Parallel(n_jobs=n_jobs)(delayed(replay)(dataset, target, policy, seed=seed, **settings)
                                   for policy in policies for seed in range(seeds))
    target = np.asarray(target).astype(int)
    system_hit = np.unique(dataset.groups[target == 1])
    summary: dict[str, Any] = {"base_rate": float(target.mean()), "rows": int(len(target)), "seeds": seeds,
                               "element_systems": int(len(np.unique(dataset.groups))),
                               "element_systems_with_a_hit": int(len(system_hit)),
                               "settings": {k: v for k, v in settings.items() if not callable(v)}, "policies": {}}
    by_policy = {p: [r for r in runs if r["policy"] == p] for p in policies}
    random_hits = np.asarray([r["hits"] for r in by_policy.get("random", [])], dtype=float)
    for policy, group in by_policy.items():
        hits = np.asarray([r["hits"] for r in group], dtype=float)
        entry = {"hits_mean": float(hits.mean()), "hits_min": float(hits.min()), "hits_max": float(hits.max()),
                 "queries_mean": float(np.mean([r["queries"] for r in group])),
                 "hit_rate": float(hits.sum() / max(sum(r["queries"] for r in group), 1)),
                 "new_system_hits_mean": float(np.mean([r["new_system_hits"] for r in group])),
                 "cumulative_hits_mean": np.cumsum(np.mean([r["hits_per_round"] for r in group], axis=0)).tolist()}
        if len(random_hits) == len(hits) and policy != "random":
            entry["enrichment_over_random"] = float(hits.mean() / max(random_hits.mean(), 1e-9))
            entry["beats_random_on_seeds"] = f"{int((hits > random_hits).sum())} of {len(hits)}"
            random_new = np.mean([r["new_system_hits"] for r in by_policy["random"]])
            entry["new_system_enrichment_over_random"] = float(entry["new_system_hits_mean"] / max(random_new, 1e-9))
        summary["policies"][policy] = entry
    return summary


def propose(dataset: Dataset, model: Any, *, policy: str = "diverse", batch: int = 25, seed: int = 0,
            eligible: np.ndarray | None = None) -> "Any":
    """Rank untested materials with a fitted model and return the next batch as a table."""
    import pandas as pd

    pool = ~dataset.labelled if eligible is None else (~dataset.labelled & eligible)
    rows = np.where(pool)[0]
    mean, members = model.predict(dataset.X[rows], tokens=(dataset.el_index[rows], dataset.el_fraction[rows]))
    picks = choose(policy, mean, members, dataset.groups[rows], batch, np.random.default_rng(seed))
    chosen = rows[picks]
    return pd.DataFrame({"rank": np.arange(1, len(chosen) + 1), "mp_id": dataset.ids[chosen],
                         "formula": dataset.formulas[chosen], "element_system": dataset.groups[chosen],
                         "p_superconductor": np.round(mean[picks], 4),
                         "ensemble_spread": np.round(members.std(axis=0)[picks], 4), "policy": policy})
