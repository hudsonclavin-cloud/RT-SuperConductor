#!/usr/bin/env python3
"""Rung 4, shortlist 3: shortlist 2's filtered pool, verdicts carried forward, ranked by cycle 6 (``docs/rung4-shortlist3.md``).

    python scripts/rung4_shortlist3.py

Needs shortlist 2's filtered pool (``python scripts/rung4_shortlist2.py filters``) and cycle 6's five final models
(``python scripts/run_learning_cycle.py --name cycle6 ...``). Writes ``data/benchmarks/rung4/shortlist3*.csv`` and
``docs/rung4/shortlist3-summary.json``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import rung4 as R
import rung4_shortlist2 as S2
import scratchnet_common as C
from scratchnet_common import ROOT

OUT = ROOT / "data/benchmarks/rung4"
DOCS = ROOT / "docs/rung4"
CYCLE = "cycle6"
PREVIOUS = "cycle3"                         # the model that ranked shortlist 2
LITERATURE = [OUT / "literature-check.csv", OUT / "literature-check2.csv", OUT / "second-read.csv"]
PRECHECKED = {"Re3Ir", "NaMgPb2", "LiMgPb2", "SbTe2", "Cs(PIr)2"}     # pre-checked for Hazel (docs/hazel-candidates.md)
SOLID_SOLUTION_SYSTEMS = {"Ir-Re", "Re-Rh", "Os-Ru", "Os-Re"}         # readers found alloys, not compounds
FIELDS = [*S2.FIELDS, "rank_after_filters", "p_superconductor", "tc_if_superconductor_k", "rank_under_previous",
          "champion_percentile"]


def models(cycle: str) -> list:
    import joblib

    paths = sorted((ROOT / f"data/processed/learning-cycles/models/{cycle}").glob("boosting-fold*.joblib"))
    if len(paths) != 5:
        raise SystemExit(f"expected five final models for {cycle}, found {len(paths)}")
    return [joblib.load(path) for path in paths]


def score(rows: list[dict], cycle: str) -> tuple[np.ndarray, np.ndarray]:
    X, _ = C.formula_features([C.formula_text(R.fractions(r["formula"])) for r in rows])
    applied = [C.boosting_apply(model, X) for model in models(cycle)]
    return np.mean([a[0] for a in applied], axis=0), C.log_mean_tc([a[1] for a in applied])


def order(rows: list[dict], p: np.ndarray, tc: np.ndarray) -> list[int]:
    return sorted(range(len(rows)), key=lambda i: (-round(float(p[i]), 3), -round(float(p[i]), 3) * round(float(tc[i]), 2)))


def checked() -> tuple[set[str], dict[str, str]]:
    ids, verdicts = set(), {}
    for path in LITERATURE:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                ids.add(row["mp_id"])
                verdicts[row["mp_id"]] = row.get("verdict", "")
    return ids, verdicts


def main() -> int:
    rows = S2.read_filtered()
    p6, tc6 = score(rows, CYCLE)
    p3, tc3 = score(rows, PREVIOUS)
    rank6 = {i: k + 1 for k, i in enumerate(order(rows, p6, tc6))}
    rank3 = {i: k + 1 for k, i in enumerate(order(rows, p3, tc3))}
    for i, row in enumerate(rows):
        row.update(p_superconductor=round(float(p6[i]), 3), tc_if_superconductor_k=round(float(tc6[i]), 2),
                   rank_after_filters=rank6[i], rank_under_previous=rank3[i])
    opinion_path = OUT / "shortlist2-second-opinion.csv"
    if opinion_path.is_file():                 # the generation 7 champion's view; a column only
        with open(opinion_path, newline="", encoding="utf-8") as handle:
            opinion = {r["mp_id"]: r["champion_percentile"] for r in csv.DictReader(handle)}
        for row in rows:
            row["champion_percentile"] = opinion.get(row["mp_id"], "")

    ids, verdicts = checked()
    steps = {"filtered_pool": len(rows)}
    previously = sorted((r for r in rows if r["mp_id"] in ids or r["formula"] in PRECHECKED),
                        key=lambda r: r["rank_after_filters"])
    keep = [r for r in rows if r["mp_id"] not in ids and r["formula"] not in PRECHECKED]
    steps["8_verdicts_carried_forward"] = len(keep)
    keep = [r for r in keep if r["element_system"] not in SOLID_SOLUTION_SYSTEMS]
    steps["9_solid_solution_systems_dropped"] = len(keep)
    keep.sort(key=lambda r: r["rank_after_filters"])
    shortlist = S2.capped_head(keep)

    S2.write(OUT / "shortlist3.csv", shortlist, FIELDS)
    S2.write(OUT / "shortlist3-ranked.csv", sorted(rows, key=lambda r: r["rank_after_filters"]), FIELDS)
    from scipy.stats import spearmanr
    rho = float(spearmanr([rank3[i] for i in range(len(rows))], [rank6[i] for i in range(len(rows))]).correlation)
    watched = {f: next((r["rank_after_filters"] for r in rows if r["formula"] == f), None)
               for f in ("NaMgPb2", "LiMgPb2", "SbTe2", "Cs(PIr)2", "Re3Ir", "Ti2NbAl")}
    summary = {"model": CYCLE, "steps": steps, "shortlist": len(shortlist),
               "rank_correlation_with_previous": round(rho, 4),
               "shortlist_p_range": [shortlist[-1]["p_superconductor"], shortlist[0]["p_superconductor"]],
               "watched_ranks_among_filtered": watched,
               "previously_checked_in_cycle6_top20": [
                   {"formula": r["formula"], "rank": r["rank_after_filters"], "verdict": verdicts.get(r["mp_id"], "pre-checked")}
                   for r in previously if r["rank_after_filters"] <= 20]}
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "shortlist3-summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for row in shortlist:
        print(row["rank_after_filters"], row["formula"], row["prototype"], row["num_sites"], row["p_superconductor"],
              row["tc_if_superconductor_k"], row["formation_energy_ev_per_atom"], "prev", row["rank_under_previous"],
              "champ", row.get("champion_percentile"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
