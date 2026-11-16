#!/usr/bin/env python3
"""Rung 4, shortlist 2: filters first, ranking second (``docs/rung4-shortlist2.md``).

    python scripts/rung4_shortlist2.py filters     # steps 1 to 6 on the frozen pool, with the count after each
    python scripts/rung4_shortlist2.py validate    # the force-field screen on 60 made compounds
    python scripts/rung4_shortlist2.py screen      # step 7 on the survivors (a filter only if the validation passed)
    python scripts/rung4_shortlist2.py shortlist   # rank what is left with the ranking of record; cap; write the top 10

Needs the frozen pool (``python scripts/rung4.py pool``), the Materials Project structures under ``data/raw/`` and,
for ``validate`` and ``screen``, ``pip install chgnet``. Cycle 3's final models are needed for ``shortlist``.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

import rung4 as R
import scratchnet_common as C
from scratchnet_common import ROOT

OUT = ROOT / "data/benchmarks/rung4"
DOCS = ROOT / "docs/rung4"
STRUCTURES = ROOT / "data/raw/mp-structures/structures.jsonl.gz"
PP_LIST = ROOT / "data/benchmarks/candidates/pp-available.txt"
CACHE = ROOT / "data/processed/rung4/forcefield-screen.jsonl"
RADIOACTIVE = {"Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu"}
OPEN_4F = {"Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb"}
TOXIC = {"Be", "Cd", "Hg", "Tl", "As", "Os"}
FORMATION_AT_MOST = -0.10            # eV per atom
ATOMS_AT_MOST = 12
VALIDATION_SIZE, VALIDATION_SEED, VALIDATION_PASS = 60, 20261116, 0.20
SHORTLIST, PER_SYSTEM, PER_PROTOTYPE = 10, 2, 2
FIELDS = ["mp_id", "formula", "element_system", "prototype", "known_superconductors_in_prototype",
          "spacegroup_number", "num_sites", "formation_energy_ev_per_atom"]


def available_pseudopotentials() -> set[str]:
    lines = [line for line in PP_LIST.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    return set(" ".join(lines).split())


def element_steps(rows: list[dict], pseudopotentials: set[str]) -> tuple[list[dict], dict]:
    """Steps 1 to 6 in the plan's order. Each row needs on_hull, elements (a set), formation energy and num_sites."""
    steps = {"pool": len(rows)}
    keep = [r for r in rows if r["on_hull"]]
    steps["1_on_hull"] = len(keep)
    keep = [r for r in keep if not r["elements"] & (RADIOACTIVE | OPEN_4F)]
    steps["2_not_radioactive_no_open_4f"] = len(keep)
    keep = [r for r in keep if not r["elements"] & TOXIC]
    steps["3_no_toxic_element"] = len(keep)
    keep = [r for r in keep if r["elements"] <= pseudopotentials]
    steps["4_pseudopotentials_available"] = len(keep)
    keep = [r for r in keep if r["formation"] <= FORMATION_AT_MOST]
    steps["5_formation_energy"] = len(keep)
    keep = [r for r in keep if r["num_sites"] <= ATOMS_AT_MOST]
    steps["6_cell_size"] = len(keep)
    return keep, steps


def enrich(pool: list[dict], candidates: dict[str, dict]) -> list[dict]:
    rows = []
    for row in pool:
        source = candidates[row["mp_id"]]
        rows.append({**row, "on_hull": row["on_hull"] == "True", "elements": set(R.fractions(row["formula"])),
                     "formation": float(source["formation_energy_ev_per_atom"]), "num_sites": int(source["num_sites"]),
                     "formation_energy_ev_per_atom": round(float(source["formation_energy_ev_per_atom"]), 4)})
    return rows


def read_pool() -> list[dict]:
    with open(OUT / "pool.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cmd_filters() -> dict:
    candidates = {r["mp_id"]: r for r in R.load_candidates()}
    keep, steps = element_steps(enrich(read_pool(), candidates), available_pseudopotentials())
    write(OUT / "shortlist2-filtered.csv", sorted(keep, key=lambda r: r["mp_id"]), FIELDS)
    summary = {"steps": steps, "prototypes_left": len({r["prototype"] for r in keep}),
               "element_systems_left": len({r["element_system"] for r in keep}),
               "most_common_elements": Counter(e for r in keep for e in r["elements"]).most_common(12)}
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "shortlist2-filters.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return summary


# ------------------------------------------------------------------------------------------------ force-field screen

def structures(wanted: set[str]) -> dict:
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    out = {}
    with gzip.open(STRUCTURES, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["mp_id"] in wanted:
                structure = Structure.from_dict(row["structure"])
                refined = SpacegroupAnalyzer(structure, symprec=1e-2).get_primitive_standard_structure()
                out[row["mp_id"]] = refined if len(refined) == len(structure) else structure
    return out


def run_screen(rows: list[dict]) -> list[dict]:
    import forcefield_screen as F

    cells, out = structures({r["mp_id"] for r in rows}), []
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if CACHE.is_file():                                     # a stopped run picks up where it left off
        done = {r["mp_id"]: r["result"] for r in map(json.loads, CACHE.read_text(encoding="utf-8").splitlines())}
    for number, row in enumerate(rows, start=1):
        began = time.time()
        if row["mp_id"] in done:
            result = done[row["mp_id"]]
        else:
            try:
                result = F.screen(cells[row["mp_id"]])
            except Exception as error:  # noqa: BLE001 - one bad structure must not stop the screen
                result = {"error": f"{type(error).__name__}: {error}"[:200]}
            result["seconds"] = round(time.time() - began, 1)
            with open(CACHE, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"mp_id": row["mp_id"], "result": result}) + "\n")
        out.append({**row, **result})
        print(number, len(rows), row["formula"], result.get("lowest_cm1"), result.get("unstable"), flush=True)
    return out


SCREEN_FIELDS = ["lowest_cm1", "at", "lowest_at_zone_centre_cm1", "unstable_points", "points", "unstable",
                 "volume_change_percent", "seconds", "error"]


def made_controls(candidates: list[dict], prototypes: set[str], pseudopotentials: set[str]) -> list[dict]:
    """Made compounds like the pool's: metals, non-magnetic, on the hull, small, same prototypes, same element rules."""
    rows = []
    for r in candidates:
        if r["theoretical"] != "False" or not r["icsd_ids"] or r["is_metal"] != "True" or r["is_magnetic"] != "False":
            continue
        fractions = R.fractions(r["formula"])
        key = R.prototype(r["formula"], r["spacegroup_number"])
        if not fractions or key is None or R.prototype_label(key) not in prototypes:
            continue
        elements = set(fractions)
        if (R.hull(r) > 1e-9 or int(r["num_sites"]) > ATOMS_AT_MOST or elements & (RADIOACTIVE | OPEN_4F | TOXIC)
                or not elements <= pseudopotentials):
            continue
        rows.append({"mp_id": r["mp_id"], "formula": r["formula"], "element_system": "-".join(sorted(elements)),
                     "prototype": R.prototype_label(key), "num_sites": int(r["num_sites"]),
                     "formation_energy_ev_per_atom": round(float(r["formation_energy_ev_per_atom"]), 4)})
    return sorted(rows, key=lambda r: r["mp_id"])


def cmd_validate() -> dict:
    prototypes = {r["prototype"] for r in read_pool()}
    eligible = made_controls(R.load_candidates(), prototypes, available_pseudopotentials())
    rng = np.random.default_rng(VALIDATION_SEED)
    chosen = [eligible[i] for i in sorted(rng.choice(len(eligible), size=VALIDATION_SIZE, replace=False))]
    results = run_screen(chosen)
    write(OUT / "screen-validation.csv", results, ["mp_id", "formula", "prototype", "num_sites",
                                                   "formation_energy_ev_per_atom", *SCREEN_FIELDS])
    done = [r for r in results if "error" not in r]
    flagged = sum(r["unstable"] for r in done)
    summary = {"eligible_made_compounds": len(eligible), "screened": len(done), "errors": len(results) - len(done),
               "flagged_unstable": int(flagged), "flagged_fraction": round(flagged / len(done), 3),
               "pass_if_at_most": VALIDATION_PASS, "passes": bool(flagged / len(done) <= VALIDATION_PASS),
               "lowest_cm1_quartiles": [round(float(x), 1) for x in np.percentile([r["lowest_cm1"] for r in done],
                                                                                 [25, 50, 75])],
               "flagged_at_zone_centre": int(sum(r["lowest_at_zone_centre_cm1"] is not None
                                                 and r["lowest_at_zone_centre_cm1"] < -20 for r in done))}
    (DOCS / "shortlist2-screen-validation.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return summary


def cmd_screen() -> dict:
    with open(OUT / "shortlist2-filtered.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validation = json.loads((DOCS / "shortlist2-screen-validation.json").read_text(encoding="utf-8"))
    results = run_screen(rows)
    write(OUT / "shortlist2-screened.csv", results, [*FIELDS, *SCREEN_FIELDS])
    done = [r for r in results if "error" not in r]
    summary = {"screened": len(done), "errors": len(results) - len(done),
               "flagged_unstable": int(sum(r["unstable"] for r in done)),
               "used_as_filter": validation["passes"]}
    (DOCS / "shortlist2-screen.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return summary


# ------------------------------------------------------------------------------------------------ ranking

def capped_head(rows: list[dict], size: int = SHORTLIST) -> list[dict]:
    systems, prototypes, kept = Counter(), Counter(), []
    for row in rows:
        if systems[row["element_system"]] >= PER_SYSTEM or prototypes[row["prototype"]] >= PER_PROTOTYPE:
            continue
        systems[row["element_system"]] += 1
        prototypes[row["prototype"]] += 1
        kept.append(row)
        if len(kept) == size:
            break
    return kept


def cmd_shortlist() -> dict:
    import joblib

    screen = json.loads((DOCS / "shortlist2-screen.json").read_text(encoding="utf-8"))
    with open(OUT / "shortlist2-screened.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if screen["used_as_filter"]:
        rows = [r for r in rows if r["unstable"] == "False"]
    X, _ = C.formula_features([C.formula_text(R.fractions(r["formula"])) for r in rows])
    applied = [C.boosting_apply(joblib.load(path), X) for path in sorted(R.MODELS.glob("boosting-fold*.joblib"))]
    if len(applied) != 5:
        raise SystemExit(f"expected cycle 3's five final models under {R.MODELS}")
    p, tc = np.mean([a[0] for a in applied], axis=0), C.log_mean_tc([a[1] for a in applied])
    for row, p_i, tc_i in zip(rows, p, tc):
        row.update(p_superconductor=round(float(p_i), 3), tc_if_superconductor_k=round(float(tc_i), 2))
    order = sorted(rows, key=lambda r: (-r["p_superconductor"], -r["p_superconductor"] * r["tc_if_superconductor_k"]))
    for rank, row in enumerate(order, start=1):
        row["rank_after_filters"] = rank
    shortlist = capped_head(order)
    fields = [*FIELDS, "rank_after_filters", "p_superconductor", "tc_if_superconductor_k", "lowest_cm1", "at", "unstable"]
    write(OUT / "shortlist2.csv", shortlist, fields)
    write(OUT / "shortlist2-ranked.csv", order, fields)
    summary = {"ranked": len(order), "screen_used_as_filter": screen["used_as_filter"], "shortlist": len(shortlist),
               "shortlist_prototypes": len({r["prototype"] for r in shortlist}),
               "shortlist_p_range": [shortlist[-1]["p_superconductor"], shortlist[0]["p_superconductor"]],
               "ranked_median_p": float(np.median(p)),
               "shortlist_flagged_by_screen": int(sum(r["unstable"] == "True" for r in shortlist))}
    (DOCS / "shortlist2-summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for row in shortlist:
        print(row["rank_after_filters"], row["formula"], row["prototype"], row["p_superconductor"],
              row["tc_if_superconductor_k"], row["formation_energy_ev_per_atom"], row["lowest_cm1"], row["unstable"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("filters", "validate", "screen", "shortlist"))
    {"filters": cmd_filters, "validate": cmd_validate, "screen": cmd_screen,
     "shortlist": cmd_shortlist}[parser.parse_args().command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
