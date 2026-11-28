#!/usr/bin/env python3
"""Read the overnight electron-phonon run (plan: docs/eph-surrogate.md). Reads outputs; trains nothing, scores nothing new.

    python scripts/eph_read.py [data/benchmarks/local-gpu/eph-20261128]

The test rows were scored once, by eph_gpu.py, and are in result.json. Here:
  * the plan's rule is applied to those numbers;
  * group-bootstrap intervals and a spread breakdown are added (reported only, they change no decision);
  * candidates are marked and listed. Flags, fixed before the candidate tables were opened: more sites than any
    training cell, or the rarest element in fewer than 20 training cells, or an ensemble spread above the 90th
    percentile of the test rows' spreads. Two more were added after the first list showed insulators at its top (the
    "made metals" list is compounds containing a metal, not metallic compounds): not metallic in Materials Project,
    and magnetic in Materials Project. A candidate whose composition is in Alexandria gets Alexandria's value.
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_mp_candidates as M  # noqa: E402
import build_scratchnet_data as B  # noqa: E402
import eph_gpu as E  # noqa: E402

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/benchmarks/local-gpu/eph-20261128"
EPH = ROOT / "data/processed/alexandria-eph-2025/table.csv.gz"
MP = ROOT / "data/candidates/mp-candidates.csv.gz"
POOL = ROOT / "data/benchmarks/rung4/shortlist3-ranked.csv"
OUT_JSON, OUT_CSV = ROOT / "docs/eph-surrogate.json", ROOT / "docs/eph-surrogate-candidates.csv"
MIN_ELEMENT_CELLS, SPREAD_QUANTILE = 20, 0.90
QUEUE = ["LiGaIr", "SbTe2", "CsIr2P2", "NaMgPb2", "Nb2GeC", "ScTaN2", "V2GaN", "LiRuN", "TaNbRe2", "SrAl3Ag", "Re3Ir"]


def element_set(composition: str) -> str:
    return ",".join(sorted(p.split(":")[0] for p in composition.split(";")))


def tc_k(ln1p: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(ln1p), 0.0, None)


def main() -> int:
    result = json.loads((RUN / "result.json").read_text())
    alex = {}
    for r in csv.DictReader(gzip.open(EPH, "rt", newline="", encoding="utf-8")):
        alex[r["mat_id"]] = r
    by_comp: dict[str, list[dict]] = {}
    for r in alex.values():
        by_comp.setdefault(r["composition"], []).append(r)

    def alexandria_value(comp: str):
        """(Tc K, status) from Alexandria for a composition: the lowest-Tc stable entry is not taken; the first
        stable, unflagged one is. Unstable-only compositions say so."""
        entries = by_comp.get(comp, [])
        good = [e for e in entries if e["imag"] == "False" and e["tc_allen_dynes_mu010"] and e["lambda"]
                and float(e["lambda"]) <= 5 and float(e["wlog_k"]) >= 100]
        if good:
            return float(good[0]["tc_allen_dynes_mu010"]), "calculated"
        if entries:
            return None, "imaginary mode" if all(e["imag"] == "True" for e in entries) else "flagged"
        return None, ""

    # --- the plan's rule on the one scoring
    t = result["test"]
    rule = {"auc_tc_at_least_10k": t["auc_tc_at_least_10k"], "needs_at_least": 0.85,
            "tc_mae_k": t["tc_mae_k"], "baseline_mae_k": t["tc_mae_k_fit_median_baseline"],
            "needs_at_most": round(0.7 * t["tc_mae_k_fit_median_baseline"], 3)}
    rule["ensemble_used"] = bool(rule["auc_tc_at_least_10k"] >= 0.85 and rule["tc_mae_k"] <= rule["needs_at_most"])

    # --- reported only: group bootstrap and spread breakdown on the test rows
    tp = np.load(RUN / "test_predictions.npz", allow_pickle=False)
    true_tc, pred_tc = tc_k(tp["true"][:, 2]), tc_k(tp["pred"][:, 2])
    groups = np.array([element_set(alex[i]["composition"]) for i in tp["ids"]])
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(20261128)
    aucs, ratios = [], []
    base = result["test"]["tc_mae_k_fit_median_baseline"]
    for _ in range(1000):
        idx = np.concatenate([members[g] for g in uniq[rng.integers(len(uniq), size=len(uniq))]])
        aucs.append(E.auc((true_tc[idx] >= 10).astype(float), pred_tc[idx]))
        ratios.append(np.mean(np.abs(pred_tc[idx] - true_tc[idx])) / np.mean(np.abs(np.median(true_tc) - true_tc[idx])))
    q = lambda a: [round(float(np.nanpercentile(a, p)), 4) for p in (2.5, 50, 97.5)]  # noqa: E731
    spread = tp["ln1p_tc_spread"]
    cuts = np.quantile(spread, [1 / 3, 2 / 3])
    thirds = {}
    for name, sel in (("low spread", spread <= cuts[0]), ("middle", (spread > cuts[0]) & (spread <= cuts[1])),
                      ("high spread", spread > cuts[1])):
        thirds[name] = {"rows": int(sel.sum()), "auc_tc_at_least_10k": round(E.auc((true_tc[sel] >= 10).astype(float), pred_tc[sel]), 4),
                        "tc_mae_k": round(float(np.mean(np.abs(pred_tc[sel] - true_tc[sel]))), 3)}
    reported = {"element_set_groups_in_test": int(len(uniq)), "auc_tc_at_least_10k_interval": q(aucs),
                "mae_over_median_baseline_interval": q(ratios), "by_spread_third": thirds,
                "test_tc_at_least_77k": int((true_tc >= 77).sum()),
                "test_tc_at_least_77k_predicted_at_least_40k": int(((true_tc >= 77) & (pred_tc >= 40)).sum())}
    spread_cut = float(np.quantile(spread, SPREAD_QUANTILE))

    # --- candidates
    mp = {r["mp_id"]: r for r in csv.DictReader(gzip.open(MP, "rt", newline="", encoding="utf-8"))}
    pool = {r["mp_id"]: r for r in csv.DictReader(open(POOL, newline="", encoding="utf-8"))}
    out_rows, summary = [], {}
    for name in ("made_metals", "shortlist_pool"):
        c = np.load(RUN / f"candidates_{name}.npz", allow_pickle=False)
        tc = tc_k(c["pred"][:, 2])
        for k, mid in enumerate(c["keys"]):
            m = mp[str(mid)]
            fr = M.fractions(m["formula"])
            comp = B.composition_key(fr) if fr else ""
            flags = []
            if int(c["sites"][k]) > result["fit_max_sites"]:
                flags.append("larger than any training cell")
            if int(c["rarest_element_fit_graphs"][k]) < MIN_ELEMENT_CELLS:
                flags.append("rare element")
            if c["ln1p_tc_spread"][k] > spread_cut:
                flags.append("high spread")
            # Added after the first list showed 66 of its top 100 were insulators: the training rows were metals,
            # so the network has never seen an insulator and its output for one is meaningless.
            if m["is_metal"] != "True":
                flags.append("not metallic in Materials Project")
            if m["is_magnetic"] == "True":
                flags.append("magnetic in Materials Project")
            a_tc, a_status = alexandria_value(comp)
            p = pool.get(str(mid), {})
            out_rows.append({"list": name, "mp_id": str(mid), "formula": m["formula"], "composition": comp,
                             "net_lambda": round(float(c["pred"][k, 0]), 3), "net_wlog_k": round(float(np.exp(c["pred"][k, 1])), 1),
                             "net_tc_k": round(float(tc[k]), 2), "net_tc_spread_ln": round(float(c["ln1p_tc_spread"][k]), 3),
                             "sites": int(c["sites"][k]), "flags": "; ".join(flags),
                             "alexandria": "" if a_tc is None else round(a_tc, 2), "alexandria_status": a_status,
                             "alexandria_split": E.split_of(comp) if a_status else "",
                             "e_above_hull": m["energy_above_hull_ev_per_atom"], "known_match": m["known_match"],
                             "icsd": "yes" if m["icsd_ids"] else "no", "pool_rank": p.get("rank_after_filters", ""),
                             "p_superconductor": p.get("p_superconductor", ""), "ranking_tc_k": p.get("tc_if_superconductor_k", "")})
    made = [r for r in out_rows if r["list"] == "made_metals"]
    pool_rows = [r for r in out_rows if r["list"] == "shortlist_pool"]

    # transfer check: candidates whose composition is in Alexandria's validation or test element systems
    held = [r for r in out_rows if r["alexandria"] != "" and r["alexandria_split"] in ("valid", "test")]
    seen, uniq_held = set(), []
    for r in held:
        if r["composition"] not in seen:
            seen.add(r["composition"]); uniq_held.append(r)
    transfer = {"compositions": len(uniq_held)}
    if len(uniq_held) >= 20:
        a = np.array([float(r["alexandria"]) for r in uniq_held]); n_ = np.array([r["net_tc_k"] for r in uniq_held])
        transfer.update({"spearman": round(float(np.corrcoef(E.ranks(a), E.ranks(n_))[0, 1]), 3),
                         "mae_k": round(float(np.mean(np.abs(a - n_))), 3), "mae_k_median_baseline": round(float(np.mean(np.abs(a - np.median(a)))), 3),
                         "alexandria_tc_at_least_10k": int((a >= 10).sum()),
                         "auc_tc_at_least_10k": round(E.auc((a >= 10).astype(float), n_), 3) if 0 < (a >= 10).sum() < len(a) else None})
    new_made = [r for r in made if r["alexandria_status"] == "" and r["known_match"] == "" and not r["flags"]]
    new_made.sort(key=lambda r: -r["net_tc_k"])
    queue = [r for r in pool_rows if r["formula"] in QUEUE]
    summary = {"plan": "docs/eph-surrogate.md", "run": str(RUN.relative_to(ROOT)), "rule": rule, "reported_only": reported,
               "flag_rule": {"max_sites": result["fit_max_sites"], "min_element_cells": MIN_ELEMENT_CELLS,
                             "spread_cut_ln1p": round(spread_cut, 4), "spread_quantile": SPREAD_QUANTILE},
               "candidates": {"made_metals": len(made), "flagged": sum(1 for r in made if r["flags"]),
                              "not_metallic": sum(1 for r in made if "not metallic" in r["flags"]),
                              "magnetic": sum(1 for r in made if "magnetic" in r["flags"]),
                              "flagged_by_pre_set_rules": sum(1 for r in made if any(f in r["flags"] for f in ("larger", "rare", "high spread"))),
                              "already_in_alexandria": sum(1 for r in made if r["alexandria_status"]),
                              "known_superconductors": sum(1 for r in made if r["known_match"]),
                              "new_unflagged": len(new_made), "new_unflagged_net_tc_at_least_10k": sum(1 for r in new_made if r["net_tc_k"] >= 10),
                              "new_unflagged_net_tc_at_least_20k": sum(1 for r in new_made if r["net_tc_k"] >= 20),
                              "new_unflagged_net_tc_at_least_40k": sum(1 for r in new_made if r["net_tc_k"] >= 40),
                              "new_unflagged_max_net_tc_k": new_made[0]["net_tc_k"] if new_made else None},
               "transfer_check": transfer}
    cols = list(out_rows[0])
    keep = new_made[:100] + sorted(pool_rows, key=lambda r: -r["net_tc_k"])
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=["table"] + cols, lineterminator="\n")
        w.writeheader()
        for i, r in enumerate(keep):
            w.writerow({"table": "top 100 new made metals" if i < len(new_made[:100]) else "shortlist 3 pool", **r})
    OUT_JSON.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("\nQUEUE COMPOUNDS (network vs ranking of record)")
    for r in sorted(queue, key=lambda r: -r["net_tc_k"]):
        print(f"  {r['formula']:10s} {r['mp_id']:13s} net Tc {r['net_tc_k']:6.2f} K  lam {r['net_lambda']:.2f}  wlog {r['net_wlog_k']:6.1f}  "
              f"spread {r['net_tc_spread_ln']:.2f}  sites {r['sites']:3d}  flags [{r['flags']}]  pool rank {r['pool_rank']}  alex {r['alexandria']} {r['alexandria_status']}")
    print("\nTOP 25 NEW, UNFLAGGED MADE METALS (not in Alexandria, no known-superconductor match)")
    for r in new_made[:25]:
        print(f"  {r['formula']:12s} {r['mp_id']:13s} net Tc {r['net_tc_k']:6.2f} K  lam {r['net_lambda']:.2f}  wlog {r['net_wlog_k']:6.1f}  "
              f"spread {r['net_tc_spread_ln']:.2f}  sites {r['sites']:3d}  hull {float(r['e_above_hull']):.3f}  icsd {r['icsd']}")
    print("\nTOP 12 SHORTLIST-3 POOL BY NETWORK Tc")
    for r in sorted(pool_rows, key=lambda r: -r["net_tc_k"])[:12]:
        print(f"  {r['formula']:12s} net Tc {r['net_tc_k']:6.2f} K  spread {r['net_tc_spread_ln']:.2f}  flags [{r['flags']}]  pool rank {r['pool_rank']}  alex {r['alexandria']} {r['alexandria_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
