#!/usr/bin/env python3
"""The ranking of record from 2026-11-27: the 50/50 blend of cycle 6 and TabICLv2 (``docs/lockbox-blend.md``).

    python scripts/blend_ranking.py [--tabicl fm-tabicl-20261123]

Re-ranks the made metals and shortlist 3's pool with the blend:
* cycle 6's part comes from its five saved final models, applied here;
* TabICLv2's part is the candidate predictions the owner's laptop made (mean of its five fold models).
Probabilities are averaged directly and Tc on the log scale, as in every blend comparison. The made metals are ordered
exactly as each learning cycle orders them (``run_learning_cycle.made_metals_order``).

Writes ``docs/learning-cycles/blend6-made-metals-top50.csv``, ``docs/learning-cycles/blend6.json`` and
``data/benchmarks/rung4/shortlist3-blend-ranked.csv``.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json

import numpy as np

import local_gpu as G
import lockbox_blend as LB
import run_learning_cycle as L
import scratchnet_common as C
from scratchnet_common import ROOT

NAME = "blend6"
SHORTLIST_OUT = ROOT / "data/benchmarks/rung4/shortlist3-blend-ranked.csv"
WATCH_SHORTLIST = ("SbTe2", "NaMgPb2", "LiMgPb2", "LiGaIr", "SrAl3Ag", "Cs(PIr)2")


def tabicl_predictions(folder, name: str) -> dict[str, tuple[float, float]]:
    with np.load(folder / f"candidates_{name}.npz", allow_pickle=False) as saved:
        return {str(k): (float(p), float(t)) for k, p, t in zip(saved["keys"], saved["p"], saved["tc"])}


def blended(keys, record: tuple[np.ndarray, np.ndarray], tabicl: dict) -> tuple[np.ndarray, np.ndarray]:
    if set(map(str, keys)) != set(tabicl) or len(tabicl) != len(keys):
        raise SystemExit("TabICLv2's candidate predictions do not cover the same compounds as the bundle")
    tab = (np.array([tabicl[str(k)][0] for k in keys]), np.array([tabicl[str(k)][1] for k in keys]))
    return C.blend(tab, record)


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    out = np.empty(len(values), int)
    out[order] = np.arange(1, len(values) + 1)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tabicl", default="fm-tabicl-20261123", help="the TabICLv2 run whose candidate predictions to use")
    args = parser.parse_args(argv)
    folder = G.result_folder(args.tabicl)
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    manifest = json.loads((G.WORK / "bundle.json").read_text(encoding="utf-8"))
    if not G.same_bundle(run, manifest) or run.get("model") != "tabicl":
        raise SystemExit("that run is not TabICLv2 on the bundle exported here")
    ds = C.load_dataset()
    with np.load(G.WORK / "bundle.npz", allow_pickle=False) as bundle:
        LB.check_record_models(bundle)
        mm_keys, mm_X = bundle["candidates_made_metals_keys"], bundle["candidates_made_metals_X"]
        sl_keys, sl_X = bundle["candidates_shortlist_pool_keys"], bundle["candidates_shortlist_pool_X"]

    # made metals, ordered as every learning cycle orders them
    rows, compositions = L.candidate_pool(L.CANDIDATES, set(ds.table.symbols))
    if [r["formula"] for r in rows] != [str(k) for k in mm_keys]:
        raise SystemExit("the candidate pool has changed since the bundle was exported")
    record_mm = LB.record_predict(mm_X)
    order_record = L.made_metals_order(rows, compositions, *record_mm)
    with gzip.open(L.WORK / "cycle6-made-metals.json.gz", "rt", encoding="utf-8") as handle:
        saved = json.load(handle)
    if [r["formula"] for r in saved[:200]] != [r["formula"] for r in order_record[:200]]:
        raise SystemExit("cycle 6's saved models no longer reproduce cycle 6's made-metals order")
    p_mm, tc_mm = blended(mm_keys, record_mm, tabicl_predictions(folder, "made_metals"))
    order = L.made_metals_order(rows, compositions, p_mm, tc_mm)
    fields = ["rank", "formula", "mp_id", "element_system", "p_superconductor", "tc_if_superconductor_k", "expected_tc_k"]
    C.write_table(L.OUT / f"{NAME}-made-metals-top50.csv", order[:50], fields)

    # shortlist 3's pool, with both ranks side by side
    with open(LB.SHORTLIST, newline="", encoding="utf-8") as handle:
        shortlist = {row["mp_id"]: row for row in csv.DictReader(handle)}
    record_sl = LB.record_predict(sl_X)
    tab_sl = tabicl_predictions(folder, "shortlist_pool")
    p_sl, tc_sl = blended(sl_keys, record_sl, tab_sl)
    r_blend, r_record = ranks(p_sl), ranks(record_sl[0])
    table = []
    for i, key in enumerate(map(str, sl_keys)):
        row = shortlist[key]
        table.append({"rank_blend": int(r_blend[i]), "rank_cycle6": int(r_record[i]), "mp_id": key,
                      "formula": row["formula"], "element_system": row["element_system"],
                      "p_blend": round(float(p_sl[i]), 3), "p_cycle6": round(float(record_sl[0][i]), 3),
                      "p_tabicl": round(tab_sl[key][0], 3), "tc_blend_k": round(float(tc_sl[i]), 2)})
    table.sort(key=lambda r: r["rank_blend"])
    C.write_table(SHORTLIST_OUT, table, list(table[0]))

    by_formula = {r["formula"]: r for r in table}
    report = {"ranking_of_record": "50/50 blend of cycle 6 and TabICLv2, from 2026-11-27 (docs/lockbox-blend.md)",
              "tabicl_run": folder.name, "tabicl_libraries": run.get("libraries"),
              "made_metals": {"untested_made_metals": len(order),
                              "top_10": [{k: r[k] for k in ("rank", "formula", "p_superconductor", "tc_if_superconductor_k")}
                                         for r in order[:10]],
                              "against_cycle6": L.compare_orders(order_record, order)},
              "shortlist3": {"pool": len(table),
                             "top_10": [{k: r[k] for k in ("rank_blend", "rank_cycle6", "formula", "p_blend")} for r in table[:10]],
                             "cycle6_top_50_kept": int(sum(r["rank_cycle6"] <= 50 for r in table[:50])),
                             "watched": {f: {k: by_formula[f][k] for k in ("rank_blend", "rank_cycle6", "p_blend")}
                                         for f in WATCH_SHORTLIST if f in by_formula}}}
    (L.OUT / f"{NAME}.json").write_text(json.dumps(L.rounded(report), indent=1) + "\n", encoding="utf-8")
    print(json.dumps(L.rounded({k: report[k] for k in ("made_metals", "shortlist3")}), indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
