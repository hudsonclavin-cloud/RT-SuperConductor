#!/usr/bin/env python3
"""Crystal-graph bundle for the owner's GPU, and the judge of what comes back (``docs/overnight-gpu.md``).

    python scripts/structure_gpu.py export            # data/processed/local-gpu/structures.npz (+ structures.json)
    python scripts/structure_gpu.py compare <folder>  # every run of an overnight results folder, paired against cycle 6

What the bundle holds:

* **Training graphs:** one crystal graph per 3DSC (Materials Project) structure whose SuperCon composition is a
  development or train-only row of the table. The labels are the table's (cycle 6, with its corrections), not
  3DSC's. A composition with several candidate structures has several graphs, each weighted 1 / (number of graphs).
* **Folds:** cycle 6's own fit, early-stopping and test rows (``run_scratchnet.fold_rows``), restricted to the
  compositions that have a structure.
* **Pretraining graphs:** the 30,358 Materials Project structures of the candidate snapshot, with five calculated
  targets (formation energy, band gap, energy above the hull, metal or not, magnetic or not). There are no
  superconductivity labels. A structure is left out if its composition is a locked composition, or the parent of a
  locked 3DSC structure.
* **Candidates:** the made-metals pool and shortlist 3's 571 compounds, as graphs of their own structures.

Locked rows and their structures are not in the bundle. The export stops if one reaches it.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

import local_gpu as G
import local_gpu_train as TR
import run_learning_cycle as L
import scratchnet_common as C
from scratchnet_common import ROOT, T
from run_scratchnet import fold_rows
from rt_superconductor.scratchnet import data as D

THREE_D = ROOT / "data/raw/3dsc/repo/superconductors_3D"
THREE_D_TABLE = THREE_D / "data/final/MP/3DSC_MP.csv"
MP_STRUCTURES = ROOT / "data/raw/mp-structures/structures.jsonl.gz"
MP_TABLE = ROOT / "data/candidates/mp-candidates.csv.gz"
SHORTLIST_POOL = ROOT / "data/benchmarks/rung4/shortlist3-ranked.csv"
WORK = G.WORK
CUTOFF, MAX_NEIGHBORS, MAX_SPECIES = 8.0, 12, 4
PRETRAIN_TARGETS = ["formation_energy_ev_per_atom", "band_gap_ev", "energy_above_hull_ev_per_atom",
                    "is_metal", "is_magnetic"]


# ---------------------------------------------------------------------------
# Graphs

def site_species(site, index: dict[str, int]) -> tuple[list[int], list[float]]:
    """Up to MAX_SPECIES (element-table row, occupancy) pairs for one site, largest occupancy first."""
    pairs = sorted(((str(el.symbol), float(occ)) for el, occ in site.species.items()), key=lambda p: (-p[1], p[0]))
    pairs = [(index[el], occ) for el, occ in pairs if el in index][:MAX_SPECIES]
    if not pairs:
        raise KeyError(f"no element of {site.species} is in the element table")
    rows = [p[0] for p in pairs] + [0] * (MAX_SPECIES - len(pairs))
    occ = [p[1] for p in pairs] + [0.0] * (MAX_SPECIES - len(pairs))
    return rows, occ


def graph_of(structure, index: dict[str, int]) -> dict[str, np.ndarray]:
    """Nodes (species and occupancies) and the edges to each site's nearest MAX_NEIGHBORS within CUTOFF angstrom."""
    n = len(structure)
    if n > 32000:
        raise ValueError("too many sites for 16-bit local indices")
    species, occ = zip(*(site_species(site, index) for site in structure))
    centre, neighbour, _, distance = structure.get_neighbor_list(r=CUTOFF)
    src, dst, dist = [], [], []
    for i in range(n):
        mine = np.flatnonzero(centre == i)
        mine = mine[np.argsort(distance[mine], kind="stable")[:MAX_NEIGHBORS]]
        src.append(np.full(len(mine), i))
        dst.append(neighbour[mine])
        dist.append(distance[mine])
    src, dst, dist = np.concatenate(src), np.concatenate(dst), np.concatenate(dist)
    if len(src) == 0:
        raise ValueError("no neighbours within the cutoff")
    return {"species": np.array(species, np.int16), "occ": np.array(occ, np.float16),
            "src": src.astype(np.int16), "dst": dst.astype(np.int16), "dist": dist.astype(np.float16),
            "volume_per_atom": np.float32(structure.volume / n), "sites": np.int16(n)}


def stack_graphs(graphs: list[dict]) -> dict[str, np.ndarray]:
    """One store for every graph: concatenated nodes and edges with offsets; edge indices stay local to each graph."""
    nodes = np.cumsum([0] + [len(g["species"]) for g in graphs])
    edges = np.cumsum([0] + [len(g["src"]) for g in graphs])
    return {"node_offsets": nodes.astype(np.int64), "edge_offsets": edges.astype(np.int64),
            "node_species": np.concatenate([g["species"] for g in graphs]),
            "node_occ": np.concatenate([g["occ"] for g in graphs]),
            "edge_src": np.concatenate([g["src"] for g in graphs]),
            "edge_dst": np.concatenate([g["dst"] for g in graphs]),
            "edge_dist": np.concatenate([g["dist"] for g in graphs]),
            "graph_volume_per_atom": np.array([g["volume_per_atom"] for g in graphs], np.float32),
            "graph_sites": np.array([g["sites"] for g in graphs], np.int16)}


# ---------------------------------------------------------------------------
# Sources

def read_3dsc(path: Path = THREE_D_TABLE) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def key_of(formula: str) -> str | None:
    try:
        return D.composition_key(D.parse_formula(formula))
    except (ValueError, KeyError):
        return None


def load_3dsc_structure(row: dict):
    from pymatgen.io.cif import CifParser

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CifParser(str(THREE_D / row["cif"]), occupancy_tolerance=1.1).parse_structures(primitive=False)[0]


def read_mp() -> tuple[list[str], list[dict], dict[str, dict]]:
    with gzip.open(MP_TABLE, "rt", newline="", encoding="utf-8") as handle:
        table = {row["mp_id"]: row for row in csv.DictReader(handle)}
    ids, structures = [], []
    with gzip.open(MP_STRUCTURES, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            ids.append(record["mp_id"])
            structures.append(record["structure"])
    return ids, structures, table


def target_value(row: dict, name: str) -> float:
    value = row.get(name, "")
    if value in ("True", "False"):
        return 1.0 if value == "True" else 0.0
    try:
        return float(value)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Export

def build(ds: C.Dataset, X_table: np.ndarray, feature_names: list[str], three_d: list[dict],
          load_structure, mp_ids: list[str], mp_structures: list, mp_table: dict[str, dict],
          mp_X: np.ndarray, candidate_sets: dict[str, list[str]]) -> tuple[dict[str, np.ndarray], dict]:
    """All arrays of the bundle, and a report of what was left out and why. Pure apart from ``load_structure``."""
    from pymatgen.core import Structure

    index = ds.table.index
    table_row = {row["composition"]: i for i, row in enumerate(ds.rows)}
    locked_keys = {row["composition"] for row in ds.rows if row["split"] == "lockbox"}
    report = {"3dsc_rows": len(three_d), "3dsc_not_in_table": 0, "3dsc_locked": 0, "3dsc_failed": 0,
              "mp_failed": 0, "mp_left_out_of_pretraining_as_locked": 0}
    locked_parents = set()
    graphs, graph_comp_key = [], []
    for row in three_d:
        key = key_of(row["formula_sc"])
        if key is None or key not in table_row:
            report["3dsc_not_in_table"] += 1
            continue
        if ds.split[table_row[key]] == "lockbox":
            report["3dsc_locked"] += 1
            parent = key_of(row.get("orig_formula_cif", ""))
            if parent:
                locked_parents.add(parent)
            continue
        try:
            graphs.append(graph_of(load_structure(row), index))
        except (ValueError, KeyError, IndexError) as error:   # an unreadable CIF or an element outside the table
            report["3dsc_failed"] += 1
            report.setdefault("3dsc_failed_examples", []).append(f"{row['formula_sc']}: {error}"[:200])
            continue
        graph_comp_key.append(key)
    n_train_graphs = len(graphs)

    mp_graph = {}
    for mp_id, record in zip(mp_ids, mp_structures):
        try:
            structure = record if hasattr(record, "get_neighbor_list") else Structure.from_dict(record)
            graphs.append(graph_of(structure, index))
        except (ValueError, KeyError, IndexError):
            report["mp_failed"] += 1
            continue
        mp_graph[mp_id] = len(graphs) - 1
    store = stack_graphs(graphs)

    comp_keys = sorted(set(graph_comp_key))
    comp_index = {key: i for i, key in enumerate(comp_keys)}
    rows = np.array([table_row[key] for key in comp_keys])
    if (ds.split[rows] == "lockbox").any():
        raise SystemExit("a locked composition reached the structure bundle")
    per_comp = np.bincount([comp_index[k] for k in graph_comp_key], minlength=len(comp_keys))
    train_comp = np.array([comp_index[k] for k in graph_comp_key], np.int32)
    bundle = {**store,
              "train_graph": np.arange(n_train_graphs, dtype=np.int32), "train_comp": train_comp,
              "train_weight": (1.0 / per_comp[train_comp]).astype(np.float32),
              "comp_keys": np.array(comp_keys), "comp_label": ds.arrays.label[rows].astype(np.float32),
              "comp_tc_k": ds.arrays.tc_k[rows].astype(np.float32), "comp_split": np.array(ds.split[rows]),
              "comp_X": X_table[rows].astype(np.float32), "feature_names": np.array(feature_names),
              "elem_features": ds.table.features.astype(np.float32), "elem_symbols": np.array(["", *ds.table.symbols]),
              "cutoff_angstrom": np.float32(CUTOFF), "max_neighbors": np.int16(MAX_NEIGHBORS)}

    dev = ds.split == "dev"
    folds = sorted(set(ds.fold[dev].tolist()))
    for fold in folds:
        for part, table_rows in zip(("fit", "valid", "test"), fold_rows(ds, fold)):
            keys = [ds.rows[i]["composition"] for i in table_rows]
            bundle[f"fold{fold}_{part}"] = np.array(sorted(comp_index[k] for k in keys if k in comp_index), np.int32)
    bundle["folds"] = np.array(folds)

    pre_ids, pre_graph, pre_targets = [], [], []
    for mp_id, g in mp_graph.items():
        row = mp_table.get(mp_id)
        if row is None:
            continue
        key = key_of(row["formula"])
        if key in locked_keys or key in locked_parents:
            report["mp_left_out_of_pretraining_as_locked"] += 1
            continue
        pre_ids.append(mp_id)
        pre_graph.append(g)
        pre_targets.append([target_value(row, name) for name in PRETRAIN_TARGETS])
    bundle.update(pre_graph=np.array(pre_graph, np.int32), pre_mp_ids=np.array(pre_ids),
                  pre_targets=np.array(pre_targets, np.float32).reshape(-1, len(PRETRAIN_TARGETS)),
                  pre_target_names=np.array(PRETRAIN_TARGETS))

    mp_row = {mp_id: i for i, mp_id in enumerate(mp_ids)}
    for name, ids in candidate_sets.items():
        have = [i for i in ids if i in mp_graph]
        report[f"candidates_{name}_without_graph"] = len(ids) - len(have)
        bundle[f"cand_{name}_graph"] = np.array([mp_graph[i] for i in have], np.int32)
        bundle[f"cand_{name}_keys"] = np.array(have)
        bundle[f"cand_{name}_X"] = mp_X[[mp_row[i] for i in have]].astype(np.float32)
    report.update(training_graphs=n_train_graphs, compositions=len(comp_keys),
                  compositions_by_split={s: int((bundle["comp_split"] == s).sum()) for s in ("dev", "train_only")},
                  pretraining_graphs=len(pre_graph), graphs=len(graphs),
                  nodes=int(store["node_offsets"][-1]), edges=int(store["edge_offsets"][-1]))
    return bundle, report


def cmd_export() -> dict:
    ds = C.load_dataset()
    formulas = [C.formula_text(C.fractions_of(row["composition"])) for row in ds.rows]
    X_table, names = C.formula_features(formulas, cache=L.WORK / "formula-features.npz")
    mp_ids, mp_structures, mp_table = read_mp()
    mp_X, _ = C.formula_features([C.formula_text(D.parse_formula(mp_table[i]["formula"])) for i in mp_ids],
                                 cache=WORK / "mp-snapshot-features.npz")
    made_rows, _ = L.candidate_pool(L.CANDIDATES, set(ds.table.symbols))
    candidate_sets = {"made_metals": [r["mp_id"] for r in made_rows]}
    if SHORTLIST_POOL.is_file():
        with open(SHORTLIST_POOL, newline="", encoding="utf-8") as handle:
            candidate_sets["shortlist_pool"] = [r["mp_id"] for r in csv.DictReader(handle)]
    bundle, report = build(ds, X_table, names, read_3dsc(), load_3dsc_structure, mp_ids, mp_structures, mp_table,
                           mp_X, candidate_sets)
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "structures.npz"
    np.savez_compressed(path, **bundle)
    with np.load(path, allow_pickle=False) as saved:
        content = TR.content_sha256(saved)
    manifest = {"bundle": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "content_sha256": content, "megabytes": round(path.stat().st_size / 1e6, 1),
                "table_sha256": ds.sha256, "record": G.RECORD, "cutoff_angstrom": CUTOFF,
                "max_neighbors": MAX_NEIGHBORS, "locked_rows_in_bundle": 0, **report,
                "per_fold": {int(f): {p: int(len(bundle[f"fold{f}_{p}"])) for p in ("fit", "valid", "test")}
                             for f in bundle["folds"]},
                "candidates": {k[5:-6]: int(len(v)) for k, v in bundle.items() if k.startswith("cand_") and k.endswith("_graph")}}
    (WORK / "structures.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return manifest


# ---------------------------------------------------------------------------
# Compare

def paired(ds: C.Dataset, first: dict, second: dict, name_first: str, name_second: str) -> dict:
    """first minus second on the development rows both have, all rows and without cuprates."""
    dev = np.flatnonzero(ds.split == "dev")
    keys_all = [ds.rows[i]["composition"] for i in dev]
    have = np.array([k in first and k in second for k in keys_all])
    rows, keys = dev[have], [k for k, h in zip(keys_all, have) if h]
    a, b = L.aligned(first, keys), L.aligned(second, keys)
    out = {"rows": int(len(rows))}
    for scope, keep in (("all", np.ones(len(rows), bool)), ("not cuprates", ds.family[rows] != "Cuprates")):
        label, tc = ds.arrays.label[rows][keep], ds.arrays.tc_k[rows][keep]
        out[scope] = {"auc": {name_first: T.score(label, tc, a[0][keep], a[1][keep])["auc"],
                              name_second: T.score(label, tc, b[0][keep], b[1][keep])["auc"]},
                      "difference": C.paired_differences(ds, rows, a, b, keep)}
    return out


def blend_of(first: dict, second: dict) -> dict:
    keys = first.keys() & second.keys()
    return {k: (0.5 * first[k][0] + 0.5 * second[k][0],
                float(np.expm1(0.5 * np.log1p(first[k][1]) + 0.5 * np.log1p(second[k][1])))) for k in keys}


def cmd_compare(folder: str) -> dict:
    root = Path(folder)
    if not (root / "summary.json").is_file():
        root = G.RETURNED / folder
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifests = {"structure": json.loads((WORK / "structures.json").read_text(encoding="utf-8")),
                 "formula": json.loads((WORK / "bundle.json").read_text(encoding="utf-8"))}
    ds = C.load_dataset()
    record = G.load_predictions(L.WORK / f"{G.RECORD}-oof.npz")
    runs, predictions = {}, {}
    for job in summary["jobs"]:
        name = job["name"]
        oof = root / name / "oof.npz"
        run_path = root / name / "run.json"
        if job.get("status") != "done" or not oof.is_file() or not run_path.is_file():
            runs[name] = {"status": job.get("status"), "compared": False}
            continue
        run = json.loads(run_path.read_text(encoding="utf-8"))
        kind = run.get("bundle_kind", "formula")
        formula_hash = run.get("formula_bundle_content_sha256")
        if not G.same_bundle(run, manifests[kind]) or (formula_hash and formula_hash != manifests["formula"]["content_sha256"]):
            runs[name] = {"status": "trained on a different bundle", "compared": False}
            continue
        predictions[name] = G.load_predictions(oof)
        result = paired(ds, predictions[name], record, name, G.RECORD)
        result["blend_with_record"] = paired(ds, blend_of(predictions[name], record), record, "blend", G.RECORD)
        runs[name] = {"status": "done", "kind": kind, "model": run.get("settings", {}).get("model", "tabular"),
                      "mean_validation_bce": run.get("mean_validation_bce"), "minutes": run.get("minutes"),
                      "compared": True, "against_record": result}
    final = summary.get("final")
    reading = {"final_job": final,
               "rule": "The final job (chosen before any test row was scored, by mean validation loss) joins the "
                       "ranking of record only if its paired non-cuprate AUC interval against cycle 6, on the rows "
                       "it scores, lies above zero, and the owner accepts it. Every other job is exploratory."}
    if final in runs and runs[final].get("compared"):
        gap = runs[final]["against_record"]["not cuprates"]["difference"]["auc"]
        reading["final_beats_record"] = bool(gap["low"] > 0)
        blend_gap = runs[final]["against_record"]["blend_with_record"]["not cuprates"]["difference"]["auc"]
        reading["final_blend_beats_record"] = bool(blend_gap["low"] > 0)
    for job, control, label in ((final, summary.get("formula_control"), "final_minus_formula_control"),
                                (summary.get("final_secondary"), summary.get("secondary_control"),
                                 "secondary_minus_formula_control")):
        if job in predictions and control in predictions:
            reading[label] = paired(ds, predictions[job], predictions[control], job, control)
    secondary = summary.get("final_secondary")
    if secondary in runs and runs[secondary].get("compared"):
        gap = runs[secondary]["against_record"]["not cuprates"]["difference"]["auc"]
        reading["secondary_beats_record_on_its_rows"] = bool(gap["low"] > 0)
    report = {"folder": str(root), "record": G.RECORD, "env": summary.get("env"), "reading": reading, "runs": runs}
    G.DOCS.mkdir(parents=True, exist_ok=True)
    out = G.DOCS / f"{root.name}.json"
    out.write_text(json.dumps(L.rounded(report), indent=1) + "\n", encoding="utf-8")
    print(f"{'job':<28} {'kind':<9} {'rows':>5} {'val BCE':>8} {'AUC nc':>7} {'cycle6':>7} {'gap (95%)':>24}")
    for name, run in runs.items():
        if not run.get("compared"):
            print(f"{name:<28} {run['status']}")
            continue
        nc = run["against_record"]["not cuprates"]
        gap = nc["difference"]["auc"]
        print(f"{name:<28} {run['kind']:<9} {run['against_record']['rows']:>5} "
              f"{(run['mean_validation_bce'] or float('nan')):>8.4f} {nc['auc'][name]:>7.4f} {nc['auc'][G.RECORD]:>7.4f} "
              f"{gap['median']:>+8.4f} ({gap['low']:+.4f}, {gap['high']:+.4f})")
    print(json.dumps(L.rounded(reading), indent=1)[:3000])
    print(f"wrote {out}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export")
    c = sub.add_parser("compare")
    c.add_argument("folder", help="an overnight results folder holding summary.json, or its name under data/benchmarks/local-gpu/")
    args = parser.parse_args()
    if args.command == "export":
        cmd_export()
    else:
        cmd_compare(args.folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
