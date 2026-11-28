#!/usr/bin/env python3
"""Train a network on the owner's own GPU, on exactly the rows and folds the ranking of record uses (``docs/local-gpu.md``).

Two halves, run here; the training in between runs on the owner's machine:

    python scripts/local_gpu.py export            # writes data/processed/local-gpu/bundle.npz (+ bundle.json)
    #   copy bundle.npz to the GPU machine, then there:
    #   python scripts/local_gpu_train.py bundle.npz --out results/ --device cuda
    #   commit results/ as data/benchmarks/local-gpu/<tag>/ and push it (or copy it to data/processed/local-gpu/<tag>/)
    python scripts/local_gpu.py compare <tag>     # paired comparison with the ranking of record on the same rows

The bundle holds features, labels and, for each development fold, the row numbers that cycle 6's boosting fitted on,
stopped early on, and was scored on. Locked rows are not in the bundle at all, so nothing trained from it can see
them. The NIMS rows that cycle 4 added are included and follow the same per-fold rule as in ``run_learning_cycle.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import local_gpu_train as TR
import run_learning_cycle as L
import scratchnet_common as C
from scratchnet_common import ROOT, T
from run_scratchnet import fold_rows

WORK = ROOT / "data/processed/local-gpu"           # the bundle (ignored: it holds the licensed training table)
RETURNED = ROOT / "data/benchmarks/local-gpu"      # results pushed back from the GPU machine (predictions only)
DOCS = ROOT / "docs/local-gpu"
ADDITIONS = ROOT / "data/processed/learning-cycles/nims-additions.csv.gz"
RECORD = "cycle6"                         # the ranking of record the network is compared with
SHORTLIST_POOL = ROOT / "data/benchmarks/rung4/shortlist3-ranked.csv"


def build_bundle(ds: C.Dataset, X: np.ndarray, extra: dict | None, candidates: dict[str, np.ndarray] | None = None
                 ) -> dict[str, np.ndarray]:
    """Arrays for the trainer. Row numbers index the bundle's own rows (table rows without the locked ones, then added rows)."""
    keep = np.flatnonzero(ds.split != "lockbox")
    new_index = np.full(len(ds.rows), -1)
    new_index[keep] = np.arange(len(keep))
    n_table = len(keep)
    n_extra = 0 if extra is None else len(extra["keys"])

    def mapped(rows: np.ndarray) -> np.ndarray:
        out = new_index[rows]
        if (out < 0).any():
            raise SystemExit("a locked row reached the bundle")
        return out

    bundle = {
        "X": np.vstack([X[keep], extra["X"]]) if extra is not None else X[keep],
        "label": np.concatenate([ds.arrays.label[keep], extra["label"]]) if extra is not None else ds.arrays.label[keep],
        "tc_k": np.concatenate([ds.arrays.tc_k[keep], extra["tc_k"]]) if extra is not None else ds.arrays.tc_k[keep],
        "keys": np.array([ds.rows[i]["composition"] for i in keep] + ([] if extra is None else list(extra["keys"]))),
        "is_added_row": np.concatenate([np.zeros(n_table, bool), np.ones(n_extra, bool)]),
        "split": np.array([ds.split[i] for i in keep] + ["added"] * n_extra),
    }
    dev = ds.split == "dev"
    folds = sorted(set(ds.fold[dev].tolist()))
    for fold in folds:
        fit, valid, test = fold_rows(ds, fold)
        fit_b = mapped(fit)
        if extra is not None:
            use = L.added_rows_for(extra, fold, set(ds.arrays.group[valid].tolist()))
            fit_b = np.concatenate([fit_b, n_table + use])
        bundle[f"fold{fold}_fit"] = fit_b.astype(np.int64)
        bundle[f"fold{fold}_valid"] = mapped(valid).astype(np.int64)
        bundle[f"fold{fold}_test"] = mapped(test).astype(np.int64)
    bundle["folds"] = np.array(folds)
    if candidates:
        for name, value in candidates.items():
            bundle[f"candidates_{name}"] = value
    return bundle


def cmd_export() -> dict:
    ds = C.load_dataset()
    formulas = [C.formula_text(C.fractions_of(row["composition"])) for row in ds.rows]
    X, names = C.formula_features(formulas, cache=L.WORK / "formula-features.npz")
    extra = L.load_additions(ADDITIONS, ds)
    rows, compositions = L.candidate_pool(L.CANDIDATES, set(ds.table.symbols))
    X_pool, _ = C.formula_features([C.formula_text(c) for c in compositions], cache=L.WORK / "candidate-features.npz")
    candidates = {"made_metals_X": X_pool, "made_metals_keys": np.array([r["formula"] for r in rows])}
    if SHORTLIST_POOL.is_file():
        import csv
        import rung4 as R
        with open(SHORTLIST_POOL, newline="", encoding="utf-8") as handle:
            pool = list(csv.DictReader(handle))
        X_short, _ = C.formula_features([C.formula_text(R.fractions(r["formula"])) for r in pool])
        candidates.update(shortlist_pool_X=X_short, shortlist_pool_keys=np.array([r["mp_id"] for r in pool]))
    bundle = build_bundle(ds, X, extra, candidates)
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "bundle.npz"
    np.savez_compressed(path, feature_names=np.array(names), **bundle)
    with np.load(path, allow_pickle=False) as saved:
        content = TR.content_sha256(saved)
    manifest = {"bundle": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "content_sha256": content,
                "table_sha256": ds.sha256, "additions_sha256": extra["sha256"], "record": RECORD,
                "rows": int(len(bundle["keys"])), "added_rows": int(bundle["is_added_row"].sum()),
                "features": int(bundle["X"].shape[1]), "folds": [int(f) for f in bundle["folds"]],
                "locked_rows_in_bundle": 0,
                "per_fold": {int(f): {"fit": int(len(bundle[f"fold{f}_fit"])), "valid": int(len(bundle[f"fold{f}_valid"])),
                                      "test": int(len(bundle[f"fold{f}_test"]))} for f in bundle["folds"]},
                "candidates": {k[:-2]: int(len(v)) for k, v in candidates.items() if k.endswith("_X")}}
    (WORK / "bundle.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return manifest


def compare(ds: C.Dataset, record: dict[str, tuple[float, float]], network: dict[str, tuple[float, float]]) -> dict:
    """The network, cycle 6, and their 50/50 blend on the development rows both scored. Positive AUC differences favour the first."""
    dev = np.flatnonzero(ds.split == "dev")
    keys = [ds.rows[i]["composition"] for i in dev]
    have = np.array([k in network and k in record for k in keys])
    rows = dev[have]
    keys = [k for k, h in zip(keys, have) if h]
    net = L.aligned(network, keys)
    rec = L.aligned(record, keys)
    mix = C.blend(net, rec)
    not_cuprate = ds.family[rows] != "Cuprates"
    label, tc = ds.arrays.label[rows], ds.arrays.tc_k[rows]
    out = {"rows_compared": int(len(rows)), "development_rows": int(len(dev)), "scores": {}, "paired": {}}
    for name, keep in (("all", np.ones(len(rows), bool)), ("not cuprates", not_cuprate)):
        out["scores"][name] = {model: T.score(label[keep], tc[keep], pred[0][keep], pred[1][keep])
                               for model, pred in (("network", net), (RECORD, rec), ("blend", mix))}
        out["paired"][name] = {"network_minus_record": C.paired_differences(ds, rows, net, rec, keep),
                               "blend_minus_record": C.paired_differences(ds, rows, mix, rec, keep)}
    gap = out["paired"]["not cuprates"]["network_minus_record"]["auc"]
    blend_gap = out["paired"]["not cuprates"]["blend_minus_record"]["auc"]
    out["reading"] = {
        "network_beats_record": bool(gap["low"] > 0),
        "blend_beats_record": bool(blend_gap["low"] > 0),
        "rule": "A model replaces or joins the ranking of record only if its paired non-cuprate AUC interval against "
                "the record is above zero, and the owner accepts it. Otherwise it is a second opinion."}
    return out


def load_predictions(path: Path) -> dict[str, tuple[float, float]]:
    with np.load(path) as saved:
        return {str(k): (float(p), float(t)) for k, p, t in zip(saved["compositions"], saved["p"], saved["tc"])}


def same_bundle(run: dict, manifest: dict) -> bool:
    """True when the results were trained on the bundle exported here: same arrays (or, failing that, the same file)."""
    if run.get("bundle_content_sha256") and manifest.get("content_sha256"):
        return run["bundle_content_sha256"] == manifest["content_sha256"]
    return run.get("bundle_sha256") == manifest.get("sha256")


def result_folder(name: str) -> Path:
    """A folder path, or a tag under data/processed/local-gpu/ or data/benchmarks/local-gpu/ (results pushed from the GPU machine)."""
    for candidate in (Path(name), WORK / name, RETURNED / name):
        if (candidate / "run.json").is_file():
            return candidate
    raise SystemExit(f"no run.json for {name!r} in {WORK} or {RETURNED}")


def cmd_compare(name: str) -> dict:
    ds = C.load_dataset()
    result_dir = result_folder(name)
    tag = result_dir.name
    run = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
    manifest = json.loads((WORK / "bundle.json").read_text(encoding="utf-8"))
    if not same_bundle(run, manifest):
        raise SystemExit("these results were trained on a different bundle than the one exported here; "
                         "re-export and retrain, or check that the table and NIMS additions are unchanged")
    record = load_predictions(L.WORK / f"{RECORD}-oof.npz")
    network = load_predictions(result_dir / "oof.npz")
    report = {"tag": tag, "record": RECORD, "run": run, **compare(ds, record, network)}
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / f"{tag}.json"
    out.write_text(json.dumps(L.rounded(report), indent=1) + "\n", encoding="utf-8")
    print(json.dumps(L.rounded({k: report[k] for k in ("rows_compared", "scores", "reading")}), indent=1))
    print(f"wrote {out}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export")
    c = sub.add_parser("compare")
    c.add_argument("tag", help="a folder holding run.json and oof.npz, or its name under data/processed/local-gpu/ "
                               "or data/benchmarks/local-gpu/")
    args = parser.parse_args()
    if args.command == "export":
        cmd_export()
    else:
        cmd_compare(args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
