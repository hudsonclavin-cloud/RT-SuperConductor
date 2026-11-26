#!/usr/bin/env python3
"""One scoring of the TabICLv2 + cycle 6 blend on the locked rows (plan, fixed before any look: ``docs/lockbox-blend.md``).

    python scripts/lockbox_blend.py export          # data/processed/local-gpu/locked-features.npz: features, no labels
    #   on the GPU machine, with the bundle already there:
    #   python scripts/foundation_gpu_train.py data/processed/local-gpu/bundle.npz --model tabicl \\
    #       --extra-candidates locked-features.npz --candidates locked --out runs/lockbox-tabicl --device cuda
    #   push runs/lockbox-tabicl as data/benchmarks/local-gpu/lockbox-tabicl-<date>/
    python scripts/lockbox_blend.py score <folder>  # once: refuses if docs/local-gpu/lockbox-blend.json exists

Each model is scored the way it scores new compounds today:
* cycle 6: the mean of its five saved final models (``data/processed/learning-cycles/models/cycle6/``);
* TabICLv2: the mean of its five fold models, as for the shortlist candidates.
The blend is their 50/50 average. Only the export's features and composition keys leave this machine; the labels are
read here, once, by ``score``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import local_gpu as G
import run_learning_cycle as L
import scratchnet_common as C
from scratchnet_common import ROOT, T

LOCKED = G.WORK / "locked-features.npz"
LOCKED_MANIFEST = G.WORK / "locked-features.json"
OUT = G.DOCS / "lockbox-blend.json"
RECORD = "cycle6"
RECORD_MODELS = L.WORK / "models" / RECORD
SHORTLIST = ROOT / "data/benchmarks/rung4/shortlist3-ranked.csv"
RULE = ("The blend becomes the ranking of record only if its non-cuprate AUC on the locked rows is at least cycle 6's, "
        "and the owner accepts it. Scored once; the paired intervals are reported but do not decide.")


def locked_features(ds, X: np.ndarray) -> dict[str, np.ndarray]:
    """The locked rows' features and composition keys. Nothing else: no label, no Tc, no fold, no group."""
    locked = np.flatnonzero(ds.split == "lockbox")
    return {"candidates_locked_X": X[locked],
            "candidates_locked_keys": np.array([ds.rows[i]["composition"] for i in locked])}


def cmd_export() -> dict:
    ds = C.load_dataset()
    formulas = [C.formula_text(C.fractions_of(row["composition"])) for row in ds.rows]
    X, names = C.formula_features(formulas, cache=L.WORK / "formula-features.npz")
    out = locked_features(ds, X)
    with np.load(G.WORK / "bundle.npz", allow_pickle=False) as bundle:
        if list(bundle["feature_names"]) != list(names):
            raise SystemExit("the bundle's features differ from the table's; re-export the bundle first")
        if set(bundle["keys"].tolist()) & set(out["candidates_locked_keys"].tolist()):
            raise SystemExit("a locked composition is in the training bundle; refusing")
    np.savez_compressed(LOCKED, feature_names=np.array(names), **out)
    manifest = {"file": str(LOCKED.relative_to(ROOT)), "sha256": hashlib.sha256(LOCKED.read_bytes()).hexdigest(),
                "rows": int(len(out["candidates_locked_keys"])), "table_sha256": ds.sha256,
                "labels_included": False, "plan": "docs/lockbox-blend.md"}
    LOCKED_MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return manifest


def refuse_if_scored(path: Path = OUT) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists: the locked rows have been scored for this plan already, and are scored once")


def aligned(keys: list[str], found_keys, p, tc) -> tuple[np.ndarray, np.ndarray]:
    """Predictions in the order of ``keys``; refuses a file that misses or adds a locked composition."""
    found = {str(k): (float(a), float(b)) for k, a, b in zip(found_keys, p, tc)}
    if set(found) != set(keys) or len(found) != len(found_keys):
        raise SystemExit(f"the predictions cover {len(found)} compositions; the locked rows are {len(keys)}")
    return np.array([found[k][0] for k in keys]), np.array([found[k][1] for k in keys])


def record_predict(X: np.ndarray, model_dir: Path = RECORD_MODELS) -> tuple[np.ndarray, np.ndarray]:
    """Cycle 6 as it ranks new compounds: the mean of its five final models (Tc averaged on the log scale)."""
    import joblib

    paths = sorted(model_dir.glob("boosting-fold*.joblib"))
    if len(paths) != 5:
        raise SystemExit(f"expected cycle 6's five final models in {model_dir}, found {len(paths)}")
    applied = [C.boosting_apply(joblib.load(path), X) for path in paths]
    return np.mean([p for p, _ in applied], axis=0), C.log_mean_tc([t for _, t in applied])


def check_record_models(bundle) -> float:
    """The loaded models must reproduce cycle 6's published shortlist-3 probabilities (rounded to 3 decimals)."""
    with open(SHORTLIST, newline="", encoding="utf-8") as handle:
        published = {row["mp_id"]: float(row["p_superconductor"]) for row in csv.DictReader(handle)}
    p, _ = record_predict(bundle["candidates_shortlist_pool_X"])
    gap = max(abs(published[str(k)] - float(v)) for k, v in zip(bundle["candidates_shortlist_pool_keys"], p))
    if gap > 0.0006:
        raise SystemExit(f"the saved models do not reproduce cycle 6's shortlist (largest gap {gap:.4f})")
    return gap


def score_locked(ds, rows: np.ndarray, record: tuple[np.ndarray, np.ndarray],
                 tabicl: tuple[np.ndarray, np.ndarray]) -> dict:
    """Cycle 6, TabICLv2 and their blend on the locked rows, with the plan's reading."""
    mix = C.blend(tabicl, record)
    not_cuprate = ds.family[rows] != "Cuprates"
    label, tc = ds.arrays.label[rows], ds.arrays.tc_k[rows]
    out = {"locked_rows": int(len(rows)), "labelled_rows": int(np.isfinite(label).sum()), "scores": {}, "paired": {}}
    for name, keep in (("all", np.ones(len(rows), bool)), ("not cuprates", not_cuprate)):
        out["scores"][name] = {model: T.score(label[keep], tc[keep], pred[0][keep], pred[1][keep])
                               for model, pred in ((RECORD, record), ("tabicl", tabicl), ("blend", mix))}
        out["paired"][name] = {"blend_minus_record": C.paired_differences(ds, rows, mix, record, keep),
                               "tabicl_minus_record": C.paired_differences(ds, rows, tabicl, record, keep)}
    blend_auc = out["scores"]["not cuprates"]["blend"]["auc"]
    record_auc = out["scores"]["not cuprates"][RECORD]["auc"]
    out["reading"] = {"blend_auc_not_cuprates": blend_auc, "record_auc_not_cuprates": record_auc,
                      "blend_passes": bool(blend_auc >= record_auc), "rule": RULE}
    return out


def cmd_score(name: str) -> dict:
    refuse_if_scored()
    folder = G.result_folder(name)
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    manifest = json.loads((G.WORK / "bundle.json").read_text(encoding="utf-8"))
    locked_manifest = json.loads(LOCKED_MANIFEST.read_text(encoding="utf-8"))
    if run.get("model") != "tabicl" or not run.get("complete"):
        raise SystemExit("this is not a complete TabICLv2 run")
    if not G.same_bundle(run, manifest):
        raise SystemExit("this run used a different training bundle")
    if run.get("extra_candidates", {}).get("sha256") != locked_manifest["sha256"]:
        raise SystemExit("this run did not predict the locked-features file exported here")
    ds = C.load_dataset()
    rows = np.flatnonzero(ds.split == "lockbox")
    keys = [ds.rows[i]["composition"] for i in rows]
    with np.load(folder / "candidates_locked.npz", allow_pickle=False) as saved:
        tabicl = aligned(keys, saved["keys"], saved["p"], saved["tc"])
    with np.load(LOCKED, allow_pickle=False) as locked, np.load(G.WORK / "bundle.npz", allow_pickle=False) as bundle:
        gap = check_record_models(bundle)
        if [str(k) for k in locked["candidates_locked_keys"]] != keys:
            raise SystemExit("the locked-features file is out of date")
        record = record_predict(locked["candidates_locked_X"])
    report = {"plan": "docs/lockbox-blend.md", "tag": folder.name, "record": RECORD,
              "record_models_check_largest_gap": gap, "run": run, **score_locked(ds, rows, record, tabicl)}
    G.DOCS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(L.rounded(report), indent=1) + "\n", encoding="utf-8")
    print(json.dumps(L.rounded({k: report[k] for k in ("locked_rows", "scores", "reading")}), indent=1))
    print(f"wrote {OUT}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export")
    score = sub.add_parser("score")
    score.add_argument("folder", help="the TabICLv2 run that predicted the locked rows (a tag or a path)")
    args = parser.parse_args()
    if args.command == "export":
        cmd_export()
    else:
        cmd_score(args.folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
