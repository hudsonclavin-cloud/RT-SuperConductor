#!/usr/bin/env python3
"""An outside test of the ranking of record on NEMAD compositions neither model has seen (plan: docs/nemad-outside-test.md).

    python scripts/nemad_outside_test.py export          # data/processed/local-gpu/nemad-features.npz: features, no labels
    #   on the GPU machine, with the bundle already there:
    #   python scripts/foundation_gpu_train.py data/processed/local-gpu/bundle.npz --model tabicl \\
    #       --extra-candidates nemad-features.npz --candidates nemad --out runs/nemad-tabicl --device cuda
    #   push runs/nemad-tabicl as data/benchmarks/local-gpu/nemad-tabicl-<date>/
    python scripts/nemad_outside_test.py score <folder>   # once: refuses if docs/local-gpu/nemad-outside-test.json exists

The rows are the usable NEMAD subset's new compositions (scripts/build_nemad_additions.py, ``select_new``): flagged
experimental, at normal or unstated pressure, not in the table (locked rows included), not in the NIMS file, not a near
twin of either. Each model scores them the way it scores new compounds today:
* cycle 6: the mean of its five saved final models (``lockbox_blend.record_predict``);
* TabICLv2: the mean of its five fold models, run on the owner's laptop against the same bundle as before.
The blend is their 50/50 average. Only features and composition keys leave this machine; NEMAD's labels are read here,
once, by ``score``. Intervals resample major-element groups tied through near twins, this project's unit of independence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import build_mp_candidates as M
import build_nemad_additions as BA
import build_scratchnet_data as B
import local_gpu as G
import lockbox_blend as LB
import novelty_check as N
import run_learning_cycle as L
import scratchnet_common as C
from scratchnet_common import ROOT, T

FEATURES = G.WORK / "nemad-features.npz"
MANIFEST = G.WORK / "nemad-features.json"
OUT = G.DOCS / "nemad-outside-test.json"
LIST = "nemad"
RECORD_SETTINGS = {"estimators": 8, "max_context": 0, "seed": 0}   # as in fm-tabicl-20261123, the ranking of record
RULE = ("A check, not an adoption test: the ranking of record is already the blend. Read: does the blend's non-cuprate "
        "AUC on these compositions equal or beat cycle 6's? Scored once; the paired intervals are reported.")


def test_rows() -> tuple[list[str], dict[str, tuple[int, float | None]]]:
    """The compositions (sorted) and NEMAD's label and Tc for each, from the usable subset."""
    held, valid = N.held_compositions()
    with open(BA.NEMAD, newline="", encoding="utf-8") as handle:
        labels = BA.nemad_labels(list(csv.DictReader(handle)))
    keys, _ = BA.select_new(labels, held, valid)
    return keys, {k: labels[k] for k in keys}


def features_for(keys: list[str]) -> tuple[np.ndarray, list[str]]:
    return C.formula_features([C.formula_text(C.fractions_of(k)) for k in keys])


def cmd_export() -> dict:
    keys, _ = test_rows()
    X, names = features_for(keys)
    with np.load(G.WORK / "bundle.npz", allow_pickle=False) as bundle:
        if list(bundle["feature_names"]) != list(names):
            raise SystemExit("the bundle's features differ from these; re-export the bundle first")
        if set(bundle["keys"].tolist()) & set(keys):
            raise SystemExit("a test composition is in the training bundle; refusing")
    np.savez_compressed(FEATURES, feature_names=np.array(names), **{f"candidates_{LIST}_X": X,
                                                                     f"candidates_{LIST}_keys": np.array(keys)})
    manifest = {"file": str(FEATURES.relative_to(ROOT)), "sha256": hashlib.sha256(FEATURES.read_bytes()).hexdigest(),
                "rows": len(keys), "labels_included": False, "plan": "docs/nemad-outside-test.md"}
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return manifest


def refuse_if_scored(path: Path = OUT) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists: these rows have been scored for this plan already, and are scored once")


def paired(y, tc, groups, first, second, keep) -> dict:
    """first minus second: AUC (positive favours first) and Tc log error on superconductors with a Tc (negative does)."""
    y, tc, groups = y[keep], tc[keep], groups[keep]
    p1, t1, p2, t2 = first[0][keep], first[1][keep], second[0][keep], second[1][keep]
    positive = (y == 1) & np.isfinite(tc)

    def auc_gap(s):
        return T.roc_auc(y[s], p1[s]) - T.roc_auc(y[s], p2[s])

    def tc_gap(s):
        s = s[positive[s]]
        if not len(s):
            return float("nan")
        truth = np.log1p(tc[s])
        return float(np.mean(np.abs(np.log1p(t1[s]) - truth)) - np.mean(np.abs(np.log1p(t2[s]) - truth)))

    return {"auc": C.group_bootstrap(groups, auc_gap), "tc_log_mae": C.group_bootstrap(groups, tc_gap)}


def independence_groups(keys: list[str]) -> list[str]:
    """The project's unit of independence: major-element groups tied together through near twins (as in the table)."""
    comps = [C.fractions_of(k) for k in keys]
    return B.merge_groups([B.major_group(c) for c in comps], B.near_pairs(comps, B.NEAR_TOLERANCE))


def check_run(run: dict, bundle_manifest: dict) -> None:
    """Refuse a run that is not the ranking of record's TabICLv2: all five folds, the same settings, the same bundle."""
    if run.get("model") != "tabicl" or not run.get("complete"):
        raise SystemExit("this is not a complete TabICLv2 run")
    if [f["fold"] for f in run.get("folds", [])] != list(bundle_manifest["folds"]):
        raise SystemExit("this run did not use every fold; the ranking of record averages all five")
    settings = run.get("settings", {})
    for name, value in RECORD_SETTINGS.items():
        if settings.get(name) != value:
            raise SystemExit(f"this run's {name} is {settings.get(name)!r}, not the ranking of record's {value!r}")
    if not G.same_bundle(run, bundle_manifest):
        raise SystemExit("this run used a different training bundle")


def score_rows(keys: list[str], labels: dict, record: tuple, tabicl: tuple) -> dict:
    """Cycle 6, TabICLv2 and the blend on the test rows, with the plan's reading."""
    mix = C.blend(tabicl, record)
    y = np.array([float(labels[k][0]) for k in keys])
    tc = np.array([np.nan if labels[k][1] is None else float(labels[k][1]) for k in keys])
    family = np.array([M.family(set(C.fractions_of(k))) for k in keys])
    groups = np.array(independence_groups(keys))
    out = {"rows": len(keys), "superconductors": int((y == 1).sum()), "nulls": int((y == 0).sum()), "scores": {}, "paired": {}}
    for name, keep in (("all", np.ones(len(y), bool)), ("not cuprates", family != "Cuprates"),
                       ("intermetallic", family == "Other")):
        out["scores"][name] = {m: T.score(y[keep], tc[keep], p[0][keep], p[1][keep])
                               for m, p in (("cycle6", record), ("tabicl", tabicl), ("blend", mix))}
        out["paired"][name] = {"blend_minus_cycle6": paired(y, tc, groups, mix, record, keep),
                               "tabicl_minus_cycle6": paired(y, tc, groups, tabicl, record, keep)}
    b, r = out["scores"]["not cuprates"]["blend"]["auc"], out["scores"]["not cuprates"]["cycle6"]["auc"]
    out["reading"] = {"blend_auc_not_cuprates": b, "cycle6_auc_not_cuprates": r, "blend_at_least_cycle6": bool(b >= r),
                      "rule": RULE}
    return out


def cmd_score(name: str) -> dict:
    refuse_if_scored()
    folder = G.result_folder(name)
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    bundle_manifest = json.loads((G.WORK / "bundle.json").read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    check_run(run, bundle_manifest)
    if run.get("extra_candidates", {}).get("sha256") != manifest["sha256"]:
        raise SystemExit("this run did not predict the features file exported here")
    keys, labels = test_rows()
    with np.load(FEATURES, allow_pickle=False) as saved:
        if [str(k) for k in saved[f"candidates_{LIST}_keys"]] != keys:
            raise SystemExit("the features file is out of date: the test rows have changed since the export")
        record = LB.record_predict(saved[f"candidates_{LIST}_X"])
    with np.load(folder / f"candidates_{LIST}.npz", allow_pickle=False) as saved:
        tabicl = LB.aligned(keys, saved["keys"], saved["p"], saved["tc"])
    with np.load(G.WORK / "bundle.npz", allow_pickle=False) as bundle:
        gap = LB.check_record_models(bundle)
    report = {"plan": "docs/nemad-outside-test.md", "tag": folder.name, "record_models_check_largest_gap": gap,
              "run": run, **score_rows(keys, labels, record, tabicl)}
    G.DOCS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(L.rounded(report), indent=1) + "\n", encoding="utf-8")
    print(json.dumps(L.rounded({k: report[k] for k in ("rows", "scores", "reading")}), indent=1))
    print(f"wrote {OUT}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export")
    score = sub.add_parser("score")
    score.add_argument("folder", help="the TabICLv2 run that predicted the NEMAD rows (a tag or a path)")
    args = parser.parse_args()
    if args.command == "export":
        cmd_export()
    else:
        cmd_score(args.folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
