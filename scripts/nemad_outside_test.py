#!/usr/bin/env python3
"""An outside test of the ranking of record on NEMAD compositions neither model has seen (plan: docs/nemad-outside-test.md).

    python scripts/nemad_outside_test.py export          # data/processed/local-gpu/nemad-features.npz: features, no labels
    #   on the GPU machine, with the bundle already there:
    #   python scripts/foundation_gpu_train.py data/processed/local-gpu/bundle.npz --model tabicl \\
    #       --extra-candidates data/processed/local-gpu/nemad-features.npz --candidates nemad --out runs/nemad-tabicl \\
    #       --device cuda
    #   push runs/nemad-tabicl as data/benchmarks/local-gpu/nemad-tabicl-<date>/
    python scripts/nemad_outside_test.py score <folder>   # once: refuses if docs/local-gpu/nemad-outside-test.json exists

The rows are the usable NEMAD subset's new compositions (scripts/build_nemad_additions.py, ``select_new``): flagged
experimental, at normal or unstated pressure, not in the table (locked rows included), not in the NIMS file, not a near
twin of either. Each model scores them the way it scores new compounds today:
* cycle 6: the mean of its five saved final models (``lockbox_blend.record_predict``);
* TabICLv2: the mean of its five fold models, run on the owner's laptop against the same bundle as before.
The blend is their 50/50 average. Only features and composition keys leave this machine; NEMAD's labels are read here,
once, by ``score``. Intervals resample groups formed by the table's rule (major elements, tied through near twins) over the training
and test compositions together; rows whose group holds no training row form the plan's outside slice.
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
RECORD_TABICL = "fm-tabicl-20261123"   # the ranking of record's TabICLv2 run; it predicted the made-metals list
REPRODUCED_GAP = 0.02                  # largest probability gap on the made-metals overlap that counts as reproduced
OTHER = "other families (mostly intermetallic)"   # build_mp_candidates.family "Other": no Cu+O, not iron-based, no O
RULE = ("A check, not an adoption test: the ranking of record is already the blend. Two readings, blend minus cycle 6 "
        "in AUC: main = all non-cuprates; outside = non-cuprates whose group holds no training row. Outcome, first "
        "match wins: (1) either paired interval entirely below zero: propose going back to cycle 6 alone, the owner "
        "decides; (2) main and outside both at least cycle 6's: supported; (3) main at least cycle 6's, outside lower: "
        "supported on the main reading, a mark on the outside one; (4) otherwise: a mark against the blend. Scored once.")
OUTCOMES = {"propose going back": "Either paired interval lies entirely below zero: propose going back to cycle 6 alone; "
                                  "the owner decides.",
            "supported": "Main and outside readings are both at least cycle 6's: the decision is supported; nothing changes.",
            "supported on main, mark on outside": "Main at least cycle 6's, outside lower (interval includes zero): "
                                                  "nothing changes; recorded as a mark on the outside reading.",
            "mark against the blend": "Main lower (interval includes zero): nothing changes; recorded as a mark against "
                                      "the blend."}


def outside_rows() -> tuple[list[str], dict[str, tuple[int, float | None]]]:
    """The compositions (sorted) and NEMAD's label and Tc for each, from the usable subset."""
    held, valid = N.held_compositions()
    with open(BA.NEMAD, newline="", encoding="utf-8") as handle:
        labels = BA.nemad_labels(list(csv.DictReader(handle)))
    keys, _ = BA.select_new(labels, held, valid)
    return keys, {k: labels[k] for k in keys}


def features_for(keys: list[str]) -> tuple[np.ndarray, list[str]]:
    return C.formula_features([C.formula_text(C.fractions_of(k)) for k in keys])


def cmd_export() -> dict:
    keys, _ = outside_rows()
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


def grouping(keys: list[str], training_keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Each test row's group, and whether that group holds no training row.

    The table's rule (major-element groups tied together through near twins) applied to the training and test
    compositions together, so test rows chained through a training row share a group. Uses compositions only."""
    comps = [C.fractions_of(k) for k in training_keys] + [C.fractions_of(k) for k in keys]
    merged = B.merge_groups([B.major_group(c) for c in comps], B.near_pairs(comps, B.NEAR_TOLERANCE))
    trained = set(merged[:len(training_keys)])
    groups = np.array(merged[len(training_keys):])
    return groups, np.array([g not in trained for g in groups])


def training_keys(bundle) -> list[str]:
    """The compositions the models fit: labelled bundle rows (both trainers drop rows with no label)."""
    return [str(k) for k, label in zip(bundle["keys"], bundle["label"]) if np.isfinite(label)]


def outcome(reading: dict) -> str:
    """The plan's outcome for a reading block; the first matching row of the plan's table wins."""
    main, outside = reading["main"], reading["outside"]
    if main["interval_entirely_below_zero"] or outside["interval_entirely_below_zero"]:
        return "propose going back"
    if main["blend_at_least_cycle6"]:
        return "supported" if outside["blend_at_least_cycle6"] else "supported on main, mark on outside"
    return "mark against the blend"


def made_metals_reproduction(keys: list[str], tabicl: tuple, bundle, X: np.ndarray) -> dict:
    """How closely this run's TabICLv2 matches the record run's on test rows that are also made-metals candidates.

    The verdict uses only rows whose feature vectors are identical in both lists; rows whose features differ by
    rounding of the composition are reported apart, so model drift is not mixed with input drift."""
    with np.load(G.result_folder(RECORD_TABICL) / "candidates_made_metals.npz", allow_pickle=False) as record:
        p_record = dict(zip((str(k) for k in record["keys"]), record["p"].tolist()))
    position = {k: i for i, k in enumerate(keys)}
    same, rounded = [], []
    for j, formula in enumerate(str(f) for f in bundle["candidates_made_metals_keys"]):
        fractions = M.fractions(formula)
        i = position.get(B.composition_key(fractions)) if fractions else None
        if i is None or formula not in p_record:
            continue
        gap = abs(float(tabicl[0][i]) - p_record[formula])
        (same if np.array_equal(bundle["candidates_made_metals_X"][j], X[i]) else rounded).append(gap)
    out = {"rows": len(same) + len(rounded), "rows_with_identical_inputs": len(same),
           "rows_whose_inputs_differ_by_rounding": len(rounded), "threshold": REPRODUCED_GAP, "record_run": RECORD_TABICL}
    if same:
        out.update({"largest_gap": float(max(same)), "median_gap": float(np.median(same)),
                    "reproduced": bool(max(same) <= REPRODUCED_GAP)})
    if rounded:
        out["largest_gap_where_inputs_differ"] = float(max(rounded))
    return out


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


def score_rows(keys: list[str], labels: dict, record: tuple, tabicl: tuple, groups: np.ndarray,
               new_group: np.ndarray) -> dict:
    """Cycle 6, TabICLv2 and the blend on the test rows, with the plan's readings (``grouping`` gives the last two)."""
    mix = C.blend(tabicl, record)
    y = np.array([float(labels[k][0]) for k in keys])
    tc = np.array([np.nan if labels[k][1] is None else float(labels[k][1]) for k in keys])
    family = np.array([M.family(set(C.fractions_of(k))) for k in keys])
    out = {"rows": len(keys), "superconductors": int((y == 1).sum()), "nulls": int((y == 0).sum()),
           "groups": len(set(groups.tolist())), "rows_in_groups_with_no_training_row": int(new_group.sum()),
           "scores": {}, "paired": {}}
    not_cuprate = family != "Cuprates"
    for name, keep in (("all", np.ones(len(y), bool)), ("not cuprates", not_cuprate),
                       (OTHER, family == "Other"),
                       ("not cuprates, groups with no training row", not_cuprate & new_group),
                       ("not cuprates, groups with training rows", not_cuprate & ~new_group)):
        out["scores"][name] = {m: T.score(y[keep], tc[keep], p[0][keep], p[1][keep])
                               for m, p in (("cycle6", record), ("tabicl", tabicl), ("blend", mix))}
        out["paired"][name] = {"blend_minus_cycle6": paired(y, tc, groups, mix, record, keep),
                               "tabicl_minus_cycle6": paired(y, tc, groups, tabicl, record, keep)}
    out["reading"] = {"rule": RULE}
    for label, name in (("main", "not cuprates"), ("outside", "not cuprates, groups with no training row")):
        b, r = out["scores"][name]["blend"]["auc"], out["scores"][name]["cycle6"]["auc"]
        high = out["paired"][name]["blend_minus_cycle6"]["auc"]["high"]
        out["reading"][label] = {"slice": name, "blend_auc": b, "cycle6_auc": r, "blend_at_least_cycle6": bool(b >= r),
                                 "interval_entirely_below_zero": bool(high < 0)}
    out["reading"]["propose_going_back_to_cycle6"] = any(out["reading"][k]["interval_entirely_below_zero"]
                                                         for k in ("main", "outside"))
    out["reading"]["outcome"] = outcome(out["reading"])
    out["reading"]["what_follows"] = OUTCOMES[out["reading"]["outcome"]]
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
    keys, labels = outside_rows()
    with np.load(FEATURES, allow_pickle=False) as saved:
        if [str(k) for k in saved[f"candidates_{LIST}_keys"]] != keys:
            raise SystemExit("the features file is out of date: the test rows have changed since the export")
        X = saved[f"candidates_{LIST}_X"]
        record = LB.record_predict(X)
    with np.load(folder / f"candidates_{LIST}.npz", allow_pickle=False) as saved:
        tabicl = LB.aligned(keys, saved["keys"], saved["p"], saved["tc"])
    with np.load(G.WORK / "bundle.npz", allow_pickle=False) as bundle:
        gap = LB.check_record_models(bundle)
        groups, new_group = grouping(keys, training_keys(bundle))
        reproduction = made_metals_reproduction(keys, tabicl, bundle, X)
    report = {"plan": "docs/nemad-outside-test.md", "tag": folder.name, "record_models_check_largest_gap": gap,
              "tabicl_reproduces_record_on_made_metals_overlap": reproduction,
              "run": run, **score_rows(keys, labels, record, tabicl, groups, new_group)}
    G.DOCS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(L.rounded(report), indent=1) + "\n", encoding="utf-8")
    print(json.dumps(L.rounded({k: report[k] for k in ("rows", "tabicl_reproduces_record_on_made_metals_overlap",
                                                        "scores", "reading")}), indent=1))
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
