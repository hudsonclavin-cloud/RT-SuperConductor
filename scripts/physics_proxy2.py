#!/usr/bin/env python3
"""A second physics proxy, trained on the Alexandria and JARVIS electron-phonon calculations (``docs/physics-proxy2.md``).

    python scripts/physics_proxy2.py data     # parse both sets into one table, one row per composition
    python scripts/physics_proxy2.py p1       # does it learn the calculation? and does it agree with Bercx et al.?
    python scripts/physics_proxy2.py p2       # does it carry to experiments? (the 882 made metals of test T2)
    python scripts/physics_proxy2.py p3       # does it help the model? (the four NIMS selection folds)

The raw files stay under ``data/raw/`` (git ignores it). Nothing here changes the frozen champion or its inputs: the
first proxy stays in ``children.physics_feature``, and this one lives in its own files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import scratchnet_common as C
from scratchnet_common import ROOT

ALEXANDRIA = ROOT / "data/raw/alexandria-epc"
JARVIS = ROOT / "data/raw/jarvis-epc/jarvis_epc_data_figshare_1058.json.zip"
JARVIS_SHA256 = "4213651cee9b4c13a376dd74ac1f8d3bd9efcd451f1b8da615255c8df0e1487f"
WORK = ROOT / "data/processed/physics-proxy2"
DOCS = ROOT / "docs/physics-proxy2"
BATCHES = "abcdefghijklmnopq"
LAMBDA_AT_MOST = 4.0
MU_STAR = 0.10
TC_FLAG_K = 1.0
TARGETS = ("log_lambda", "log_omega_log", "log1p_tc", "tc_above_1k")
F_BLOCK = set("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Ac Th Pa U Np Pu".split())


def allen_dynes(lam: float, omega_log_k: float, mu_star: float = MU_STAR) -> float:
    denominator = lam - mu_star * (1.0 + 0.62 * lam)
    if lam <= 0 or omega_log_k <= 0 or denominator <= 0:
        return 0.0
    return omega_log_k / 1.2 * math.exp(-1.04 * (1.0 + lam) / denominator)


def read_mcmillan(text: str) -> tuple[float, float] | None:
    """Lambda and omega_log (K) from the header of an Alexandria McMillan.dat."""
    lam = re.search(r"#\s*lambda\s*=\s*([-\d.Ee+]+)", text)
    wlog = re.search(r"#\s*wlog\[K\]\s*=\s*([-\d.Ee+]+)", text)
    if not lam or not wlog:
        return None
    return float(lam.group(1)), float(wlog.group(1))


def cif_formula(text: str) -> str | None:
    found = re.search(r"_chemical_formula_sum\s+'([^']+)'", text)
    return found.group(1).replace(" ", "") if found else None


def usable(lam: float, wlog: float) -> bool:
    return math.isfinite(lam) and math.isfinite(wlog) and 0 < lam <= LAMBDA_AT_MOST and wlog > 0


def alexandria_rows() -> tuple[list[dict], Counter]:
    rows, counts = [], Counter()
    for letter in BATCHES:
        if not (ALEXANDRIA / f"batch-{letter}.done").is_file():
            raise SystemExit(f"Alexandria batch {letter} is not downloaded yet")
        for folder in sorted((ALEXANDRIA / f"batch-{letter}").glob("*/*")):
            counts["folders"] += 1
            mcmillan, cif = folder / "McMillan.dat", folder / "geo_opt.cif"
            if not mcmillan.is_file() or not cif.is_file():
                counts["no_result_files"] += 1
                continue
            values = read_mcmillan(mcmillan.read_text(encoding="utf-8", errors="replace"))
            formula = cif_formula(cif.read_text(encoding="utf-8", errors="replace"))
            if values is None or formula is None:
                counts["unreadable"] += 1
                continue
            if not usable(*values):
                counts["lambda_or_omega_out_of_range"] += 1
                continue
            rows.append({"source": "alexandria", "id": folder.name, "formula": formula, "lambda": values[0],
                         "omega_log_k": values[1]})
    return rows, counts


def jarvis_rows() -> tuple[list[dict], Counter]:
    import hashlib

    if hashlib.sha256(JARVIS.read_bytes()).hexdigest() != JARVIS_SHA256:
        raise SystemExit("the JARVIS file is not the one recorded in the plan")
    archive = zipfile.ZipFile(JARVIS)
    data = json.loads(archive.read(archive.namelist()[0]))
    rows, counts = [], Counter()
    for entry in data:
        counts["entries"] += 1
        if entry.get("stability") != "stable":
            counts["not_marked_stable"] += 1
            continue
        try:
            lam, wlog = float(entry["lamb"]), float(entry["wlog"])
        except (TypeError, ValueError):
            counts["unreadable"] += 1
            continue
        if not usable(lam, wlog):
            counts["lambda_or_omega_out_of_range"] += 1
            continue
        formula = "".join(f"{element}{count}" for element, count in sorted(Counter(entry["atoms"]["elements"]).items()))
        rows.append({"source": "jarvis", "id": entry["jid"], "formula": formula, "lambda": lam, "omega_log_k": wlog})
    return rows, counts


def by_composition(rows: list[dict], valid: set[str]) -> tuple[list[dict], Counter]:
    """One row per composition: structures of the same formula are averaged in logarithms."""
    import build_mp_candidates as M
    import build_scratchnet_data as B

    groups, counts = defaultdict(list), Counter()
    for row in rows:
        fractions = M.fractions(row["formula"])
        if not fractions or not set(fractions) <= valid:
            counts["formula_not_usable"] += 1
            continue
        groups[B.composition_key(fractions)].append(row)
    out = []
    for key, group in sorted(groups.items()):
        lam = math.exp(np.mean([math.log(r["lambda"]) for r in group]))
        wlog = math.exp(np.mean([math.log(r["omega_log_k"]) for r in group]))
        tc = allen_dynes(lam, wlog)
        fractions = C.fractions_of(key)
        out.append({"composition": key, "lambda": round(lam, 5), "omega_log_k": round(wlog, 3), "tc_k": round(tc, 4),
                    "tc_above_1k": int(tc > TC_FLAG_K), "structures": len(group),
                    "sources": "+".join(sorted({r["source"] for r in group})),
                    "element_system": "-".join(sorted(fractions)),
                    "has_f_element": int(bool(set(fractions) & F_BLOCK))})
    return out, counts


def cmd_data() -> dict:
    valid = set(C.load_dataset().table.symbols)
    alexandria, a_counts = alexandria_rows()
    jarvis, j_counts = jarvis_rows()
    table, t_counts = by_composition(alexandria + jarvis, valid)
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "table.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lam = np.array([r["lambda"] for r in table])
    summary = {"alexandria": {"kept": len(alexandria), **a_counts}, "jarvis": {"kept": len(jarvis), **j_counts},
               "compositions": len(table), **t_counts,
               "element_systems": len({r["element_system"] for r in table}),
               "tc_above_1k": int(sum(r["tc_above_1k"] for r in table)),
               "with_an_f_element": int(sum(r["has_f_element"] for r in table)),
               "in_both_sets": int(sum(r["sources"] == "alexandria+jarvis" for r in table)),
               "lambda_quartiles": [round(float(x), 3) for x in np.percentile(lam, [25, 50, 75])]}
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "data.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return summary


# ------------------------------------------------------------------------------------------------ the proxy

def read_table() -> list[dict]:
    with open(WORK / "table.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def targets_of(table: list[dict]) -> np.ndarray:
    return np.array([[math.log(float(r["lambda"])), math.log(float(r["omega_log_k"])), math.log1p(float(r["tc_k"])),
                      float(r["tc_above_1k"])] for r in table])


def fit_models(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, seed: int = 0) -> list:
    """Three regressors (log lambda, log omega_log, log(1 + Tc)) and one classifier (Tc above 1 K)."""
    import children_knowledge as G
    import children_nims as N

    models = [G.fit_regressor(X, Y[:, column], groups, seed=seed) for column in range(3)]
    models.append(N.fit_boost(X, Y[:, 3].astype(int), groups, G.BOOST, seed=seed))
    return models


def apply_models(models: list, X: np.ndarray) -> np.ndarray:
    return np.column_stack([*(m.predict(X) for m in models[:3]), models[3].predict_proba(X)[:, 1]])


def features_of(fractions: list[dict]) -> np.ndarray:
    X, _ = C.formula_features([C.formula_text(f) for f in fractions])
    return X


def proxy_features(fractions: list[dict], table: list[dict] | None = None, leave_out: set[str] | None = None) -> np.ndarray:
    """Cross-fitted proxy outputs: a composition is scored by the model that never saw its element system.

    ``leave_out`` removes compositions from the training table as well (used when a test set shares formulas).
    """
    import rung4 as R

    table = read_table() if table is None else table
    if leave_out:
        table = [r for r in table if r["composition"] not in leave_out]
    Xb = features_of([C.fractions_of(r["composition"]) for r in table])
    Yb = targets_of(table)
    gb = np.array([r["element_system"] for r in table])
    fold_b = np.array([R.system_fold(s) for s in gb])
    Xc = features_of(fractions)
    fold_c = np.array([R.system_fold("-".join(sorted(f))) for f in fractions])
    out = np.full((len(fractions), len(TARGETS)), np.nan)
    for fold in range(5):
        if not (fold_c == fold).any():
            continue
        rows = np.flatnonzero(fold_b != fold)
        out[fold_c == fold] = apply_models(fit_models(Xb[rows], Yb[rows], gb[rows], seed=fold), Xc[fold_c == fold])
    return out


def cmd_p1() -> dict:
    import build_scratchnet_data as B
    import children_nims as N
    import rung4 as R
    from scipy.stats import spearmanr

    table = read_table()
    fractions = [C.fractions_of(r["composition"]) for r in table]
    Y = targets_of(table)
    groups = np.array([r["element_system"] for r in table])
    oof = proxy_features(fractions, table)                       # each row scored without its own element system
    flag = Y[:, 3].astype(int)
    interval = R.group_bootstrap(flag, [oof[:, 3]], groups)[0]
    f_block = np.array([r["has_f_element"] == "1" for r in table])
    p1 = {"compositions": len(table), "tc_above_1k": int(flag.sum()), "element_systems": int(len(set(groups))),
          "auc": round(N.auc(flag, oof[:, 3]), 4), "interval": interval, "pass_mark": 0.70,
          "rank_correlation_lambda": round(float(spearmanr(Y[:, 0], oof[:, 0])[0]), 4),
          "rank_correlation_omega_log": round(float(spearmanr(Y[:, 1], oof[:, 1])[0]), 4),
          "auc_with_an_f_element": round(N.auc(flag[f_block], oof[f_block, 3]), 4) if f_block.sum() > 20 else None,
          "auc_without": round(N.auc(flag[~f_block], oof[~f_block, 3]), 4)}
    p1["passes"] = bool(p1["auc"] >= p1["pass_mark"])

    bercx = R.bercx_rows()
    seen = {}
    for row in bercx:                                            # one row per composition, as the proxy sees them
        key = B.composition_key(row["fractions"])
        seen[key] = max(seen.get(key, 0), row["target"])
    keys = sorted(seen)
    predicted = proxy_features([C.fractions_of(k) for k in keys], table, leave_out=set(keys))
    yb = np.array([seen[k] for k in keys])
    gb = np.array(["-".join(sorted(C.fractions_of(k))) for k in keys])
    shared = len(set(keys) & {r["composition"] for r in table})
    p1b = {"bercx_compositions": len(keys), "flagged_by_bercx": int(yb.sum()),
           "formulas_shared_with_the_new_data_and_left_out_of_training": shared,
           "auc": round(N.auc(yb, predicted[:, 3]), 4), "interval": R.group_bootstrap(yb, [predicted[:, 3]], gb)[0],
           "first_proxy_within_bercx": 0.8248}
    report = {"p1_learns_the_calculation": p1, "p1b_agrees_with_bercx": p1b}
    (DOCS / "p1.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


def cmd_p2() -> dict:
    import build_mp_candidates as M
    import build_scratchnet_data as B
    import children_nims as N
    import rung4 as R

    table = read_table()
    table_b = {r["key"] for r in R.bercx_rows()}
    test = [r for r in R.experimental_rows(R.load_candidates(), table_b) if not r["in_table_b"]]
    keys = [B.composition_key(M.fractions(r["formula"])) for r in test]
    in_new = {r["composition"] for r in table}
    keep = np.array([k not in in_new for k in keys])
    predicted = proxy_features([r["fractions"] for r in test], table)
    y = np.array([r["label"] for r in test])
    groups = np.array([r["system"] for r in test])
    f_block = np.array([bool(set(r["fractions"]) & F_BLOCK) for r in test])

    def block(mask: np.ndarray) -> dict:
        return {"rows": int(mask.sum()), "superconductors": int(y[mask].sum()),
                "auc_tc_flag": round(N.auc(y[mask], predicted[mask, 3]), 4),
                "auc_predicted_lambda": round(N.auc(y[mask], predicted[mask, 0]), 4),
                "interval": R.group_bootstrap(y[mask], [predicted[mask, 3]], groups[mask])[0]}

    report = {"same_882_rows_as_t2": len(test), "left_out_because_the_formula_is_in_the_new_data": int((~keep).sum()),
              "primary": block(keep), "pass_mark": 0.60, "first_proxy_on_these_rows": 0.5309,
              "with_an_f_element": block(keep & f_block), "without": block(keep & ~f_block),
              "all_882_including_shared_formulas": block(np.ones(len(test), dtype=bool))}
    report["passes"] = bool(report["primary"]["auc_tc_flag"] >= report["pass_mark"])
    (DOCS / "p2.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


def nims_features() -> np.ndarray:
    """The proxy's four outputs for every composition of the NIMS table (row order of ``table.json``), cached."""
    import children_knowledge as G

    cache = WORK / "nims-proxy2.npz"
    if not cache.is_file():
        nims = json.loads((G.WORK / "table.json").read_text(encoding="utf-8"))["keys"]
        np.savez_compressed(cache, proxy2=proxy_features([C.fractions_of(k) for k in nims]))
    with np.load(cache) as saved:
        return saved["proxy2"]


def cmd_features() -> None:
    print(json.dumps({"nims_rows": int(len(nims_features())), "columns": list(TARGETS)}))


def cmd_p3() -> dict:
    import children_knowledge as G

    if not os.environ.get("PROXY2_WHOLE_FOLDS"):
        raise SystemExit("superseded (see the amendment in docs/physics-proxy2.md): P3 is scored on the selection "
                         "halves inside `children_evolve.py gen8`, so that the lockbox halves stay unseen.")
    proxy2 = nims_features()
    children = {"proxy": (("proxy",), False), "proxy2": (("proxy",), True),
                "all": (G.CHANGES, False), "all2": (G.CHANGES, True)}
    folds, report = {}, {"folds": {}, "children": {}}
    for fold in G.FOLDS:
        data = G.load(fold)
        swapped = {**data, "blocks": {**data["blocks"], "proxy": proxy2}}
        scores = {"baseline": G.score_child((), data)}
        for name, (changes, new) in children.items():
            scores[name] = G.score_child(tuple(changes), swapped if new else data)
            print(fold, name, round(G.endpoint(data["y_test"], scores[name], data["test_family"], G.PRIMARY_FAMILY), 4),
                  flush=True)
        folds[fold] = {"y": data["y_test"], "groups": data["test_groups"], "family": data["test_family"],
                       "scores": scores}
        report["folds"][fold] = {n: {"primary": round(G.endpoint(data["y_test"], s, data["test_family"], G.PRIMARY_FAMILY), 4),
                                     "all": round(G.endpoint(data["y_test"], s, data["test_family"], None), 4)}
                                 for n, s in scores.items()}
    names = ["baseline", *children]
    intervals = G.fold_bootstrap(folds, names)
    for name in names:
        entry = {"primary": round(float(np.mean([report["folds"][f][name]["primary"] for f in G.FOLDS])), 4),
                 "all": round(float(np.mean([report["folds"][f][name]["all"] for f in G.FOLDS])), 4)}
        if name != "baseline":
            entry.update(intervals[name])
        report["children"][name] = entry
    base = report["children"]["baseline"]
    for name in children:
        report["children"][name]["primary_gain"] = round(report["children"][name]["primary"] - base["primary"], 4)
    versus = {f: {**v, "scores": {"baseline": v["scores"]["all"], "all2": v["scores"]["all2"]}} for f, v in folds.items()}
    report["all2_minus_all"] = {"primary_gain": round(report["children"]["all2"]["primary"]
                                                      - report["children"]["all"]["primary"], 4),
                                **G.fold_bootstrap(versus, ["baseline", "all2"])["all2"]}
    report["reproduces_generations_6_and_7"] = {"baseline": base["primary"] == 0.6963,
                                                "proxy": report["children"]["proxy"]["primary"] == 0.7042,
                                                "all": report["children"]["all"]["primary"] == 0.7516}
    report["p3_passes"] = bool(report["children"]["proxy2"]["primary_gain_99"][0] > 0)
    report["new_proxy_gain_larger_than_the_first"] = bool(report["children"]["proxy2"]["primary_gain"]
                                                          > report["children"]["proxy"]["primary_gain"])
    (DOCS / "p3.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "folds"}, indent=1))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("data", "p1", "p2", "features", "p3"))
    {"data": cmd_data, "p1": cmd_p1, "p2": cmd_p2, "features": cmd_features, "p3": cmd_p3}[parser.parse_args().command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
