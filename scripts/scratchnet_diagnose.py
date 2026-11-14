#!/usr/bin/env python3
"""Two experiments on why the from-scratch network trails gradient boosting.

``learning-curve``  Train both on an eighth, a quarter, a half and all of the training groups and test on
                    the same development folds. If the network closes on boosting as data grows, more data
                    would help it; if the gap stays, more data of this kind would not. Only the labelled
                    rows are thinned: a network with the Materials Project side task (hybrid-multitask)
                    gets all of the side-task rows (26,911) at every share, because those are not limited
                    the way SuperCon is.

``orderings``       On held-out rows outside the copper and iron families, compare ordering by the chance
                    of superconducting with ordering by chance x Tc: how many picks superconduct, and how
                    high their Tc is.

``inputs``          Give boosting only what the plain network sees (raw element amounts, or each element's
                    properties with its amount) in place of its 155 hand-made formula features. If boosting
                    falls to the network's level, the gap is the features; if it does not, it is the learner.

Development rows only; the locked test set is not read. Results go to
``docs/scratchnet-learning-curve.json``, ``docs/scratchnet-inputs.json`` and ``docs/scratchnet-orderings.json``.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

import run_scratchnet as R
import scratchnet_common as C
from scratchnet_common import ROOT, T

from rt_superconductor.scratchnet.data import stable_hash

SHARES = (0.125, 0.25, 0.5, 1.0)


def subsample(ds: C.Dataset, rows: np.ndarray, share: float) -> np.ndarray:
    """A share of the training rows: whole groups, chosen by a hash of the group name.

    The one training-only group (a third of the training rows) is thinned row by row instead, so that
    every share keeps the same mix of that group and the rest. A smaller share is always a subset of a
    larger one.
    Only labelled rows pass through here; side-task rows are added later, in full, by the network fit.
    """
    if share >= 1.0:
        return rows
    cut = int(round(share * 1000))
    keep = [stable_hash("curve-row:" + ds.rows[int(row)]["composition"]) % 1000 < cut
            if ds.split[row] == "train_only"
            else stable_hash("curve-group:" + str(ds.arrays.group[row])) % 1000 < cut for row in rows]
    return rows[np.array(keep, dtype=bool)]


def element_blocks(ds: C.Dataset) -> np.ndarray:
    """What the plain network is given, laid out for trees: for each of up to eight elements, most abundant
    first, its amount and its fixed properties. Empty slots are left blank (boosting handles blanks)."""
    arrays = ds.arrays
    blocks = np.concatenate([arrays.fraction[..., None], ds.table.features[arrays.index]], axis=2).astype(float)
    blocks[~arrays.mask] = np.nan
    return blocks.reshape(len(arrays), -1)


def _curve_task(task: tuple[float, int, str, int]) -> dict:
    share, fold, model, seed = task
    ds = R._STATE["ds"]
    fit, valid, test = R.fold_rows(ds, fold)
    fit, valid = subsample(ds, fit, share), subsample(ds, valid, share)
    started = time.time()
    if model == "boosting":
        p, tc, _ = C.boosting_predict(R._STATE["X"], ds.arrays, fit, valid, R._STATE["X"][test], seed=fold,
                                      **R._STATE["boosting"])
    else:
        (p, tc), _ = R.fit_network(model, seed, fit, valid, test)
    print(f"  {model} share {share:g} fold {fold} seed {seed}: {len(fit)} fit rows, {time.time() - started:.0f} s",
          flush=True)
    return {"task": task, "p": p, "tc": tc, "fit_rows": int(len(fit)), "valid_rows": int(len(valid))}


def boosting_settings(out_dir: Path) -> dict:
    """The boosting settings chosen in round 2, so every experiment uses the tuned baseline."""
    chosen = json.loads((out_dir / "scratchnet-round2.json").read_text(encoding="utf-8"))["selection"]["boosting"]
    return {"classifier_settings": chosen["classifier"], "regressor_settings": chosen["regressor"]}


def run_learning_curve(networks: list[str], seeds: list[int], jobs: int, out_dir: Path,
                       shares: tuple[float, ...] = SHARES) -> dict:
    started = time.time()
    ds = R.prepare(networks)
    R._STATE["boosting"] = boosting_settings(out_dir)
    dev = np.flatnonzero(ds.split == "dev")
    position = {int(row): i for i, row in enumerate(dev)}
    folds = sorted(set(ds.fold[dev].tolist()))
    tasks = [(share, fold, "boosting", 0) for share in shares for fold in folds]
    tasks += [(share, fold, network, seed) for network in networks for seed in seeds for share in shares
              for fold in folds]
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        results = pool.map(_curve_task, tasks, chunksize=1)
    predictions: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]] = {}
    rows_used: dict[float, int] = {}
    for model in ["boosting"] + networks:
        predictions[model] = {}
        for share in shares:
            per_seed = []
            for seed in ([0] if model == "boosting" else seeds):
                p_all, tc_all = np.full(len(dev), np.nan), np.full(len(dev), np.nan)
                for result in results:
                    s, fold, m, sd = result["task"]
                    if (s, m, sd) == (share, model, seed):
                        where = np.array([position[int(row)] for row in R.fold_rows(ds, fold)[2]])
                        p_all[where], tc_all[where] = result["p"], result["tc"]
                        if model == "boosting":
                            rows_used[share] = rows_used.get(share, 0) + result["fit_rows"] + result["valid_rows"]
                per_seed.append((p_all, tc_all))
            predictions[model][share] = (np.mean([p for p, _ in per_seed], axis=0),
                                         C.log_mean_tc([tc for _, tc in per_seed]))
    label, tc = ds.arrays.label[dev], ds.arrays.tc_k[dev]
    curve = {}
    for model, by_share in predictions.items():
        curve[model] = {}
        for share, prediction in by_share.items():
            score = T.score(label, tc, *prediction)
            spread = C.intervals(ds, dev, prediction)
            curve[model][f"{share:g}"] = {
                "mean_training_rows_per_fold": rows_used[share] / len(folds),
                "auc": score["auc"], "auc_interval": [spread["auc"]["low"], spread["auc"]["high"]],
                "tc_log_mae": score["tc_log_mae"],
                "tc_log_mae_interval": [spread["tc_log_mae"]["low"], spread["tc_log_mae"]["high"]],
                "expected_tc_spearman": score["expected_tc_spearman"]}
    gaps = {}
    for network in networks:
        gaps[f"network {network} minus boosting"] = {
            f"{share:g}": C.paired_differences(ds, dev, predictions[network][share], predictions["boosting"][share])
            for share in shares}
        # Does the gap close as data grows? The change in the gap from the smallest share to all the data,
        # with an interval from resampling whole test groups.
        first, last = shares[0], shares[-1]
        labelled = np.isfinite(label)

        def closing(sample: np.ndarray) -> float:
            sample = sample[labelled[sample]]

            def gap(share: float) -> float:
                return (T.roc_auc(label[sample], predictions[network][share][0][sample])
                        - T.roc_auc(label[sample], predictions["boosting"][share][0][sample]))

            return gap(last) - gap(first)

        gaps[f"network {network} minus boosting"]["change_in_auc_gap_smallest_to_all"] = C.group_bootstrap(
            ds.arrays.group[dev], closing, resamples=300)
    report = {"data_sha256": ds.sha256, "rows": "development rows, out-of-fold; the locked test set is not read",
              "shares": list(shares), "networks": networks, "seeds": seeds, "folds": folds,
              "boosting_settings": R._STATE["boosting"], "curve": curve, "gaps": gaps,
              "minutes": round((time.time() - started) / 60.0, 1)}
    path = out_dir / "scratchnet-learning-curve.json"
    path.write_text(json.dumps(R.rounded(report), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return report


def run_inputs(out_dir: Path) -> dict:
    started = time.time()
    ds = R.prepare([])
    settings = boosting_settings(out_dir)
    dev = np.flatnonzero(ds.split == "dev")
    position = {int(row): i for i, row in enumerate(dev)}
    folds = sorted(set(ds.fold[dev].tolist()))
    inputs = {"155 hand-made formula features": R._STATE["X"],
              "raw amounts of each element (90 numbers)": ds.dense[:, 1:],
              "each element's amount and properties (8 x 29 numbers)": element_blocks(ds)}
    predictions = {}
    for name, X in inputs.items():
        p_all, tc_all = np.full(len(dev), np.nan), np.full(len(dev), np.nan)
        for fold in folds:
            fit, valid, test = R.fold_rows(ds, fold)
            where = np.array([position[int(row)] for row in test])
            p_all[where], tc_all[where], _ = C.boosting_predict(X, ds.arrays, fit, valid, X[test], seed=fold, **settings)
        predictions[name] = (p_all, tc_all)
        print(f"  boosting on {name}: done", flush=True)
    label, tc = ds.arrays.label[dev], ds.arrays.tc_k[dev]
    reference = "155 hand-made formula features"
    table = {}
    for name, prediction in predictions.items():
        score = T.score(label, tc, *prediction)
        table[name] = {"columns": int(inputs[name].shape[1]), "auc": score["auc"], "tc_log_mae": score["tc_log_mae"],
                       "tc_mae_k": score["tc_mae_k"], "expected_tc_spearman": score["expected_tc_spearman"]}
        if name != reference:
            table[name]["minus_formula_features"] = C.paired_differences(ds, dev, prediction, predictions[reference])
    # The network results for the same rows, from round 2, for the comparison the question is about.
    round2 = json.loads((out_dir / "scratchnet-round2.json").read_text(encoding="utf-8"))["scores"]["all"]
    networks = {name: {"auc": round2[name]["auc"], "tc_log_mae": round2[name]["tc_log_mae"]}
                for name in ("network tuned", "network hybrid", "network multitask", "network hybrid-multitask")
                if name in round2}
    report = {"data_sha256": ds.sha256, "rows": "development rows, out-of-fold; the locked test set is not read",
              "boosting_settings": settings, "boosting_by_input": table, "networks_in_round_2": networks,
              "note": "The plain network ('network tuned') sees each element's amount and properties and nothing "
                      "else; 'hybrid' also gets the formula features; 'multitask' adds the Materials Project side task.",
              "minutes": round((time.time() - started) / 60.0, 1)}
    path = out_dir / "scratchnet-inputs.json"
    path.write_text(json.dumps(R.rounded(report), indent=1) + "\n", encoding="utf-8")
    print(json.dumps(R.rounded(report), indent=1))
    return report


def ordering_table(label: np.ndarray, tc: np.ndarray, groups: np.ndarray, probability: np.ndarray,
                   tc_if: np.ndarray, shares: tuple[float, ...] = (0.05, 0.1, 0.2), resamples: int = 1000,
                   seed: int = 0) -> dict:
    """Two ways to order the same rows: by the chance of superconducting, or by chance x Tc.

    For each share, the picks of both orders are scored on how many superconduct (``hit_rate``) and on the
    measured Tc averaged over the picks with non-superconductors counted as zero (``mean_tc_k``). The
    differences (by chance minus by chance x Tc) carry intervals from resampling whole groups, with the picks
    chosen again inside every resample.
    """
    hit = (label == 1).astype(float)
    realised = np.where(label == 1, np.nan_to_num(tc), 0.0)
    expected = probability * tc_if
    everything = np.arange(len(label))

    def picks(rows: np.ndarray, share: float) -> tuple[np.ndarray, np.ndarray]:
        return C.top_rows(probability, rows, share), C.top_rows(expected, rows, share)

    out: dict = {"rows": int(len(label)), "base_rate": float(hit.mean()) if len(label) else None, "shares": {}}
    if not len(label):
        return out
    for share in shares:
        by_p, by_expected = picks(everything, share)
        entry = {"picks": int(len(by_p)),
                 "by_probability": {"hit_rate": float(hit[by_p].mean()), "mean_tc_k": float(realised[by_p].mean()),
                                    "median_tc_of_hits_k": float(np.median(tc[by_p][label[by_p] == 1]))
                                    if (label[by_p] == 1).any() else None},
                 "by_expected_tc": {"hit_rate": float(hit[by_expected].mean()),
                                    "mean_tc_k": float(realised[by_expected].mean()),
                                    "median_tc_of_hits_k": float(np.median(tc[by_expected][label[by_expected] == 1]))
                                    if (label[by_expected] == 1).any() else None},
                 "picks_in_common": int(len(set(by_p.tolist()) & set(by_expected.tolist())))}

        def gap(values: np.ndarray):
            def statistic(rows: np.ndarray) -> float:
                first, second = picks(rows, share)
                return float(values[first].mean() - values[second].mean())
            return statistic
        entry["hit_rate_difference"] = {"value": entry["by_probability"]["hit_rate"] - entry["by_expected_tc"]["hit_rate"],
                                        **C.group_bootstrap(groups, gap(hit), resamples, seed)}
        entry["mean_tc_difference_k"] = {"value": entry["by_probability"]["mean_tc_k"] - entry["by_expected_tc"]["mean_tc_k"],
                                         **C.group_bootstrap(groups, gap(realised), resamples, seed)}
        out["shares"][f"{share:g}"] = entry
    return out


def run_orderings(name: str, out_dir: Path, resamples: int = 1000) -> dict:
    """On held-out development rows outside the copper and iron families: order by chance, or by chance x Tc?"""
    ds = C.load_dataset()
    saved = np.load(C.WORK / f"{name}-oof.npz", allow_pickle=True)
    rows, distance = saved["rows"], saved["distance"]
    probability, tc_if = saved["p::boosting"], saved["tc::boosting"]
    label, tc, groups = ds.arrays.label[rows], ds.arrays.tc_k[rows], ds.arrays.group[rows]
    copper = C.has_element(ds, rows, "Cu")
    keep = np.isfinite(label) & np.isfinite(probability) & np.isin(ds.family[rows], ["Other", "Oxides"]) & ~copper
    report = {"name": name, "model": "boosting, out-of-fold on the development rows",
              "rows_kept": "labelled, family Other or Oxides, no copper",
              "bands": {}}
    for band, low in (("all distances", 0.0), ("0.1 or more from any training composition", 0.1),
                      ("0.3 or more from any training composition", 0.3)):
        use = keep & (distance >= low)
        report["bands"][band] = ordering_table(label[use], tc[use], groups[use], probability[use], tc_if[use],
                                               resamples=resamples)
    out = out_dir / "scratchnet-orderings.json"
    out.write_text(json.dumps(R.rounded(report), indent=1) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    curve = sub.add_parser("learning-curve", help="network and boosting trained on growing shares of the data")
    curve.add_argument("--networks", default="tuned,hybrid-multitask")
    curve.add_argument("--seeds", default="0,1")
    curve.add_argument("--jobs", type=int, default=4)
    curve.add_argument("--out-dir", type=Path, default=ROOT / "docs")
    inputs = sub.add_parser("inputs", help="boosting on the network's raw inputs instead of formula features")
    inputs.add_argument("--out-dir", type=Path, default=ROOT / "docs")
    orderings = sub.add_parser("orderings", help="order held-out rows by chance or by chance x Tc: what each finds")
    orderings.add_argument("--name", default="round2")
    orderings.add_argument("--out-dir", type=Path, default=ROOT / "docs")
    args = parser.parse_args()
    if args.command == "inputs":
        run_inputs(args.out_dir)
        return 0
    if args.command == "orderings":
        report = run_orderings(args.name, args.out_dir)
        for band, table in report["bands"].items():
            for share, entry in table["shares"].items():
                print(band, share, entry["by_probability"], entry["by_expected_tc"],
                      {k: (round(entry[k]["value"], 3), round(entry[k]["low"], 3), round(entry[k]["high"], 3))
                       for k in ("hit_rate_difference", "mean_tc_difference_k")})
        return 0
    networks = args.networks.split(",")
    unknown = [name for name in networks if name not in R.VARIANTS]
    if unknown:
        raise SystemExit(f"unknown networks {unknown}; known: {sorted(R.VARIANTS)}")
    report = run_learning_curve(networks, [int(s) for s in args.seeds.split(",")], args.jobs, args.out_dir)
    for model, by_share in report["curve"].items():
        print(model, {share: (round(v["auc"], 3), round(v["tc_log_mae"], 3)) for share, v in by_share.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
