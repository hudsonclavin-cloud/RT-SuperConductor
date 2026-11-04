#!/usr/bin/env python3
"""Rank unrelaxed prototype substitutions with the EPC target models.

Every generated row is a hypothesis. Stability values are exact-formula MP
data only when available; the known prototype's source stability is kept in a
separate proxy field. DFT verification is always marked pending.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from train_physics_targets import (
    DATASET_FILE, _formula_and_structure, _make_matrix, _write_csv,
    allen_dynes_tc_k, build_characteristics, fit_physics_models,
)


def physics_point_prediction(lambda_trees, omega_log_trees):
    lam=float(np.mean(lambda_trees));omega=float(np.exp(np.mean(np.log(omega_log_trees))))
    return lam,omega,allen_dynes_tc_k(lam,omega)


def _group(element: str) -> int | None:
    from pymatgen.core import Element
    return Element(element).group


def _same_metal_nonmetal_class(left: str, right: str) -> bool:
    from pymatgen.core import Element

    a, b = Element(left), Element(right)
    # Metalloids are kept as their own class and can substitute only for other
    # metalloids. This also avoids silently treating them as nonmetals.
    category = lambda element: "metal" if element.is_metal else "metalloid" if element.is_metalloid else "nonmetal"
    return category(a) == category(b)


def _charge_balances_with_common_oxidation_states(formula: str) -> bool:
    from pymatgen.core import Composition

    try:
        return bool(Composition(formula).oxi_state_guesses(all_oxi_states=False))
    except (ValueError, TypeError):
        return False


def is_low_confidence(tc_tree_sd_k: float, predicted_tc_k: float) -> bool:
    return tc_tree_sd_k >= predicted_tc_k


def select_by_confidence_group(output: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Assign independent ranks and reserve top-k slots for both confidence tiers."""

    groups = {name: [] for name in ("standard", "low_confidence")}
    for row in output:
        groups[row["ranking_group"]].append(row)
    for grouped in groups.values():
        grouped.sort(key=lambda row: (-row["acquisition_ucb_k"], -row["predicted_tc_k"], row["formula"]))
        for index, row in enumerate(grouped, 1):
            row["group_rank"] = index
    standard_budget = (top_k + 1) // 2
    low_budget = top_k // 2
    selected = groups["standard"][:standard_budget] + groups["low_confidence"][:low_budget]
    remaining = top_k - len(selected)
    if remaining:
        selected_ids = {row["formula"] for row in selected}
        remaining_rows = sorted(
            (row for row in output if row["formula"] not in selected_ids),
            key=lambda row: (-row["acquisition_ucb_k"], -row["predicted_tc_k"], row["formula"]),
        )
        selected.extend(remaining_rows[:remaining])
    for row in selected:
        row["review_rank"] = row["group_rank"] if row["ranking_group"] == "standard" else ""
    return sorted(selected, key=lambda row: (row["ranking_group"] != "standard", row["group_rank"]))


def generate_substitutions(
    rows: list[dict[str, Any]], max_prototypes: int = 180,
    drop_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    from pymatgen.core import Element
    from chemistry_policy import substitution_rejection

    known = {_formula_and_structure(source)[0] for source in rows}
    observed_elements = {str(element) for row in rows for element in row["atoms"]["elements"]}
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts = {
        "proposals_considered": 0,
        "dropped_hydrogen_substitution": 0,
        "dropped_metal_nonmetal_class_swap": 0,
        "dropped_no_common_oxidation_state_balance": 0,
        "dropped_known_formula": 0,
        "dropped_duplicate_formula": 0,
        "accepted": 0,
        "dropped_group_or_oxidation_mismatch": 0,
        "dropped_insulating_parent": 0,
        "dropped_ionic_charge_imbalance": 0,
    }
    prototypes = sorted(rows, key=lambda row: (str(row.get("stability")), str(row.get("jid"))))[:max_prototypes]
    for prototype in prototypes:
        elements = list(prototype["atoms"]["elements"])
        distinct = sorted(set(elements))
        for source_element in distinct:
            source_group = _group(source_element)
            replacements = [el.symbol for el in Element if el.symbol != source_element and el.symbol in observed_elements]
            for replacement in replacements:
                counts["proposals_considered"] += 1
                if source_element == "H" or replacement == "H":
                    counts["dropped_hydrogen_substitution"] += 1
                    continue
                if not _same_metal_nonmetal_class(source_element, replacement):
                    counts["dropped_metal_nonmetal_class_swap"] += 1
                    continue
                changed = [replacement if element == source_element else element for element in elements]
                candidate = dict(prototype)
                candidate["atoms"] = dict(prototype["atoms"], elements=changed)
                candidate["jid"] = f"candidate-{prototype['jid']}-{source_element}-{replacement}"
                formula = _formula_and_structure(candidate)[0]
                if formula in known:
                    counts["dropped_known_formula"] += 1
                    continue
                if formula in seen:
                    counts["dropped_duplicate_formula"] += 1
                    continue
                reason = substitution_rejection(source_element, replacement, _formula_and_structure(prototype)[0], formula, prototype.get("band_gap", prototype.get("bandgap")))
                if reason:
                    counts["dropped_"+reason] = counts.get("dropped_"+reason, 0) + 1
                    if reason == 'ionic_charge_imbalance':counts["dropped_no_common_oxidation_state_balance"] += 1
                    continue
                seen.add(formula)
                output.append({
                    "candidate_jid": candidate["jid"], "jid": candidate["jid"], "formula": formula,
                    "prototype_jid": prototype["jid"], "prototype_formula": _formula_and_structure(prototype)[0],
                    "source_stability_proxy": prototype.get("stability", "unknown"),
                    "parent_band_gap_ev": prototype.get("band_gap", prototype.get("bandgap")),
                    "parent_gap_source": prototype.get("parent_gap_source", "not_available"),
                    "substitution": f"{source_element}->{replacement}",
                    "structure_status": "unrelaxed parent-prototype substitution",
                    "atoms": candidate["atoms"], "lamb": None, "wlog": None, "Tc": None, "is_generated_candidate": True,
                    "press": "unknown",
                })
    counts["accepted"] = len(output)
    if drop_counts is not None:
        drop_counts.clear()
        drop_counts.update(counts)
    return output


def rank_candidates(
    source_rows: list[dict[str, Any]], models: dict[str, Any],
    feature_names: list[str], *, mp_candidate_cache: Path | None = None,
    top_k: int = 20, drop_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    candidates = generate_substitutions(source_rows, drop_counts=drop_counts)
    if not candidates:
        return []
    characteristics, _, _ = build_characteristics(candidates)
    X = _make_matrix(characteristics, feature_names)
    X = models["imputer"].transform(X)
    lam_by_tree = np.asarray([tree.predict(X) for tree in models["lambda"].estimators_])
    wlog_by_tree = np.exp(np.asarray([tree.predict(X) for tree in models["log_omega_log_k"].estimators_]))
    tc_by_tree = np.empty_like(lam_by_tree)
    for tree_index in range(len(lam_by_tree)):
        tc_by_tree[tree_index] = [allen_dynes_tc_k(lam, wlog) for lam, wlog in zip(
            lam_by_tree[tree_index], wlog_by_tree[tree_index]
        )]
    # An upper-confidence acquisition score deliberately balances high mean Tc
    # against model spread. It is a screening priority, not a calibrated bound.
    output = []
    for idx, cand in enumerate(candidates):
        lam_values = lam_by_tree[:, idx]
        wlog_values = wlog_by_tree[:, idx]
        tc_values = tc_by_tree[:, idx]
        mean_lambda, point_omega, point_tc = physics_point_prediction(lam_values, wlog_values)
        low_confidence = is_low_confidence(float(np.std(tc_values)), point_tc)
        output.append({
            "candidate_id": cand["candidate_jid"], "chemical_family": "Phonon-mediated prototype",
            "formula": cand["formula"], "prototype_jid": cand["prototype_jid"],
            "reasons": f"H unchanged; metal/nonmetal/metalloid class preserved; same group or overlapping common oxidation state; ionic charge balance screened when applicable; reference-parent gap={cand.get('parent_band_gap_ev', 'unknown')} eV (not a candidate property); candidate gap and stability require DFT; unrelaxed geometry; tree spread not calibrated",
            "parent_band_gap_ev": cand.get("parent_band_gap_ev"),
            "parent_gap_source": cand.get("parent_gap_source"),
            "review_rank": "" if low_confidence else 1,
            "confidence_group": "low_confidence" if low_confidence else "confident",
            "tc_estimator": "Allen-Dynes at mean lambda and geometric-mean omega_log; tree spread is an uncalibrated diagnostic",
            "prototype_formula": cand["prototype_formula"], "substitution": cand["substitution"],
            "predicted_lambda": mean_lambda, "lambda_tree_sd": float(np.std(lam_values)),
            "predicted_omega_log_k": point_omega, "omega_log_tree_sd_k": float(np.std(wlog_values)),
            "predicted_tc_k": point_tc, "tc_tree_sd_k": float(np.std(tc_values)),
            "acquisition_ucb_k": float(point_tc + 0.5 * np.std(tc_values)),
            "ranking_group": "low_confidence" if low_confidence else "standard",
            "prototype_stability_proxy": cand["source_stability_proxy"],
            "candidate_mp_energy_above_hull_ev_atom": "", "candidate_mp_stability_status": "not_provided",
            "candidate_stability": "unknown; requires MP phase stability and DFT relaxation",
            "structure_status": cand["structure_status"], "dft_verification": "pending",
        })
    if mp_candidate_cache and mp_candidate_cache.is_file():
        with mp_candidate_cache.open(newline="", encoding="utf-8-sig") as handle:
            cache = {row["formula"]: row for row in csv.DictReader(handle)}
        for row in output:
            match = cache.get(row["formula"])
            if match:
                row["candidate_mp_energy_above_hull_ev_atom"] = match.get("energy_above_hull_ev_atom", "")
                row["candidate_mp_stability_status"] = match.get("status", "exact_composition_match_requires_structure_review")
                row["candidate_stability"] = (
                    "unknown; Materials Project query unavailable"
                    if str(match.get("status", "")).startswith(("api_unavailable", "blocked_"))
                    else "Materials Project exact-composition result; phase correspondence must be checked"
                )
    # Keep both confidence tiers visible in the top-20 package when both have
    # candidates, reserving half the slots for each and backfilling shortages.
    return select_by_confidence_group(output, top_k)


def _load_raw(path: Path, parent_properties: Path | None = None) -> list[dict[str, Any]]:
    import zipfile
    with zipfile.ZipFile(path) as zf:
        rows=json.load(zf.open("jarvis_epc_data_figshare_1058.json"))
    parent_properties=parent_properties or ROOT/'data/raw/jarvis-supercond/mp-characteristics.csv'
    if parent_properties.is_file():
        with parent_properties.open() as f:properties={r['jid']:r for r in csv.DictReader(f)}
        for r in rows:
            d=properties.get(str(r['jid']),{});gap=d.get('mp_band_gap_ev')
            if gap not in (None,''):
                value=float(gap)
                if math.isfinite(value):
                    r['band_gap']=value;r['parent_gap_source']='MP exact-composition reference; phase correspondence remains a proxy'
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-zip", type=Path, default=Path("data/raw/jarvis-supercond") / DATASET_FILE)
    parser.add_argument("--model", type=Path, default=Path("data/processed/physics-target/electron-phonon-target-models.joblib"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/physics-target/top-20-prototype-substitutions.csv"))
    parser.add_argument("--mp-candidate-cache", type=Path, default=Path("data/raw/jarvis-supercond/mp-candidate-stability.csv"))
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    rows = _load_raw(args.data_zip)
    models = joblib.load(args.model)
    search_counts: dict[str, int] = {}
    result = rank_candidates(rows, models, models["feature_names"], mp_candidate_cache=args.mp_candidate_cache,
                             top_k=args.top_k, drop_counts=search_counts)
    _write_csv(args.output, result)
    counts_by_group = {group: sum(row["ranking_group"] == group for row in result)
                       for group in ("standard", "low_confidence")}
    summary = {"candidate_filter_counts": search_counts,
               "counter_aliases": {"dropped_no_common_oxidation_state_balance": "dropped_ionic_charge_imbalance"},
               "search_limits": {"max_prototypes": 180, "source_prototypes": len(rows)}, "candidates_written": len(result),
               "written_by_confidence_group": counts_by_group, "output": str(args.output),
                      "all_dft_verification_pending": True,
                      "top_predicted_tc_k": max((row["predicted_tc_k"] for row in result), default=None)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).with_name("physics-search-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
