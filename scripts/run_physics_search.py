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
    return tc_tree_sd_k > predicted_tc_k


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
    return sorted(selected, key=lambda row: (row["ranking_group"] != "standard", row["group_rank"]))


def generate_substitutions(
    rows: list[dict[str, Any]], max_prototypes: int = 180,
    drop_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    from pymatgen.core import Element

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
    }
    prototypes = sorted(rows, key=lambda row: (str(row.get("stability")), str(row.get("jid"))))[:max_prototypes]
    for prototype in prototypes:
        elements = list(prototype["atoms"]["elements"])
        distinct = sorted(set(elements))
        for source_element in distinct:
            source_group = _group(source_element)
            if source_group is None:
                continue
            replacements = [el.symbol for el in Element if el.group == source_group
                            and el.symbol != source_element and el.symbol in observed_elements]
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
                seen.add(formula)
                if not _charge_balances_with_common_oxidation_states(formula):
                    counts["dropped_no_common_oxidation_state_balance"] += 1
                    continue
                output.append({
                    "candidate_jid": candidate["jid"], "jid": candidate["jid"], "formula": formula,
                    "prototype_jid": prototype["jid"], "prototype_formula": _formula_and_structure(prototype)[0],
                    "source_stability_proxy": prototype.get("stability", "unknown"),
                    "substitution": f"{source_element}->{replacement}",
                    "structure_status": "unrelaxed parent-prototype substitution",
                    "atoms": candidate["atoms"], "lamb": 0.0, "wlog": 1.0, "Tc": 0.0,
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
        low_confidence = is_low_confidence(float(np.std(tc_values)), float(np.mean(tc_values)))
        output.append({
            "formula": cand["formula"], "prototype_jid": cand["prototype_jid"],
            "prototype_formula": cand["prototype_formula"], "substitution": cand["substitution"],
            "predicted_lambda": float(np.mean(lam_values)), "lambda_tree_sd": float(np.std(lam_values)),
            "predicted_omega_log_k": float(np.mean(wlog_values)), "omega_log_tree_sd_k": float(np.std(wlog_values)),
            "predicted_tc_k": float(np.mean(tc_values)), "tc_tree_sd_k": float(np.std(tc_values)),
            "acquisition_ucb_k": float(np.mean(tc_values) + 0.5 * np.std(tc_values)),
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


def _load_raw(path: Path) -> list[dict[str, Any]]:
    import zipfile
    with zipfile.ZipFile(path) as zf:
        return json.load(zf.open("jarvis_epc_data_figshare_1058.json"))


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
    summary = {"candidate_filter_counts": search_counts, "candidates_written": len(result),
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
