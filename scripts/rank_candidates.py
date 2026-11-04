#!/usr/bin/env python3
"""Generate, screen, and rank substitution candidates from a trained model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rt_superconductor.harmonized import HarmonizedRecord, load_harmonized
from rt_superconductor.materials_project import audit_materials_project_queries, normalize_fixed_formula, validated_candidates
from rt_superconductor.search import CandidateHypothesis, CandidateProposal, propose_batch
from rt_superconductor.targets import ATMOSPHERIC_PRESSURE_GPA

from train_pressure_aware import build_features, select_training_records
from validate_ferrite_shortlist import load_measurement_evidence, match_measurement_evidence, _reduced_formula


HULL_CUTOFF_EV_PER_ATOM = 0.05
MAX_METALLIC_BAND_GAP_EV = 0.1
FAMILY_PREDICTION_WEIGHT = 0.80
FAMILY_UNCERTAINTY_WEIGHT = 0.20
TOP_K_COLUMNS = (
    "rank", "candidate_id", "formula", "predicted_tc_kelvin", "uncertainty_k",
    "chemical_family", "energy_above_hull", "band_gap_ev", "is_metal", "mp_id", "mp_exists",
    "3dsc_exists", "3dsc_stanev_exists", "uci_exists", "supercon2_exists", "supermat_exists", "known_in_database",
    "measured_tc_kelvin", "measured_tc_sources", "validation_match_basis",
    "ranking_group", "group_rank", "acquisition_score", "plausibility", "pressure_gpa",
    "reasons", "status", "model_dataset_sha256", "confidence_group", "review_rank", "parent_record_id", "parent_formula", "substitution", "parent_mp_id", "parent_energy_above_hull", "parent_property_role", "insulator_flag", "insulator_reason", "lab_access_flags",
)


def low_confidence(tree_spread_k: float, predicted_tc_k: float) -> bool:
    """Keep only positive predictions with tree spread strictly below half Tc."""
    prediction = float(predicted_tc_k)
    spread = float(tree_spread_k)
    return prediction <= 0 or spread >= 0.5 * prediction


def candidate_confidence_group(tree_spread_k: float, predicted_tc_k: float, *, lab_access_flags: str | list[str] | None = None) -> str:
    """Return the review tier, including explicit lab-access exclusions."""
    return "low_confidence" if low_confidence(tree_spread_k, predicted_tc_k) or bool(lab_access_flags) else "confident"


def _elements(formula: str) -> dict[str, float]:
    from pymatgen.core import Composition

    return {str(element): float(amount) for element, amount in Composition(formula).get_el_amt_dict().items()}


def _candidate_formula(amounts: dict[str, float], old: str, replacement: str) -> str | None:
    from pymatgen.core import Composition
    from rt_superconductor.formula import canonicalize_formula

    changed = dict(amounts)
    amount = changed.pop(old)
    changed[replacement] = changed.get(replacement, 0.0) + amount
    formula = Composition(changed).reduced_formula
    result = canonicalize_formula(formula)
    return result.canonical if result.status == "parsed" else None


def _canonical_formula(value: str) -> str | None:
    from rt_superconductor.formula import canonicalize_formula

    result = canonicalize_formula(value.strip())
    return result.canonical if result.status == "parsed" else None


def _read_formula_set(path: Path | None, fields: tuple[str, ...]) -> set[str] | None:
    """Read formulas from an optional source without inventing unavailable rows."""

    if path is None or not path.is_file():
        return None
    formulas: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            for field in fields:
                value = (row.get(field) or "").strip()
                if not value:
                    continue
                formula = _reduced_formula(value)
                if formula:
                    formulas.add(formula)
                    break
    return formulas


def _default_source_path(*names: str) -> Path | None:
    for name in names:
        path = ROOT / name
        if path.is_file():
            return path
    return None


def _database_flags(
    formula: str,
    records: list[HarmonizedRecord],
    *,
    uci_path: Path | None,
    stanev_path: Path | None,
    supercon2_path: Path | None,
    supermat_path: Path | None,
    mp_exists: bool | None,
    source_sets: dict[str, set[str] | None] | None = None,
) -> dict[str, bool | None]:
    canonical = _reduced_formula(formula) or formula
    if source_sets is None:
        source_sets = {
            "3dsc": {_reduced_formula(record.formula) or record.formula for record in records if record.source_db == "3dsc"},
            "3dsc_stanev": _read_formula_set(stanev_path, ("name",)),
            "uci": _read_formula_set(uci_path, ("material", "formula")),
            "supercon2": _read_formula_set(supercon2_path, ("formula", "rawMaterial", "material")),
            "supermat": _read_formula_set(supermat_path, ("material", "formula")),
        }
    flags: dict[str, bool | None] = {
        "mp_exists": mp_exists,
        "3dsc_exists": canonical in source_sets["3dsc"],
        "3dsc_stanev_exists": None if source_sets["3dsc_stanev"] is None else canonical in source_sets["3dsc_stanev"],
    }
    for source in ("uci", "supercon2", "supermat"):
        values = source_sets[source]
        flags[f"{source}_exists"] = None if values is None else canonical in values
    return flags


def _status_from_flags(flags: dict[str, bool | None]) -> tuple[str, str]:
    known_sources = [name for name, value in flags.items() if value is True]
    unknown_sources = [name for name, value in flags.items() if value is None]
    if known_sources:
        return "known", ", ".join(sorted(known_sources))
    if unknown_sources:
        return "unknown", ", ".join(sorted(unknown_sources))
    return "new", "none of the available databases"


def _text_bool(value: bool | None) -> str:
    return "" if value is None else ("true" if value else "false")


def enumerate_substitution_candidates(
    records: list[HarmonizedRecord], *, family: str | None = None,
    max_bases_per_family: int = 40, max_candidates: int = 3000,
    drop_counts: dict | None = None, structure_index_path: Path | None = None,
) -> list[dict[str, str]]:
    from chemistry_policy import substitution_rejection, uniform_site_coordination
    from collections import Counter
    counts=Counter({key:0 for key in ('considered','hydrogen','family_defining_element','active_layer_element','site_role_mismatch','no_common_oxidation_state','ionic_radii_unavailable','ionic_radius_mismatch','oxide_anion_row_swap','insulating_parent','ionic_charge_imbalance','invalid_formula','known_formula','duplicate','accepted')});by_family=defaultdict(list);observed={r.formula for r in records}
    feature_path=ROOT/'data/processed/structure_features.csv'
    gaps={}
    if feature_path.is_file():
        with feature_path.open() as f:
            for r in csv.DictReader(f):
                try:
                    value=float(r['band_gap_2'])
                    if np.isfinite(value):gaps[r['record_id']]=value
                except (KeyError,ValueError):pass
    structure_index = {}
    index_candidates = [structure_index_path] if structure_index_path else []
    index_candidates.extend(ROOT / value for value in ('data/processed/3dsc-structure-index.csv', 'data/processed/active-learning/3dsc-structure-index.csv', 'data/processed/continuous-training/3dsc-structure-index.csv', 'data/processed/structure-index.csv'))
    index_path = next((path for path in index_candidates if path and path.is_file()), None)
    if index_path:
        with index_path.open(newline='') as handle:
            structure_index = {row['record_id']:row['cif_path'] for row in csv.DictReader(handle)}
    for record in select_training_records(records):
        if family is None or record.chemical_family==family:by_family[record.chemical_family].append(record)
    if max_candidates < 1 or max_bases_per_family < 1:raise ValueError("Candidate and parent limits must be positive")
    candidates={}
    for fam,family_records in sorted(by_family.items()):
        if len(candidates)>=max_candidates:break
        bases=sorted(family_records,key=lambda r:r.record_id)[:max_bases_per_family]
        elements=sorted({el for r in bases for el in _elements(r.formula)})
        for r in bases:
            for old in _elements(r.formula):
                for new in elements:
                    if old==new:continue
                    counts['considered']+=1
                    formula=_candidate_formula(_elements(r.formula),old,new)
                    if not formula:counts['invalid_formula']+=1;continue
                    cif_path=structure_index.get(r.record_id)
                    if cif_path and not Path(cif_path).is_absolute(): cif_path=str(ROOT / cif_path)
                    cn=uniform_site_coordination(str(cif_path),old) if cif_path and Path(cif_path).is_file() else None
                    reason=substitution_rejection(old,new,r.formula,formula,gaps.get(r.record_id),family=fam,coordination_number=cn)
                    if reason:counts[reason]+=1;continue
                    if formula in observed:counts['known_formula']+=1;continue
                    key=(fam,formula)
                    if key in candidates:counts['duplicate']+=1;continue
                    candidates[key]={'formula':formula,'chemical_family':fam,'parent_record_id':r.record_id,'parent_formula':r.formula,'substitution':old+'->'+new,'parent_mp_id':r.mp_id,'parent_energy_above_hull':r.energy_above_hull,
                       'chemistry_reason':f'{old}->{new}; family-defining/active-layer elements preserved; cation/anion role retained; common oxidation state and Shannon-radius mismatch <=15% checked at coordination={cn or "unavailable"}; ionic charge balance passed when applicable; parent reference gap={gaps.get(r.record_id,"unknown")} eV'}
                    counts['accepted']+=1
                    if len(candidates)>=max_candidates:break
                if len(candidates)>=max_candidates:break
            if len(candidates)>=max_candidates:break
    if drop_counts is not None:drop_counts.update(counts)
    return [candidates[k] for k in sorted(candidates)]


def _mp_matches(formulas: list[str], api_key: str | None) -> tuple[dict[str, dict[str, Any]], str, str | None]:
    if not api_key:
        return {}, "degraded_no_mp_api_key", "MP_API_KEY is absent; MP existence, hull, band gap, and metallicity remain unknown"
    try:
        from mp_api.client import MPRester
    except ImportError:
        return {}, "degraded_mp_api_unavailable", "mp-api is not installed; hull filtering was not applied"
    matches: dict[str, dict[str, Any]] = {}
    try:
        with MPRester(api_key) as client:
            audit = audit_materials_project_queries(
                [{"record_id": f"candidate:{index}", "formula": formula} for index, formula in enumerate(sorted(set(formulas)))],
                summary_client=client.materials.summary,
                error_secrets=(api_key,),
            )
        safe_by_formula = validated_candidates(audit)
        for formula in sorted(set(formulas)):
            parsed = normalize_fixed_formula(formula)
            alternatives = safe_by_formula.get(parsed.normalized or "", [])
            if not alternatives:
                continue
            finite_hull = [item for item in alternatives if item.get("energy_above_hull") is not None]
            selected = min(finite_hull, key=lambda item: float(item["energy_above_hull"])) if finite_hull else alternatives[0]
            matches[formula] = {
                "mp_id": selected.get("material_id", ""),
                "band_gap_method": selected.get("band_gap_method") or selected.get("run_type") or selected.get("calc_type"),
                "energy_above_hull": selected.get("energy_above_hull"),
                "band_gap_ev": selected.get("band_gap_ev"),
                "is_metal": None if selected.get("is_metal") is None else bool(selected.get("is_metal")),
                "spacegroup_number": selected.get("spacegroup_number"),
                "mp_alternatives": alternatives,
            }
    except Exception as exc:
        return matches, "degraded_mp_query_error", f"Materials Project query failed: {type(exc).__name__}: {exc}"
    note = (
        "Materials Project composition audit: "
        f"queried={audit['queried_formula_count']}, safe_matches={audit['safe_match_count']}, "
        f"rejected_mismatches={audit['rejected_mismatch_count']}, "
        f"skipped_ambiguous={audit['skipped_ambiguous_formula_count']}, "
        f"api_errors={audit['api_error_count']}"
    )
    return matches, "ready", note


def _write_empty(path: Path, report: dict[str, Any], *additional_paths: Path) -> None:
    for target in (path, *additional_paths):
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(TOP_K_COLUMNS)
    report_path = path.with_name("candidate-ranking-report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_screen_rejections(path: Path, rows: list[dict[str, str]]) -> None:
    """Keep composition/property screen exclusions inspectable beside a shortlist."""
    target = path.with_name(path.stem + "-screen-rejections.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("formula", "chemical_family", "parent_formula", "reason"))
        writer.writeheader()
        writer.writerows(rows)


def model_feature_matrix(matrix, feature_names, artifact):
    """Use the saved formula descriptor schema and its train-fitted imputer."""
    if isinstance(artifact, dict) and artifact.get('feature_names'):
        indices={name:i for i,name in enumerate(feature_names)}
        missing=set(artifact['feature_names'])-set(indices)
        if missing:raise ValueError('Model descriptor schema is unavailable: '+str(sorted(missing)))
        matrix=matrix[:,[indices[name] for name in artifact['feature_names']]]
    if isinstance(artifact, dict) and artifact.get('imputer') is not None:
        matrix=artifact['imputer'].transform(matrix)
    return matrix


def rank_candidates(
    *,
    input_path: Path,
    model_dir: Path,
    output_path: Path,
    batch_size: int = 25,
    max_candidates: int = 3000,
    require_hull: bool = False,
    api_key: str | None = None,
    family: str | None = None,
    uci_path: Path | None = None,
    stanev_path: Path | None = None,
    supercon2_path: Path | None = None,
    supermat_path: Path | None = None,
    validation_output_path: Path | None = None,
    hypotheses_output_path: Path | None = None,
    structure_index_path: Path | None = None,
) -> dict[str, Any]:
    if family is not None:
        family_slug = "".join(character.lower() if character.isalnum() else "_" for character in family).strip("_")
        validation_output_path = validation_output_path or output_path.with_name(f"{family_slug}_validation.csv")
        hypotheses_output_path = hypotheses_output_path or output_path.with_name(f"{family_slug}_hypotheses.csv")
    records = load_harmonized(input_path)
    model_report_path = model_dir / "pressure-aware-report.json"
    model_report = json.loads(model_report_path.read_text(encoding="utf-8")) if model_report_path.is_file() else {}
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if model_report.get("training_status") != "ready":
        report = {
            "status": "blocked_model_not_ready",
            "blocked_reason": model_report.get("blocked_reason", "pressure-aware report is missing or not ready"),
            "dataset_sha256": digest,
            "model_report": str(model_report_path),
        }
        _write_empty(output_path, report, *(path for path in (validation_output_path, hypotheses_output_path) if path))
        return report
    model_path = model_dir / "pressure-aware-model.joblib"
    if not model_path.is_file():
        report = {"status": "blocked_model_missing", "dataset_sha256": digest, "model": str(model_path)}
        _write_empty(output_path, report, *(path for path in (validation_output_path, hypotheses_output_path) if path))
        return report

    if family is not None and not any(record.chemical_family == family for record in select_training_records(records)):
        report = {
            "status": "blocked_family_not_found",
            "family": family,
            "dataset_sha256": digest,
            "model_report": str(model_report_path),
        }
        _write_empty(output_path, report, *(path for path in (validation_output_path, hypotheses_output_path) if path))
        return report

    uci_path = uci_path or _default_source_path("data/raw/uci/extracted/unique_m.csv")
    stanev_path = stanev_path or _default_source_path("data/raw/3dsc/Supercon_data_by_2018_Stanev.csv")
    supercon2_path = supercon2_path or _default_source_path("data/raw/supercon2.csv", "data/raw/supercon2/SuperCon2.csv")
    supermat_path = supermat_path or _default_source_path("data/raw/SuperMat-2.0.csv", "data/raw/supermat/SuperMat-2.0.csv")
    chemistry_counts = {}
    candidates = enumerate_substitution_candidates(records, family=family, max_candidates=max_candidates, drop_counts=chemistry_counts, structure_index_path=structure_index_path)
    matches, mp_status, mp_note = _mp_matches([candidate["formula"] for candidate in candidates], api_key or os.environ.get("MP_API_KEY"))
    measurement_evidence, measurement_source_status = load_measurement_evidence(
        uci_path=uci_path,
        stanev_path=stanev_path,
        supercon2_path=supercon2_path,
        supermat_path=supermat_path,
    )
    source_sets = {
        "3dsc": {_reduced_formula(record.formula) or record.formula for record in records if record.source_db == "3dsc"},
        "3dsc_stanev": _read_formula_set(stanev_path, ("name",)),
        "uci": _read_formula_set(uci_path, ("material", "formula")),
        "supercon2": _read_formula_set(supercon2_path, ("formula", "rawMaterial", "material")),
        "supermat": _read_formula_set(supermat_path, ("material", "formula")),
    }
    from chemistry_policy import insulator_rejection, radioactive_elements
    screened: list[dict[str, Any]] = []
    rejected_hull = 0
    rejected_nonmetal = 0
    rejected_insulator_rows: list[dict[str, str]] = []
    mp_query_available = mp_status == "ready"
    for candidate in candidates:
        match = matches.get(candidate["formula"], {})
        hull = match.get("energy_above_hull")
        if hull is not None and hull >= HULL_CUTOFF_EV_PER_ATOM:
            rejected_hull += 1
            continue
        if require_hull and hull is None:
            rejected_hull += 1
            continue
        band_gap = match.get("band_gap_ev")
        is_metal = match.get("is_metal")
        insulator_reason = insulator_rejection(candidate["formula"], band_gap, match.get("band_gap_method"))
        if insulator_reason:
            rejected_insulator_rows.append({"formula": candidate["formula"], "chemical_family": candidate["chemical_family"], "reason": insulator_reason, "parent_formula": candidate.get("parent_formula", "")})
            continue
        lab_flags = radioactive_elements(candidate["formula"])
        plausibility = max(0.0, 1.0 - float(hull) / HULL_CUTOFF_EV_PER_ATOM) if hull is not None else 0.0
        flags = _database_flags(
            candidate["formula"], records,
            uci_path=uci_path,
            stanev_path=stanev_path,
            supercon2_path=supercon2_path,
            supermat_path=supermat_path,
            mp_exists=(candidate["formula"] in matches) if mp_query_available else None,
            source_sets=source_sets,
        )
        known_status, known_sources = _status_from_flags(flags)
        exact_measurements, _ = match_measurement_evidence(candidate["formula"], measurement_evidence)
        screened.append({
            **candidate, **match, "plausibility": plausibility,
            "insulator_flag": "false", "insulator_reason": "", "lab_access_flags": lab_flags,
            "database_flags": flags, "known_status": known_status, "known_sources": known_sources,
            "exact_measurements": exact_measurements,
        })
    _write_screen_rejections(output_path, rejected_insulator_rows)

    if not screened:
        # No ensemble predictions exist in this branch, so confidence movement
        # is explicitly unavailable rather than reading uninitialized counts.
        report = {
            "status": "blocked_no_plausible_candidates",
            "dataset_sha256": digest,
            "chemistry_filter_counts": chemistry_counts,
            "search_limits": {"max_bases_per_family": 40, "max_candidates": max_candidates},
        "enumerated_candidate_count": len(candidates),
            "rejected_hull_count": rejected_hull,
            "rejected_nonmetal_count": rejected_nonmetal,
        "rejected_insulator_count": len(rejected_insulator_rows),
        "rejected_insulator_rows": rejected_insulator_rows,
            "screen_rejections_csv": str(output_path.with_name(output_path.stem + "-screen-rejections.csv")),
        "confidence_threshold": "uncertainty_k < 0.5 * predicted_tc_kelvin; radioactive rows are low-confidence",
        "old_confidence_threshold": "uncertainty_k < predicted_tc_kelvin",
        "old_low_confidence_count_all_scored": None,
        "new_low_confidence_count_all_scored": None,
        "moved_to_low_confidence_count_all_scored": None,
        "radioactive_candidate_count_all_scored": sum(bool(radioactive_elements(item["formula"])) for item in candidates),
            "mp_status": mp_status,
            "mp_note": mp_note,
            "require_hull": require_hull,
        }
        _write_empty(output_path, report, *(path for path in (validation_output_path, hypotheses_output_path) if path))
        return report

    artifact = joblib.load(model_path)
    model = artifact["model"] if isinstance(artifact, dict) else artifact
    hull_fill = artifact.get("hull_fill") if isinstance(artifact, dict) else None
    candidate_records = [HarmonizedRecord(
        record_id=hashlib.sha256(f"candidate|{item['chemical_family']}|{item['formula']}".encode()).hexdigest()[:24],
        source_db="3dsc",
        formula=item["formula"],
        tc_kelvin=None,  # Feature-only holder: the candidate has no measured Tc.
        applied_pressure_gpa=ATMOSPHERIC_PRESSURE_GPA,
        measurement_method="unknown",
        sample_form="unknown",
        doi="candidate-hypothesis",
        review_status="pending",
        provenance="curated",
        is_synthetic=1,  # Never eligible for empirical fitting.
        chemical_family=item["chemical_family"],
        mp_id=item.get("mp_id", ""),
        energy_above_hull=item.get("energy_above_hull"),
    ) for item in screened]
    feature_matrix, current_feature_names, _ = build_features(candidate_records, hull_fill=hull_fill)
    feature_matrix = model_feature_matrix(feature_matrix, current_feature_names, artifact)
    tree_predictions = np.asarray([tree.predict(feature_matrix) for tree in model.estimators_])
    predicted = tree_predictions.mean(axis=0)
    uncertainty = tree_predictions.std(axis=0)
    hypotheses = []
    for item, prediction, spread in zip(screened, predicted, uncertainty):
        hypotheses.append(CandidateHypothesis(
            candidate_id=hashlib.sha256(f"candidate|{item['chemical_family']}|{item['formula']}".encode()).hexdigest()[:24],
            composition=item["formula"],
            predicted_tc_k=max(0.0, float(prediction)),
            tc_uncertainty_k=max(0.0, float(spread)),
            pressure_gpa=ATMOSPHERIC_PRESSURE_GPA,
            structural_plausibility=float(item["plausibility"]),
            synthesis_feasibility=0.2 if item["plausibility"] > 0 else 0.0,
            novelty=1.0,
            provenance=("model:pressure-aware-random-forest", "status:hypothesis"),
        ))
    # Select the confidence tier before applying the requested output limit.
    old_low_confidence_count = 0
    new_low_confidence_count = 0
    moved_to_low_confidence_count = 0
    radioactive_count = 0
    for item, prediction, spread in zip(screened, predicted, uncertainty):
        old_low = float(prediction) <= 0 or float(spread) >= max(float(prediction), 0.0)
        new_low = candidate_confidence_group(float(spread), max(float(prediction), 0.0), lab_access_flags=item.get("lab_access_flags")) == "low_confidence"
        old_low_confidence_count += int(old_low)
        new_low_confidence_count += int(new_low)
        moved_to_low_confidence_count += int(new_low and not old_low)
        radioactive_count += int(bool(item.get("lab_access_flags")))
    proposals = propose_batch(hypotheses, batch_size=len(hypotheses))
    if family is not None:
        family_proposals: list[CandidateProposal] = []
        for item, hypothesis in zip(screened, hypotheses):
            prediction = hypothesis.predicted_tc_k
            uncertainty_score = min(hypothesis.tc_uncertainty_k / 100.0, 1.0) * item["plausibility"]
            prediction_score = min(max(prediction / 300.0, 0.0), 1.0)
            score = FAMILY_PREDICTION_WEIGHT * prediction_score + FAMILY_UNCERTAINTY_WEIGHT * uncertainty_score
            reasons = (
                f"family mode ({family}): score uses predicted Tc ({FAMILY_PREDICTION_WEIGHT:g}) and capped within-family uncertainty ({FAMILY_UNCERTAINTY_WEIGHT:g})",
                "family mode does not use distance to the 273 K target window",
            )
            family_proposals.append(CandidateProposal(candidate=hypothesis, score=score, reasons=reasons))
        proposals = family_proposals

    def group_order(proposal: CandidateProposal) -> tuple[int, float, str]:
        item = lookup.get(proposal.candidate.candidate_id, {})
        order = {"new": 0, "known": 1, "unknown": 2}.get(item.get("known_status", "unknown"), 2)
        low = candidate_confidence_group(proposal.candidate.tc_uncertainty_k, proposal.candidate.predicted_tc_k, lab_access_flags=item.get("lab_access_flags")) == "low_confidence"
        return (int(low)*10+order), -proposal.score, proposal.candidate.candidate_id

    lookup = {hypothesis.candidate_id: item for hypothesis, item in zip(hypotheses, screened)}
    proposals = sorted(proposals, key=group_order)[:batch_size]
    group_ranks: dict[str, int] = defaultdict(int)
    ranked_rows: list[dict[str, object]] = []
    for rank, proposal in enumerate(proposals, start=1):
        item = lookup[proposal.candidate.candidate_id]
        reasons = list(proposal.reasons)
        reasons.append(item.get("chemistry_reason", "necessary chemistry screens passed; stability not established"))
        radioactive = item.get("lab_access_flags", [])
        low = candidate_confidence_group(proposal.candidate.tc_uncertainty_k, proposal.candidate.predicted_tc_k, lab_access_flags=radioactive) == "low_confidence"
        reasons.append("low confidence: tree spread is at least half predicted Tc or prediction is nonpositive" if low else "confident: tree spread is strictly below half predicted Tc; calibration still required")
        if radioactive:
            reasons.append("radioactive elements flagged: " + ", ".join(radioactive))
        if item.get("energy_above_hull") is None:
            reasons.append("Materials Project hull energy is unknown; stability review is required")
        else:
            reasons.append(f"Materials Project energy above hull={float(item['energy_above_hull']):g} eV/atom")
        if item.get("band_gap_ev") is None:
            reasons.append("Materials Project band gap is unknown; electronic screening requires review")
        else:
            reasons.append(f"Materials Project band gap={float(item['band_gap_ev']):g} eV")
        if item.get("is_metal") is None:
            reasons.append("Materials Project metallicity is unknown; electronic screening requires review")
        elif item["is_metal"]:
            reasons.append("Materials Project marks the lowest-hull entry metallic")
        flags = item["database_flags"]
        if item["known_status"] == "known":
            reasons.append(f"formula is present in {item['known_sources']}")
        elif item["known_status"] == "new":
            reasons.append("formula is absent from all available database snapshots")
        else:
            reasons.append(f"database novelty is unknown for {item['known_sources']}; additional source access is required")
        exact_measurements = item.get("exact_measurements", [])
        measured_tc = None
        measured_sources = ""
        if exact_measurements:
            measured_tc = sum(float(measurement["tc_kelvin"]) for measurement in exact_measurements) / len(exact_measurements)
            measured_sources = ", ".join(sorted({str(measurement["source"]) for measurement in exact_measurements}))
            reasons.append(f"exact measured Tc available from {measured_sources}; validation output")
        else:
            reasons.append("no exact measured compound match in the supplied validation sources; hypothesis output")
        if mp_note:
            reasons.append(mp_note)
        group = item["known_status"]
        group_ranks[group] += 1
        ranked_rows.append({
            "rank": "" if low else rank,
            "confidence_group": "low_confidence" if low else "confident",
            "review_rank": "" if low else rank,
            "candidate_id": proposal.candidate.candidate_id,
            "formula": proposal.candidate.composition,
            "parent_record_id": item.get("parent_record_id", ""),
            "parent_formula": item.get("parent_formula", ""),
            "substitution": item.get("substitution", ""),
            "parent_mp_id": item.get("parent_mp_id", ""),
            "parent_energy_above_hull": item.get("parent_energy_above_hull"),
            "parent_property_role": "reference_parent_proxy_not_candidate",
            "predicted_tc_kelvin": f"{proposal.candidate.predicted_tc_k:.8g}",
            "uncertainty_k": f"{proposal.candidate.tc_uncertainty_k:.8g}",
            "chemical_family": item["chemical_family"],
            "energy_above_hull": "" if item.get("energy_above_hull") is None else f"{float(item['energy_above_hull']):.8g}",
            "band_gap_ev": "" if item.get("band_gap_ev") is None else f"{float(item['band_gap_ev']):.8g}",
            "is_metal": _text_bool(item.get("is_metal")),
            "mp_id": item.get("mp_id", ""),
            "mp_exists": _text_bool(flags.get("mp_exists")),
            "3dsc_exists": _text_bool(flags.get("3dsc_exists")),
            "3dsc_stanev_exists": _text_bool(flags.get("3dsc_stanev_exists")),
            "uci_exists": _text_bool(flags.get("uci_exists")),
            "supercon2_exists": _text_bool(flags.get("supercon2_exists")),
            "supermat_exists": _text_bool(flags.get("supermat_exists")),
            "known_in_database": item["known_status"],
            "measured_tc_kelvin": "" if measured_tc is None else f"{measured_tc:.8g}",
            "measured_tc_sources": measured_sources,
            "validation_match_basis": "exact_reduced_formula" if measured_tc is not None else "",
            "ranking_group": group,
            "group_rank": group_ranks[group],
            "acquisition_score": f"{proposal.score:.8g}",
            "plausibility": f"{item['plausibility']:.8g}",
            "pressure_gpa": ATMOSPHERIC_PRESSURE_GPA,
            "reasons": "; ".join(reasons),
            "status": "review_required_validation" if measured_tc is not None else "review_required_hypothesis",
            "model_dataset_sha256": artifact.get("dataset_sha256", digest) if isinstance(artifact, dict) else digest,
            "insulator_flag": "false", "insulator_reason": "",
            "lab_access_flags": ",".join(radioactive),
        })

    def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TOP_K_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    write_rows(output_path, ranked_rows)
    validation_rows = [row for row in ranked_rows if row["measured_tc_kelvin"] != ""]
    hypothesis_rows = [row for row in ranked_rows if row["measured_tc_kelvin"] == ""]
    if family is not None:
        write_rows(validation_output_path, validation_rows)
        write_rows(hypotheses_output_path, hypothesis_rows)
    report = {
        "status": "ready" if mp_status == "ready" else mp_status,
        "input": str(input_path),
        "dataset_sha256": digest,
        "model": str(model_path),
        "model_dataset_sha256": artifact.get("dataset_sha256", digest) if isinstance(artifact, dict) else digest,
        "chemistry_filter_counts": chemistry_counts,
            "search_limits": {"max_bases_per_family": 40, "max_candidates": max_candidates},
        "enumerated_candidate_count": len(candidates),
        "rejected_hull_count": rejected_hull,
        "rejected_nonmetal_count": rejected_nonmetal,
        "rejected_insulator_count": len(rejected_insulator_rows),
        "rejected_insulator_rows": rejected_insulator_rows,
        "screen_rejections_csv": str(output_path.with_name(output_path.stem + "-screen-rejections.csv")),
        "confidence_threshold": "uncertainty_k < 0.5 * predicted_tc_kelvin; radioactive rows are low-confidence",
        "old_confidence_threshold": "uncertainty_k < predicted_tc_kelvin",
        "old_low_confidence_count_all_scored": old_low_confidence_count,
        "new_low_confidence_count_all_scored": new_low_confidence_count,
        "moved_to_low_confidence_count_all_scored": moved_to_low_confidence_count,
        "radioactive_candidate_count_all_scored": radioactive_count,
        "screened_candidate_count": len(screened),
        "ranked_count": len(proposals),  # Legacy total-output field.
        "written_count": len(ranked_rows),
        "review_ranked_count": sum(row["review_rank"] != "" for row in ranked_rows),
        "low_confidence_count": sum(row["confidence_group"] == "low_confidence" for row in ranked_rows),
        "mp_status": mp_status,
        "mp_note": mp_note,
        "hull_cutoff_ev_per_atom": HULL_CUTOFF_EV_PER_ATOM,
        "require_hull": require_hull,
        "family": family,
        "score_mode": "within_family_predicted_tc_and_uncertainty" if family is not None else "general_acquisition",
        "measurement_source_status": measurement_source_status,
        "validation_output": str(validation_output_path) if family is not None else None,
        "hypotheses_output": str(hypotheses_output_path) if family is not None else None,
        "validation_count": len(validation_rows) if family is not None else 0,
        "hypotheses_count": len(hypothesis_rows) if family is not None else len(ranked_rows),
        "database_source_paths": {
            "uci": str(uci_path) if uci_path else None,
            "3dsc_stanev": str(stanev_path) if stanev_path else None,
            "supercon2": str(supercon2_path) if supercon2_path else None,
            "supermat": str(supermat_path) if supermat_path else None,
        },
        "known_count": sum(item["known_status"] == "known" for item in screened),
        "new_count": sum(item["known_status"] == "new" for item in screened),
        "unknown_database_status_count": sum(item["known_status"] == "unknown" for item in screened),
        "output": str(output_path),
        "interpretation": "Every row is a model hypothesis requiring human review and structure/experimental validation; no row is evidence of superconductivity.",
    }
    output_path.with_name("candidate-ranking-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/training/harmonized_supercon.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/processed/pressure-aware"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/candidates/top_k.csv"))
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-candidates", type=int, default=3000)
    parser.add_argument("--require-hull", action="store_true")
    parser.add_argument("--family", help="restrict substitutions and ranking to one chemical family")
    parser.add_argument("--uci", type=Path, help="optional UCI CSV for database-existence flags")
    parser.add_argument("--stanev", type=Path, help="optional 3DSC Supercon_data_by_2018_Stanev.csv")
    parser.add_argument("--supercon2", type=Path, help="optional SuperCon2 export for database-existence flags")
    parser.add_argument("--supermat", type=Path, help="optional SuperMat export for database-existence flags")
    parser.add_argument("--validation-output", type=Path, help="family-mode CSV for exact measured compounds")
    parser.add_argument("--hypotheses-output", type=Path, help="family-mode CSV for candidates without an exact measured compound")
    parser.add_argument("--structure-index", type=Path, help="optional record_id to CIF path table for site coordination checks")
    parser.add_argument("--api-key", default=os.environ.get("MP_API_KEY"))
    args = parser.parse_args()
    report = rank_candidates(
        input_path=args.input,
        model_dir=args.model_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        max_candidates=args.max_candidates,
        require_hull=args.require_hull,
        api_key=args.api_key,
        family=args.family,
        uci_path=args.uci,
        stanev_path=args.stanev,
        supercon2_path=args.supercon2,
        supermat_path=args.supermat,
        validation_output_path=args.validation_output,
        hypotheses_output_path=args.hypotheses_output,
        structure_index_path=args.structure_index,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
