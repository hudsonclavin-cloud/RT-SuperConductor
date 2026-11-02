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
from rt_superconductor.search import CandidateHypothesis, CandidateProposal, propose_batch
from rt_superconductor.targets import ATMOSPHERIC_PRESSURE_GPA

from train_pressure_aware import build_features, select_training_records


HULL_CUTOFF_EV_PER_ATOM = 0.05
MAX_METALLIC_BAND_GAP_EV = 0.1
FAMILY_PREDICTION_WEIGHT = 0.80
FAMILY_UNCERTAINTY_WEIGHT = 0.20
TOP_K_COLUMNS = (
    "rank", "candidate_id", "formula", "predicted_tc_kelvin", "uncertainty_k",
    "chemical_family", "energy_above_hull", "band_gap_ev", "is_metal", "mp_id", "mp_exists",
    "3dsc_exists", "uci_exists", "supercon2_exists", "supermat_exists", "known_in_database",
    "ranking_group", "group_rank", "acquisition_score", "plausibility", "pressure_gpa",
    "reasons", "status", "model_dataset_sha256",
)


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
                formula = _canonical_formula(value)
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
    supercon2_path: Path | None,
    supermat_path: Path | None,
    mp_exists: bool | None,
    source_sets: dict[str, set[str] | None] | None = None,
) -> dict[str, bool | None]:
    canonical = _canonical_formula(formula) or formula
    if source_sets is None:
        source_sets = {
            "3dsc": {record.formula for record in records if record.source_db == "3dsc"},
            "uci": _read_formula_set(uci_path, ("material", "formula")),
            "supercon2": _read_formula_set(supercon2_path, ("formula", "rawMaterial", "material")),
            "supermat": _read_formula_set(supermat_path, ("material", "formula")),
        }
    flags: dict[str, bool | None] = {
        "mp_exists": mp_exists,
        "3dsc_exists": canonical in source_sets["3dsc"],
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
    max_bases_per_family: int = 40, max_candidates: int = 3000
) -> list[dict[str, str]]:
    """Enumerate deterministic same-family elemental substitutions with pymatgen."""

    by_family: dict[str, list[HarmonizedRecord]] = defaultdict(list)
    observed = {record.formula for record in records}
    for record in select_training_records(records):
        if family is not None and record.chemical_family != family:
            continue
        by_family[record.chemical_family].append(record)
    candidates: dict[tuple[str, str], dict[str, str]] = {}
    for family in sorted(by_family):
        family_records = sorted(by_family[family], key=lambda record: record.record_id)[:max_bases_per_family]
        family_elements = sorted({element for record in family_records for element in _elements(record.formula)})
        for record in family_records:
            amounts = _elements(record.formula)
            for old in sorted(amounts):
                try:
                    from pymatgen.core.periodic_table import Element

                    old_group = Element(old).group
                except Exception:
                    old_group = None
                for replacement in family_elements:
                    if replacement == old:
                        continue
                    try:
                        replacement_group = Element(replacement).group
                    except Exception:
                        replacement_group = None
                    if old_group is not None and replacement_group != old_group:
                        continue
                    formula = _candidate_formula(amounts, old, replacement)
                    if not formula or formula in observed:
                        continue
                    key = (family, formula)
                    candidates[key] = {
                        "formula": formula,
                        "chemical_family": family,
                        "parent_record_id": record.record_id,
                    }
                    if len(candidates) >= max_candidates:
                        return [candidates[key] for key in sorted(candidates)]
    return [candidates[key] for key in sorted(candidates)]


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
            for formula in sorted(set(formulas)):
                try:
                    documents = client.materials.summary.search(
                        formula=formula,
                        fields=["material_id", "energy_above_hull", "symmetry", "band_gap", "is_metal"],
                    )
                    usable = list(documents)
                    if not usable:
                        continue
                    with_hull = [document for document in usable if getattr(document, "energy_above_hull", None) is not None]
                    selected = min(with_hull, key=lambda document: float(document.energy_above_hull)) if with_hull else usable[0]
                    symmetry = getattr(selected, "symmetry", None)
                    band_gap = getattr(selected, "band_gap", None)
                    is_metal = getattr(selected, "is_metal", None)
                    matches[formula] = {
                        "mp_id": str(getattr(selected, "material_id", "")),
                        "energy_above_hull": None if getattr(selected, "energy_above_hull", None) is None else float(selected.energy_above_hull),
                        "band_gap_ev": None if band_gap is None else float(band_gap),
                        "is_metal": None if is_metal is None else bool(is_metal),
                        "spacegroup_number": getattr(symmetry, "number", None),
                    }
                except Exception:
                    continue
    except Exception as exc:
        return matches, "degraded_mp_query_error", f"Materials Project query failed: {type(exc).__name__}: {exc}"
    return matches, "ready", None


def _write_empty(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(TOP_K_COLUMNS)
    report_path = path.with_name("candidate-ranking-report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    supercon2_path: Path | None = None,
    supermat_path: Path | None = None,
) -> dict[str, Any]:
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
        _write_empty(output_path, report)
        return report
    model_path = model_dir / "pressure-aware-model.joblib"
    if not model_path.is_file():
        report = {"status": "blocked_model_missing", "dataset_sha256": digest, "model": str(model_path)}
        _write_empty(output_path, report)
        return report

    if family is not None and not any(record.chemical_family == family for record in select_training_records(records)):
        report = {
            "status": "blocked_family_not_found",
            "family": family,
            "dataset_sha256": digest,
            "model_report": str(model_report_path),
        }
        _write_empty(output_path, report)
        return report

    uci_path = uci_path or _default_source_path("data/raw/uci/extracted/unique_m.csv")
    supercon2_path = supercon2_path or _default_source_path("data/raw/supercon2.csv", "data/raw/supercon2/SuperCon2.csv")
    supermat_path = supermat_path or _default_source_path("data/raw/SuperMat-2.0.csv", "data/raw/supermat/SuperMat-2.0.csv")
    candidates = enumerate_substitution_candidates(records, family=family, max_candidates=max_candidates)
    matches, mp_status, mp_note = _mp_matches([candidate["formula"] for candidate in candidates], api_key or os.environ.get("MP_API_KEY"))
    source_sets = {
        "3dsc": {record.formula for record in records if record.source_db == "3dsc"},
        "uci": _read_formula_set(uci_path, ("material", "formula")),
        "supercon2": _read_formula_set(supercon2_path, ("formula", "rawMaterial", "material")),
        "supermat": _read_formula_set(supermat_path, ("material", "formula")),
    }
    screened: list[dict[str, Any]] = []
    rejected_hull = 0
    rejected_nonmetal = 0
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
        if (band_gap is not None and band_gap > MAX_METALLIC_BAND_GAP_EV) or is_metal is False:
            rejected_nonmetal += 1
            continue
        plausibility = max(0.0, 1.0 - float(hull) / HULL_CUTOFF_EV_PER_ATOM) if hull is not None else 0.0
        flags = _database_flags(
            candidate["formula"], records,
            uci_path=uci_path,
            supercon2_path=supercon2_path,
            supermat_path=supermat_path,
            mp_exists=(candidate["formula"] in matches) if mp_query_available else None,
            source_sets=source_sets,
        )
        known_status, known_sources = _status_from_flags(flags)
        screened.append({
            **candidate, **match, "plausibility": plausibility,
            "database_flags": flags, "known_status": known_status, "known_sources": known_sources,
        })

    if not screened:
        report = {
            "status": "blocked_no_plausible_candidates",
            "dataset_sha256": digest,
            "enumerated_candidate_count": len(candidates),
            "rejected_hull_count": rejected_hull,
            "rejected_nonmetal_count": rejected_nonmetal,
            "mp_status": mp_status,
            "mp_note": mp_note,
            "require_hull": require_hull,
        }
        _write_empty(output_path, report)
        return report

    artifact = joblib.load(model_path)
    model = artifact["model"] if isinstance(artifact, dict) else artifact
    hull_fill = artifact.get("hull_fill") if isinstance(artifact, dict) else None
    candidate_records = [HarmonizedRecord(
        record_id=hashlib.sha256(f"candidate|{item['chemical_family']}|{item['formula']}".encode()).hexdigest()[:24],
        source_db="3dsc",
        formula=item["formula"],
        tc_kelvin=0.0,
        applied_pressure_gpa=ATMOSPHERIC_PRESSURE_GPA,
        measurement_method="unknown",
        sample_form="unknown",
        doi="candidate-hypothesis",
        review_status="pending",
        provenance="curated",
        is_synthetic=0,
        chemical_family=item["chemical_family"],
        mp_id=item.get("mp_id", ""),
        energy_above_hull=item.get("energy_above_hull"),
    ) for item in screened]
    feature_matrix, _, _ = build_features(candidate_records, hull_fill=hull_fill)
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
    proposals = propose_batch(hypotheses, batch_size=batch_size)
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
        return order, -proposal.score, proposal.candidate.candidate_id

    lookup = {hypothesis.candidate_id: item for hypothesis, item in zip(hypotheses, screened)}
    proposals = sorted(proposals, key=group_order)[:batch_size]
    group_ranks: dict[str, int] = defaultdict(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TOP_K_COLUMNS)
        writer.writeheader()
        for rank, proposal in enumerate(proposals, start=1):
            item = lookup[proposal.candidate.candidate_id]
            reasons = list(proposal.reasons)
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
            if mp_note:
                reasons.append(mp_note)
            group = item["known_status"]
            group_ranks[group] += 1
            writer.writerow({
                "rank": rank,
                "candidate_id": proposal.candidate.candidate_id,
                "formula": proposal.candidate.composition,
                "predicted_tc_kelvin": f"{proposal.candidate.predicted_tc_k:.8g}",
                "uncertainty_k": f"{proposal.candidate.tc_uncertainty_k:.8g}",
                "chemical_family": item["chemical_family"],
                "energy_above_hull": "" if item.get("energy_above_hull") is None else f"{float(item['energy_above_hull']):.8g}",
                "band_gap_ev": "" if item.get("band_gap_ev") is None else f"{float(item['band_gap_ev']):.8g}",
                "is_metal": _text_bool(item.get("is_metal")),
                "mp_id": item.get("mp_id", ""),
                "mp_exists": _text_bool(flags.get("mp_exists")),
                "3dsc_exists": _text_bool(flags.get("3dsc_exists")),
                "uci_exists": _text_bool(flags.get("uci_exists")),
                "supercon2_exists": _text_bool(flags.get("supercon2_exists")),
                "supermat_exists": _text_bool(flags.get("supermat_exists")),
                "known_in_database": item["known_status"],
                "ranking_group": group,
                "group_rank": group_ranks[group],
                "acquisition_score": f"{proposal.score:.8g}",
                "plausibility": f"{item['plausibility']:.8g}",
                "pressure_gpa": ATMOSPHERIC_PRESSURE_GPA,
                "reasons": "; ".join(reasons),
                "status": "review_required_hypothesis",
                "model_dataset_sha256": artifact.get("dataset_sha256", digest) if isinstance(artifact, dict) else digest,
            })
    report = {
        "status": "ready" if mp_status == "ready" else mp_status,
        "input": str(input_path),
        "dataset_sha256": digest,
        "model": str(model_path),
        "model_dataset_sha256": artifact.get("dataset_sha256", digest) if isinstance(artifact, dict) else digest,
        "enumerated_candidate_count": len(candidates),
        "rejected_hull_count": rejected_hull,
        "rejected_nonmetal_count": rejected_nonmetal,
        "screened_candidate_count": len(screened),
        "ranked_count": len(proposals),
        "mp_status": mp_status,
        "mp_note": mp_note,
        "hull_cutoff_ev_per_atom": HULL_CUTOFF_EV_PER_ATOM,
        "require_hull": require_hull,
        "family": family,
        "score_mode": "within_family_predicted_tc_and_uncertainty" if family is not None else "general_acquisition",
        "database_source_paths": {
            "uci": str(uci_path) if uci_path else None,
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
    parser.add_argument("--supercon2", type=Path, help="optional SuperCon2 export for database-existence flags")
    parser.add_argument("--supermat", type=Path, help="optional SuperMat export for database-existence flags")
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
        supercon2_path=args.supercon2,
        supermat_path=args.supermat,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
