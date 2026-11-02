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
from rt_superconductor.search import CandidateHypothesis, propose_batch
from rt_superconductor.targets import ATMOSPHERIC_PRESSURE_GPA

from train_pressure_aware import build_features, select_training_records


HULL_CUTOFF_EV_PER_ATOM = 0.05
TOP_K_COLUMNS = (
    "rank", "candidate_id", "formula", "predicted_tc_kelvin", "uncertainty_k",
    "chemical_family", "energy_above_hull", "mp_id", "acquisition_score",
    "plausibility", "pressure_gpa", "reasons", "status", "model_dataset_sha256",
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


def enumerate_substitution_candidates(
    records: list[HarmonizedRecord], *, max_bases_per_family: int = 40, max_candidates: int = 3000
) -> list[dict[str, str]]:
    """Enumerate deterministic same-family elemental substitutions with pymatgen."""

    by_family: dict[str, list[HarmonizedRecord]] = defaultdict(list)
    observed = {record.formula for record in records}
    for record in select_training_records(records):
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
        return {}, "degraded_no_mp_api_key", "MP_API_KEY is absent; hull filtering was not applied"
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
                        fields=["material_id", "energy_above_hull", "symmetry"],
                    )
                    usable = [document for document in documents if getattr(document, "energy_above_hull", None) is not None]
                    if not usable:
                        continue
                    selected = min(usable, key=lambda document: float(document.energy_above_hull))
                    symmetry = getattr(selected, "symmetry", None)
                    matches[formula] = {
                        "mp_id": str(getattr(selected, "material_id", "")),
                        "energy_above_hull": float(selected.energy_above_hull),
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

    candidates = enumerate_substitution_candidates(records, max_candidates=max_candidates)
    matches, mp_status, mp_note = _mp_matches([candidate["formula"] for candidate in candidates], api_key or os.environ.get("MP_API_KEY"))
    screened: list[dict[str, Any]] = []
    rejected_hull = 0
    for candidate in candidates:
        match = matches.get(candidate["formula"], {})
        hull = match.get("energy_above_hull")
        if hull is not None and hull >= HULL_CUTOFF_EV_PER_ATOM:
            rejected_hull += 1
            continue
        if require_hull and hull is None:
            rejected_hull += 1
            continue
        plausibility = max(0.0, 1.0 - float(hull) / HULL_CUTOFF_EV_PER_ATOM) if hull is not None else 0.0
        screened.append({**candidate, **match, "plausibility": plausibility})

    if not screened:
        report = {
            "status": "blocked_no_plausible_candidates",
            "dataset_sha256": digest,
            "enumerated_candidate_count": len(candidates),
            "rejected_hull_count": rejected_hull,
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
    lookup = {hypothesis.candidate_id: item for hypothesis, item in zip(hypotheses, screened)}
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
            if mp_note:
                reasons.append(mp_note)
            writer.writerow({
                "rank": rank,
                "candidate_id": proposal.candidate.candidate_id,
                "formula": proposal.candidate.composition,
                "predicted_tc_kelvin": f"{proposal.candidate.predicted_tc_k:.8g}",
                "uncertainty_k": f"{proposal.candidate.tc_uncertainty_k:.8g}",
                "chemical_family": item["chemical_family"],
                "energy_above_hull": "" if item.get("energy_above_hull") is None else f"{float(item['energy_above_hull']):.8g}",
                "mp_id": item.get("mp_id", ""),
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
        "screened_candidate_count": len(screened),
        "ranked_count": len(proposals),
        "mp_status": mp_status,
        "mp_note": mp_note,
        "hull_cutoff_ev_per_atom": HULL_CUTOFF_EV_PER_ATOM,
        "require_hull": require_hull,
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
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
