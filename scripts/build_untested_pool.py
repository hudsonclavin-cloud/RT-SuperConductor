#!/usr/bin/env python3
"""Build and score an MP pool of stable metallic compounds absent from known SC data.

All Materials Project queries are intended to run in GitHub Actions with
``MP_API_KEY``. A candidate is called "existing but untested" only relative to
the five named data exports; this does not establish that no paper or
measurement exists elsewhere.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rt_superconductor.harmonized import HarmonizedRecord, load_harmonized
from train_pressure_aware import build_features, select_training_records, _model


HULL_MAX_EV_PER_ATOM = 0.025
MAX_SITES = 20
METALLIC_GAP_MAX_EV = 1e-8
MIN_EMBARGOED_SPEARMAN = 0.5
DEFAULT_TOP_PER_FAMILY = 50
ALLOWED_RUN_TYPES = {"ggau", "r2scan", "r2scanu"}
OUTPUT_COLUMNS = (
    "rank_within_family", "mp_id", "formula", "predicted_tc_kelvin", "uncertainty_k",
    "conservative_score_tc_minus_sd_k", "chemical_family", "family_embargoed_spearman",
    "energy_above_hull_ev_per_atom", "summary_band_gap_ev", "metallic_method_band_gap_ev", "metallic_run_types",
    "spacegroup_number", "num_sites", "nearest_known_superconductor_formula",
    "nearest_known_superconductor_tc_kelvin", "nearest_known_superconductor_record_id",
    "nearest_known_superconductor_family", "nearest_known_superconductor_distance",
    "nearest_training_formula", "family_assignment_distance", "unresolved_source_formula_overlap", "chemistry_policy_flagged",
    "chemistry_policy_flags", "dft_verification",
)


@dataclass(frozen=True)
class FormulaSourceIndex:
    formulas: frozenset[str]
    unresolved_element_systems: frozenset[tuple[str, ...]]
    stats: dict[str, dict[str, int | str]]


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def reduced_formula(value: str | None) -> str | None:
    """Normalize a fixed composition to pymatgen's reduced formula."""

    if not value or not str(value).strip():
        return None
    try:
        from pymatgen.core import Composition

        composition = Composition(str(value).strip())
        if not composition or not composition.elements:
            return None
        return composition.reduced_formula
    except Exception:
        return None


def formula_elements(value: str) -> tuple[str, ...]:
    """Extract valid element symbols from a source formula, including variables."""

    from pymatgen.core.periodic_table import Element

    result = set()
    for symbol in re.findall(r"[A-Z][a-z]?", str(value)):
        try:
            Element(symbol)
        except Exception:
            continue
        result.add(symbol)
    return tuple(sorted(result))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        first = handle.readline()
        if first and not first.lstrip().startswith("#"):
            handle.seek(0)
        return list(csv.DictReader(handle, skipinitialspace=True))


def read_formula_source(path: Path, fields: tuple[str, ...], source_name: str) -> tuple[set[str], set[tuple[str, ...]], dict[str, int | str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required formula source {source_name!r} is missing: {path}")
    formulas: set[str] = set()
    unresolved_systems: set[tuple[str, ...]] = set()
    unparsed_values = 0
    rows = _csv_rows(path)
    for row in rows:
        found_parsed_formula = False
        unresolved_for_row: list[tuple[str, ...]] = []
        for field in fields:
            value = (row.get(field) or "").strip()
            if not value:
                continue
            normalized = reduced_formula(value)
            if normalized:
                formulas.add(normalized)
                found_parsed_formula = True
            else:
                elements = formula_elements(value)
                if elements:
                    unresolved_for_row.append(elements)
        if not found_parsed_formula:
            unparsed_values += max(1, len(unresolved_for_row))
            unresolved_systems.update(unresolved_for_row)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return formulas, unresolved_systems, {
        "path": str(path),
        "sha256": digest,
        "rows": len(rows),
        "distinct_reduced_formulas": len(formulas),
        "unparsed_formula_values": unparsed_values,
        "unresolved_element_systems": len(unresolved_systems),
    }


def load_formula_source_index(paths: dict[str, Path]) -> FormulaSourceIndex:
    """Load formulas from each required superconductivity data export."""

    specifications = {
        "3dsc": ("formula_sc", "formula_2", "norm_formula_sc", "norm_formula_2", "orig_formula_cif"),
        "uci": ("material", "formula"),
        "stanev": ("name", "formula", "material"),
        "supercon2": ("formula", "rawMaterial", "material"),
        "supermat": ("material", "formula"),
    }
    missing = sorted(set(specifications) - set(paths))
    if missing:
        raise ValueError(f"missing formula-source paths: {missing}")
    formulas: set[str] = set()
    unresolved: set[tuple[str, ...]] = set()
    stats: dict[str, dict[str, int | str]] = {}
    for source, fields in specifications.items():
        source_formulas, source_unresolved, source_stats = read_formula_source(paths[source], fields, source)
        formulas.update(source_formulas)
        unresolved.update(source_unresolved)
        stats[source] = source_stats
    stats["combined"] = {
        "distinct_reduced_formulas": len(formulas),
        "unresolved_element_systems": len(unresolved),
    }
    return FormulaSourceIndex(frozenset(formulas), frozenset(unresolved), stats)


def _structure(doc: Any) -> Any:
    structure = _get(doc, "structure")
    if isinstance(structure, dict):
        try:
            from pymatgen.core import Structure

            return Structure.from_dict(structure)
        except Exception:
            return None
    return structure


def _ordered(doc: Any) -> bool:
    structure = _structure(doc)
    if structure is None:
        return False
    return bool(_get(structure, "is_ordered", False))


def _is_summary_metallic(doc: Any) -> bool:
    if _get(doc, "is_metal") is True:
        return True
    gap = _get(doc, "band_gap")
    try:
        return gap is not None and math.isfinite(float(gap)) and float(gap) <= METALLIC_GAP_MAX_EV
    except (TypeError, ValueError):
        return False


def _site_count(doc: Any) -> int | None:
    value = _get(doc, "nsites", _get(doc, "num_sites"))
    if value is None:
        structure = _structure(doc)
        value = len(structure) if structure is not None else None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _summary_formula(doc: Any) -> str | None:
    composition = _get(doc, "composition_reduced")
    reduced = _get(composition, "reduced_formula") if composition is not None else None
    return reduced_formula(str(reduced or _get(doc, "formula_pretty", "")))


def summary_filter_candidates(
    docs: Iterable[Any],
    known_formulas: set[str] | frozenset[str],
    unresolved_element_systems: set[tuple[str, ...]] | frozenset[tuple[str, ...]] = frozenset(),
) -> tuple[list[Any], dict[str, int]]:
    """Apply explicit, cumulative MP summary filters without a network call."""

    items = list(docs)
    counts: dict[str, int] = {"mp_summary_returned": len(items)}
    stable = []
    for doc in items:
        hull = _get(doc, "energy_above_hull")
        try:
            if hull is not None and math.isfinite(float(hull)) and 0 <= float(hull) <= HULL_MAX_EV_PER_ATOM:
                stable.append(doc)
        except (TypeError, ValueError):
            continue
    counts["energy_above_hull_le_0p025"] = len(stable)
    small = [doc for doc in stable if _site_count(doc) is not None and 1 <= _site_count(doc) <= MAX_SITES]
    counts["site_count_le_20"] = len(small)
    ordered = [doc for doc in small if _ordered(doc)]
    counts["ordered_structures"] = len(ordered)
    metallic = [doc for doc in ordered if _is_summary_metallic(doc)]
    counts["summary_metallic_prefilter"] = len(metallic)
    unknown_formula = 0
    known_match = 0
    unresolved_overlap = 0
    unmatched = []
    for doc in metallic:
        formula = _summary_formula(doc)
        if formula is None:
            unknown_formula += 1
            continue
        if formula in known_formulas:
            known_match += 1
            continue
        elements = formula_elements(formula)
        if elements and elements in unresolved_element_systems:
            unresolved_overlap += 1
        unmatched.append(doc)
    counts["formula_unparseable"] = unknown_formula
    counts["formula_matches_known_superconductor_source"] = known_match
    counts["ambiguous_formula_source_system_overlap"] = unresolved_overlap
    counts["unmatched_reduced_formula"] = len(unmatched)
    return unmatched, counts


def _run_type_token(value: Any) -> str:
    raw = getattr(value, "value", value)
    token = str(raw).casefold().replace(" ", "").replace("_", "").replace("-", "").replace("+", "")
    return token


def _task_is_metallic_in_supported_method(task: Any) -> tuple[bool, str | None, float | None]:
    run_type = _get(task, "run_type")
    token = _run_type_token(run_type)
    if token not in ALLOWED_RUN_TYPES:
        return False, None, None
    output = _get(task, "output")
    gap = _get(output, "bandgap", _get(output, "band_gap"))
    try:
        gap_value = float(gap) if gap is not None else None
        metallic = gap_value is not None and math.isfinite(gap_value) and abs(gap_value) <= METALLIC_GAP_MAX_EV
    except (TypeError, ValueError):
        metallic = False
        gap_value = None
    return metallic, str(getattr(run_type, "value", run_type)) if metallic else None, gap_value if metallic else None


def qualify_by_method(docs: Iterable[Any], task_docs: Iterable[Any]) -> tuple[list[tuple[Any, tuple[tuple[str, float], ...]]], dict[str, int]]:
    """Require a task-level metallic result from GGA+U or r2SCAN."""

    task_by_id = {str(_get(task, "task_id", "")): task for task in task_docs}
    qualified: list[tuple[Any, tuple[tuple[str, float], ...]]] = []
    missing_task_refs = 0
    no_qualifying_method = 0
    for doc in docs:
        task_ids = _get(doc, "task_ids") or []
        source_task = _get(doc, "es_source_calc_id")
        ids = [str(value) for value in task_ids]
        if source_task:
            ids.append(str(source_task))
        methods: dict[str, float] = {}
        found_task = False
        for task_id in set(ids):
            task = task_by_id.get(task_id)
            if task is None:
                continue
            found_task = True
            metallic, method, gap = _task_is_metallic_in_supported_method(task)
            if metallic and method and gap is not None:
                methods[method] = min(gap, methods.get(method, gap))
        if not found_task:
            missing_task_refs += 1
        if methods:
            qualified.append((doc, tuple(sorted(methods.items()))))
        else:
            no_qualifying_method += 1
    return qualified, {
        "method_qualified_metallic": len(qualified),
        "no_supported_metallic_method": no_qualifying_method,
        "missing_task_records": missing_task_refs,
    }


def _spacegroup_number(doc: Any) -> int | None:
    symmetry = _get(doc, "symmetry")
    value = _get(symmetry, "number") if symmetry is not None else None
    try:
        number = int(value)
        if 1 <= number <= 230:
            return number
    except (TypeError, ValueError):
        pass
    structure = _structure(doc)
    if structure is None:
        return None
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        return int(SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number())
    except Exception:
        return None


@lru_cache(maxsize=256)
def _crystal_system(number: int | None) -> str | None:
    if number is None:
        return None
    try:
        from pymatgen.symmetry.groups import SpaceGroup

        return str(SpaceGroup.from_int_number(int(number)).crystal_system)
    except Exception:
        return None


def _composition_fractions(formula: str) -> dict[str, float]:
    from pymatgen.core import Composition

    amounts = Composition(formula).get_el_amt_dict()
    total = sum(float(value) for value in amounts.values())
    return {str(element): float(amount) / total for element, amount in amounts.items()}


def composition_structure_distance(
    left_formula: str,
    left_spacegroup: int | None,
    right_formula: str,
    right_spacegroup: int | None,
) -> float:
    """Weighted total-variation chemistry distance plus categorical symmetry distance."""

    left, right = _composition_fractions(left_formula), _composition_fractions(right_formula)
    return _fraction_structure_distance(left, left_spacegroup, right, right_spacegroup)


def _fraction_structure_distance(
    left: dict[str, float],
    left_spacegroup: int | None,
    right: dict[str, float],
    right_spacegroup: int | None,
) -> float:
    composition_distance = 0.5 * sum(abs(left.get(element, 0.0) - right.get(element, 0.0))
                                    for element in set(left) | set(right))
    if left_spacegroup is None or right_spacegroup is None:
        structure_distance = 0.5
    elif left_spacegroup == right_spacegroup:
        structure_distance = 0.0
    elif _crystal_system(left_spacegroup) == _crystal_system(right_spacegroup):
        structure_distance = 0.5
    else:
        structure_distance = 1.0
    return 0.8 * composition_distance + 0.2 * structure_distance


def nearest_training_record(
    formula: str,
    spacegroup: int | None,
    records: list[HarmonizedRecord],
    *,
    superconducting_only: bool = False,
) -> tuple[HarmonizedRecord, float] | None:
    candidates = [record for record in records if not superconducting_only or record.tc_kelvin > 0]
    if not candidates:
        return None
    ranked = [
        (composition_structure_distance(formula, spacegroup, record.formula, record.spacegroup_number), record.record_id, record)
        for record in candidates
    ]
    distance, _, record = min(ranked, key=lambda value: (value[0], value[1]))
    return record, float(distance)


def _nearest_training_records_precomputed(
    formula: str,
    spacegroup: int | None,
    records: list[tuple[HarmonizedRecord, dict[str, float]]],
) -> tuple[tuple[HarmonizedRecord, float] | None, tuple[HarmonizedRecord, float] | None]:
    candidate_fractions = _composition_fractions(formula)
    nearest_any: tuple[float, str, HarmonizedRecord] | None = None
    nearest_superconductor: tuple[float, str, HarmonizedRecord] | None = None
    for record, fractions in records:
        distance = _fraction_structure_distance(candidate_fractions, spacegroup, fractions, record.spacegroup_number)
        ranked = (distance, record.record_id, record)
        if nearest_any is None or ranked[:2] < nearest_any[:2]:
            nearest_any = ranked
        if record.tc_kelvin > 0 and (nearest_superconductor is None or ranked[:2] < nearest_superconductor[:2]):
            nearest_superconductor = ranked
    any_result = (nearest_any[2], float(nearest_any[0])) if nearest_any else None
    superconductor_result = (
        (nearest_superconductor[2], float(nearest_superconductor[0])) if nearest_superconductor else None
    )
    return any_result, superconductor_result


def eligible_family_metrics(report: dict[str, Any], threshold: float = MIN_EMBARGOED_SPEARMAN) -> dict[str, float]:
    """Return only ready metrics explicitly marked as shared-MP-embargoed."""

    within = report.get("within_family_evaluation", {})
    eligible: dict[str, float] = {}
    for family, metric in within.get("families", {}).items():
        rho = metric.get("spearman_rho")
        if (metric.get("status") == "ready" and metric.get("shared_mp_ids_embargoed") is True
                and rho is not None and math.isfinite(float(rho)) and float(rho) > threshold):
            eligible[str(family)] = float(rho)
    return eligible


def chemistry_policy_flags(formula: str) -> list[str]:
    """Return review flags applicable to existing MP entries, never reject them."""

    from pymatgen.core import Composition, Element

    composition = Composition(formula)
    symbols = {element.symbol for element in composition.elements}
    flags: list[str] = []
    radioactive = {"Tc", "Pm", "Po", "Th", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"}
    radioactive_present = sorted(symbols & radioactive, key=lambda symbol: Element(symbol).Z)
    if radioactive_present:
        flags.append("radioactive_or_actinide_lab_access:" + ";".join(radioactive_present))
    anions = {"O", "F", "Cl", "Br", "I"}
    transition_metals = {element.symbol for element in composition.elements if element.is_transition_metal}
    if len(symbols) == 2 and len(transition_metals) == 1 and symbols & anions:
        flags.append("binary_transition_metal_oxide_or_halide_review")
    return flags


def _candidate_record(formula: str, mp_id: str, hull: float, family: str) -> HarmonizedRecord:
    from rt_superconductor.targets import ATMOSPHERIC_PRESSURE_GPA

    digest = hashlib.sha256(f"mp-existing:{mp_id}:{formula}".encode()).hexdigest()[:24]
    return HarmonizedRecord(
        record_id=digest,
        source_db="uci",
        formula=formula,
        tc_kelvin=0.0,
        applied_pressure_gpa=ATMOSPHERIC_PRESSURE_GPA,
        measurement_method="unknown",
        sample_form="unknown",
        doi=f"Materials Project:{mp_id}",
        review_status="pending",
        provenance="curated",
        is_synthetic=0,
        chemical_family=family,
        mp_id=mp_id,
        energy_above_hull=hull,
    )


def _stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big") % (2**31 - 1)


def score_family_candidates(
    qualified: list[tuple[Any, tuple[tuple[str, float], ...]]],
    training_records: list[HarmonizedRecord],
    eligible_metrics: dict[str, float],
    unresolved_element_systems: set[tuple[str, ...]] | frozenset[tuple[str, ...]] = frozenset(),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Assign nearest family, then score only families passing embargoed rho > 0.5."""

    family_records: dict[str, list[HarmonizedRecord]] = defaultdict(list)
    for record in training_records:
        family_records[record.chemical_family].append(record)
    record_distance_cache = [(record, _composition_fractions(record.formula)) for record in training_records]
    prepared: list[dict[str, Any]] = []
    assigned_counts: dict[str, int] = defaultdict(int)
    skipped_no_nearest = 0
    skipped_family_gate = 0
    for doc, methods in qualified:
        formula = _summary_formula(doc)
        if formula is None:
            skipped_no_nearest += 1
            continue
        mp_id = str(_get(doc, "material_id", ""))
        try:
            hull = float(_get(doc, "energy_above_hull"))
        except (TypeError, ValueError):
            skipped_no_nearest += 1
            continue
        sg = _spacegroup_number(doc)
        nearest, nearest_sc = _nearest_training_records_precomputed(formula, sg, record_distance_cache)
        if nearest is None:
            skipped_no_nearest += 1
            continue
        nearest_record, family_distance = nearest
        family = nearest_record.chemical_family
        assigned_counts[family] += 1
        if family not in eligible_metrics:
            skipped_family_gate += 1
            continue
        row_record = _candidate_record(formula, mp_id, hull, family)
        nearest_sc_record, nearest_sc_distance = nearest_sc if nearest_sc else (None, None)
        prepared.append({
            "doc": doc,
            "record": row_record,
            "mp_id": mp_id,
            "formula": formula,
            "hull": hull,
            "spacegroup_number": sg,
            "num_sites": _site_count(doc),
            "summary_band_gap_ev": _get(doc, "band_gap"),
            "metallic_run_types": ";".join(method for method, _ in methods),
            "metallic_method_band_gap_ev": min(gap for _, gap in methods),
            "chemical_family": family,
            "family_embargoed_spearman": eligible_metrics[family],
            "family_assignment_distance": family_distance,
            "nearest_training_record": nearest_record,
            "nearest_known_superconductor": nearest_sc_record,
            "nearest_known_superconductor_distance": nearest_sc_distance,
            "unresolved_source_formula_overlap": formula_elements(formula) in unresolved_element_systems,
        })
    counts = {
        "family_assigned": sum(assigned_counts.values()),
        "family_gate_passed": len(prepared),
        "family_gate_skipped": skipped_family_gate,
        "no_nearest_training_record": skipped_no_nearest,
        **{f"family_assigned:{family}": count for family, count in sorted(assigned_counts.items())},
    }
    if not prepared:
        return [], counts

    scored: list[dict[str, Any]] = []
    by_family_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(prepared):
        by_family_indices[row["chemical_family"]].append(index)
    for family, indices in sorted(by_family_indices.items()):
        records = family_records[family]
        known_hulls = [record.energy_above_hull for record in records if record.energy_above_hull is not None]
        hull_fill = float(np.median(known_hulls)) if known_hulls else 0.0
        train_X, feature_names, _ = build_features(records, hull_fill=hull_fill)
        candidates = [prepared[index]["record"] for index in indices]
        candidate_X, candidate_names, _ = build_features(candidates, hull_fill=hull_fill)
        if feature_names != candidate_names:
            raise RuntimeError(f"feature schema changed between training and pool for family {family}")
        targets = np.asarray([record.tc_kelvin for record in records], dtype=float)
        model = _model(_stable_seed(f"untested-pool:{family}"))
        model.fit(train_X, targets)
        tree_predictions = np.asarray([tree.predict(candidate_X) for tree in model.estimators_])
        means = tree_predictions.mean(axis=0)
        uncertainties = tree_predictions.std(axis=0)
        for local, index in enumerate(indices):
            row = prepared[index]
            predicted = float(means[local])
            uncertainty = float(uncertainties[local])
            nearest_sc = row["nearest_known_superconductor"]
            flags = chemistry_policy_flags(row["formula"])
            scored.append({
                "mp_id": row["mp_id"],
                "formula": row["formula"],
                "predicted_tc_kelvin": predicted,
                "uncertainty_k": uncertainty,
                "conservative_score_tc_minus_sd_k": predicted - uncertainty,
                "chemical_family": family,
                "family_embargoed_spearman": row["family_embargoed_spearman"],
                "energy_above_hull_ev_per_atom": row["hull"],
                "summary_band_gap_ev": row["summary_band_gap_ev"],
                "metallic_method_band_gap_ev": row["metallic_method_band_gap_ev"],
                "metallic_run_types": row["metallic_run_types"],
                "spacegroup_number": row["spacegroup_number"],
                "num_sites": row["num_sites"],
                "nearest_known_superconductor_formula": nearest_sc.formula if nearest_sc else "",
                "nearest_known_superconductor_tc_kelvin": nearest_sc.tc_kelvin if nearest_sc else "",
                "nearest_known_superconductor_record_id": nearest_sc.record_id if nearest_sc else "",
                "nearest_known_superconductor_family": nearest_sc.chemical_family if nearest_sc else "",
                "nearest_known_superconductor_distance": row["nearest_known_superconductor_distance"],
                "nearest_training_formula": row["nearest_training_record"].formula,
                "family_assignment_distance": row["family_assignment_distance"],
                "unresolved_source_formula_overlap": row["unresolved_source_formula_overlap"],
                "chemistry_policy_flagged": bool(flags),
                "chemistry_policy_flags": ";".join(flags),
                "dft_verification": "pending",
            })
    return scored, counts


def select_top_per_family(rows: Iterable[dict[str, Any]], limit: int = DEFAULT_TOP_PER_FAMILY) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["chemical_family"])].append(row)
    result: list[dict[str, Any]] = []
    for family in sorted(grouped):
        ranked = sorted(
            grouped[family],
            key=lambda row: (-float(row["conservative_score_tc_minus_sd_k"]),
                             -float(row["predicted_tc_kelvin"]), str(row["mp_id"])),
        )[:limit]
        for rank, row in enumerate(ranked, start=1):
            result.append({"rank_within_family": rank, **row})
    return result


def _write_shortlist(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in OUTPUT_COLUMNS})


def _fetch_mp(api_key: str, source_index: FormulaSourceIndex) -> tuple[list[tuple[Any, tuple[tuple[str, float], ...]]], dict[str, int]]:
    if not api_key:
        raise ValueError("MP_API_KEY is required; this script must run in the configured CI workflow")
    from mp_api.client import MPRester

    summary_fields = [
        "material_id", "formula_pretty", "composition_reduced", "energy_above_hull", "nsites",
        "band_gap", "is_metal", "structure", "symmetry", "task_ids", "es_source_calc_id",
    ]
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            energy_above_hull=(0, HULL_MAX_EV_PER_ATOM),
            num_sites=(1, MAX_SITES),
            deprecated=False,
            fields=summary_fields,
            all_fields=False,
            chunk_size=1000,
        )
        unmatched, counts = summary_filter_candidates(
            docs, set(source_index.formulas), set(source_index.unresolved_element_systems)
        )
        task_ids: set[str] = set()
        for doc in unmatched:
            for task_id in (_get(doc, "task_ids") or []):
                if task_id:
                    task_ids.add(str(task_id))
            source_task = _get(doc, "es_source_calc_id")
            if source_task:
                task_ids.add(str(source_task))
        tasks: list[Any] = []
        ordered_ids = sorted(task_ids)
        for start in range(0, len(ordered_ids), 100):
            tasks.extend(mpr.materials.tasks.search(
                task_ids=ordered_ids[start:start + 100],
                fields=["task_id", "run_type", "output"],
                all_fields=False,
                chunk_size=100,
            ))
    qualified, method_counts = qualify_by_method(unmatched, tasks)
    counts.update(method_counts)
    counts["task_ids_queried"] = len(task_ids)
    counts["task_documents_returned"] = len(tasks)
    return qualified, counts


def build_untested_pool(
    *,
    input_path: Path,
    model_dir: Path,
    report_path: Path,
    output_path: Path,
    source_paths: dict[str, Path],
    api_key: str | None = None,
    top_per_family: int = DEFAULT_TOP_PER_FAMILY,
) -> dict[str, Any]:
    source_index = load_formula_source_index(source_paths)
    pressure_report_path = model_dir / "pressure-aware-report.json"
    if not pressure_report_path.is_file():
        raise FileNotFoundError(f"trained pressure-aware report not found: {pressure_report_path}")
    pressure_report = json.loads(pressure_report_path.read_text(encoding="utf-8"))
    if pressure_report.get("training_status") != "ready":
        raise ValueError("pressure-aware training report is not ready; pool scoring is blocked")
    eligible = eligible_family_metrics(pressure_report)
    family_metrics = pressure_report.get("within_family_evaluation", {}).get("families", {})
    family_gate_decisions = {
        family: {
            "spearman_rho": metric.get("spearman_rho"),
            "row_count": metric.get("row_count"),
            "shared_mp_ids_embargoed": metric.get("shared_mp_ids_embargoed") is True,
            "passed": family in eligible,
            "reason": "passed" if family in eligible else (
                "not_ready_or_not_embargoed" if metric.get("status") != "ready" or metric.get("shared_mp_ids_embargoed") is not True
                else "spearman_not_strictly_above_0.5"
            ),
        }
        for family, metric in sorted(family_metrics.items())
    }
    records = select_training_records(load_harmonized(input_path))
    if not records:
        raise ValueError("there are no empirical training records to assign families or score candidates")

    qualified, filter_counts = _fetch_mp(api_key or os.environ.get("MP_API_KEY", ""), source_index)
    scored, score_counts = score_family_candidates(
        qualified, records, eligible, source_index.unresolved_element_systems
    )
    shortlist = select_top_per_family(scored, top_per_family)
    _write_shortlist(output_path, shortlist)
    family_summary: dict[str, dict[str, Any]] = {}
    for family, rho in sorted(eligible.items()):
        family_summary[family] = {
            "embargoed_within_family_spearman": rho,
            "gate": "passed (> 0.5)",
            "pool_count": score_counts.get(f"family_assigned:{family}", 0),
            "scored_count": sum(row["chemical_family"] == family for row in scored),
            "shortlist_count": sum(row["chemical_family"] == family for row in shortlist),
            "top_shortlist_chemistry_policy_flagged_count": sum(
                row["chemical_family"] == family and row["chemistry_policy_flagged"] for row in shortlist
            ),
        }
    top_flagged = sum(bool(row["chemistry_policy_flagged"]) for row in shortlist)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "ready" if shortlist else ("ready_empty_after_family_gate" if eligible else "blocked_no_family_passed_embargoed_spearman_gate"),
        "interpretation": "Existing MP materials absent by reduced formula from normalized rows in the five specified data exports; not proof that no experiment has ever tested them.",
        "criteria": {
            "energy_above_hull_ev_per_atom_max": HULL_MAX_EV_PER_ATOM,
            "metallic_methods": ["GGA+U", "r2SCAN", "r2SCAN+U"],
            "metallic_band_gap_ev_max": METALLIC_GAP_MAX_EV,
            "ordered_structure_required": True,
            "max_sites": MAX_SITES,
            "known_sources": sorted(source_paths),
            "family_gate": "embargoed within-family Spearman strictly greater than 0.5",
            "family_assignment": "nearest empirical record using 80% total-variation composition distance and 20% crystal-system/space-group distance",
            "ranking": "family-specific random forest, ordered by predicted Tc minus tree-spread standard deviation",
            "chemistry_policy_scope": "review flags only for extant MP entries; substitution-only rules are not applied as rejection tests",
        },
        "source_formula_index": source_index.stats,
        "training_data": {
            "path": str(input_path),
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "empirical_records": len(records),
        },
        "within_family_report": str(pressure_report_path),
        "embargoed_family_metrics": family_metrics,
        "family_gate_decisions": family_gate_decisions,
        "eligible_family_metrics": eligible,
        "filter_counts": filter_counts,
        "scoring_counts": score_counts,
        "pool_size_after_all_materials_project_and_database_filters": filter_counts.get("method_qualified_metallic", 0),
        "pool_size_assigned_to_families": score_counts.get("family_assigned", 0),
        "pool_size_scored_after_family_gate": len(scored),
        "shortlist_rows": len(shortlist),
        "top_shortlist_chemistry_policy_flagged_count": top_flagged,
        "top_shortlist_chemistry_policy_unflagged_count": len(shortlist) - top_flagged,
        "families": family_summary,
        "output_csv": str(output_path),
        "output_csv_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "limitations": [
            "The term untested is relative to 3DSC, UCI, the Stanev export, SuperCon2, and SuperMat; it does not establish a global literature absence.",
            "Some source formulas contain variables or markup that cannot be reduced; same-element-system overlaps are retained but visibly flagged for source review.",
            "The family assignment uses formula fractions and crystallographic symmetry because harmonized training rows do not all have full structures.",
            "Predictions are family-specific random-forest estimates; tree spread is a screening uncertainty proxy and is not calibrated probability.",
            "Materials Project stability and electronic labels are computational; DFT and experimental verification remain pending.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/training/harmonized_supercon.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/processed/continuous-training/pressure-aware"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/continuous-training/untested-pool-shortlist.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/processed/continuous-training/untested-pool-report.json"))
    parser.add_argument("--3dsc", dest="threedsc", type=Path, default=Path("data/raw/3dsc/MP/3DSC_MP.csv"))
    parser.add_argument("--uci", type=Path, default=Path("data/raw/uci/extracted/unique_m.csv"))
    parser.add_argument("--stanev", type=Path, default=Path("data/raw/3dsc/Supercon_data_by_2018_Stanev.csv"))
    parser.add_argument("--supercon2", type=Path, default=Path("/tmp/supercon2.csv"))
    parser.add_argument("--supermat", type=Path, default=Path("/tmp/supermat/SuperMat-2.0.csv"))
    parser.add_argument("--top-per-family", type=int, default=DEFAULT_TOP_PER_FAMILY)
    args = parser.parse_args()
    paths = {
        "3dsc": args.threedsc,
        "uci": args.uci,
        "stanev": args.stanev,
        "supercon2": args.supercon2,
        "supermat": args.supermat,
    }
    report = build_untested_pool(
        input_path=args.input,
        model_dir=args.model_dir,
        report_path=args.report,
        output_path=args.output,
        source_paths=paths,
        top_per_family=args.top_per_family,
    )
    print(json.dumps({
        "status": report["status"],
        "filter_counts": report["filter_counts"],
        "pool_size": report["pool_size_after_all_materials_project_and_database_filters"],
        "scored_pool_size": report["pool_size_scored_after_family_gate"],
        "shortlist_rows": report["shortlist_rows"],
        "top_shortlist_chemistry_policy_flagged_count": report["top_shortlist_chemistry_policy_flagged_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
