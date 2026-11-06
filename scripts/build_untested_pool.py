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
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
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
from untested_pool_policy import dos_ef_rejection, untested_pool_composition_rejection


HULL_MAX_EV_PER_ATOM = 0.025
MAX_SITES = 20
MIN_DOS_EF_STATES_EV_PER_ATOM = 0.1
MIN_FAMILY_SPEARMAN = 0.5
MAX_FAMILY_DISTANCE = 0.3
DEFAULT_TOP_PER_FAMILY = 30
OUTPUT_COLUMNS = (
    "rank_within_family", "mp_id", "formula", "predicted_tc_kelvin", "uncertainty_k",
    "conservative_score_tc_minus_sd_k", "chemical_family", "within_family_spearman",
    "energy_above_hull_ev_per_atom", "dos_ef_states_ev_per_atom", "dos_task_type",
    "dos_functional_run_type", "dos_task_id", "dos_retrieval_method",
    "spacegroup_number", "num_sites", "nearest_known_superconductor_formula",
    "nearest_known_superconductor_tc_kelvin", "nearest_known_superconductor_record_id",
    "nearest_known_superconductor_family", "nearest_known_superconductor_distance",
    "nearest_training_formula", "family_assignment_distance", "unresolved_source_formula_overlap", "dft_verification",
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
) -> tuple[list[Any], dict[str, int], dict[str, str]]:
    """Apply stability, size, exact-source and composition screens to MP summaries.

    No summary band gap or relaxation gap is used as a metallicity label. DOS is
    queried separately and is mandatory for acceptance.
    """

    items = list(docs)
    counts: dict[str, int] = {"mp_summary_returned": len(items)}
    rejected: dict[str, str] = {}
    stable = []
    for doc in items:
        mid = str(_get(doc, "material_id", ""))
        hull = _get(doc, "energy_above_hull")
        try:
            if hull is not None and math.isfinite(float(hull)) and 0 <= float(hull) <= HULL_MAX_EV_PER_ATOM:
                stable.append(doc)
            elif mid:
                rejected[mid] = "energy_above_hull_gt_0.025_or_missing"
        except (TypeError, ValueError):
            if mid:
                rejected[mid] = "energy_above_hull_invalid"
    counts["energy_above_hull_le_0p025"] = len(stable)
    small = []
    for doc in stable:
        mid = str(_get(doc, "material_id", ""))
        sites = _site_count(doc)
        if sites is not None and 1 <= sites <= MAX_SITES:
            small.append(doc)
        elif mid:
            rejected[mid] = "site_count_missing_or_gt_20"
    counts["site_count_le_20"] = len(small)
    unknown_formula = 0
    known_match = 0
    unresolved_overlap = 0
    composition_rejected = 0
    unmatched = []
    for doc in small:
        mid = str(_get(doc, "material_id", ""))
        formula = _summary_formula(doc)
        if formula is None:
            unknown_formula += 1
            if mid:
                rejected[mid] = "formula_unparseable"
            continue
        if formula in known_formulas:
            known_match += 1
            if mid:
                rejected[mid] = "formula_matches_known_superconductor_source"
            continue
        composition_reason = untested_pool_composition_rejection(formula)
        if composition_reason:
            composition_rejected += 1
            if mid:
                rejected[mid] = composition_reason
            continue
        elements = formula_elements(formula)
        if elements and elements in unresolved_element_systems:
            unresolved_overlap += 1
        unmatched.append(doc)
    counts["formula_unparseable"] = unknown_formula
    counts["formula_matches_known_superconductor_source"] = known_match
    counts["ambiguous_formula_source_system_overlap"] = unresolved_overlap
    counts["composition_policy_rejected"] = composition_rejected
    counts["unmatched_reduced_formula"] = len(unmatched)
    return unmatched, counts, rejected


def _summary_filter_without_structures(
    docs: Iterable[Any],
    known_formulas: set[str] | frozenset[str],
    unresolved_element_systems: set[tuple[str, ...]] | frozenset[tuple[str, ...]] = frozenset(),
) -> tuple[list[Any], dict[str, int], dict[str, str]]:
    """Compatibility name for the inexpensive MP summary filters."""
    return summary_filter_candidates(docs, known_formulas, unresolved_element_systems)


def _descriptor_formula(formula: str) -> str:
    """Flatten valid grouped formulas into the simple syntax used by descriptors."""

    from pymatgen.core import Composition

    composition = Composition(formula)
    # Composition.formula expands groups (for example Ba(PdO2)2) into plain
    # element counts while retaining any fractional stoichiometry.
    return composition.formula.replace(" ", "")


def _run_type_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    return re.sub(r"[^a-z]", "", str(raw).casefold())


def _is_static_or_uniform(run_type: Any) -> bool:
    token = _run_type_text(run_type)
    if "relax" in token or "optimization" in token or "optimize" in token:
        return False
    return "static" in token or "uniform" in token


def _task_is_static_or_uniform(task: Any) -> bool:
    """Use task/calculation type, not the functional-only MP ``run_type``."""
    task_type = _get(task, "task_type")
    calc_type = _get(task, "calc_type")
    if task_type is not None:
        return _is_static_or_uniform(task_type)
    if calc_type is not None:
        return _is_static_or_uniform(calc_type)
    # Some older task documents encode the task kind in run_type. Current
    # MP RunType values (GGA, GGA+U, r2SCAN, ...) alone are never sufficient.
    return _is_static_or_uniform(_get(task, "run_type"))


def _electronic_structure_task_ids(doc: Any) -> list[str]:
    metadata = _get(doc, "dos") or {}
    task_ids: list[str] = []
    direct = _get(metadata, "task_id")
    if direct:
        task_ids.append(str(direct))
    listed = _get(metadata, "task_ids") or []
    if isinstance(listed, str):
        listed = [listed]
    task_ids.extend(str(task_id) for task_id in listed if task_id)
    total = _get(metadata, "total") or {}
    for spin_doc in (total.values() if isinstance(total, dict) else []):
        task_id = _get(spin_doc, "task_id")
        if task_id:
            task_ids.append(str(task_id))
    return list(dict.fromkeys(task_ids))


def _search_electronic_structure_raw(route: Any, material_ids: list[str], *, chunk_size: int = 500) -> list[Any]:
    """Read MP DOS metadata as dictionaries to tolerate incomplete nested docs.

    The MP endpoint has returned DOS summaries without nested ``task_id``
    values. The client model currently marks those fields required and aborts
    an otherwise valid batch. Raw dictionaries preserve available references;
    missing task IDs are conservatively rejected downstream.
    """
    previous = getattr(route, "use_document_model", None)
    if previous is None:
        return route.search(
            material_ids=material_ids,
            fields=["material_id", "dos"],
            all_fields=False,
            chunk_size=chunk_size,
        )
    route.use_document_model = False
    try:
        return route.search(
            material_ids=material_ids,
            fields=["material_id", "dos"],
            all_fields=False,
            chunk_size=chunk_size,
        )
    finally:
        route.use_document_model = previous


def _search_tasks_raw(route: Any, **kwargs: Any) -> list[Any]:
    """Search task docs as dictionaries so missing optional fields stay nullable."""
    previous = getattr(route, "use_document_model", None)
    if previous is None:
        return route.search(**kwargs)
    route.use_document_model = False
    try:
        return route.search(**kwargs)
    finally:
        route.use_document_model = previous


@contextmanager
def material_level_dos_lookup(mpr: Any):
    """Fetch DOS by material identity through its eligible task references.

    MP's ``get_dos_by_material_id`` assumes a nested ``total.1.task_id`` that
    current responses may omit. The deep-enrichment path instead resolves a
    material to its exact calculation task and retrieves that parsed curve.
    Keep raw mode enabled and return both the curve and the task ID used.
    """
    route = mpr.materials.electronic_structure_dos
    previous = getattr(route, "use_document_model", None)
    if previous is not None:
        route.use_document_model = False

    def load_for_material(material_id: str, eligible_task_ids: list[str]):
        from enrich_deep_mp import dos_curve_for_task

        failures = []
        for task_id in eligible_task_ids:
            try:
                dos = dos_curve_for_task(route, str(task_id))
                if dos is not None:
                    return dos, str(task_id)
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise failures[-1]
        raise ValueError(f"No eligible calculation task has DOS for {material_id}")

    try:
        yield load_for_material
    finally:
        if previous is not None:
            route.use_document_model = previous


def _candidate_dos_task_refs_by_formula(
    task_route: Any,
    candidate_docs: list[Any],
    electronic_structure_docs: list[Any],
    *,
    formula_batch_size: int = 100,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Recover absent ES task references via formula plus exact structure match.

    Current Materials Project task documents do not expose ``material_id`` as
    a searchable field. Formula alone is not a safe identity key, so task
    results are retained only when their output structure matches the exact
    candidate structure. Static/uniform task metadata is checked here and
    again before DOS payload retrieval.
    """
    es_by_material = {
        str(_get(doc, "material_id", "")): doc
        for doc in electronic_structure_docs
        if _get(doc, "material_id")
    }
    missing: dict[str, Any] = {}
    for candidate in candidate_docs:
        material_id = str(_get(candidate, "material_id", ""))
        if not material_id:
            continue
        es_doc = es_by_material.get(material_id)
        if es_doc is None or not _electronic_structure_task_ids(es_doc):
            missing[material_id] = candidate

    formulas_to_materials: dict[str, set[str]] = defaultdict(set)
    candidates_by_id = {}
    for material_id, candidate in missing.items():
        formula = _summary_formula(candidate)
        if formula and _structure(candidate) is not None:
            formulas_to_materials[formula].add(material_id)
            candidates_by_id[material_id] = candidate

    by_material: dict[str, set[str]] = defaultdict(set)
    errors: Counter[str] = Counter()
    error_samples: list[dict[str, str]] = []
    task_docs_returned = 0
    non_static_uniform_tasks_rejected = 0
    task_structures_missing = 0
    structure_comparisons = 0
    structure_matches = 0
    formulas = sorted(formulas_to_materials)
    for start in range(0, len(formulas), max(1, formula_batch_size)):
        batch = formulas[start:start + max(1, formula_batch_size)]
        try:
            tasks = _search_tasks_raw(
                task_route,
                formula=batch,
                fields=["task_id", "formula_pretty", "task_type", "calc_type", "run_type", "structure"],
                all_fields=False,
                chunk_size=100,
            )
        except Exception as exc:
            errors[type(exc).__name__] += 1
            if len(error_samples) < 5:
                error_samples.append({
                    "error_type": type(exc).__name__,
                    "message": str(exc).replace("\n", " ")[:240],
                    "formula_batch_size": str(len(batch)),
                })
            continue
        task_docs_returned += len(tasks)
        for task in tasks:
            task_id = _get(task, "task_id")
            if not task_id:
                continue
            if not _task_is_static_or_uniform(task):
                non_static_uniform_tasks_rejected += 1
                continue
            task_formula = reduced_formula(str(_get(task, "formula_pretty", "")))
            structure = _structure(task)
            if structure is None:
                task_structures_missing += 1
                continue
            task_formula = task_formula or reduced_formula(structure.composition.reduced_formula)
            if not task_formula:
                continue
            for material_id in formulas_to_materials.get(task_formula, set()):
                candidate_structure = _structure(candidates_by_id[material_id])
                if candidate_structure is not None:
                    structure_comparisons += 1
                if candidate_structure is not None and _same_structure(candidate_structure, structure):
                    structure_matches += 1
                    by_material[material_id].add(str(task_id))

    return (
        {material_id: sorted(task_ids) for material_id, task_ids in sorted(by_material.items())},
        {
            "materials_missing_es_task_refs": len(missing),
            "formula_queries": math.ceil(len(formulas) / max(1, formula_batch_size)),
            "materials_recovered": len(by_material),
            "task_refs_recovered": sum(len(task_ids) for task_ids in by_material.values()),
            "candidate_formulas_without_structures": sum(
                1 for material_id in missing if material_id not in candidates_by_id
            ),
            "task_documents_returned": task_docs_returned,
            "non_static_uniform_tasks_rejected": non_static_uniform_tasks_rejected,
            "task_structures_missing": task_structures_missing,
            "structure_comparisons": structure_comparisons,
            "structure_matches": structure_matches,
            "error_counts": dict(sorted(errors.items())),
            "error_samples": error_samples,
        },
    )


def _same_structure(left: Any, right: Any) -> bool:
    """Conservatively match same-phase MP structures while allowing rounding."""
    try:
        from pymatgen.analysis.structure_matcher import StructureMatcher
        from pymatgen.core import Structure

        def as_structure(value: Any):
            if isinstance(value, Structure):
                return value
            if isinstance(value, dict):
                return Structure.from_dict(value)
            return None

        left_structure, right_structure = as_structure(left), as_structure(right)
        if left_structure is None or right_structure is None:
            return False
        if len(left_structure) != len(right_structure):
            return False
        if left_structure.composition.reduced_formula != right_structure.composition.reduced_formula:
            return False
        matcher = StructureMatcher(
            ltol=0.05, stol=0.1, angle_tol=1.0, primitive_cell=False,
            scale=False, attempt_supercell=False,
        )
        return bool(matcher.fit(left_structure, right_structure))
    except Exception:
        return False


def dos_at_fermi_states_ev_per_atom(dos: Any, nsites: int | None = None) -> float:
    """Interpolate total DOS at Ef and normalize cell DOS by the site count.

    MP's total DOS endpoint returns a ``Dos`` object without an attached
    structure. Prefer the candidate's independently retrieved MP ``nsites``;
    fall back to ``CompleteDos.structure`` for callers that have one.
    """
    energies = np.asarray(dos.energies, dtype=float)
    densities = np.asarray(dos.get_densities(), dtype=float)
    efermi = float(dos.efermi)
    if nsites is None:
        structure = getattr(dos, "structure", None)
        nsites = len(structure) if structure is not None else 0
    try:
        nsites = int(nsites)
    except (TypeError, ValueError):
        nsites = 0
    if (energies.ndim != 1 or densities.shape != energies.shape or len(energies) < 2
            or nsites < 1 or not np.isfinite(energies).all() or not np.isfinite(densities).all()
            or not math.isfinite(efermi) or np.any(np.diff(energies) <= 0)
            or not energies[0] <= efermi <= energies[-1]):
        raise ValueError("DOS grid, Fermi level, or site count is invalid")
    value = float(np.interp(efermi, energies, densities)) / nsites
    if not math.isfinite(value) or value < 0:
        raise ValueError("DOS at Fermi level is invalid")
    return value


def qualify_by_dos(
    docs: Iterable[Any],
    electronic_structure_docs: Iterable[Any],
    task_docs: Iterable[Any],
    dos_loader: Any,
) -> tuple[list[tuple[Any, dict[str, Any]]], dict[str, int], dict[str, str], dict[str, Any]]:
    """Require DOS(Ef)>0.1 states/eV/atom from a static or uniform MP task."""
    es_by_id = {str(_get(doc, "material_id", "")): doc for doc in electronic_structure_docs}
    task_by_id = {str(_get(task, "task_id", "")): task for task in task_docs}
    qualified: list[tuple[Any, dict[str, Any]]] = []
    rejected: dict[str, str] = {}
    validation_error_counts: Counter[str] = Counter()
    validation_error_samples: list[dict[str, str]] = []
    counts = {
        "dos_metadata_returned": len(es_by_id), "dos_task_references": 0,
        "dos_static_or_uniform_tasks": 0, "dos_non_static_uniform_tasks_rejected": 0,
        "dos_curves_retrieved": 0, "dos_missing_task_or_curve": 0,
        "dos_below_or_equal_0p1_states_ev_atom": 0, "dos_qualified_gt_0p1_states_ev_atom": 0,
    }
    for doc in docs:
        mid = str(_get(doc, "material_id", ""))
        es_doc = es_by_id.get(mid)
        task_ids = _electronic_structure_task_ids(es_doc) if es_doc is not None else []
        if not task_ids:
            counts["dos_missing_task_or_curve"] += 1
            rejected[mid] = "no_static_or_uniform_dos_task_reference"
            continue
        counts["dos_task_references"] += 1
        tasks_for_material = [task_by_id[task_id] for task_id in task_ids if task_id in task_by_id]
        if not tasks_for_material:
            counts["dos_missing_task_or_curve"] += 1
            rejected[mid] = "dos_task_record_missing"
            continue
        counts["dos_non_static_uniform_tasks_rejected"] += sum(
            not _task_is_static_or_uniform(task) for task in tasks_for_material
        )
        allowed_tasks = [task for task in tasks_for_material if _task_is_static_or_uniform(task)]
        if not allowed_tasks:
            rejected[mid] = "dos_task_not_static_or_uniform"
            continue
        counts["dos_static_or_uniform_tasks"] += len(allowed_tasks)
        values: list[tuple[str, Any, float]] = []
        for task in allowed_tasks:
            task_id = str(_get(task, "task_id", ""))
            try:
                dos = dos_loader(task_id)
                if dos is None:
                    raise ValueError("Materials Project returned no DOS curve")
                value = dos_at_fermi_states_ev_per_atom(dos, nsites=_site_count(doc))
                counts["dos_curves_retrieved"] += 1
                values.append((task_id, task, value))
            except Exception as exc:
                response = getattr(exc, "response", {})
                code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
                error_name = f"{type(exc).__name__}:{code}" if code else type(exc).__name__
                validation_error_counts[error_name] += 1
                if len(validation_error_samples) < 5:
                    validation_error_samples.append({
                        "task_id": task_id,
                        "error_type": type(exc).__name__,
                        "error_code": str(code or ""),
                        "message": str(exc).replace("\n", " ")[:240],
                    })
                continue
        if not values:
            counts["dos_missing_task_or_curve"] += 1
            rejected[mid] = "static_or_uniform_dos_curve_missing_or_invalid"
            continue
        qualifying_values = [item for item in values if dos_ef_rejection(item[2]) is None]
        if not qualifying_values:
            counts["dos_below_or_equal_0p1_states_ev_atom"] += 1
            rejected[mid] = "dos_ef_le_0.1_states_per_ev_per_atom"
            continue
        counts["dos_qualified_gt_0p1_states_ev_atom"] += 1
        task_id, selected_task, value = max(
            (item for item in values if item[2] > MIN_DOS_EF_STATES_EV_PER_ATOM),
            key=lambda item: item[2],
        )
        selected_task_type = _get(selected_task, "task_type") or _get(selected_task, "calc_type") or ""
        selected_run_type = _get(selected_task, "run_type") or ""
        qualified.append((doc, {
            "dos_ef_states_ev_per_atom": value,
            "dos_task_id": task_id,
            "dos_task_type": str(getattr(selected_task_type, "value", selected_task_type)),
            "dos_functional_run_type": str(getattr(selected_run_type, "value", selected_run_type)),
        }))
    diagnostics = {
        "error_counts": dict(sorted(validation_error_counts.items())),
        "error_samples": validation_error_samples,
    }
    return qualified, counts, rejected, diagnostics


def qualify_by_material_level_dos(
    docs: Iterable[Any],
    electronic_structure_docs: Iterable[Any],
    task_docs: Iterable[Any],
    dos_loader: Any,
    *,
    max_workers: int = 8,
    progress_every: int = 250,
) -> tuple[list[tuple[Any, dict[str, Any]]], dict[str, int], dict[str, str], dict[str, Any]]:
    """Use MP's material-ID DOS endpoint after checking eligible task metadata.

    The DOS curve is retrieved by material ID, not by a possibly stale or
    missing task ID. We still require the material's DOS metadata to reference
    at least one Static/Uniform calculation and record that the material-level
    endpoint does not identify which eligible task supplied its curve.
    """
    es_by_id = {str(_get(doc, "material_id", "")): doc for doc in electronic_structure_docs}
    task_by_id = {str(_get(task, "task_id", "")): task for task in task_docs}
    qualified: list[tuple[Any, dict[str, Any]]] = []
    rejected: dict[str, str] = {}
    errors: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    counts = {
        "dos_metadata_returned": len(es_by_id), "dos_task_references": 0,
        "dos_static_or_uniform_tasks": 0, "dos_non_static_uniform_tasks_rejected": 0,
        "dos_curves_retrieved": 0, "dos_missing_task_or_curve": 0,
        "dos_below_or_equal_0p1_states_ev_atom": 0, "dos_qualified_gt_0p1_states_ev_atom": 0,
        "dos_material_ids_queried": 0,
    }
    eligible_docs: list[tuple[Any, str, list[Any]]] = []
    for doc in docs:
        material_id = str(_get(doc, "material_id", ""))
        refs = _electronic_structure_task_ids(es_by_id.get(material_id))
        if not refs:
            counts["dos_missing_task_or_curve"] += 1
            rejected[material_id] = "no_static_or_uniform_dos_task_reference"
            continue
        counts["dos_task_references"] += 1
        tasks = [task_by_id[task_id] for task_id in refs if task_id in task_by_id]
        if not tasks:
            counts["dos_missing_task_or_curve"] += 1
            rejected[material_id] = "dos_task_record_missing"
            continue
        allowed = [task for task in tasks if _task_is_static_or_uniform(task)]
        counts["dos_non_static_uniform_tasks_rejected"] += len(tasks) - len(allowed)
        if not allowed:
            rejected[material_id] = "dos_task_not_static_or_uniform"
            continue
        counts["dos_static_or_uniform_tasks"] += len(allowed)
        eligible_docs.append((doc, material_id, allowed))

    # Resolve each material to its eligible calculation task(s), then fetch
    # the parsed DOS curve through the exact task route used by deep enrichment.
    dos_by_material: dict[str, tuple[Any, str]] = {}
    error_by_material: dict[str, Exception] = {}
    workers = max(1, min(int(max_workers), len(eligible_docs))) if eligible_docs else 0
    if eligible_docs:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    dos_loader,
                    material_id,
                    [str(_get(task, "task_id")) for task in allowed if _get(task, "task_id")],
                ): material_id
                for _doc, material_id, allowed in eligible_docs
            }
            for index, future in enumerate(as_completed(futures), start=1):
                material_id = futures[future]
                try:
                    dos_by_material[material_id] = future.result()
                except Exception as exc:
                    error_by_material[material_id] = exc
                if progress_every > 0 and (index % progress_every == 0 or index == len(futures)):
                    print(f"Fetched material-level DOS curves for {index}/{len(futures)} eligible candidates...", flush=True)
    counts["dos_material_ids_queried"] = len(eligible_docs)

    for doc, material_id, allowed in eligible_docs:
        try:
            if material_id in error_by_material:
                raise error_by_material[material_id]
            dos, selected_task_id = dos_by_material.get(material_id, (None, ""))
            if dos is None:
                raise ValueError("No parsed DOS curve returned for the material's eligible tasks")
            value = dos_at_fermi_states_ev_per_atom(dos, nsites=_site_count(doc))
            counts["dos_curves_retrieved"] += 1
        except Exception as exc:
            response = getattr(exc, "response", {})
            code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
            errors[f"{type(exc).__name__}:{code}" if code else type(exc).__name__] += 1
            if len(samples) < 5:
                samples.append({"material_id": material_id, "error_type": type(exc).__name__,
                               "error_code": str(code or ""), "message": str(exc).replace("\n", " ")[:240]})
            counts["dos_missing_task_or_curve"] += 1
            rejected[material_id] = "material_level_dos_curve_missing_or_invalid"
            continue
        if dos_ef_rejection(value):
            counts["dos_below_or_equal_0p1_states_ev_atom"] += 1
            rejected[material_id] = "dos_ef_le_0.1_states_per_ev_per_atom"
            continue
        counts["dos_qualified_gt_0p1_states_ev_atom"] += 1
        task_types = sorted({str(getattr(_get(task, "task_type") or _get(task, "calc_type") or "", "value", _get(task, "task_type") or _get(task, "calc_type") or "")) for task in allowed})
        run_types = sorted({str(getattr(_get(task, "run_type") or "", "value", _get(task, "run_type") or "")) for task in allowed})
        qualified.append((doc, {
            "dos_ef_states_ev_per_atom": value,
            "dos_task_id": selected_task_id,
            "dos_task_type": ";".join(task_types),
            "dos_functional_run_type": ";".join(run_types),
            "dos_retrieval_method": "material_id -> verified static/uniform task_id -> parsed DOS",
        }))
    return qualified, counts, rejected, {
        "error_counts": dict(sorted(errors.items())), "error_samples": samples,
        "retrieval_method": "material_id_to_verified_task_id", "workers": workers,
    }


def prefetch_static_dos_curves(
    electronic_structure_docs: Iterable[Any],
    task_docs: Iterable[Any],
    dos_loader: Any,
    *,
    max_workers: int = 8,
    progress_every: int = 1000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch unique eligible DOS payloads concurrently with a bounded pool.

    MP task metadata is still checked before a DOS request is made. Keeping
    concurrency bounded avoids serial multi-hour S3 retrievals while remaining
    gentle on the public-data endpoint.
    """
    task_by_id = {str(_get(task, "task_id", "")): task for task in task_docs}
    task_ids = sorted({
        task_id
        for doc in electronic_structure_docs
        for task_id in _electronic_structure_task_ids(doc)
        if task_id in task_by_id and _task_is_static_or_uniform(task_by_id[task_id])
    })
    results: dict[str, Any] = {}
    error_counts: Counter[str] = Counter()
    error_samples: list[dict[str, str]] = []
    if not task_ids:
        return results, {
            "eligible_task_ids": 0,
            "curves_returned": 0,
            "error_counts": {},
            "error_samples": [],
        }
    workers = max(1, min(int(max_workers), len(task_ids)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(dos_loader, task_id): task_id for task_id in task_ids}
        for index, future in enumerate(as_completed(futures), start=1):
            task_id = futures[future]
            try:
                results[task_id] = future.result()
            except Exception as exc:
                # qualify_by_dos records missing/invalid payloads by material.
                results[task_id] = None
                response = getattr(exc, "response", {})
                code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
                error_counts[f"{type(exc).__name__}:{code}" if code else type(exc).__name__] += 1
                if len(error_samples) < 5:
                    error_samples.append({
                        "task_id": task_id,
                        "error_type": type(exc).__name__,
                        "error_code": str(code or ""),
                        "message": str(exc).replace("\n", " ")[:240],
                    })
            if progress_every > 0 and (index % progress_every == 0 or index == len(task_ids)):
                print(f"Processed {index}/{len(task_ids)} eligible DOS fetches...", flush=True)
    diagnostics = {
        "eligible_task_ids": len(task_ids),
        "curves_returned": sum(value is not None for value in results.values()),
        "error_counts": dict(sorted(error_counts.items())),
        "error_samples": error_samples,
    }
    print(
        "DOS fetch diagnostics: " + json.dumps({
            key: value for key, value in diagnostics.items() if key != "error_samples"
        }, sort_keys=True),
        flush=True,
    )
    return results, diagnostics


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


def eligible_family_metrics(report: dict[str, Any], threshold: float = MIN_FAMILY_SPEARMAN) -> dict[str, float]:
    """Read the main evaluator's within-family Spearman values for the gate."""

    within = report.get("within_family_evaluation", {})
    eligible: dict[str, float] = {}
    for family, metric in within.get("families", {}).items():
        rho = metric.get("spearman_rho")
        if (metric.get("status") == "ready" and rho is not None
                and math.isfinite(float(rho)) and float(rho) > threshold):
            eligible[str(family)] = float(rho)
    return eligible


def family_definer_rejection(family: str, formula: str) -> str | None:
    """Prevent assignment to an iron-based or cuprate family without its core."""
    elements = set(formula_elements(formula))
    normalized = family.casefold()
    if "ferrite" in normalized and "Fe" not in elements:
        return "family_definer_missing:Fe"
    if "cuprate" in normalized and not {"Cu", "O"}.issubset(elements):
        return "family_definer_missing:Cu_and_O"
    return None


def _candidate_record(formula: str, mp_id: str, hull: float, family: str) -> HarmonizedRecord:
    from rt_superconductor.targets import ATMOSPHERIC_PRESSURE_GPA

    digest = hashlib.sha256(f"mp-existing:{mp_id}:{formula}".encode()).hexdigest()[:24]
    return HarmonizedRecord(
        record_id=digest,
        source_db="uci",
        formula=_descriptor_formula(formula),
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
    qualified: list[tuple[Any, dict[str, Any]]],
    training_records: list[HarmonizedRecord],
    eligible_metrics: dict[str, float],
    unresolved_element_systems: set[tuple[str, ...]] | frozenset[tuple[str, ...]] = frozenset(),
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, str]]:
    """Assign and score only close candidates in families passing main's rho gate."""

    family_records: dict[str, list[HarmonizedRecord]] = defaultdict(list)
    for record in training_records:
        family_records[record.chemical_family].append(record)
    record_distance_cache = [(record, _composition_fractions(record.formula)) for record in training_records]
    prepared: list[dict[str, Any]] = []
    assigned_counts: dict[str, int] = defaultdict(int)
    skipped_no_nearest = 0
    skipped_family_gate = 0
    rejected: dict[str, str] = {}
    unassigned = 0
    distance_rejected = 0
    definer_rejected = 0
    for doc, dos_data in qualified:
        formula = _summary_formula(doc)
        mp_id = str(_get(doc, "material_id", ""))
        if formula is None:
            skipped_no_nearest += 1
            if mp_id:
                rejected[mp_id] = "formula_unparseable_at_family_assignment"
            continue
        try:
            hull = float(_get(doc, "energy_above_hull"))
        except (TypeError, ValueError):
            skipped_no_nearest += 1
            if mp_id:
                rejected[mp_id] = "hull_missing_at_family_assignment"
            continue
        sg = _spacegroup_number(doc)
        nearest, nearest_sc = _nearest_training_records_precomputed(formula, sg, record_distance_cache)
        if nearest is None:
            skipped_no_nearest += 1
            if mp_id:
                rejected[mp_id] = "no_nearest_training_record"
            continue
        nearest_record, family_distance = nearest
        family = nearest_record.chemical_family
        definer_reason = family_definer_rejection(family, formula)
        if definer_reason:
            unassigned += 1
            definer_rejected += 1
            rejected[mp_id] = definer_reason
            continue
        if family_distance > MAX_FAMILY_DISTANCE + 1e-12:
            unassigned += 1
            distance_rejected += 1
            rejected[mp_id] = f"unassigned_family_distance_gt_{MAX_FAMILY_DISTANCE:.1f}"
            continue
        assigned_counts[family] += 1
        if family not in eligible_metrics:
            skipped_family_gate += 1
            rejected[mp_id] = "assigned_family_failed_within_family_spearman_gate"
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
            "dos_ef_states_ev_per_atom": dos_data["dos_ef_states_ev_per_atom"],
            "dos_task_type": dos_data["dos_task_type"],
            "dos_functional_run_type": dos_data["dos_functional_run_type"],
            "dos_task_id": dos_data["dos_task_id"],
            "dos_retrieval_method": dos_data.get("dos_retrieval_method", "task_id"),
            "chemical_family": family,
            "within_family_spearman": eligible_metrics[family],
            "family_assignment_distance": family_distance,
            "nearest_training_record": nearest_record,
            "nearest_known_superconductor": nearest_sc_record,
            "nearest_known_superconductor_distance": nearest_sc_distance,
            "unresolved_source_formula_overlap": formula_elements(formula) in unresolved_element_systems,
        })
    counts = {
        "family_assigned_within_distance_and_definers": sum(assigned_counts.values()),
        "family_gate_passed": len(prepared),
        "family_gate_skipped": skipped_family_gate,
        "no_nearest_training_record": skipped_no_nearest,
        "family_unassigned": unassigned,
        "family_assignment_distance_rejected": distance_rejected,
        "family_definer_rejected": definer_rejected,
        **{f"family_assigned:{family}": count for family, count in sorted(assigned_counts.items())},
    }
    if not prepared:
        return [], counts, rejected

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
            scored.append({
                "mp_id": row["mp_id"],
                "formula": row["formula"],
                "predicted_tc_kelvin": predicted,
                "uncertainty_k": uncertainty,
                "conservative_score_tc_minus_sd_k": predicted - uncertainty,
                "chemical_family": family,
                "within_family_spearman": row["within_family_spearman"],
                "energy_above_hull_ev_per_atom": row["hull"],
                "dos_ef_states_ev_per_atom": row["dos_ef_states_ev_per_atom"],
                "dos_task_type": row["dos_task_type"],
                "dos_functional_run_type": row["dos_functional_run_type"],
                "dos_task_id": row["dos_task_id"],
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
                "dft_verification": "pending",
            })
    return scored, counts, rejected


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


def pool_run_status(
    shortlist_rows: int,
    filter_counts: dict[str, Any],
    eligible_family_metrics: dict[str, float],
) -> str:
    """Distinguish a valid empty shortlist from a blocked DOS data fetch."""
    if shortlist_rows:
        return "ready"
    if (filter_counts.get("ordered_structures", 0)
            and filter_counts.get("dos_task_references", 0) == 0):
        return "blocked_dos_task_reference_missing"
    if (filter_counts.get("dos_static_or_uniform_tasks", 0)
            and filter_counts.get("dos_curves_retrieved", 0) == 0):
        return "blocked_dos_curve_retrieval"
    if eligible_family_metrics:
        return "ready_empty_after_family_gate"
    return "blocked_no_family_passed_spearman_gate"


def _write_shortlist(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in OUTPUT_COLUMNS})


def _read_old_shortlist(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    return _csv_rows(path)


def _render_summary(path: Path, report: dict[str, Any]) -> None:
    """Write the requested human-readable filter and before/after audit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Existing-but-untested Materials Project shortlist (v2)",
        "",
        "This is a computational screening list, not evidence that any entry superconducts or is globally untested. The source comparison covers 3DSC, UCI, Stanev, SuperCon2, and SuperMat. Metallic acceptance requires a material-level Materials Project DOS(Ef) above 0.1 states/eV/atom, plus at least one referenced task whose task_type or calc_type is Static or Uniform. The material-level endpoint does not identify which task supplied the curve; task IDs are therefore left blank. Relaxation DOS and band gaps are not substitutes.",
        "",
        f"Run status: **{report['status']}**. Shortlist size: **{report['shortlist_rows']}**; eligible families: **{len(report['eligible_family_metrics'])}**.",
        "",
        "## Counts per filter",
        "",
        "Counts are cumulative at each stage unless the stage is explicitly a rejection counter.",
        "",
        "| Stage | Count |",
        "|---|---:|",
    ]
    for key, value in report["filter_counts"].items():
        lines.append(f"| `{key}` | {value} |")
    diagnostics = report.get("dos_fetch_diagnostics", {})
    lines += [
        "",
        "## DOS payload retrieval diagnostics",
        "",
        f"Material-level DOS retrievals attempted: {diagnostics.get('eligible_material_ids', 0)}; "
        f"curves returned: {diagnostics.get('curves_returned', 0)}; "
        f"fetch errors by type: `{json.dumps(diagnostics.get('error_counts', {}), sort_keys=True)}`.",
        f"Curve-validation errors by type: "
        f"`{json.dumps(diagnostics.get('curve_validation_error_counts', {}), sort_keys=True)}`.",
    ]
    fallback = diagnostics.get("formula_task_reference_fallback", {})
    if fallback:
        lines.append(
            "Formula-based task lookup recovered "
            f"{fallback.get('task_refs_recovered', 0)} task references for "
            f"{fallback.get('materials_recovered', 0)} exact-structure matches from "
            f"{fallback.get('materials_missing_es_task_refs', 0)} candidates missing DOS references; "
            f"task documents returned: {fallback.get('task_documents_returned', 0)}; "
            f"non-static/uniform tasks rejected: {fallback.get('non_static_uniform_tasks_rejected', 0)}; "
            f"structure comparisons/matches: {fallback.get('structure_comparisons', 0)}/"
            f"{fallback.get('structure_matches', 0)}; "
            f"lookup errors: `{json.dumps(fallback.get('error_counts', {}), sort_keys=True)}`."
        )
        for sample in fallback.get("error_samples", []):
            lines.append(
                f"- Formula lookup error: {sample.get('error_type', '')} — {sample.get('message', '')}"
            )
    for sample in diagnostics.get("error_samples", []) + diagnostics.get("curve_validation_error_samples", []):
        lines.append(
            f"- Example `{sample.get('material_id') or sample.get('task_id', '')}`: {sample.get('error_type', '')} "
            f"{sample.get('error_code', '')} — {sample.get('message', '')}"
        )
    lines += ["", "## Family gate and shortlist", "", "| Family | Main within-family Spearman | Assigned candidates | Scored | Top 30 |", "|---|---:|---:|---:|---:|"]
    for family, values in sorted(report["families"].items()):
        lines.append(
            f"| {family} | {values['within_family_spearman']:.4f} | {values['assigned_count']} | "
            f"{values['scored_count']} | {values['shortlist_count']} |"
        )
    if not report["families"]:
        lines.append("| *(none passed the main within-family Spearman gate)* | — | — | — | 0 |")
    lines += ["", "### Assignment/scoring outcomes", "", "| Outcome | Count |", "|---|---:|"]
    for key, value in report["scoring_counts"].items():
        lines.append(f"| `{key}` | {value} |")
    lines += ["", "Rejected candidate counts are keyed by MP ID in the report JSON under `candidate_assignment_outcomes`; unassigned records carry a reason and are never scored.", ""]
    lines += [
        "",
        "The family gate uses `within_family_evaluation.families[*].spearman_rho` from `train_pressure_aware.py`. That evaluator keeps shared Materials Project IDs in the same validation fold. Candidates are assigned only when the nearest training record is within distance 0.3; Ferrites require Fe, and Cuprates require both Cu and O.",
        "",
        "## Previous shortlist: before and after", "",
        f"Previous rows: {report['old_shortlist_rows']}; retained in this top-30 output: {report['old_shortlist_retained']}; removed: {report['old_shortlist_removed']}.",
        "",
        "| Old rank | MP ID | Formula | Old family | V2 disposition | Reason |",
        "|---:|---|---|---|---|---|",
    ]
    for row in report["old_shortlist_comparison"]:
        lines.append(
            f"| {row.get('rank_within_family', '')} | {row.get('mp_id', '')} | {row.get('formula', '')} | "
            f"{row.get('chemical_family', '')} | {row['disposition']} | {row['reason']} |"
        )
    if not report["old_shortlist_comparison"]:
        lines.append("| — | — | — | — | baseline unavailable | Prior shortlist CSV was not provided to the CI run. |")
    lines += [
        "",
        "## Limits", "",
        "Materials Project formation energies, ordering, and DOS are DFT outputs. The random-forest Tc values are family-specific screening predictions with tree-spread uncertainty, not calibrated probabilities or discoveries. The term ‘untested’ only means absent by reduced formula from the five indexed exports; a literature review is still required. Candidate ranking is pending independent DFT and experimental validation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _fetch_mp_with_client(
    mpr: Any,
    source_index: FormulaSourceIndex,
) -> tuple[list[tuple[Any, dict[str, Any]]], dict[str, int], dict[str, str], dict[str, Any]]:
    """Query MP, retain ordered structures, and require static/uniform DOS(Ef)."""
    summary_fields = [
        "material_id", "formula_pretty", "composition_reduced", "energy_above_hull", "nsites",
        "symmetry",
    ]
    print("Querying Materials Project summaries for stable materials with at most 20 sites...", flush=True)
    docs = mpr.materials.summary.search(
        energy_above_hull=(0, HULL_MAX_EV_PER_ATOM),
        num_sites=(1, MAX_SITES),
        deprecated=False,
        fields=summary_fields,
        all_fields=False,
        chunk_size=1000,
    )
    unmatched, counts, rejected = _summary_filter_without_structures(
        docs, set(source_index.formulas), set(source_index.unresolved_element_systems)
    )
    print(
        f"Summary query returned {counts['mp_summary_returned']} records; "
        f"{len(unmatched)} remain after stability, size, known-source, and composition screens.",
        flush=True,
    )
    material_ids = sorted({str(_get(doc, "material_id")) for doc in unmatched if _get(doc, "material_id")})
    structures: dict[str, Any] = {}
    structure_batches = list(range(0, len(material_ids), 1000))
    for batch_number, start in enumerate(structure_batches, start=1):
        structure_docs = mpr.materials.summary.search(
            material_ids=material_ids[start:start + 1000],
            fields=["material_id", "structure"],
            all_fields=False,
            chunk_size=1000,
        )
        for structure_doc in structure_docs:
            material_id = _get(structure_doc, "material_id")
            structure = _get(structure_doc, "structure")
            if material_id and structure is not None:
                structures[str(material_id)] = structure
        print(
            f"Fetched structures batch {batch_number}/{len(structure_batches)}; "
            f"{len(structures)} structures returned so far.",
            flush=True,
        )
    ordered_unmatched = []
    for doc in unmatched:
        material_id = str(_get(doc, "material_id", ""))
        structure = structures.get(material_id)
        if structure is None:
            rejected[material_id] = "structure_missing"
            continue
        merged_doc = {
            name: _get(doc, name)
            for name in (
                "material_id", "formula_pretty", "composition_reduced", "energy_above_hull",
                "nsites", "symmetry",
            )
        }
        merged_doc["structure"] = structure
        if _ordered(merged_doc):
            ordered_unmatched.append(merged_doc)
        else:
            rejected[material_id] = "structure_disordered"
    counts["structure_material_ids_queried"] = len(material_ids)
    counts["structure_documents_returned"] = len(structures)
    counts["missing_structure_records"] = len(material_ids) - len(structures)
    counts["ordered_structures"] = len(ordered_unmatched)
    counts["ordered_and_source_unmatched"] = len(ordered_unmatched)
    material_ids = sorted({str(_get(doc, "material_id")) for doc in ordered_unmatched})
    es_docs: list[Any] = []
    for start in range(0, len(material_ids), 500):
        batch_ids = material_ids[start:start + 500]
        if batch_ids:
            es_docs.extend(_search_electronic_structure_raw(
                mpr.materials.electronic_structure, batch_ids, chunk_size=500
            ))
        print(f"Fetched DOS metadata for {min(start + 500, len(material_ids))}/{len(material_ids)} structures...", flush=True)
    fallback_task_refs, fallback_diagnostics = _candidate_dos_task_refs_by_formula(
        mpr.materials.tasks, ordered_unmatched, es_docs
    )
    # DOS summaries sometimes omit their nested task IDs. Formula lookup only
    # recovers references for the exact candidate MP material ID; it never
    # borrows another polymorph's calculation.
    es_docs.extend(
        {"material_id": material_id, "dos": {"task_ids": task_ids}}
        for material_id, task_ids in sorted(fallback_task_refs.items())
    )
    counts["dos_formula_fallback_materials_missing_refs"] = fallback_diagnostics[
        "materials_missing_es_task_refs"
    ]
    counts["dos_formula_fallback_materials_recovered"] = fallback_diagnostics["materials_recovered"]
    counts["dos_formula_fallback_task_refs_recovered"] = fallback_diagnostics["task_refs_recovered"]
    counts["dos_formula_fallback_query_errors"] = sum(fallback_diagnostics["error_counts"].values())
    counts["dos_formula_fallback_task_documents_returned"] = fallback_diagnostics["task_documents_returned"]
    counts["dos_formula_fallback_non_static_uniform_tasks_rejected"] = fallback_diagnostics[
        "non_static_uniform_tasks_rejected"
    ]
    counts["dos_formula_fallback_structure_comparisons"] = fallback_diagnostics["structure_comparisons"]
    counts["dos_formula_fallback_structure_matches"] = fallback_diagnostics["structure_matches"]
    dos_task_ids = sorted({task_id for doc in es_docs for task_id in _electronic_structure_task_ids(doc)})
    tasks: list[Any] = []
    for batch_number, start in enumerate(range(0, len(dos_task_ids), 100), start=1):
        tasks.extend(mpr.materials.tasks.search(
            task_ids=dos_task_ids[start:start + 100],
            fields=["task_id", "run_type", "task_type", "calc_type"],
            all_fields=False,
            chunk_size=100,
        ))
        print(
            f"Fetched calculation metadata batch {batch_number}/{math.ceil(len(dos_task_ids) / 100)}; "
            f"{len(tasks)} task documents returned so far.",
            flush=True,
        )
    # Map each material to its exact eligible task. MP's convenience
    # get_dos_by_material_id currently assumes a nested task_id that can be
    # absent, so use the proven deep-enrichment task resolver and record the ID.
    with material_level_dos_lookup(mpr) as load_material_dos:
        qualified, dos_counts, dos_rejected, dos_validation_diagnostics = qualify_by_material_level_dos(
            ordered_unmatched,
            es_docs,
            tasks,
            load_material_dos,
        )
    dos_fetch_diagnostics = {
        "retrieval_method": "candidate material_id -> eligible static/uniform task_id -> parsed DOS",
        "eligible_material_ids": dos_counts["dos_material_ids_queried"],
        "curves_returned": dos_counts["dos_curves_retrieved"],
        "error_counts": dos_validation_diagnostics["error_counts"],
        "error_samples": dos_validation_diagnostics["error_samples"],
        "formula_task_reference_fallback": fallback_diagnostics,
        "curve_validation_error_counts": dos_validation_diagnostics["error_counts"],
        "curve_validation_error_samples": dos_validation_diagnostics["error_samples"],
        "dos_curve_task_linkage": "exact candidate material-to-task reference; selected DOS task ID recorded",
        "material_level_request_workers": dos_validation_diagnostics["workers"],
    }
    counts.update(dos_counts)
    counts["dos_task_ids_queried"] = len(dos_task_ids)
    counts["task_documents_returned"] = len(tasks)
    counts["dos_material_ids_queried"] = dos_counts["dos_material_ids_queried"]
    counts["dos_payload_fetch_failures"] = max(
        0, dos_fetch_diagnostics["eligible_material_ids"] - dos_fetch_diagnostics["curves_returned"]
    )
    counts["dos_curve_validation_failures"] = sum(dos_validation_diagnostics["error_counts"].values())
    rejected.update(dos_rejected)
    counts["source_formula_unmatched"] = len(unmatched)
    counts["metallic_dos_qualified"] = len(qualified)
    return qualified, counts, rejected, dos_fetch_diagnostics


def _fetch_mp(api_key: str, source_index: FormulaSourceIndex) -> tuple[list[tuple[Any, dict[str, Any]]], dict[str, int], dict[str, str], dict[str, Any]]:
    if not api_key:
        raise ValueError("MP_API_KEY is required; this script must run in the configured CI workflow")
    from mp_api.client import MPRester

    with MPRester(api_key) as mpr:
        return _fetch_mp_with_client(mpr, source_index)


def build_untested_pool(
    *,
    input_path: Path,
    model_dir: Path,
    report_path: Path,
    output_path: Path,
    source_paths: dict[str, Path],
    api_key: str | None = None,
    top_per_family: int = DEFAULT_TOP_PER_FAMILY,
    old_shortlist_path: Path | None = None,
    summary_path: Path | None = None,
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
                "family_metric_not_ready" if metric.get("status") != "ready"
                else "spearman_not_strictly_above_0.5"
            ),
        }
        for family, metric in sorted(family_metrics.items())
    }
    records = select_training_records(load_harmonized(input_path))
    if not records:
        raise ValueError("there are no empirical training records to assign families or score candidates")

    old_rows = _read_old_shortlist(old_shortlist_path)
    qualified, filter_counts, filter_rejections, dos_fetch_diagnostics = _fetch_mp(
        api_key or os.environ.get("MP_API_KEY", ""), source_index
    )
    scored, score_counts, score_rejections = score_family_candidates(
        qualified, records, eligible, source_index.unresolved_element_systems
    )
    shortlist = select_top_per_family(scored, top_per_family)
    _write_shortlist(output_path, shortlist)
    qualified_ids = {str(_get(doc, "material_id", "")) for doc, _ in qualified}
    scored_ids = {row["mp_id"] for row in scored}
    shortlist_ids = {row["mp_id"] for row in shortlist}
    old_comparison: list[dict[str, str]] = []
    for old in old_rows:
        mid = old.get("mp_id", "")
        if mid in shortlist_ids:
            disposition, reason = "retained", "still ranks within the v2 top 30 for its family"
        elif mid in scored_ids:
            disposition, reason = "removed", "accepted and scored, but ranked below the v2 top 30 for its family"
        elif mid in score_rejections:
            disposition, reason = "removed", score_rejections[mid]
        elif mid in filter_rejections:
            disposition, reason = "removed", filter_rejections[mid]
        elif mid not in qualified_ids:
            disposition, reason = "removed", "not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query"
        else:
            disposition, reason = "removed", "excluded during current family assignment or scoring"
        old_comparison.append({
            **old,
            "disposition": disposition,
            "reason": reason,
        })
    scored_family_by_id = {row["mp_id"]: row["chemical_family"] for row in scored}
    candidate_outcomes = {
        str(_get(doc, "material_id", "")): f"scored:{scored_family_by_id[str(_get(doc, 'material_id', ''))]}"
        for doc, _ in qualified if str(_get(doc, "material_id", "")) in scored_family_by_id
    }
    candidate_outcomes.update(score_rejections)
    filter_rejection_counts: dict[str, int] = defaultdict(int)
    for reason in filter_rejections.values():
        filter_rejection_counts[reason] += 1
    score_rejection_counts: dict[str, int] = defaultdict(int)
    for reason in score_rejections.values():
        score_rejection_counts[reason] += 1
    family_summary: dict[str, dict[str, Any]] = {}
    for family, rho in sorted(eligible.items()):
        family_summary[family] = {
            "within_family_spearman": rho,
            "gate": "passed (> 0.5)",
            "assigned_count": score_counts.get(f"family_assigned:{family}", 0),
            "scored_count": sum(row["chemical_family"] == family for row in scored),
            "shortlist_count": sum(row["chemical_family"] == family for row in shortlist),
        }
    report: dict[str, Any] = {
        "schema_version": 2,
        "status": pool_run_status(len(shortlist), filter_counts, eligible),
        "interpretation": "Existing MP materials absent by reduced formula from normalized rows in the five specified data exports; not proof that no experiment has ever tested them.",
        "criteria": {
            "energy_above_hull_ev_per_atom_max": HULL_MAX_EV_PER_ATOM,
            "metallicity": f"Materials Project DOS(Ef) > {MIN_DOS_EF_STATES_EV_PER_ATOM} states/eV/atom from static or uniform task; relaxation tasks are rejected",
            "metallicity_task_type_fields": ["task_type", "calc_type"],
            "insulator_composition_policy": "shared chemistry_policy.insulator_rejection applied before scoring",
            "composition_exclusions": ["main chemistry_policy radioactive and insulator screens plus Tc/Pm/Po/Ra/Ac/Th/Pa/U/Np and Z>=93", "carbonates", "sulfates", "hydroxides", "nitrates", "phosphates", "alkali/alkaline-earth transition-metal fluorides", "stoichiometric BaCuO2 parent composition"],
            "ordered_structure_required": True,
            "max_sites": MAX_SITES,
            "known_sources": sorted(source_paths),
            "family_gate": "main train_pressure_aware.py within-family Spearman strictly greater than 0.5; shared MP IDs embargoed by that evaluator",
            "family_assignment": "nearest empirical record using 80% total-variation composition distance and 20% crystal-system/space-group distance, at distance <=0.3; Ferrites require Fe and Cuprates require Cu+O",
            "ranking": "family-specific random forest, ordered by predicted Tc minus tree-spread standard deviation",
            "top_per_eligible_family": top_per_family,
        },
        "source_formula_index": source_index.stats,
        "training_data": {
            "path": str(input_path),
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "empirical_records": len(records),
        },
        "within_family_report": str(pressure_report_path),
        "main_within_family_metrics": family_metrics,
        "family_gate_decisions": family_gate_decisions,
        "eligible_family_metrics": eligible,
        "filter_counts": filter_counts,
        "dos_fetch_diagnostics": dos_fetch_diagnostics,
        "filter_rejection_reason_counts": dict(sorted(filter_rejection_counts.items())),
        "scoring_counts": score_counts,
        "scoring_rejection_reason_counts": dict(sorted(score_rejection_counts.items())),
        "candidate_assignment_outcomes": candidate_outcomes,
        "pool_size_after_all_materials_project_and_database_filters": filter_counts.get("metallic_dos_qualified", 0),
        "pool_size_assigned_to_families": score_counts.get("family_assigned_within_distance_and_definers", 0),
        "pool_size_scored_after_family_gate": len(scored),
        "shortlist_rows": len(shortlist),
        "families": family_summary,
        "old_shortlist_rows": len(old_rows),
        "old_shortlist_retained": sum(row["disposition"] == "retained" for row in old_comparison),
        "old_shortlist_removed": sum(row["disposition"] == "removed" for row in old_comparison),
        "old_shortlist_comparison": old_comparison,
        "output_csv": str(output_path),
        "output_csv_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "output_dft_results_status": "pending",
        "limitations": [
            "The term untested is relative to 3DSC, UCI, the Stanev export, SuperCon2, and SuperMat; it does not establish a global literature absence.",
            "Some source formulas contain variables or markup that cannot be reduced; same-element-system overlaps are retained but visibly flagged for source review.",
            "Family assignment uses formula fractions and symmetry-number proxies because harmonized records do not all have CIFs; family distances above 0.3 remain unassigned.",
            "Predictions are family-specific random-forest estimates; tree spread is a screening uncertainty proxy and is not calibrated probability.",
            "Materials Project stability and DOS labels are computational; shortlisted candidate calculations and experimental verification remain pending.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if summary_path is not None:
        _render_summary(summary_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/training/harmonized_supercon.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/processed/continuous-training/pressure-aware"))
    parser.add_argument("--output", type=Path, default=Path("docs/untested-pool-shortlist.csv"))
    parser.add_argument("--report", type=Path, default=Path("data/processed/continuous-training/untested-pool-report.json"))
    parser.add_argument("--summary", type=Path, default=Path("docs/untested-pool-summary.md"))
    parser.add_argument("--old-shortlist", type=Path, default=Path("/tmp/untested-pool-v1.csv"))
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
        old_shortlist_path=args.old_shortlist,
        summary_path=args.summary,
    )
    print(json.dumps({
        "status": report["status"],
        "filter_counts": report["filter_counts"],
        "pool_size": report["pool_size_after_all_materials_project_and_database_filters"],
        "scored_pool_size": report["pool_size_scored_after_family_gate"],
        "shortlist_rows": report["shortlist_rows"],
        "old_shortlist_removed": report["old_shortlist_removed"],
        "family_counts": report["families"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
