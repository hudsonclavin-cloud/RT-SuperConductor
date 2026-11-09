import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_untested_pool import (  # noqa: E402
    FormulaSourceIndex,
    _candidate_record,
    _candidate_dos_task_refs_by_formula,
    _electronic_structure_task_ids,
    _search_electronic_structure_raw,
    _task_is_static_or_uniform,
    _render_summary,
    _summary_filter_without_structures,
    _fetch_mp,
    _source_index_fingerprint,
    _write_mp_snapshot,
    composition_structure_distance,
    dos_at_fermi_states_ev_per_atom,
    epw_eligibility_reason,
    eligible_family_metrics,
    family_definer_rejection,
    magnetism_classification,
    oxygen_agnostic_formula_key,
    material_level_dos_lookup,
    prefetch_static_dos_curves,
    pool_run_status,
    qualify_by_dos,
    qualify_by_material_level_dos,
    reduced_formula,
    score_family_candidates,
    summary_metallicity_evidence,
    select_top_per_family,
)
from untested_pool_policy import (  # noqa: E402
    radioactive_elements,
    untested_pool_candidate_rejection,
    untested_pool_composition_rejection,
)
from rt_superconductor.harmonized import HarmonizedRecord  # noqa: E402
from train_pressure_aware import purge_shared_mp_ids  # noqa: E402


def _summary(mp_id, formula, *, hull=0.01, sites=2, sg=225, structure=None,
             is_metal=True, band_gap=None, is_magnetic=None, ordering=None, total_magnetization=None):
    values = dict(
        material_id=mp_id,
        formula_pretty=formula,
        composition_reduced=None,
        energy_above_hull=hull,
        nsites=sites,
        symmetry=SimpleNamespace(number=sg),
        is_metal=is_metal,
        band_gap=band_gap,
        is_magnetic=is_magnetic,
        ordering=ordering,
        total_magnetization=total_magnetization,
    )
    if structure is not None:
        values["structure"] = structure
    return SimpleNamespace(**values)


class _Dos:
    def __init__(self, density, nsites=1):
        self.energies = np.asarray([0.0, 1.0, 2.0])
        self.efermi = 1.0
        self._densities = np.asarray([density, density, density])
        self.structure = [None] * nsites

    def get_densities(self):
        return self._densities


class UntestedPoolV2Tests(unittest.TestCase):
    def test_formula_normalization_uses_reduced_composition(self):
        self.assertEqual(reduced_formula("Fe2Se2"), "FeSe")
        self.assertEqual(reduced_formula("Nb3Sn"), "Nb3Sn")
        self.assertIsNone(reduced_formula("not a formula"))

    def test_oxygen_variants_of_known_source_formulas_are_rejected(self):
        self.assertEqual(oxygen_agnostic_formula_key("YBa2Cu3O6"), "Ba2YCu3")
        kept, counts, rejected = _summary_filter_without_structures(
            [_summary("mp-ybco-oxygen-variant", "YBa2Cu3O7")],
            known_formulas={"YBa2Cu3O6"},
        )
        self.assertEqual(kept, [])
        self.assertEqual(counts["oxygen_variant_of_known"], 1)
        self.assertEqual(rejected["mp-ybco-oxygen-variant"], "oxygen_variant_of_known")

    def test_mp_magnetism_is_copied_and_unknown_is_not_assumed_non_magnetic(self):
        self.assertEqual(magnetism_classification(_summary("m", "FeSe", is_magnetic=True)), "magnetic")
        self.assertEqual(magnetism_classification(_summary("m", "FeSe", ordering="AFM")), "magnetic")
        self.assertEqual(magnetism_classification(_summary("m", "Nb3Sn", ordering="NM")), "non_magnetic")
        self.assertEqual(magnetism_classification(_summary("m", "Nb3Sn", is_magnetic=False)), "non_magnetic")
        self.assertEqual(magnetism_classification(_summary("m", "Nb3Sn")), "unknown")

    def test_epw_list_requires_other_family_small_site_count_and_explicit_nonmagnetism(self):
        base = {
            "chemical_family": "Other", "num_sites": 20,
            "magnetism_classification": "non_magnetic",
        }
        self.assertEqual(epw_eligibility_reason(base), "eligible_non_magnetic_other_max_20_sites")
        self.assertEqual(epw_eligibility_reason({**base, "magnetism_classification": "magnetic"}), "magnetic_excluded_from_epw")
        self.assertEqual(epw_eligibility_reason({**base, "magnetism_classification": "unknown"}), "magnetism_unknown_excluded_from_epw")
        self.assertEqual(epw_eligibility_reason({**base, "num_sites": 21}), "site_count_gt_20")
        self.assertEqual(epw_eligibility_reason({**base, "chemical_family": "Cuprates"}), "family_not_other")

    def test_mp_summary_gate_does_not_use_band_gap_as_metallicity(self):
        docs = [
            _summary("mp-ok", "Nb3Sn"),
            _summary("mp-unstable", "FeSe", hull=0.03),
            _summary("mp-large", "Al", sites=21),
            _summary("mp-known", "CuO"),
            _summary("mp-radioactive", "ThFeAsO"),
        ]
        kept, counts, rejected = _summary_filter_without_structures(
            docs, known_formulas={"CuO"}
        )
        self.assertEqual([doc.material_id for doc in kept], ["mp-ok"])
        self.assertEqual(counts["mp_summary_returned"], 5)
        self.assertEqual(counts["energy_above_hull_le_0p025"], 4)
        self.assertEqual(counts["site_count_le_20"], 3)
        self.assertEqual(counts["composition_policy_rejected"], 1)
        self.assertEqual(rejected["mp-radioactive"], "radioactive_element:Th")

    def test_summary_metallicity_labels_are_diagnostic_not_pool_qualification(self):
        docs = [
            _summary("mp-is-metal", "Nb3Sn", is_metal=True, band_gap=0.4),
            _summary("mp-zero-gap", "Nb3Sn", is_metal=False, band_gap=0.0),
            _summary("mp-both", "Nb3Sn", is_metal=True, band_gap=0.0),
            _summary("mp-insulator", "Nb3Sn", is_metal=False, band_gap=0.2),
            _summary("mp-no-summary", "Nb3Sn", is_metal=None, band_gap=None),
        ]
        # These labels are exposed for diagnostics only. Production qualification
        # is exercised below via material-level DOS and static/uniform tasks.
        self.assertEqual(summary_metallicity_evidence(docs[2]), "is_metal_true;band_gap_zero")

    def test_required_composition_rejections(self):
        for formula, reason in (
            ("FeCO3", "carbonate_salt"),
            ("CoCO3", "carbonate_salt"),
            ("Ni(OH)2", "hydroxide_salt"),
            ("K2CuF4", "alkali_alkaline_earth_transition_metal_fluoride"),
            ("BaCuO2", "undoped_bacuO2_parent_composition"),
        ):
            with self.subTest(formula=formula):
                self.assertEqual(untested_pool_composition_rejection(formula), reason)
        self.assertEqual(untested_pool_composition_rejection("CuO"),
                         "binary_transition_metal_oxide_or_halide")
        self.assertEqual(untested_pool_candidate_rejection("BaCuO2", 1.0),
                         "undoped_bacuO2_parent_composition")
        self.assertEqual(untested_pool_candidate_rejection("Nb3Sn", 0.1),
                         "dos_ef_le_0.1_states_per_ev_per_atom")
        self.assertIsNone(untested_pool_candidate_rejection("Nb3Sn", 0.10001))

    def test_bacuO2_parent_exclusion_is_reduced_formula_aware(self):
        self.assertEqual(untested_pool_composition_rejection("Ba2Cu2O4"),
                         "undoped_bacuO2_parent_composition")
        self.assertIsNone(untested_pool_composition_rejection("Ba2YCu3O7"))

    def test_radioactive_exclusion_includes_named_elements_and_all_heavier(self):
        for formula, symbol in (("TcFe2", "Tc"), ("PmNi2", "Pm"), ("PoFe", "Po"),
                                ("RaCuO2", "Ra"), ("AcFeAsO", "Ac"), ("PaFeAsO", "Pa"),
                                ("UFeAsO", "U"), ("NpO2", "Np"), ("OgFe", "Og")):
            with self.subTest(formula=formula):
                self.assertIn(symbol, radioactive_elements(formula))

    def test_dos_is_normalized_per_atom_and_static_or_uniform_only(self):
        self.assertEqual(dos_at_fermi_states_ev_per_atom(_Dos(2.0, nsites=4)), 0.5)
        # MP's total DOS is a Dos object without its own crystal structure.
        mp_dos = SimpleNamespace(
            energies=np.asarray([-1.0, 0.0, 1.0]),
            efermi=0.0,
            get_densities=lambda: np.asarray([1.0, 2.0, 3.0]),
        )
        self.assertEqual(dos_at_fermi_states_ev_per_atom(mp_dos, nsites=4), 0.5)
        self.assertTrue(_task_is_static_or_uniform(SimpleNamespace(task_type="Static", run_type="GGA")))
        self.assertTrue(_task_is_static_or_uniform(SimpleNamespace(task_type="NSCF Uniform", run_type="GGA")))
        self.assertFalse(_task_is_static_or_uniform(SimpleNamespace(task_type="Structure Optimization", run_type="GGA")))
        self.assertFalse(_task_is_static_or_uniform(SimpleNamespace(task_type="Structure Optimization", calc_type="GGA Static", run_type="GGA")))
        self.assertFalse(_task_is_static_or_uniform(SimpleNamespace(task_type=None, calc_type=None, run_type="GGA+U")))

    def test_electronic_structure_search_uses_raw_dicts_for_incomplete_mp_dos(self):
        incomplete = {"material_id": "mp-1", "dos": {"total": {"1": {"band_gap": 0.0}}}}
        calls = []

        def search(**kwargs):
            calls.append(kwargs)
            return [incomplete]

        route = SimpleNamespace(use_document_model=True, search=search)
        docs = _search_electronic_structure_raw(route, ["mp-1"])
        self.assertEqual(docs, [incomplete])
        self.assertTrue(route.use_document_model)
        self.assertEqual(calls[0]["fields"], ["material_id", "dos"])
        self.assertEqual(_electronic_structure_task_ids(docs[0]), [])

    def test_formula_task_fallback_requires_same_structure_and_static_task(self):
        from pymatgen.core import Lattice, Structure

        candidate_structure = Structure(
            Lattice.cubic(4.0), ["Nb", "Nb", "Nb", "Sn"],
            [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        )
        other_polymorph = Structure(
            Lattice.cubic(5.0), ["Nb", "Nb", "Nb", "Sn"],
            [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        )
        candidate = _summary("mp-1", "Nb3Sn", structure=candidate_structure)
        unrelated = {
            "task_id": "wrong-phase-task", "formula_pretty": "Nb3Sn",
            "task_type": "Static", "structure": other_polymorph.as_dict(),
        }
        matched = {
            "task_id": "static-task", "formula_pretty": "Nb3Sn",
            "task_type": "Static", "structure": candidate_structure.as_dict(),
        }
        relaxation = {
            "task_id": "relax-task", "formula_pretty": "Nb3Sn",
            "task_type": "Structure Optimization", "structure": candidate_structure.as_dict(),
        }
        calls = []

        def search(**kwargs):
            calls.append((kwargs, route.use_document_model))
            return [matched, unrelated, relaxation]

        route = SimpleNamespace(use_document_model=True, search=search)
        refs, diagnostics = _candidate_dos_task_refs_by_formula(
            route, [candidate], [{"material_id": "mp-1", "dos": {"total": {"1": {"band_gap": 0.0}}}}]
        )
        self.assertEqual(refs, {"mp-1": ["static-task"]})
        self.assertEqual(diagnostics["materials_missing_es_task_refs"], 1)
        self.assertEqual(diagnostics["task_refs_recovered"], 1)
        self.assertEqual(diagnostics["non_static_uniform_tasks_rejected"], 1)
        self.assertEqual(diagnostics["structure_comparisons"], 2)
        self.assertEqual(diagnostics["structure_matches"], 1)
        self.assertFalse(calls[0][1])
        self.assertNotIn("material_id", calls[0][0]["fields"])
        self.assertTrue(route.use_document_model)

    def test_dos_qualification_rejects_relaxation_even_when_dos_is_high(self):
        docs = [_summary("mp-relax", "Nb3Sn"), _summary("mp-static", "AlCu")]
        es_docs = [
            SimpleNamespace(material_id="mp-relax", dos={"task_id": "relax-task"}),
            SimpleNamespace(material_id="mp-static", dos={"task_id": "static-task"}),
        ]
        tasks = [
            SimpleNamespace(task_id="relax-task", run_type="GGA", task_type="Structure Optimization"),
            SimpleNamespace(task_id="static-task", run_type="GGA", task_type="Static"),
        ]
        qualified, counts, rejected, diagnostics = qualify_by_dos(
            docs, es_docs, tasks, lambda _task_id: _Dos(1.0)
        )
        self.assertEqual([row[0].material_id for row in qualified], ["mp-static"])
        self.assertEqual(rejected["mp-relax"], "dos_task_not_static_or_uniform")
        self.assertEqual(counts["dos_non_static_uniform_tasks_rejected"], 1)
        self.assertEqual(diagnostics["error_counts"], {})

    def test_material_level_dos_retrieval_uses_material_id_after_task_metadata_gate(self):
        docs = [_summary("mp-static", "Nb3Sn", sites=2), _summary("mp-relax", "AlCu", sites=2)]
        es_docs = [
            {"material_id": "mp-static", "dos": {"task_id": "static-task"}},
            {"material_id": "mp-relax", "dos": {"task_id": "relax-task"}},
        ]
        tasks = [
            {"task_id": "static-task", "task_type": "Static", "run_type": "GGA"},
            {"task_id": "relax-task", "task_type": "Structure Optimization", "run_type": "GGA"},
        ]
        dos_calls = []

        def get_by_material_and_tasks(material_id, task_ids):
            dos_calls.append(material_id)
            self.assertEqual(task_ids, ["static-task"])
            return _Dos(1.0, nsites=2), "static-task"

        qualified, counts, rejected, diagnostics = qualify_by_material_level_dos(
            docs,
            es_docs,
            tasks,
            lambda material_id, _task_ids: get_by_material_and_tasks(material_id, ["static-task"]),
        )
        self.assertEqual(dos_calls, ["mp-static"])
        self.assertEqual([item[0].material_id for item in qualified], ["mp-static"])
        self.assertEqual(qualified[0][1]["dos_retrieval_method"],
                         "material_id -> verified static/uniform task_id -> parsed DOS")
        self.assertEqual(qualified[0][1]["dos_task_id"], "static-task")
        self.assertEqual(rejected["mp-relax"], "dos_task_not_static_or_uniform")
        self.assertEqual(counts["dos_material_ids_queried"], 1)
        self.assertEqual(diagnostics["error_counts"], {})

    def test_low_dos_is_rejected_by_the_required_gate(self):
        doc = _summary("mp-low-dos", "Nb3Sn", sites=2)
        task = {"task_id": "static-task", "task_type": "Static", "run_type": "GGA"}
        enriched, counts, rejected, _ = qualify_by_material_level_dos(
            [doc],
            [{"material_id": "mp-low-dos", "dos": {"task_id": "static-task"}}],
            [task],
            lambda _mid, _task_ids: (_Dos(0.1, nsites=2), "static-task"),
        )
        self.assertEqual(enriched, [])
        self.assertEqual(counts["dos_below_or_equal_0p1_states_ev_atom"], 1)
        self.assertEqual(rejected["mp-low-dos"], "dos_ef_le_0.1_states_per_ev_per_atom")

    def test_material_level_dos_lookup_uses_raw_route_and_restores_mode(self):
        route = SimpleNamespace(use_document_model=True)

        def get_from_task_id(task_id):
            self.assertEqual(task_id, "task-raw")
            self.assertFalse(route.use_document_model)
            return "raw-dos-payload"

        route.get_dos_from_task_id = get_from_task_id

        mpr = SimpleNamespace(
            materials=SimpleNamespace(electronic_structure_dos=route),
        )
        with material_level_dos_lookup(mpr) as loader:
            self.assertFalse(route.use_document_model)
            self.assertEqual(loader("mp-raw", ["task-raw"]), ("raw-dos-payload", "task-raw"))
        self.assertTrue(route.use_document_model)

    def test_dos_validation_errors_are_retained_for_run_diagnostics(self):
        invalid_dos = SimpleNamespace(
            energies=np.asarray([0.0, 1.0]),
            efermi=0.5,
            structure=[None],
            get_densities=lambda: np.asarray([1.0]),
        )
        qualified, counts, rejected, diagnostics = qualify_by_dos(
            [_summary("mp-invalid", "Nb3Sn")],
            [SimpleNamespace(material_id="mp-invalid", dos={"task_id": "static-task"})],
            [SimpleNamespace(task_id="static-task", task_type="Static")],
            lambda _task_id: invalid_dos,
        )
        self.assertEqual(qualified, [])
        self.assertEqual(rejected["mp-invalid"], "static_or_uniform_dos_curve_missing_or_invalid")
        self.assertEqual(diagnostics["error_counts"], {"ValueError": 1})
        self.assertIn("DOS grid", diagnostics["error_samples"][0]["message"])

    def test_dos_prefetch_only_loads_unique_static_or_uniform_tasks(self):
        es_docs = [
            SimpleNamespace(material_id="mp-1", dos={"task_id": "static"}),
            SimpleNamespace(material_id="mp-2", dos={"task_id": "static"}),
            SimpleNamespace(material_id="mp-3", dos={"task_id": "relax"}),
        ]
        tasks = [
            SimpleNamespace(task_id="static", task_type="Static"),
            SimpleNamespace(task_id="relax", task_type="Structure Optimization"),
        ]
        loaded = []

        def loader(task_id):
            loaded.append(task_id)
            return task_id

        cache, diagnostics = prefetch_static_dos_curves(
            es_docs, tasks, loader, max_workers=2, progress_every=0
        )
        self.assertEqual(cache, {"static": "static"})
        self.assertEqual(loaded, ["static"])
        self.assertEqual(diagnostics["curves_returned"], 1)
        _empty, empty_diagnostics = prefetch_static_dos_curves([], [], loader, progress_every=0)
        self.assertEqual(empty_diagnostics["curves_returned"], 0)

    def test_dos_prefetch_preserves_failure_type_for_audit(self):
        es_docs = [SimpleNamespace(material_id="mp-1", dos={"task_id": "missing"})]
        tasks = [SimpleNamespace(task_id="missing", task_type="Static")]

        def loader(_task_id):
            raise FileNotFoundError("missing public DOS object")

        _cache, diagnostics = prefetch_static_dos_curves(
            es_docs, tasks, loader, max_workers=1, progress_every=0
        )
        self.assertEqual(diagnostics["error_counts"], {"FileNotFoundError": 1})
        self.assertEqual(diagnostics["error_samples"][0]["message"], "missing public DOS object")

    def test_dos_threshold_is_strict(self):
        self.assertEqual(untested_pool_candidate_rejection("Nb3Sn", 0.1),
                         "dos_ef_le_0.1_states_per_ev_per_atom")
        self.assertIsNone(untested_pool_candidate_rejection("Nb3Sn", 0.100001))

    def test_family_definers_and_distance_cap_are_enforced(self):
        self.assertEqual(family_definer_rejection("Ferrites", "CoO"), "family_definer_missing:Fe")
        self.assertEqual(family_definer_rejection("Cuprates", "BaFeO2"), "family_definer_missing:Cu_and_O")
        self.assertIsNone(family_definer_rejection("Cuprates", "BaCuO2"))
        self.assertGreater(composition_structure_distance("FeS", 129, "FeSe", 129), 0.3)

    def test_main_family_metrics_drive_gate_without_builder_embargo_copy(self):
        report = {"within_family_evaluation": {"families": {
            "Pass": {"status": "ready", "spearman_rho": 0.51},
            "Threshold": {"status": "ready", "spearman_rho": 0.5},
            "Blocked": {"status": "blocked", "spearman_rho": 0.9},
        }}}
        self.assertEqual(eligible_family_metrics(report), {"Pass": 0.51})

    def test_pool_status_reports_required_dos_gate(self):
        counts = {
            "ordered_and_source_unmatched": 12,
            "dos_static_or_uniform_tasks": 12,
            "dos_curves_retrieved": 0,
        }
        self.assertEqual(pool_run_status(0, counts, {"Other": 0.7}), "ready_empty_after_required_dos_gate")
        self.assertEqual(pool_run_status(
            0, {"ordered_structures": 100, "dos_metadata_returned": 100, "dos_task_references": 0}, {"Other": 0.7}
        ), "ready_empty_after_family_gate")
        self.assertEqual(pool_run_status(1, counts, {"Other": 0.7}), "ready")
        self.assertEqual(pool_run_status(0, {"dos_curves_retrieved": 0}, {"Other": 0.7}), "ready_empty_after_family_gate")
        self.assertEqual(pool_run_status(
            0, {"ordered_and_source_unmatched": 100, "dos_ef_gt_0p1_required_gate": 0}, {}
        ), "ready_empty_after_required_dos_gate")

    def test_summary_documents_verified_material_to_task_dos_resolution(self):
        report = {
            "status": "ready_empty_after_required_dos_gate",
            "shortlist_rows": 0,
            "eligible_family_metrics": {"Other": 0.6},
            "filter_counts": {},
            "dos_fetch_diagnostics": {
                "eligible_material_ids": 2,
                "curves_returned": 0,
                "error_counts": {"OSError": 2},
                "curve_validation_error_counts": {},
                "retrieval_method": "candidate material_id -> eligible task_id -> parsed DOS",
                "formula_task_reference_fallback": {},
            },
            "families": {},
            "scoring_counts": {},
            "old_shortlist_rows": 0,
            "old_shortlist_retained": 0,
            "old_shortlist_removed": 0,
            "old_shortlist_comparison": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.md"
            _render_summary(path, report)
            summary = path.read_text(encoding="utf-8")
        self.assertIn("DOS(E_F) strictly above 0.1 states/eV/atom", summary)
        self.assertIn("Missing curves, low DOS, and relaxation/optimization tasks are rejected", summary)
        self.assertIn("candidate material_id -> eligible task_id -> parsed DOS", summary)
        self.assertNotIn("task IDs are therefore left blank", summary)
    def test_mp_snapshot_replays_offline_and_checks_source_fingerprint(self):
        from pymatgen.core import Lattice, Structure

        source_index = FormulaSourceIndex(
            formulas=frozenset({"CuO", "FeSe"}),
            unresolved_element_systems=frozenset({("Fe", "O")}),
            stats={"3dsc": {"sha256": "source-checksum"}},
        )
        structure = Structure(
            Lattice.cubic(6.0), ["Nb", "Nb", "Nb", "Sn"],
            [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        )
        doc = _summary("mp-123", "Nb3Sn", hull=0.01, sites=4, sg=221)
        task = SimpleNamespace(
            task_id="task-static", task_type="Static", calc_type=None, run_type="GGA+U"
        )
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "mp-query-cache.json.gz"
            _write_mp_snapshot(
                cache_path,
                source_index,
                raw_query_results={
                    "summary_query": [doc, _summary("mp-rejected", "ThFeAsO")],
                    "structure_query": [SimpleNamespace(material_id="mp-123", structure=structure)],
                    "electronic_structure_query": [SimpleNamespace(
                        material_id="mp-123", dos={"task_id": "task-static"}
                    )],
                    "formula_task_fallback_query": [],
                    "task_metadata_query": [task],
                    "dos_curves": {"task-static": _Dos(1.68, nsites=4)},
                },
                qualified=[(doc, {"dos_ef_states_ev_per_atom": 0.42})],
                counts={"dos_ef_gt_0p1_required_gate": 1},
                rejected={"mp-rejected": "radioactive_element:Th"},
                diagnostics={"api_errors": 0},
            )
            qualified, counts, rejected, diagnostics = _fetch_mp(
                "", source_index, mp_cache_path=cache_path
            )
            self.assertEqual(qualified[0][0]["material_id"], "mp-123")
            self.assertAlmostEqual(qualified[0][1]["dos_ef_states_ev_per_atom"], 0.42)
            self.assertEqual(counts["dos_ef_gt_0p1_required_gate"], 1)
            self.assertEqual(rejected["mp-rejected"], "radioactive_element:Th")
            self.assertEqual(diagnostics["mp_query_cache"]["mode"], "replayed_offline")
            self.assertEqual(diagnostics["mp_query_cache"]["raw_result_counts"]["summary_query"], 2)

            changed_sources = FormulaSourceIndex(
                formulas=frozenset({"CuO"}),
                unresolved_element_systems=frozenset(),
                stats={"3dsc": {"sha256": "changed-checksum"}},
            )
            self.assertNotEqual(_source_index_fingerprint(source_index), _source_index_fingerprint(changed_sources))
            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                _fetch_mp("", changed_sources, mp_cache_path=cache_path)

    def test_mp_summary_metal_and_gap_do_not_qualify_without_static_uniform_dos(self):
        from pymatgen.core import Lattice, Structure

        source_index = FormulaSourceIndex(
            formulas=frozenset({"CuO"}),
            unresolved_element_systems=frozenset(),
            stats={"3dsc": {"sha256": "source-checksum"}},
        )
        structure = Structure(
            Lattice.cubic(6.0), ["Nb", "Nb", "Nb", "Sn"],
            [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        )
        doc = _summary("mp-no-dos", "Nb3Sn", sites=4, sg=221, structure=structure, is_metal=True, band_gap=0.0)
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "mp-query-cache.json.gz"
            _write_mp_snapshot(
                cache_path,
                source_index,
                raw_query_results={
                    "summary_query": [doc],
                    "structure_query": [SimpleNamespace(material_id="mp-no-dos", structure=structure)],
                    "electronic_structure_query": [],
                    "formula_task_fallback_query": [],
                    "task_metadata_query": [],
                    "dos_curves": {},
                },
                qualified=[], counts={}, rejected={}, diagnostics={},
            )
            qualified, counts, rejected, _diagnostics = _fetch_mp(
                "", source_index, mp_cache_path=cache_path
            )
        self.assertEqual(qualified, [])
        self.assertEqual(rejected["mp-no-dos"], "no_static_or_uniform_dos_task_reference")
        self.assertEqual(counts["dos_ef_gt_0p1_required_gate"], 0)
        self.assertEqual(counts["dos_curves_retrieved"], 0)

    def test_main_evaluator_embargoes_shared_mp_ids(self):
        rows = [
            HarmonizedRecord("a" * 24, "3dsc", "FeSe", 10.0, 0.0, "unknown", "unknown", "10.0000/a", "confirmed", "literature", 0, chemical_family="Ferrites", mp_id="mp-shared"),
            HarmonizedRecord("b" * 24, "3dsc", "CoSe", 11.0, 0.0, "unknown", "unknown", "10.0000/b", "confirmed", "literature", 0, chemical_family="Ferrites", mp_id="mp-shared"),
            HarmonizedRecord("c" * 24, "3dsc", "Nb3Sn", 18.0, 0.0, "unknown", "unknown", "10.0000/c", "confirmed", "literature", 0, chemical_family="Ferrites", mp_id="mp-other"),
        ]
        safe_train, purged = purge_shared_mp_ids(rows, np.asarray([0, 2]), np.asarray([1]))
        self.assertEqual(safe_train.tolist(), [2])
        self.assertEqual(purged, ["a" * 24])

    def test_assignment_does_not_score_out_of_range_or_missing_definer(self):
        records = [
            HarmonizedRecord("a" * 24, "3dsc", "FeSe", 20.0, 0.0, "unknown", "unknown", "10.0000/a", "confirmed", "literature", 0, chemical_family="Ferrites", mp_id="mp-train", energy_above_hull=0.01, spacegroup_number=129),
        ]
        candidate_doc = _summary(
            "mp-new", "FeSe", sg=129, is_magnetic=True, ordering="FM", total_magnetization=2.5
        )
        dos_record = {
            "dos_ef_states_ev_per_atom": 0.5,
            "dos_task_id": "task",
            "dos_task_type": "Static",
            "dos_functional_run_type": "GGA",
        }
        qualified = [(candidate_doc, dos_record)]
        scored, counts, rejected = score_family_candidates(qualified, records, {"Ferrites": 0.7})
        self.assertEqual(len(scored), 1)
        self.assertEqual(scored[0]["dft_verification"], "pending")
        self.assertEqual(scored[0]["metallicity_source"], "mp_static_or_uniform_dos_ef")
        self.assertTrue(scored[0]["is_magnetic"])
        self.assertEqual(scored[0]["ordering"], "FM")
        self.assertEqual(scored[0]["total_magnetization"], 2.5)
        self.assertEqual(scored[0]["magnetism_classification"], "magnetic")
        self.assertEqual(counts["family_assigned_within_distance_and_definers"], 1)

        non_fe = _summary("mp-no-fe", "CoSe", sg=129)
        scored, counts, rejected = score_family_candidates(
            [(non_fe, dos_record)], records, {"Ferrites": 0.7}
        )
        self.assertEqual(scored, [])
        self.assertEqual(rejected["mp-no-fe"], "family_definer_missing:Fe")

        far = _summary("mp-far", "FeO", sg=129)
        scored, counts, rejected = score_family_candidates(
            [(far, dos_record)], records, {"Ferrites": 0.7}
        )
        self.assertEqual(scored, [])
        self.assertEqual(rejected["mp-far"], "unassigned_family_distance_gt_0.3")

    def test_grouped_formula_is_flattened_for_training_descriptor(self):
        from rt_superconductor.descriptors import formula_descriptor_vector
        record = _candidate_record("Ba(PdO2)2", "mp-1", 0.01, "Other")
        self.assertEqual(record.formula, "Ba1Pd2O4")
        values, _ = formula_descriptor_vector(record.formula)
        self.assertTrue(all(np.isfinite(value) for value in values))

    def test_top_30_per_family_use_conservative_score(self):
        rows = [
            {"chemical_family": "Other", "mp_id": f"mp-{index}",
             "conservative_score_tc_minus_sd_k": float(index), "predicted_tc_kelvin": float(index)}
            for index in range(40)
        ]
        top = select_top_per_family(rows, limit=30)
        self.assertEqual(len(top), 30)
        self.assertEqual(top[0]["mp_id"], "mp-39")
        self.assertEqual(top[-1]["rank_within_family"], 30)


if __name__ == "__main__":
    unittest.main()
