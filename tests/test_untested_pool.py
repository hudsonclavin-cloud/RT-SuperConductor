import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_untested_pool import (  # noqa: E402
    _candidate_record,
    _task_is_static_or_uniform,
    _summary_filter_without_structures,
    composition_structure_distance,
    dos_at_fermi_states_ev_per_atom,
    eligible_family_metrics,
    family_definer_rejection,
    prefetch_static_dos_curves,
    pool_run_status,
    qualify_by_dos,
    reduced_formula,
    score_family_candidates,
    select_top_per_family,
)
from chemistry_policy import (  # noqa: E402
    radioactive_elements,
    untested_pool_candidate_rejection,
    untested_pool_composition_rejection,
)
from rt_superconductor.harmonized import HarmonizedRecord  # noqa: E402
from train_pressure_aware import embargoed_family_groups  # noqa: E402


def _summary(mp_id, formula, *, hull=0.01, sites=2, sg=225):
    return SimpleNamespace(
        material_id=mp_id,
        formula_pretty=formula,
        composition_reduced=None,
        energy_above_hull=hull,
        nsites=sites,
        symmetry=SimpleNamespace(number=sg),
    )


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

    def test_required_composition_rejections(self):
        for formula, reason in (
            ("FeCO3", "carbonate_salt"),
            ("CoCO3", "carbonate_salt"),
            ("Ni(OH)2", "hydroxide_salt"),
            ("K2CuF4", "alkali_alkaline_earth_transition_metal_fluoride"),
        ):
            with self.subTest(formula=formula):
                self.assertEqual(untested_pool_composition_rejection(formula), reason)
        self.assertEqual(untested_pool_composition_rejection("CuO"),
                         "binary_transition_metal_oxide_or_halide")
        # BaCuO2 is rejected when its measured MP DOS fails the mandatory gate.
        self.assertIsNone(untested_pool_composition_rejection("BaCuO2"))
        self.assertEqual(untested_pool_candidate_rejection("BaCuO2", 0.1),
                         "dos_ef_le_0.1_states_per_ev_per_atom")
        self.assertIsNone(untested_pool_candidate_rejection("Nb3Sn", 0.10001))

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
        self.assertEqual(untested_pool_candidate_rejection("BaCuO2", 0.1),
                         "dos_ef_le_0.1_states_per_ev_per_atom")
        self.assertIsNone(untested_pool_candidate_rejection("BaCuO2", 0.100001))

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

    def test_pool_status_does_not_call_missing_dos_a_valid_empty_search(self):
        counts = {"dos_static_or_uniform_tasks": 12, "dos_curves_retrieved": 0}
        self.assertEqual(pool_run_status(0, counts, {"Other": 0.7}), "blocked_dos_curve_retrieval")
        self.assertEqual(pool_run_status(1, counts, {"Other": 0.7}), "ready")
        self.assertEqual(pool_run_status(0, {"dos_curves_retrieved": 0}, {"Other": 0.7}), "ready_empty_after_family_gate")

    def test_main_evaluator_groups_shared_mp_ids_together(self):
        rows = [
            HarmonizedRecord("a" * 24, "3dsc", "FeSe", 10.0, 0.0, "unknown", "unknown", "10.0000/a", "confirmed", "literature", 0, chemical_family="Ferrites", mp_id="mp-shared"),
            HarmonizedRecord("b" * 24, "3dsc", "CoSe", 11.0, 0.0, "unknown", "unknown", "10.0000/b", "confirmed", "literature", 0, chemical_family="Ferrites", mp_id="mp-shared"),
            HarmonizedRecord("c" * 24, "3dsc", "Nb3Sn", 18.0, 0.0, "unknown", "unknown", "10.0000/c", "confirmed", "literature", 0, chemical_family="Ferrites", mp_id="mp-other"),
        ]
        groups = embargoed_family_groups(rows)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])

    def test_assignment_does_not_score_out_of_range_or_missing_definer(self):
        records = [
            HarmonizedRecord("a" * 24, "3dsc", "FeSe", 20.0, 0.0, "unknown", "unknown", "10.0000/a", "confirmed", "literature", 0, chemical_family="Ferrites", mp_id="mp-train", energy_above_hull=0.01, spacegroup_number=129),
        ]
        candidate_doc = _summary("mp-new", "FeSe", sg=129)
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
