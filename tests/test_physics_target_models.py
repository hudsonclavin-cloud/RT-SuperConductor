import math
import sys
import unittest
import csv
import tempfile
from types import SimpleNamespace

from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_physics_targets import BOND_FEATURE_NAMES, _make_matrix, allen_dynes_tc_k, build_characteristics, fit_physics_models
from run_physics_search import generate_substitutions, select_by_confidence_group, is_low_confidence
from enrich_candidate_stability import enrich as enrich_candidate_stability
from enrich_physics_mp import enrich as enrich_physics_mp


def _record(jid, elements, stability="stable"):
    return {
        "jid": jid,
        "atoms": {
            "lattice_mat": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            "coords": [[float(i) / max(1, len(elements)), 0.0, 0.0] for i, _ in enumerate(elements)],
            "elements": elements,
        },
        "stability": stability,
        "lamb": 1.0,
        "wlog": 200.0,
        "Tc": 10.0,
        "press": "0 GPa",
    }


class PhysicsTargetModelTests(unittest.TestCase):
    def test_allen_dynes_uses_mu_star_point_one_and_kelvin_omega_log(self):
        self.assertAlmostEqual(allen_dynes_tc_k(0.7572, 164.9922), 6.882208886874397, places=10)
        self.assertEqual(allen_dynes_tc_k(0.01, 100.0), 0.0)
        self.assertTrue(math.isnan(allen_dynes_tc_k(float("nan"), 100.0)))

    def test_characteristics_include_magpie_shell_lewis_and_separate_mp_unknowns(self):
        rows, feature_names, status = build_characteristics([_record("JVASP-test", ["Nb", "Nb", "Nb", "Sn"])])
        row = rows[0]
        self.assertEqual(status, "not_requested")
        self.assertEqual(row["formula"], "Nb3Sn")
        self.assertIn("lewis_dot_proxy_avg_outer_electrons", feature_names)
        self.assertIn("shell__descriptor_shell_5_fill_pct", feature_names)
        self.assertTrue(any(name.startswith("magpie__") for name in feature_names))
        self.assertIsNone(row["mp_dos_ef_states_per_ev_per_cell"])
        self.assertIsNone(row["mp_band_gap_ev"])
        self.assertIsNone(row["mp_is_metal"])
        self.assertIsNone(row["mp_density_g_cm3"])
        self.assertGreater(row["structure_density_g_cm3"], 0)
        self.assertTrue(all(name in row for name in BOND_FEATURE_NAMES))

    def test_missing_physics_features_remain_nan_for_fold_local_imputation(self):
        matrix = _make_matrix([{"x": 1.0}, {"x": None}, {"x": ""}], ["x"])
        self.assertEqual(matrix[0, 0], 1.0)
        self.assertTrue(math.isnan(matrix[1, 0]))
        self.assertTrue(math.isnan(matrix[2, 0]))

    def test_fitted_physics_model_uses_jarvis_bond_descriptors(self):
        rows, base_names, _ = build_characteristics([_record("JVASP-Nb3Sn", ["Nb", "Nb", "Nb", "Sn"])])
        models, _ = fit_physics_models(rows, base_names + BOND_FEATURE_NAMES)
        self.assertEqual(models["feature_names"][-len(BOND_FEATURE_NAMES):], BOND_FEATURE_NAMES)

    def test_search_substitutions_keep_parent_stability_as_a_proxy(self):
        rows = [
            _record("JVASP-KCl", ["K", "Cl"]),
            _record("JVASP-Na", ["Na"]),
        ]
        candidates = generate_substitutions(rows)
        candidate = next(row for row in candidates if row["formula"] == "NaCl")
        self.assertEqual(candidate["prototype_formula"], "KCl")
        self.assertEqual(candidate["source_stability_proxy"], "stable")
        self.assertEqual(candidate["structure_status"], "unrelaxed parent-prototype substitution")

    def test_substitution_filter_reports_hydrogen_class_and_charge_balance_drops(self):
        rows = [
            _record("JVASP-KCl", ["K", "Cl"]),
            _record("JVASP-Na", ["Na"]),
            _record("JVASP-HCl", ["H", "Cl"]),
            _record("JVASP-Li", ["Li"]),
            _record("JVASP-Nb3Sn", ["Nb", "Nb", "Nb", "Sn"]),
            _record("JVASP-Ta", ["Ta"]),
            _record("JVASP-BCl3", ["B", "Cl", "Cl", "Cl"]),
            # Deliberately unbalanced unit fixture; never empirical training data.
            _record("UNIT-unbalanced-KCl2", ["K", "Cl", "Cl"]),
            _record("JVASP-Al", ["Al"]),
        ]
        counts = {}
        generated = generate_substitutions(rows, drop_counts=counts)
        formulas = {row["formula"] for row in generated}
        self.assertNotIn("HCl", formulas)
        self.assertTrue(all(not r["substitution"].startswith("H->") and not r["substitution"].endswith("->H") for r in generated))
        self.assertGreater(counts["dropped_hydrogen_substitution"], 0)
        self.assertGreater(counts["dropped_metal_nonmetal_class_swap"], 0)
        self.assertGreater(counts["dropped_no_common_oxidation_state_balance"], 0)

    def test_low_confidence_rows_are_ranked_in_their_own_group(self):
        self.assertTrue(is_low_confidence(8.1, 8.0))
        self.assertTrue(is_low_confidence(8.0, 8.0))  # Review requires SD strictly below Tc.
        self.assertFalse(is_low_confidence(7.9, 8.0))
        rows = [
            {"formula": "S1", "ranking_group": "standard", "acquisition_ucb_k": 10, "predicted_tc_k": 9},
            {"formula": "S2", "ranking_group": "standard", "acquisition_ucb_k": 8, "predicted_tc_k": 7},
            {"formula": "L1", "ranking_group": "low_confidence", "acquisition_ucb_k": 99, "predicted_tc_k": 2},
            {"formula": "L2", "ranking_group": "low_confidence", "acquisition_ucb_k": 90, "predicted_tc_k": 1},
        ]
        selected = select_by_confidence_group(rows, top_k=4)
        self.assertEqual([row["ranking_group"] for row in selected], [
            "standard", "standard", "low_confidence", "low_confidence"
        ])
        self.assertEqual([row["group_rank"] for row in selected], [1, 2, 1, 2])
        self.assertTrue(all(row["review_rank"]=="" for row in selected if row["ranking_group"]=="low_confidence"))

    def test_materials_project_failure_preserves_candidate_stability_as_unknown(self):
        class FailingMPRester:
            def __init__(self, _key):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            class materials:
                class summary:
                    @staticmethod
                    def search(**_kwargs):
                        raise RuntimeError("HTTP 403")

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "candidates.csv"
            output_path = Path(tmp) / "stability.csv"
            input_path.write_text("formula\nNb3Sn\n", encoding="utf-8")
            result = enrich_candidate_stability(
                input_path, output_path, api_key="test", rester_factory=FailingMPRester
            )
            with output_path.open(encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(result["status_counts"], {"api_unavailable_403": 1})
        self.assertEqual(row["status"], "api_unavailable_403")
        self.assertEqual(row["energy_above_hull_ev_atom"], "")

    def test_physics_mp_enrichment_records_exact_summary_and_dos_fields(self):
        metadata_queries=[];spectrum_queries=[];clients=[]
        class FakeDOS:
            efermi=0.0
            energies = [-1.0, 0.0, 1.0]

            @staticmethod
            def get_densities():
                return [0.5, 4.0, 0.25]

        class FakeMPRester:
            def __init__(self, _key):
                clients.append(self)
                self.materials = SimpleNamespace(
                    summary=SimpleNamespace(search=self._summary),
                    electronic_structure=SimpleNamespace(search=self._electronic,use_document_model=True),
                    electronic_structure_dos=SimpleNamespace(use_document_model=True,get_dos_from_task_id=self._dos_from_task),
                )

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def _dos_from_task(self, task_id):
                assert task_id=='task-fixture-NaCl'
                assert self.materials.electronic_structure.use_document_model is False
                spectrum_queries.append('mp-test-NaCl')
                return FakeDOS()

            def get_dos_by_material_id(self, material_id):
                spectrum_queries.append(material_id)
                assert self.materials.electronic_structure_dos.use_document_model is False
                return FakeDOS()

            @staticmethod
            def _summary(*, formula, fields):
                assert formula == ["NaCl"] and "density" in fields
                return [type("Doc", (), {
                    "formula_pretty": "NaCl", "material_id": "mp-test-NaCl",
                    "density": 2.16, "band_gap": 5.0, "is_metal": False,
                })()]

            def _electronic(self, *, material_ids, fields):
                metadata_queries.append(material_ids)
                assert material_ids == ["mp-test-NaCl"] and "dos" in fields
                if self.materials.electronic_structure.use_document_model is False:
                    return [{"dos":{"task_id":"task-fixture-NaCl","total":{"1":{"efermi":0.0}}}}]
                return [type("EDoc", (), {
                    "material_id": "mp-test-NaCl", "dos": FakeDOS(), "efermi": 0.0,
                })()]

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "features.csv"
            output_path = Path(tmp) / "mp.csv"
            input_path.write_text("jid,formula\nJVASP-test,NaCl\n", encoding="utf-8")
            result = enrich_physics_mp(
                input_path, output_path, api_key="test", rester_factory=FakeMPRester
            )
            with output_path.open(encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(result["status_counts"], {"ready": 1})
        self.assertEqual(row["mp_material_id"], "mp-test-NaCl")
        self.assertEqual(float(row["mp_dos_ef_states_per_ev_per_cell"]), 4.0)
        self.assertEqual(float(row["mp_density_g_cm3"]), 2.16)
        self.assertEqual(row["mp_is_metal"], "False")
        self.assertEqual(metadata_queries,[["mp-test-NaCl"]])  # Metadata selects the spectrum task; it is not the curve.
        self.assertEqual(spectrum_queries,["mp-test-NaCl"])
        self.assertTrue(clients[0].materials.electronic_structure_dos.use_document_model)
        self.assertTrue(clients[0].materials.electronic_structure.use_document_model)


if __name__ == "__main__":
    unittest.main()
