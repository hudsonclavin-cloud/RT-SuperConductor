import math
import sys
import unittest
import csv
import tempfile
from types import SimpleNamespace

from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_physics_targets import allen_dynes_tc_k, build_characteristics
from run_physics_search import generate_substitutions, select_by_confidence_group, is_low_confidence
from enrich_candidate_stability import enrich as enrich_candidate_stability
from enrich_physics_mp import enrich as enrich_physics_mp


def _record(jid, elements, stability="stable"):
    return {
        "jid": jid,
        "atoms": {
            "lattice_mat": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            "coords": [[0.0, 0.0, 0.0] for _ in elements],
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
            _record("JVASP-Al", ["Al"]),
        ]
        counts = {}
        generated = generate_substitutions(rows, drop_counts=counts)
        formulas = {row["formula"] for row in generated}
        self.assertNotIn("HCl", formulas)
        self.assertGreater(counts["dropped_hydrogen_substitution"], 0)
        self.assertGreater(counts["dropped_metal_nonmetal_class_swap"], 0)
        self.assertGreater(counts["dropped_no_common_oxidation_state_balance"], 0)

    def test_low_confidence_rows_are_ranked_in_their_own_group(self):
        self.assertTrue(is_low_confidence(8.1, 8.0))
        self.assertFalse(is_low_confidence(8.0, 8.0))
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
        class FakeDOS:
            energies = [-1.0, 0.0, 1.0]

            @staticmethod
            def get_densities():
                return [0.5, 4.0, 0.25]

        class FakeMPRester:
            def __init__(self, _key):
                self.materials = SimpleNamespace(
                    summary=SimpleNamespace(search=self._summary),
                    electronic_structure=SimpleNamespace(search=self._electronic),
                )

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def _summary(*, formula, fields):
                assert formula == ["NaCl"] and "density" in fields
                return [type("Doc", (), {
                    "formula_pretty": "NaCl", "material_id": "mp-test-NaCl",
                    "density": 2.16, "band_gap": 5.0, "is_metal": False,
                })()]

            @staticmethod
            def _electronic(*, material_ids, fields):
                assert material_ids == ["mp-test-NaCl"] and "dos" in fields
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


if __name__ == "__main__":
    unittest.main()
