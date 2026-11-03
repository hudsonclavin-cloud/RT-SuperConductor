import math
import sys
import unittest
import csv
import tempfile

from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_physics_targets import allen_dynes_tc_k, build_characteristics
from run_physics_search import generate_substitutions
from enrich_candidate_stability import enrich as enrich_candidate_stability


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
            _record("JVASP-Nb3Sn", ["Nb", "Nb", "Nb", "Sn"]),
            _record("JVASP-Ta", ["Ta"]),
        ]
        candidates = generate_substitutions(rows)
        candidate = next(row for row in candidates if row["formula"] == "Ta3Sn")
        self.assertEqual(candidate["prototype_formula"], "Nb3Sn")
        self.assertEqual(candidate["source_stability_proxy"], "stable")
        self.assertEqual(candidate["structure_status"], "unrelaxed parent-prototype substitution")

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


if __name__ == "__main__":
    unittest.main()
