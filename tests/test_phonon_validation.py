import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_phonon_screen import (SETTINGS, STABLE_CONTROLS, UNSTABLE_CONTROLS,
                                    control_structure, score_setting)


class PhononReferenceValidationTests(unittest.TestCase):
    def test_reference_set_has_required_controls_and_three_settings(self):
        self.assertGreaterEqual(len(STABLE_CONTROLS), 20)
        self.assertGreaterEqual(len(UNSTABLE_CONTROLS), 5)
        self.assertTrue({"Al_fcc", "Cu_fcc", "Nb_bcc", "Nb3Sn_A15", "MgB2_AlB2"} <= set(STABLE_CONTROLS))
        self.assertEqual([item[1:] for item in SETTINGS], [(3, 3), (4, 4), (5, 5)])

    def test_benchmark_structures_have_expected_formulas(self):
        self.assertEqual(control_structure("Al_fcc").composition.reduced_formula, "Al")
        self.assertEqual(control_structure("Nb3Sn_A15").composition.reduced_formula, "Nb3Sn")
        self.assertEqual(control_structure("MgB2_AlB2").composition.reduced_formula, "MgB2")
        auzn = control_structure("AuZn_B2_0K").composition.get_el_amt_dict()
        self.assertEqual(auzn, {"Au": 1, "Zn": 1})
        niti = control_structure("NiTi_B2_0K").composition.get_el_amt_dict()
        self.assertEqual(niti, {"Ni": 1, "Ti": 1})

    def test_confusion_counts_blocked_rows_separately_and_enforces_al(self):
        rows = [
            {"control_id": "Al_fcc", "setting": "current_3x3", "minimum_frequency_thz": -.6},
            {"control_id": "Cu_fcc", "setting": "current_3x3", "minimum_frequency_thz": .2},
            {"control_id": "Zr_bcc_0K", "setting": "current_3x3", "minimum_frequency_thz": -.8},
            {"control_id": "Ti_bcc_0K", "setting": "current_3x3", "status": "blocked_calculation"},
        ]
        r = score_setting(rows, "current_3x3")
        self.assertEqual(r["false_positive_stable"], 1)
        self.assertEqual(r["true_positive_unstable"], 1)
        self.assertEqual(r["blocked_unstable"], len(UNSTABLE_CONTROLS) - 2)
        self.assertEqual(r["condition_mismatch_unstable_controls"], 1)
        self.assertTrue(r["al_false_positive"])
        self.assertFalse(r["passes_user_criterion"])
        self.assertEqual(r["screen_classification"], "diagnostic only")


if __name__ == "__main__":
    unittest.main()
