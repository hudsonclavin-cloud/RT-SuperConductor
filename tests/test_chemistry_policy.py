import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from chemistry_policy import (
    substitution_rejection, active_layer_elements, insulator_rejection,
    radioactive_elements,
)


class ChemistryPolicyRegressionTests(unittest.TestCase):
    def test_rejects_fe_to_sm_in_active_iron_pnictide_family(self):
        self.assertEqual(
            substitution_rejection('Fe', 'Sm', 'As2EuFe1.8Ni0.2', 'As2EuNi0.2Sm1.8', family='Ferrites'),
            'family_defining_element',
        )

    def test_rejects_ni_to_ba_for_shannon_radius_mismatch(self):
        self.assertEqual(
            substitution_rejection('Ni', 'Ba', 'BaNi2P2', 'Ba3P2', family='Other'),
            'ionic_radius_mismatch',
        )

    def test_rejects_oxygen_to_selenium_in_oxide_spacer(self):
        self.assertEqual(
            substitution_rejection('O', 'Se', 'AsFeOSm', 'AsFeSeSm', family='Other'),
            'oxide_anion_row_swap',
        )

    def test_preserves_the_complete_active_layer_signature(self):
        self.assertEqual(active_layer_elements('BaFe2As2'), {'Fe', 'As'})
        self.assertEqual(active_layer_elements('La2CuO4'), {'Cu', 'O'})
        self.assertEqual(active_layer_elements('LiFeSe'), {'Fe', 'Se'})
        self.assertEqual(active_layer_elements('NiAs'), {'Ni', 'As'})

    def test_allows_similar_shannon_radius_cation_substitution(self):
        self.assertIsNone(
            substitution_rejection('Ba', 'Sr', 'BaFe2As2', 'SrFe2As2', family='Ferrites')
        )

    def test_flags_binary_transition_metal_oxides_and_halides(self):
        for formula in ('NiO', 'CoO', 'MnO', 'FeO', 'CuO', 'NiCl2'):
            with self.subTest(formula=formula):
                self.assertEqual(insulator_rejection(formula), 'binary_transition_metal_oxide_or_halide')
        self.assertIsNone(insulator_rejection('La2CuO4'))

    def test_only_method_labeled_gga_u_or_r2scan_gaps_trigger_property_filter(self):
        self.assertEqual(insulator_rejection('LiNbO2', 0.5, 'GGA+U'), 'mp_gga_u_gap_gt_0p1_ev')
        self.assertEqual(insulator_rejection('LiNbO2', 0.5, 'r2SCAN'), 'mp_r2scan_gap_gt_0p1_ev')
        self.assertIsNone(insulator_rejection('LiNbO2', 0.5, 'PBE'))
        self.assertIsNone(insulator_rejection('LiNbO2', 0.05, 'r2SCAN'))

    def test_radioactive_candidates_are_identified_for_lab_access_flag(self):
        self.assertEqual(radioactive_elements('ThFeAsO'), ['Th'])
        self.assertEqual(radioactive_elements('PmNi2B2C'), ['Pm'])
        self.assertIn('Np', radioactive_elements('NpO2'))
        self.assertEqual(radioactive_elements('Nb3Sn'), [])

    def test_confidence_uses_strict_half_prediction_cutoff(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        from rank_candidates import low_confidence, candidate_confidence_group
        self.assertFalse(low_confidence(4.9, 10.0))
        self.assertTrue(low_confidence(5.0, 10.0))
        self.assertTrue(low_confidence(1.0, 0.0))
        self.assertEqual(candidate_confidence_group(1.0, 10.0, lab_access_flags=['Tc']), 'low_confidence')
        self.assertEqual(candidate_confidence_group(1.0, 10.0), 'confident')


if __name__ == '__main__':
    unittest.main()
