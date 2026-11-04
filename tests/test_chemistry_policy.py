import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from chemistry_policy import substitution_rejection, active_layer_elements


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


if __name__ == '__main__':
    unittest.main()
