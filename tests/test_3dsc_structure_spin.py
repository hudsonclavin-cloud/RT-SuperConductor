import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from evaluate_3dsc_structure_spin import _cif_features, _spin_features
from train_physics_targets import BOND_FEATURE_NAMES, _jarvis_bond_features


class ThreeDSCStructureSpinTests(unittest.TestCase):
    def test_spin_values_are_normalized_per_atom_and_ordering_is_one_hot(self) -> None:
        features = _spin_features({
            "is_magnetic_2": "True", "ordering_2": "AFM", "nsites_2": "4",
            "total_magnetization_2": "2.0", "num_magnetic_sites_2": "2",
        })
        self.assertEqual(features["is_magnetic_2"], 1.0)
        self.assertEqual(features["total_magnetization_2_per_atom"], 0.5)
        self.assertEqual(features["num_magnetic_sites_2_per_atom"], 0.5)
        self.assertEqual(features["ordering_2"], "AFM")
        self.assertEqual([features[f"ordering_2_{v}"] for v in ("NM", "FM", "AFM", "FiM")], [0.0, 0.0, 1.0, 0.0])

    def test_spin_unknowns_stay_unknown(self) -> None:
        features = _spin_features({"ordering_2": "unknown", "nsites_2": "0"})
        self.assertIsNone(features["is_magnetic_2"])
        self.assertIsNone(features["total_magnetization_2_per_atom"])
        self.assertTrue(all(features[f"ordering_2_{v}"] is None for v in ("NM", "FM", "AFM", "FiM")))

    def test_cif_bond_and_pattern_descriptors_are_computed(self) -> None:
        from pymatgen.core import Lattice, Structure

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "CsCl.cif"
            structure = Structure(Lattice.cubic(4.0), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
            structure.to(filename=path)
            features = _cif_features(path)
        self.assertGreater(features["nearest_neighbor_min_angstrom"], 0)
        self.assertGreater(features["nearest_neighbor_mean_angstrom"], 0)
        self.assertGreater(features["metal_nonmetal_mean_distance_angstrom"], 0)
        self.assertGreater(features["crystalnn_mean_coordination_number"], 0)
        self.assertIn(features["dimensionality"], range(4))
        self.assertEqual(sum(features[f"dimensionality_{n}d"] for n in range(4)), 1.0)

    def test_jarvis_model_bond_features_have_defined_schema(self) -> None:
        features = _jarvis_bond_features({
            "lattice_mat": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            "elements": ["Cs", "Cl"], "coords": [[0, 0, 0], [0.5, 0.5, 0.5]], "cartesian": False,
        })
        self.assertEqual(set(features), set(BOND_FEATURE_NAMES))
        self.assertIsNotNone(features["bond_crystalnn_mean_coordination_number"])
        self.assertGreater(features["bond_nearest_neighbor_mean_angstrom"], 0)


if __name__ == "__main__":
    unittest.main()
