"""Small ASE adapter smoke test; ASE remains optional for normal test runs."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("ase"), "ASE is optional; phonon adapter test skipped")
class PhononValidationOptionalDependencyTests(unittest.TestCase):
    def test_ase_structure_round_trip_preserves_composition(self):
        from ase import Atoms
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = Atoms(
            symbols=["Nb", "Nb", "Nb", "Sn"],
            positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            cell=[4.0, 4.0, 4.0],
            pbc=True,
        )
        structure = AseAtomsAdaptor.get_structure(atoms)
        restored = AseAtomsAdaptor.get_atoms(structure)

        self.assertEqual(structure.composition.reduced_formula, "Nb3Sn")
        self.assertEqual(restored.get_chemical_formula(), "Nb3Sn")


if __name__ == "__main__":
    unittest.main()
