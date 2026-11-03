import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from deep_physics_features import assert_safe, table_features, cif_features
class DeepPhysicsTests(unittest.TestCase):
    def test_forbidden_fields_fail_matrix_audit(self):
        for name in ('synth_doped','totreldiff','formula_similarity','correct_formula_frac','formula_frac','weight','created_at_2','updated_2','run_type_2','tc'):
            with self.assertRaises(ValueError): assert_safe(['band_gap_2',name])
    def test_atomic_weight_descriptor_is_not_source_weight(self):
        assert_safe(['MagpieData mean AtomicWeight'])
    def test_table_ratios_and_provenance_flag(self):
        d=table_features({'lata_2':'2','latb_2':'4','latc_2':'6','cell_volume_2':'48','nsites_2':'2','num_magnetic_sites_2':'1','ordering_2':'AFM','run_type_2':'GGA'})
        self.assertEqual(d['volume_per_atom'],24);self.assertEqual(d['c_over_a'],3)
        self.assertEqual(d['magnetic_sites_per_atom'],.5);self.assertEqual(d['ordering_AFM'],1)
    def test_cif_features_are_finite_and_missing_ewald_is_not_zero(self):
        import tempfile
        from pymatgen.core import Lattice,Structure
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'s.cif';Structure(Lattice.cubic(3),['Cu'],[[0,0,0]]).to(filename=p)
            r=cif_features(p)
        self.assertGreater(r['nearest_min'],0);self.assertIsNone(r['ewald_ev_per_atom'])
