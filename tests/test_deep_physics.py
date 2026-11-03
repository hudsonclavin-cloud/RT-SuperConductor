import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from deep_physics_features import assert_safe, table_features, cif_features
class DeepPhysicsTests(unittest.TestCase):
    def test_forbidden_fields_fail_matrix_audit(self):
        for name in ('synth_doped','totreldiff','formula_similarity','correct_formula_frac','formula_frac','weight','created_at_2','updated_2','last_updated_2','source__weight','source__formula_frac','run_type_2','tc'):
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

class DeepAblationTests(unittest.TestCase):
    def test_feature_groups_use_allowlist(self):
        from deep_ablation import groups_for
        r=groups_for(['band_gap_2','synth_doped','nearest_min','ordering_NM','run_type_2'])
        self.assertEqual(r['electronic'],['band_gap_2']);self.assertEqual(r['bonding'],['nearest_min'])
        self.assertFalse(any('synth_doped' in v or 'run_type_2' in v for v in r.values()))
    def test_actual_matrix_entry_rejects_leak(self):
        import numpy as np
        from deep_ablation import evaluate
        with self.assertRaises(ValueError):
            evaluate(np.zeros((2,1)),np.array([1,2]),np.array(['A','B']),np.array(['A','B']),['formula_similarity'])
    def test_auc_blocks_unscored_or_single_class(self):
        import numpy as np
        from deep_ablation import auc
        self.assertIsNone(auc(np.array([False,True]),np.array([0.,np.nan])))
        self.assertIsNone(auc(np.array([True,True]),np.array([0.,1.])))
        self.assertEqual(auc(np.array([False,True]),np.array([.5,.5])),.5)

class StructureEncoderTests(unittest.TestCase):
    def test_frozen_encoder_batches_and_detaches(self):
        try:import torch
        except ImportError:self.skipTest('optional torch dependency')
        import numpy as np
        from deep_encoders import encode,BUDGET_CPU_SECONDS
        class Encoder:
            def eval(self):return self
            def __call__(self,graphs,**kwargs):return {'crystal_fea':torch.ones((len(graphs),64))}
        x=encode(Encoder(),list(range(19)))
        self.assertEqual(x.shape,(19,64));self.assertTrue(np.all(x==1));self.assertEqual(BUDGET_CPU_SECONDS,21600)
    def test_scope_checkpoint_reuse_requires_identical_inputs(self):
        import numpy as np
        import tempfile
        from deep_ablation import evaluate
        x=np.array([[0.],[1.],[2.],[3.]])
        y=np.array([0.,1.,0.,2.]);family=np.array(['A','A','B','B']);calls=[]
        def transform(tr,te):calls.append(1);return x[tr],x[te]
        with tempfile.TemporaryDirectory() as t:
            a,_=evaluate(x,y,family,family,['safe_feature'],trees=2,fold_transform=transform,checkpoint_dir=Path(t))
            self.assertEqual(len(calls),2);calls.clear()
            b,_=evaluate(x,y,family,family,['safe_feature'],trees=2,fold_transform=transform,checkpoint_dir=Path(t))
            self.assertEqual(calls,[]);self.assertEqual(__import__("json").loads(__import__("json").dumps(a)),b)
            evaluate(x+1,y,family,family,['safe_feature'],trees=2,fold_transform=transform,checkpoint_dir=Path(t))
            self.assertEqual(len(calls),2)
    def test_cpu_budget_survives_restart_and_stops_at_six_hours(self):
        try:import torch
        except ImportError:self.skipTest('optional torch dependency')
        import tempfile,json
        from unittest.mock import patch
        from deep_encoders import budget_start,check_budget
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'cpu-budget.json').write_text(json.dumps({'cpu_seconds':30}))
            with patch('deep_encoders.time.process_time',return_value=10),patch.dict('os.environ',{'DEEP_PREVIOUS_CPU_SECONDS':'7'}):
                self.assertEqual(budget_start(p),-20)
            with patch('deep_encoders.time.process_time',return_value=21601):
                with self.assertRaises(TimeoutError):check_budget(p,0)
