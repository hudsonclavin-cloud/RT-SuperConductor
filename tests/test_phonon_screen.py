"""Integrity, numerical, and restart guards for the ML-potential screen."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from deep_physics_checks import ScreenBudget,candidate_structure,phonon_status,screen_structure

class PhononIntegrityTests(unittest.TestCase):
    def test_nonfinite_modes_or_forces_cannot_pass(self):
        for values in [(float('nan'),.01,0),(.1,float('inf'),0),(.1,.01,float('nan'))]:
            with self.assertRaises(ValueError):phonon_status(*values)
    def test_candidate_formula_and_occupancy_are_preserved(self):
        from pymatgen.core import Structure,Lattice
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'candidate.cif'
            Structure(Lattice.cubic(4),[{'Mg':.5,'Zn':.5},'O'],[[0,0,0],[.5,.5,.5]]).to(filename=path)
            with self.assertRaisesRegex(ValueError,'Occupational ordering required'):
                candidate_structure({'formula':'Mg0.5Zn0.5O','candidate_cif_path':str(path)})
            Structure(Lattice.cubic(4),['Mg','O'],[[0,0,0],[.5,.5,.5]]).to(filename=path)
            with self.assertRaisesRegex(ValueError,'composition differs'):
                candidate_structure({'formula':'CaO','candidate_cif_path':str(path)})
            with self.assertRaisesRegex(ValueError,'content hash'):
                candidate_structure({'formula':'MgO','candidate_cif_path':str(path),'candidate_cif_sha256':'not-the-hash'})
    def test_budget_is_cumulative_across_resumes_and_can_be_extended(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'cpu.json'
            with patch('deep_physics_checks.time.process_time',return_value=10):budget=ScreenBudget(path,'protocol',100)
            with patch('deep_physics_checks.time.process_time',return_value=20):budget.save()
            with patch('deep_physics_checks.time.process_time',return_value=30):resumed=ScreenBudget(path,'protocol',200)
            with patch('deep_physics_checks.time.process_time',return_value=32):self.assertEqual(resumed.elapsed,12)
            with patch('deep_physics_checks.time.process_time',return_value=300):
                with self.assertRaises(TimeoutError):resumed.check()
            self.assertEqual(json.loads(path.read_text())['signature'],'protocol')
    def test_completed_exact_calculation_can_resume_without_force_evaluation(self):
        with tempfile.TemporaryDirectory() as t:
            directory=Path(t);(directory/'same-key.json').write_text(json.dumps({'result':{'status':'no_instability_detected_ml','minimum_frequency_thz':0.}}))
            result=screen_structure(None,None,None,{},directory,'same-key')
            self.assertEqual(result['status'],'no_instability_detected_ml')
            self.assertTrue(result['reused_calculation'])

@unittest.skipUnless(importlib.util.find_spec('phonopy'),'Optional phonopy package not installed')
class PhonopyAPITests(unittest.TestCase):
    def test_translational_spring_fixture_gamma_and_coarse_mesh(self):
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
        phonon=Phonopy(PhonopyAtoms(symbols=['Si'],cell=np.eye(3)*5,scaled_positions=[[0,0,0]]),supercell_matrix=np.diag([2,2,2]))
        size=len(phonon.supercell);force_constants=np.zeros((size,size,3,3))
        for i in range(size):
            for j in range(size):force_constants[i,j]=np.eye(3)*((size-1) if i==j else -1)
        phonon.force_constants=force_constants;phonon.symmetrize_force_constants();phonon.run_qpoints([[0,0,0]]);phonon.run_mesh([3,3,3],is_gamma_center=True)
        self.assertEqual(phonon.qpoints.frequencies.shape,(1,3));self.assertEqual(int(np.sum(phonon.mesh.weights)),27)
        self.assertTrue(np.isfinite(phonon.mesh.frequencies).all());self.assertGreaterEqual(float(phonon.mesh.frequencies.min()),-1e-6)
        self.assertGreater(phonon.unit_conversion_factor,0)

class NumericalCheckpointTests(unittest.TestCase):
    def test_numpy_prediction_values_roundtrip_without_losing_checkpoint(self):
        from deep_physics_checks import atomic_screen_json
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'state.json';atomic_screen_json(path,{'force':np.float32(.01),'coords':np.zeros((1,3))})
            state=json.loads(path.read_text());self.assertAlmostEqual(state['force'],.01,places=6);self.assertEqual(state['coords'],[[0.,0.,0.]])
