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

class PhysicalChecksTests(unittest.TestCase):
    def test_metal_oxygen_bridge_angles_and_neutral_d_count(self):
        import tempfile
        from pymatgen.core import Lattice,Structure
        from deep_physics_checks import oxide_angles,mean_d_electrons
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bridge.cif';Structure(Lattice.cubic(10),['O','Mn','Mn'],[[0,0,0],[.25,0,0],[.75,0,0]]).to(filename=p)
            r=oxide_angles(p)
        self.assertAlmostEqual(r['mom_angle_min_deg'],180,places=5)
        self.assertAlmostEqual(mean_d_electrons('FeO'),3)
    def test_phonon_threshold_and_unconverged_relaxation(self):
        from deep_physics_checks import phonon_status
        self.assertEqual(phonon_status(-.5,.01),'no_instability_detected_ml')
        self.assertEqual(phonon_status(-.5001,.01),'dynamically_unstable_ml')
        self.assertEqual(phonon_status(.1,.06),'blocked_relaxation_not_converged')
        self.assertEqual(phonon_status(.1,.01,.2),'blocked_relaxation_not_converged')

class ProjectedDOSTests(unittest.TestCase):
    def test_orbital_fractions_and_zero_window(self):
        import numpy as np
        from types import SimpleNamespace
        from pymatgen.core import Composition
        from enrich_deep_mp import orbital_weights,dos_at_fermi
        class DOS:
            efermi=0.;energies=np.array([-1.,0.,1.]);structure=SimpleNamespace(composition=Composition('Fe'))
            def get_densities(self):return np.array([4.,4.,4.])
            def get_element_spd_dos(self,element):return {k:SimpleNamespace(get_densities=lambda v=v:np.repeat(v,3)) for k,v in [('s',2.),('p',1.),('d',1.)]}
        r=orbital_weights(DOS());self.assertEqual(r['orbital_s_fraction'],.5);self.assertEqual(r['orbital_d_fraction'],.25);self.assertEqual(dos_at_fermi(DOS()),4)
        d=DOS();d.energies=np.array([-.1,0,.1]);self.assertEqual(orbital_weights(d),{})
        d.efermi=2.
        with self.assertRaises(ValueError):dos_at_fermi(d)
    def test_enrichment_cannot_query_mp_outside_actions(self):
        import tempfile,csv
        from unittest.mock import patch
        from enrich_deep_mp import enrich
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'input.csv').write_text('test');(p/'features.csv').write_text('record_id\nx\n')
            with patch('enrich_deep_mp.load_harmonized',return_value=[]),patch.dict('os.environ',{'GITHUB_ACTIONS':'false','MP_API_KEY':'test-only'}):
                r=enrich(p/'input.csv',p/'features.csv',p/'output.csv')
            self.assertEqual(r['queries'],0);self.assertEqual(r['orbital_available'],0)

class MechanismComparisonTests(unittest.TestCase):
    def test_global_model_excludes_all_held_out_elemental_systems(self):
        import numpy as np
        from deep_mechanisms import global_training_indices
        systems=np.array(['Fe-As','Cu-O','Fe-As','Nb-Sn'])
        selected=global_training_indices(systems,np.array([1]),np.array([0]),'Ferrites')
        self.assertEqual(selected.tolist(),[1,3])
        self.assertEqual(global_training_indices(systems,np.array([1]),np.array([0]),'cross-family').tolist(),[1])
    def test_dos_metadata_selects_task_then_fetches_real_curve(self):
        from types import SimpleNamespace
        from enrich_deep_mp import complete_dos_with_task
        calls=[];curve=object()
        def get_task(task):calls.append(task);return curve
        client=SimpleNamespace(materials=SimpleNamespace(electronic_structure=SimpleNamespace(search=lambda **kwargs:[{'dos':{'total':{'1':{'task_id':'mp-task-fixture'}}}}]),electronic_structure_dos=SimpleNamespace(get_dos_from_task_id=get_task)))
        dos,task=complete_dos_with_task(client,'mp-fixture')
        self.assertIs(dos,curve);self.assertEqual(task,'mp-task-fixture');self.assertEqual(calls,['mp-task-fixture'])
        client.materials.electronic_structure.search=lambda **kwargs:[{'dos':{'total':{}}}]
        with self.assertRaises(ValueError):complete_dos_with_task(client,'mp-fixture')

class CandidateStructureTests(unittest.TestCase):
    def test_ordered_parent_substitution_and_disorder_are_distinct(self):
        import tempfile,csv
        from pymatgen.core import Lattice,Structure
        from deep_physics_checks import prepare_candidate_structures
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);parent=d/'parent.cif';Structure(Lattice.cubic(4.2),['Mg','O'],[[0,0,0],[.5,.5,.5]]).to(filename=parent)
            (d/'index.csv').write_text('record_id,cif_path\np,'+str(parent)+'\n')
            (d/'shortlist.csv').write_text('candidate_id,formula,parent_record_id,parent_formula,substitution\nx,CaO,p,MgO,Mg->Ca\n')
            rows=prepare_candidate_structures(d/'shortlist.csv',d/'index.csv',d/'out.csv',d/'structures')
            self.assertEqual(rows[0]['candidate_structure_status'],'unrelaxed_ordered_parent_substitution')
            candidate=Structure.from_file(rows[0]['candidate_cif_path']);self.assertEqual(candidate.composition.reduced_formula,'CaO')
            Structure(Lattice.cubic(4.2),[{'Mg':.5,'Zn':.5},'O'],[[0,0,0],[.5,.5,.5]]).to(filename=parent)
            (d/'shortlist.csv').write_text('candidate_id,formula,parent_record_id,parent_formula,substitution\nx,Ca0.5Zn0.5O,p,Mg0.5Zn0.5O,Mg->Ca\n')
            rows=prepare_candidate_structures(d/'shortlist.csv',d/'index.csv',d/'out.csv',d/'structures')
            self.assertEqual(rows[0]['candidate_structure_status'],'blocked_occupational_ordering_required')
            self.assertFalse(Structure.from_file(rows[0]['candidate_cif_path']).is_ordered)

class ExpandedFeatureGroupTests(unittest.TestCase):
    def test_physics_group_membership_keeps_source_metadata_out(self):
        from deep_ablation import groups_for
        from deep_physics_features import assert_safe
        g=groups_for(['is_gga_plus_u','mean_d_electron_count','mom_angle_mean_deg','orbital_d_fraction','mp_dos_ef_states_ev_cell','run_type_2'])
        self.assertEqual(set(g['correlation']),{'is_gga_plus_u','mean_d_electron_count'})
        self.assertIn('mom_angle_mean_deg',g['lattice']);self.assertIn('orbital_d_fraction',g['orbital'])
        self.assertNotIn('run_type_2',sum(g.values(),[]));assert_safe(sum(g.values(),[]))

class CollectionDiagnosticTests(unittest.TestCase):
    def test_endpoint_error_sample_redacts_api_key(self):
        from unittest.mock import patch
        from enrich_deep_mp import safe_error
        with patch.dict('os.environ',{'MP_API_KEY':'unit-only-not-a-real-key'}):
            text=safe_error(ValueError('request unit-only-not-a-real-key failed'))
        self.assertNotIn('unit-only-not-a-real-key',text);self.assertIn('[redacted]',text)

class CurrentMPMetadataTests(unittest.TestCase):
    def test_current_root_task_id_is_supported_beside_legacy_spin_task(self):
        from types import SimpleNamespace
        from enrich_deep_mp import complete_dos_with_task
        calls=[];curve=object()
        def getter(task):calls.append(task);return curve
        client=SimpleNamespace(materials=SimpleNamespace(electronic_structure=SimpleNamespace(search=lambda **kw:[{'dos':{'task_id':'root-task-fixture','total':{'1':{'efermi':0.0}}}}]),electronic_structure_dos=SimpleNamespace(get_dos_from_task_id=getter)))
        dos,task=complete_dos_with_task(client,'mp-fixture')
        self.assertIs(dos,curve);self.assertEqual(task,'root-task-fixture');self.assertEqual(calls,['root-task-fixture'])

class AblationSelectionTests(unittest.TestCase):
    def test_subset_respects_paired_order_and_rejects_missing_variants(self):
        from deep_ablation import select_variants
        variants={'baseline':1,'plus_spin':2,'all':3}
        self.assertEqual(list(select_variants(variants,['all','baseline'])),['baseline','all'])
        self.assertIs(select_variants(variants),variants)
        with self.assertRaises(ValueError):select_variants(variants,['plus_missing'])

class DOSStorageCompatibilityTests(unittest.TestCase):
    def test_numeric_normalization_preserves_legacy_s3_prefix(self):
        from enrich_deep_mp import dos_storage_keys
        self.assertEqual(dos_storage_keys('alpha-fixture',normalize=lambda task:'1676349'),['dos/mp-1676349.json.gz','dos/alpha-fixture.json.gz'])
    def test_missing_numeric_key_retries_literal_equivalent_task_keys(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from enrich_deep_mp import dos_curve_for_task
        curve=object();calls=[]
        def missing(task):raise OSError('NoSuchKey')
        def get(**kwargs):calls.append(kwargs);return ([{'data':curve}],1)
        route=SimpleNamespace(get_dos_from_task_id=missing,_query_open_data=get)
        with patch('enrich_deep_mp.dos_storage_keys',return_value=['dos/mp-1676349.json.gz']):result=dos_curve_for_task(route,'alpha-fixture')
        self.assertIs(result,curve);self.assertEqual(calls,[{'bucket':'materialsproject-parsed','key':'dos/mp-1676349.json.gz'}])

class OrbitalIntegrationPortabilityTests(unittest.TestCase):
    def test_integration_does_not_depend_on_removed_numpy_trapz(self):
        import numpy as np
        from types import SimpleNamespace
        from unittest.mock import patch
        from pymatgen.core import Composition
        from enrich_deep_mp import orbital_weights
        class DOS:
            energies=np.array([-1.,0.,1.]);efermi=0.;structure=SimpleNamespace(composition=Composition('Fe'))
            def get_element_spd_dos(self,element):return {'s':SimpleNamespace(get_densities=lambda:np.ones(3)),'d':SimpleNamespace(get_densities=lambda:np.ones(3)*3)}
        with patch.object(np,'trapz',None,create=True):r=orbital_weights(DOS())
        self.assertAlmostEqual(r['orbital_s_fraction'],.25);self.assertAlmostEqual(r['orbital_d_fraction'],.75)

class EnrichmentMergeTests(unittest.TestCase):
    def test_reference_properties_preserve_geometry_and_row_order(self):
        import tempfile,csv
        from merge_physics_enrichment import merge
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);base=root/'base.csv';source=root/'mp.csv';output=root/'merged.csv'
            base.write_text('record_id,mom_angle_mean_deg\na,180\nb,90\n')
            source.write_text('record_id,mom_angle_mean_deg,reference_mp_id,orbital_d_fraction\nb,,mp-b,0.5\na,,mp-a,0.75\n')
            report=merge(base,source,output,{'a':'mp-a','b':'mp-b'})
            rows=list(csv.DictReader(output.open()))
            self.assertEqual([r['record_id'] for r in rows],['a','b'])
            self.assertEqual(rows[0]['mom_angle_mean_deg'],'180')
            self.assertEqual(rows[0]['orbital_d_fraction'],'0.75')
            self.assertTrue(report['reference_ids_checked'])
    def test_mismatched_reference_or_cohort_is_rejected(self):
        import tempfile
        from merge_physics_enrichment import merge
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);base=root/'base.csv';source=root/'mp.csv';output=root/'merged.csv'
            base.write_text('record_id\na\n');source.write_text('record_id,reference_mp_id\na,mp-wrong\n')
            with self.assertRaises(ValueError):merge(base,source,output,{'a':'mp-correct'})
            source.write_text('record_id,reference_mp_id\nb,mp-correct\n')
            with self.assertRaises(ValueError):merge(base,source,output)
            source.write_text('record_id,reference_mp_id\na,mp-correct\na,mp-correct\n')
            with self.assertRaises(ValueError):merge(base,source,output)

class ReferenceIDCoverageTests(unittest.TestCase):
    def test_mp_and_mvc_ids_are_included_but_demo_ids_are_excluded(self):
        from enrich_deep_mp import select_reference_ids
        values=['mp-10003','mvc-11697','mp-demo-1',None,'mp-10003']
        self.assertEqual(select_reference_ids(values),['mp-10003','mvc-11697'])
        self.assertEqual(select_reference_ids(values,{'mvc-11697'}),['mvc-11697'])
    def test_unknown_explicit_reference_is_rejected(self):
        from enrich_deep_mp import select_reference_ids
        with self.assertRaises(ValueError):select_reference_ids(['mp-10003'],{'mp-unknown'})

class LegacyIdentityQuarantineTests(unittest.TestCase):
    def test_legacy_dos_lookup_is_blocked_before_any_endpoint(self):
        from types import SimpleNamespace
        from enrich_deep_mp import complete_dos_with_task
        with self.assertRaisesRegex(ValueError,'manual identity resolution'):complete_dos_with_task(SimpleNamespace(),'mvc-11697')
    def test_legacy_numeric_properties_become_missing_with_rejected_audit(self):
        import tempfile,json
        from enrich_deep_mp import flush_output
        row={'record_id':'x','mp_dos_ef_states_ev_cell':3.14,'orbital_d_fraction':.75}
        with tempfile.TemporaryDirectory() as t:
            report=flush_output([row],{'x':'mvc-11697'},{},Path(t)/'out.csv',{})
        self.assertIsNone(row['mp_dos_ef_states_ev_cell']);self.assertIsNone(row['orbital_d_fraction'])
        self.assertEqual(row['mp_dos_status'],'blocked_legacy_namespace')
        self.assertEqual(json.loads(row['rejected_legacy_properties_json'])['mp_dos_ef_states_ev_cell'],3.14)
        self.assertEqual(report['dos_available'],0)

class ChemistryPolicyTests(unittest.TestCase):
    def test_hydrogen_metal_class_valence_and_parent_gap_rules(self):
        from chemistry_policy import substitution_rejection as reject
        self.assertEqual(reject('H','Li','HCl','LiCl'),'hydrogen')
        self.assertEqual(reject('Na','Cl','NaCl','Cl2'),'site_role_mismatch')
        self.assertEqual(reject('Na','Mg','NaCl','MgCl'),'no_common_oxidation_state')
        # These counters distinguish valence, coordination/radius, and layer-role rejection.
        self.assertEqual(reject('Na','K','NaCl','KCl',.2),'insulating_parent')
        self.assertEqual(reject('Na','K','NaCl','KCl2'),'ionic_charge_imbalance')
        self.assertIsNone(reject('Nb','Ta','Nb3Sn','Ta3Sn'))
    def test_epc_point_estimator_matches_reported_lambda_and_omega(self):
        from run_physics_search import physics_point_prediction
        from train_physics_targets import allen_dynes_tc_k
        lam,omega,tc=physics_point_prediction([1,2],[100,400])
        self.assertAlmostEqual(lam,1.5);self.assertAlmostEqual(omega,200)
        self.assertAlmostEqual(tc,allen_dynes_tc_k(lam,omega))
    def test_generated_unknown_targets_are_not_zero_or_trainable(self):
        from train_physics_targets import build_characteristics,fit_physics_models
        source={'jid':'candidate-unit','is_generated_candidate':True,'lamb':None,'wlog':None,'Tc':None,'atoms':{'lattice_mat':[[3,0,0],[0,3,0],[0,0,3]],'coords':[[0,0,0],[.5,.5,.5]],'elements':['Nb','Sn']}}
        rows,names,_=build_characteristics([source]);r=rows[0]
        self.assertIsNone(r['lambda']);self.assertIsNone(r['omega_log_k']);self.assertIsNone(r['source_tc_k']);self.assertIsNone(r['allen_dynes_tc_mu0p1_k'])
        with self.assertRaises(ValueError):fit_physics_models(rows,names)

class CandidatePolicyIntegrityTests(unittest.TestCase):
    def test_global_generation_limit_is_not_exceeded_between_families(self):
        import tempfile
        from unittest.mock import patch
        from rank_candidates import enumerate_substitution_candidates
        from rt_superconductor.harmonized import HarmonizedRecord
        records=[HarmonizedRecord(record_id=character*24,source_db='3dsc',formula=formula,tc_kelvin=10.,applied_pressure_gpa=0.,measurement_method='unknown',sample_form='unknown',doi='unit-fixture',review_status='curated_database',provenance='curated',is_synthetic=0,chemical_family=family) for character,formula,family in [('a','NiO','Ferrites'),('b','ZnCl2','Ferrites'),('c','KCl','Other'),('d','Na2O','Other')]]
        counts={}
        with tempfile.TemporaryDirectory() as t,patch('rank_candidates.ROOT',Path(t)):
            rows=enumerate_substitution_candidates(records,max_candidates=1,drop_counts=counts)
        self.assertEqual(len(rows),1);self.assertEqual(counts['accepted'],1)
        self.assertEqual(counts['considered'],sum(v for k,v in counts.items() if k!='considered'))
    def test_generated_source_cannot_copy_parent_labels_even_if_flag_omitted(self):
        from train_physics_targets import build_characteristics,assert_reference_targets
        source={'jid':'candidate-unit-stale-parent','lamb':1.,'wlog':200.,'Tc':10.,'atoms':{'lattice_mat':[[3,0,0],[0,3,0],[0,0,3]],'coords':[[0,0,0]],'elements':['Nb']}}
        rows,_,_=build_characteristics([source]);row=rows[0]
        for key in ('lambda','omega_log_k','source_tc_k','allen_dynes_tc_mu0p1_k'):self.assertIsNone(row[key])
        with self.assertRaises(ValueError):assert_reference_targets([{'jid':'candidate-cached-unit','lambda':1.,'omega_log_k':200.}])
    def test_physics_drop_counts_reconcile_without_legacy_alias(self):
        from run_physics_search import generate_substitutions
        def source(jid,symbols):return {'jid':jid,'atoms':{'lattice_mat':[[3,0,0],[0,3,0],[0,0,3]],'coords':[[i/len(symbols),0,0] for i in range(len(symbols))],'elements':symbols},'stability':'stable','lamb':1.,'wlog':200.,'Tc':10.}
        counts={};generate_substitutions([source('unit-KCl2',['K','Cl','Cl']),source('unit-Na',['Na']),source('unit-HCl',['H','Cl'])],drop_counts=counts)
        exclusive=sum(value for name,value in counts.items() if name.startswith('dropped_') and name!='dropped_no_common_oxidation_state_balance')
        self.assertEqual(counts['proposals_considered'],counts['accepted']+exclusive)
        self.assertEqual(counts['dropped_ionic_charge_imbalance'],counts['dropped_no_common_oxidation_state_balance'])

class CandidateScoringPipelineTests(unittest.TestCase):
    def test_saved_descriptor_selection_and_imputer_are_used(self):
        import numpy as np
        from sklearn.impute import SimpleImputer
        from rank_candidates import model_feature_matrix
        imputer=SimpleImputer().fit(np.array([[8.,2.],[10.,4.]]))
        result=model_feature_matrix(np.array([[4.,np.nan,8.]]),['a','b','c'],{'feature_names':['c','b'],'imputer':imputer})
        np.testing.assert_allclose(result,[[8.,3.]])
        with self.assertRaises(ValueError):model_feature_matrix(result,['a','b'],{'feature_names':['missing']})
    def test_general_mode_preserves_confident_candidates_before_output_limit(self):
        import csv,json,tempfile
        import numpy as np
        from types import SimpleNamespace
        from unittest.mock import patch
        import rank_candidates as ranker
        from rt_superconductor.harmonized import HarmonizedRecord,write_harmonized
        from rt_superconductor.search import CandidateProposal
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input.csv';model_dir=root/'model';model_dir.mkdir();output=root/'candidates.csv'
            record=HarmonizedRecord(record_id='a'*24,source_db='3dsc',formula='Nb',tc_kelvin=10.,applied_pressure_gpa=0.,measurement_method='unknown',sample_form='unknown',doi='unit-fixture',review_status='curated_database',provenance='curated',is_synthetic=0,chemical_family='Other')
            write_harmonized(source,[record]);(model_dir/'pressure-aware-report.json').write_text(json.dumps({'training_status':'ready'}));(model_dir/'pressure-aware-model.joblib').write_bytes(b'unit-only placeholder')
            class Tree:
                def __init__(self,values):self.values=np.array(values)
                def predict(self,matrix):return self.values
            artifact={'model':SimpleNamespace(estimators_=[Tree([0.,20.]),Tree([100.,20.])])}
            candidates=[{'formula':formula,'chemical_family':'Other','parent_record_id':record.record_id} for formula in ['CuS','ZnS']]
            def features(records,**kwargs):
                self.assertTrue(all(r.tc_kelvin is None and r.is_synthetic==1 for r in records))
                return np.zeros((2,1)),['unit-feature'],'unit-fixture'
            def acquisition(hypotheses,*,batch_size):
                return [CandidateProposal(candidate=hypotheses[0],score=100.,reasons=('unit low-confidence priority',)),CandidateProposal(candidate=hypotheses[1],score=1.,reasons=('unit confident priority',))][:batch_size]
            with patch.object(ranker,'enumerate_substitution_candidates',return_value=candidates),patch.object(ranker,'_mp_matches',return_value=({},'degraded_no_mp_api_key',None)),patch.object(ranker.joblib,'load',return_value=artifact),patch.object(ranker,'build_features',side_effect=features),patch.object(ranker,'propose_batch',side_effect=acquisition):
                report=ranker.rank_candidates(input_path=source,model_dir=model_dir,output_path=output,batch_size=1)
            with output.open() as handle:rows=list(csv.DictReader(handle))
        self.assertEqual(rows[0]['formula'],'ZnS');self.assertEqual(rows[0]['confidence_group'],'confident')
        self.assertEqual(report['review_ranked_count'],1);self.assertEqual(report['low_confidence_count'],0)
