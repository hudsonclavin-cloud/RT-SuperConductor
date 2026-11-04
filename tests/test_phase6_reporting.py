"""Scientific-report and reuse boundaries; all materials here are test fixtures."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))


class UnknownMeasurementPressureTests(unittest.TestCase):
    def record(self, pressure):
        from rt_superconductor.harmonized import HarmonizedRecord
        return HarmonizedRecord(record_id='a'*24, source_db='3dsc', formula='FeSe', tc_kelvin=8.0,
                                applied_pressure_gpa=pressure, measurement_method='unknown', sample_form='unknown',
                                doi='unit-fixture', review_status='curated_database', provenance='curated', is_synthetic=0,
                                chemical_family='Ferrites')

    def test_unknown_measurement_pressure_round_trips_and_is_not_zero(self):
        from rt_superconductor.harmonized import write_harmonized, load_harmonized, stable_record_id
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'records.csv'
            write_harmonized(path, [self.record(None)])
            with path.open() as handle:
                self.assertEqual(next(csv.DictReader(handle))['applied_pressure_gpa'], '')
            self.assertIsNone(load_harmonized(path)[0].applied_pressure_gpa)
        self.assertNotEqual(stable_record_id('3dsc','FeSe','unit-fixture',8,None), stable_record_id('3dsc','FeSe','unit-fixture',8,0))

    def test_pressure_repair_does_not_change_deep_formula_baseline(self):
        import numpy as np
        from deep_ablation import load
        from rt_superconductor.harmonized import write_harmonized
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);features=root/'features.csv';records=root/'records.csv'
            features.write_text('record_id,band_gap_2\n'+'a'*24+',0\n')
            write_harmonized(records, [self.record(0.000101325)])
            _,known,names,_=load(records,features)
            write_harmonized(records, [self.record(None)])
            _,unknown,new_names,_=load(records,features)
        self.assertEqual(names,new_names)
        self.assertNotIn('applied_pressure_gpa',names)
        np.testing.assert_allclose(known,unknown,equal_nan=True)

    def test_unknown_pressure_never_enters_ambient_or_high_pressure_gate(self):
        from promotion_gate import evaluate_gate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'predictions.csv';model=root/'model.joblib';model.write_bytes(b'unit-fixture')
            source.write_text('formula,chemical_family,tc_kelvin,predicted_tc_kelvin,applied_pressure_gpa,review_status,provenance,is_synthetic,source_db\n'
                              'FeSe,Ferrites,8,7,0.000101325,curated_database,curated,0,3dsc\n'
                              'H3S,Hydrides,200,100,,curated_database,curated,0,3dsc\n')
            result=evaluate_gate(predictions_path=source,model_path=model)
        self.assertEqual(result['unknown_pressure_rows'],1)
        self.assertEqual(result['ambient_eligible_rows'],1)
        self.assertEqual(result['ambient_mae_k'],1)
        self.assertEqual(result['hydride_benchmark_rows']['H3S'],0)

    def test_unknown_reviewed_pressure_bridge_is_idempotent(self):
        from rt_superconductor.harmonized import bridge_reviewed_records, load_harmonized, write_harmonized
        with (ROOT/'data/training/reviewed/reviewed_records.csv').open(newline='',encoding='utf-8') as handle:
            reader=csv.DictReader(handle);fields=reader.fieldnames;reviewed=next(reader)
        reviewed['pressure_gpa']='';reviewed['pressure_condition']='unknown'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);harmonized=root/'harmonized.csv';folder=root/'reviewed';folder.mkdir();reviewed_path=folder/'reviewed_records.csv'
            write_harmonized(harmonized,[])
            with reviewed_path.open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerow(reviewed)
            first=bridge_reviewed_records(harmonized,reviewed_path)
            before=harmonized.read_bytes()
            second=bridge_reviewed_records(harmonized,reviewed_path)
            result=load_harmonized(harmonized)
            self.assertEqual(harmonized.read_bytes(),before)
        self.assertEqual(first['bridged_record_count'],1)
        self.assertEqual(second['bridged_record_count'],0)
        self.assertEqual(len(result),1)
        self.assertIsNone(result[0].applied_pressure_gpa)

    def test_nonfinite_pressure_is_invalid_instead_of_an_observation(self):
        from promotion_gate import measured_pressure
        for value in ('nan','inf','-1'):
            with self.assertRaises(ValueError):measured_pressure({'applied_pressure_gpa':value})
        self.assertIsNone(measured_pressure({'applied_pressure_gpa':''}))


class ReproducibleStudyTests(unittest.TestCase):
    def test_library_version_change_invalidates_scope_cache(self):
        import numpy as np
        from deep_ablation import evaluate
        x=np.array([[0.],[1.],[2.],[3.]])
        y=np.array([0.,1.,0.,2.]);families=np.array(['A','A','B','B']);calls=[]
        def transform(train,test):calls.append(1);return x[train],x[test]
        with tempfile.TemporaryDirectory() as tmp:
            with patch('deep_ablation.evaluation_environment',return_value={'software':{'numpy':'fixture-1'}}):
                evaluate(x,y,families,families,['safe_feature'],trees=2,fold_transform=transform,checkpoint_dir=Path(tmp))
                self.assertEqual(len(calls),2);calls.clear()
                evaluate(x,y,families,families,['safe_feature'],trees=2,fold_transform=transform,checkpoint_dir=Path(tmp))
                self.assertEqual(calls,[])
            with patch('deep_ablation.evaluation_environment',return_value={'software':{'numpy':'fixture-2'}}):
                evaluate(x,y,families,families,['safe_feature'],trees=2,fold_transform=transform,checkpoint_dir=Path(tmp))
                self.assertEqual(len(calls),2)

    def test_study_metadata_has_input_and_cif_hashes_and_units(self):
        from deep_ablation import write_study_metadata
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);input_path=root/'input.csv';features=root/'features.csv';output=root/'study.json'
            input_path.write_text('unit-fixture')
            features.write_text('record_id,cif_sha256\na,'+'f'*64+'\n')
            report=write_study_metadata(input_path,features,[SimpleNamespace(record_id='a')],['safe_feature'],output)
            self.assertEqual(json.loads(output.read_text())['cohort_rows'],1)
        for name in ('harmonized_sha256','feature_table_sha256','record_ids_sha256','cif_hashes_sha256'):
            self.assertEqual(len(report[name]),64)
        self.assertEqual(report['target_units'],'K')
        self.assertEqual(report['metric_units']['auc0'],'dimensionless')
        self.assertIn('not ambient',report['pressure_scope'])

    def test_mvc_numeric_artifact_is_quarantined_but_blank_rows_are_allowed(self):
        from merge_physics_enrichment import merge
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'geometry.csv';source=root/'mp.csv';output=root/'merged.csv'
            base.write_text('record_id,mom_angle_mean_deg\na,180\n')
            source.write_text('record_id,reference_mp_id,mp_dos_ef_states_ev_cell\na,mvc-11697,10\n')
            with self.assertRaisesRegex(ValueError,'Unresolved MVC'):merge(base,source,output,{'a':'mvc-11697'})
            source.write_text('record_id,reference_mp_id,mp_dos_ef_states_ev_cell\na,mvc-11697,\n')
            merge(base,source,output,{'a':'mvc-11697'})
            with output.open() as handle:self.assertEqual(next(csv.DictReader(handle))['mom_angle_mean_deg'],'180')

    def test_workflow_reuses_table_and_keeps_literature_queue_refresh(self):
        content=(ROOT/'.github/workflows/continuous-training.yml').read_text()
        self.assertIn('deep_mp_reference_run:',content)
        self.assertIn('deep_mp_ids:',content)
        self.assertIn('actions/download-artifact@v5',content)
        self.assertIn('Refresh SuperCon2 and SuperMat literature review queues',content)
        self.assertIn('python scripts/merge_physics_enrichment.py',content)
        self.assertIn('data/processed/deep-ablation/*.csv',content)
        self.assertIn('data/processed/ARTIFACT-INDEX.txt',content)
        self.assertNotIn('actions/checkout@v4',content)
        self.assertNotIn('actions/setup-python@v5',content)

class ArtifactReuseProvenanceTests(unittest.TestCase):
    def test_reuse_retains_standard_metadata_name_without_claiming_geometry_run(self):
        text=(ROOT/'.github/workflows/continuous-training.yml').read_text()
        self.assertIn("Path('data/processed/structure_features_mp.json').write_text",text)
        self.assertNotIn('geometry recomputed in this run',text)
        self.assertIn("metadata['queries_this_run']=0",text)
