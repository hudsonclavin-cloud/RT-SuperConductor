#!/usr/bin/env python3
"""Create evidence-separated screening lists; never label observations as predictions."""
import argparse,csv,json,re
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--harmonized',type=Path,required=True);p.add_argument('--registry',type=Path,required=True);p.add_argument('--calculations',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 records=list(csv.DictReader(a.harmonized.open())); structures=list(csv.DictReader(a.registry.open())); jobs=list(csv.DictReader(a.calculations.open())); valid={r['mp_id']:r for r in structures}; job={r['mp_id']:r for r in jobs}
 categories={'ambient_observations':[],'low_pressure_observations':[],'high_pressure_controls':[],'unresolved_or_synthetic':[],'experimental_validation':[]}
 for r in records:
  pressure=float(r['applied_pressure_gpa']) if r.get('applied_pressure_gpa') else None
  row={'record_id':r['record_id'],'formula':r['formula'],'reported_tc_kelvin':r['tc_kelvin'],'reported_pressure_gpa':r.get('applied_pressure_gpa',''),'doi':r.get('doi',''),'review_status':r.get('review_status',''),'provenance':r.get('provenance',''),'is_synthetic':r.get('is_synthetic',''),'mp_id':r.get('mp_id',''),'structure_id':valid.get(r.get('mp_id',''),{}).get('structure_id',''),'structure_status':'retrieved' if r.get('mp_id') in valid else 'unresolved_or_demo','calculation_status':job.get(r.get('mp_id',''),{}).get('status','blocked'),'predicted_tc_kelvin':'','evidence_type':'reported_record_not_model_prediction'}
  if r.get('is_synthetic') in ('1','true','True') or row['structure_status']!='retrieved': categories['unresolved_or_synthetic'].append(row)
  elif pressure is not None and pressure>=10: categories['high_pressure_controls'].append(row)
  elif pressure is not None and pressure<=0.001: categories['ambient_observations'].append(row)
  elif pressure is not None and pressure<10: categories['low_pressure_observations'].append(row)
  else: categories['unresolved_or_synthetic'].append(row)
  if r.get('review_status')=='confirmed' and r.get('is_synthetic')=='0': categories['experimental_validation'].append(row)
 a.output_dir.mkdir(parents=True,exist_ok=True)
 for name,rows in categories.items():
  path=a.output_dir/f'{name}.csv'
  cols=list(rows[0]) if rows else ['record_id','formula','evidence_type']
  with path.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
 empirical=[r for r in records if r.get('review_status')=='confirmed' and r.get('is_synthetic')=='0']
 families=sorted({r.get('chemical_family','Unknown') or 'Unknown' for r in empirical})
 groups=sorted({r.get('mp_id','') for r in empirical if r.get('mp_id')})
 model_audit={'eligible_empirical_rows':len(empirical),'independent_chemical_families':families,'family_count':len(families),'unique_mp_structure_count':len(groups),'family_held_out_status':'ready' if len(families)>=2 else 'blocked_insufficient_family_diversity','chronological_test_status':'blocked_publication_dates_not_available_in_harmonized_schema','structure_model_status':'blocked_structure_features_not_joined_to_training_boundary','reason':'Do not fit/evaluate an extrapolative model until independent empirical families and leakage-safe grouping are available.'}
 (a.output_dir/'model-readiness.json').write_text(json.dumps(model_audit,indent=2)+'\n')
 required_elements=sorted({e for r in structures for e in re.findall(r'[A-Z][a-z]?',r.get('formula',''))})
 with (a.output_dir/'pseudopotential-needs.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['element','required_for_unique_structures','pseudo_file','sha256','functional','source_license','status']);w.writeheader()
  for element in required_elements:w.writerow({'element':element,'required_for_unique_structures':sum(element in re.findall(r'[A-Z][a-z]?',r.get('formula','')) for r in structures),'pseudo_file':'','sha256':'','functional':'','source_license':'','status':'needed_user_or_compute_admin'})
 report={'records':len(records),'unique_structures':len(structures),'empirical_confirmed_records':len(categories['experimental_validation']),'lists':{k:len(v) for k,v in categories.items()},'model_readiness':model_audit,'required_pseudopotential_elements':required_elements,'model_status':model_audit['family_held_out_status'],'no_ml_predictions_generated':True,'next_actions':['review unresolved formula-to-structure mappings','obtain compatible pseudopotentials and compute access','add independent confirmed nonsynthetic families before extrapolative model claims']}
 (a.output_dir/'status.json').write_text(json.dumps(report,indent=2)+'\n');(a.output_dir/'README.md').write_text('# Research decision package\n\nCSV lists contain reported records, not predicted Tc values. Synthetic and unresolved records are separated. No DFT or electronic-density calculation results are represented. See status.json for counts and blockers.\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
