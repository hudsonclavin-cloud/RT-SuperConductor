#!/usr/bin/env python3
"""Create evidence-separated screening lists; never label observations as predictions."""
import argparse,csv,json
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
 report={'records':len(records),'unique_structures':len(structures),'empirical_confirmed_records':len(categories['experimental_validation']),'lists':{k:len(v) for k,v in categories.items()},'model_status':'not evaluated here; current harmonized data contains synthetic demonstration rows and limited empirical family diversity','no_ml_predictions_generated':True,'next_actions':['review unresolved formula-to-structure mappings','supply validated pseudopotentials and external compute settings','obtain independent confirmed nonsynthetic families before family-held-out model claims']}
 (a.output_dir/'status.json').write_text(json.dumps(report,indent=2)+'\n');(a.output_dir/'README.md').write_text('# Research decision package\n\nCSV lists contain reported records, not predicted Tc values. Synthetic and unresolved records are separated. No DFT or electronic-density calculation results are represented. See status.json for counts and blockers.\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
