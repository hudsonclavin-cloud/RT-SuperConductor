#!/usr/bin/env python3
"""Deduplicate, validate, and featurize CIF structures from a structure artifact."""
from __future__ import annotations
import argparse, csv, json, hashlib
from collections import defaultdict, Counter
from pathlib import Path

def run(root: Path, harmonized: Path, out: Path) -> dict:
    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        raise RuntimeError("Install project extras with pip install -e '.[ml]'") from exc
    manifest=json.loads((root/'structure-manifest.json').read_text())
    with harmonized.open(newline='',encoding='utf-8-sig') as f: records=list(csv.DictReader(f))
    lineage=defaultdict(list)
    for r in records:
        if r.get('mp_id') and not r['mp_id'].startswith('mp-demo-'):
            lineage[r['mp_id']].append(r)
    entries={}
    qc=[]; features=[]
    for item in manifest.get('entries',[]):
        mp=item.get('mp_id','')
        if item.get('structure_status')!='retrieved' or not mp or mp.startswith('mp-demo-'): continue
        # Reuse one CIF per MP id, preserving all record-to-structure lineage.
        target=entries.setdefault(mp, {'mp_id':mp,'formula':item['formula'],'cif_path':item['cif_path'],'record_ids':[]})
        target['record_ids'].append(item['record_id'])
    for mp,entry in sorted(entries.items()):
        cif=root/'structures'/entry['cif_path']
        sid='mp-'+hashlib.sha256(mp.encode()).hexdigest()[:16]
        row={'structure_id':sid,**entry,'record_ids':';'.join(sorted(set(entry['record_ids'])))}
        linked=lineage.get(mp,[])
        row['source_record_count']=len(linked)
        row['synthetic_only']=int(bool(linked) and all(r.get('is_synthetic') in ('1','true','True') for r in linked))
        row['source_provenance']=';'.join(sorted({r.get('provenance','unknown') for r in linked}))
        try:
            s=Structure.from_file(cif)
            comp=s.composition
            qcrow={'structure_id':sid,'mp_id':mp,'status':'valid','formula_cif':comp.reduced_formula,'formula_source':entry['formula'],'num_sites':len(s),'volume_angstrom3':s.volume,'spacegroup_number':'','partial_occupancy':int(any(abs(sum(site.species.values())-1)>1e-6 for site in s)),'min_distance_angstrom':'','warnings':''}
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                analyzer=SpacegroupAnalyzer(s)
                qcrow['spacegroup_number']=analyzer.get_space_group_number()
            except Exception as exc: qcrow['warnings']=f'symmetry:{type(exc).__name__}'
            distances=s.distance_matrix
            positive=distances[distances>1e-5]
            qcrow['min_distance_angstrom']=float(positive.min()) if positive.size else ''
            if qcrow['partial_occupancy']: qcrow['status']='warning'; qcrow['warnings']='partial_occupancy'
            if qcrow['min_distance_angstrom'] and qcrow['min_distance_angstrom']<0.5: qcrow['status']='invalid'; qcrow['warnings']='unphysical_short_distance'
            qcrow['formula_match_reduced']=int(comp.reduced_formula==entry['formula'])
            atoms=len(s); density=float(s.density)
            feat={'structure_id':sid,'mp_id':mp,'formula':entry['formula'],'num_sites':atoms,'volume_angstrom3':float(s.volume),'volume_per_site_angstrom3':float(s.volume/atoms),'density_g_cm3':density,'spacegroup_number':qcrow['spacegroup_number'],'min_distance_angstrom':qcrow['min_distance_angstrom'],'partial_occupancy':qcrow['partial_occupancy'],'energy_above_hull_eV_atom':next((r.get('energy_above_hull','') for r in linked if r.get('energy_above_hull')), ''),'formation_energy_eV_atom':'','coordination_mean':'','coordination_status':'not_computed','dimensionality':'not_computed','feature_source':'CIF+pymatgen; MP fields from harmonized source'}
            qc.append(qcrow);features.append(feat)
        except Exception as exc:
            qc.append({'structure_id':sid,'mp_id':mp,'status':'invalid','formula_source':entry['formula'],'warnings':f'{type(exc).__name__}: {exc}'})
        entry.update(row)
    out.mkdir(parents=True,exist_ok=True)
    def write(name,rows):
        if not rows:return
        cols=list(dict.fromkeys(k for r in rows for k in r))
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    structs=list(entries.values())
    write('unique-structures.csv',structs)
    write('structure-qc.csv',qc);write('structure-features.csv',features)
    lin=[]
    for mp,rs in lineage.items():
        sid='mp-'+hashlib.sha256(mp.encode()).hexdigest()[:16]
        for r in rs: lin.append({'record_id':r['record_id'],'structure_id':sid,'mp_id':mp,'formula':r['formula'],'tc_kelvin':r['tc_kelvin'],'pressure_gpa':r['applied_pressure_gpa'],'doi':r['doi'],'review_status':r['review_status'],'provenance':r['provenance'],'is_synthetic':r['is_synthetic']})
    write('structure-lineage.csv',lin)
    report={'unique_mp_structures':len(entries),'retrieved_manifest_records':manifest.get('retrieved_count',0),'qc_valid':sum(r.get('status')=='valid' for r in qc),'qc_warning':sum(r.get('status')=='warning' for r in qc),'qc_invalid':sum(r.get('status')=='invalid' for r in qc),'unresolved_records':[{'record_id':i['record_id'],'formula':i['formula'],'status':i['structure_status']} for i in manifest.get('entries',[]) if i.get('structure_status')!='retrieved'],'limitations':['Coordination, dimensionality, formation energy, and prototype similarity require additional algorithms/data and are explicitly uncomputed.','MP reference structures do not establish experimental phase identity.']}
    (out/'deduplication-report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--harmonized',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.artifact_root,a.harmonized,a.output_dir),indent=2))
