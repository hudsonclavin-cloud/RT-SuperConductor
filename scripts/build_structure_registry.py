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
    manifest_path=root/'structure-manifest.json'
    if not manifest_path.exists():
        manifest_path=root/'structures'/'structure-manifest.json'
    manifest=json.loads(manifest_path.read_text())
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
            from pymatgen.core import Composition
            expected=Composition(entry['formula']).reduced_composition
            actual=comp.reduced_composition
            qcrow['formula_match_reduced']=int(expected.almost_equals(actual))
            if not qcrow['formula_match_reduced']:
                qcrow['status']='warning'
                qcrow['warnings']='CIF composition differs from reported formula; phase/stoichiometry review required'
            # Approximate first shell using the local distance gap. Keep method
            # visible: this is a geometry descriptor, not validated chemistry.
            cn_values=[]; bond_lengths=[]
            for site in s:
                neighbors=sorted(float(n.nn_distance) for n in s.get_neighbors(site,5.0))
                if neighbors:
                    cutoff=neighbors[0]*1.25
                    neighbors=[d for d in neighbors if d<=cutoff]
                cn_values.append(len(neighbors));bond_lengths.extend(neighbors)
            atoms=len(s); density=float(s.density)
            first=linked[0] if linked else {}
            feat={'structure_id':sid,'mp_id':mp,'formula':entry['formula'],'num_sites':atoms,'volume_angstrom3':float(s.volume),'volume_per_site_angstrom3':float(s.volume/atoms),'density_g_cm3':density,'spacegroup_number':qcrow['spacegroup_number'],'min_distance_angstrom':qcrow['min_distance_angstrom'],'partial_occupancy':qcrow['partial_occupancy'],'energy_above_hull_eV_atom':first.get('energy_above_hull',''),'formation_energy_eV_atom':first.get('formation_energy_per_atom',''),'band_gap_eV':first.get('band_gap_ev',''),'is_metal_MP':first.get('is_metal',''),'total_magnetization_muB_MP':first.get('total_magnetization_muB',''),'density_g_cm3_MP':first.get('density_g_cm3',''),'coordination_mean':sum(cn_values)/len(cn_values) if cn_values else '','bond_length_mean_angstrom':sum(bond_lengths)/len(bond_lengths) if bond_lengths else '','coordination_method':'nearest_shell_distance_gap_factor_1.25','coordination_status':'approximate_geometry_only','dimensionality':'not_computed','feature_source':'CIF+pymatgen geometry; MP summary fields as calculated database properties'}
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
    unresolved=[]
    for item in manifest.get('entries',[]):
        if item.get('structure_status')!='retrieved':
            unresolved.append({'record_id':item.get('record_id',''),'formula':item.get('formula',''),'current_status':item.get('structure_status',''),'resolution_decision':'unresolved','proposed_mapping':'','reason':'No defensible experimental/computed structure was supplied by current composition match; review before mapping.','review_confidence':'low','source':'Materials Project enrichment/structure retrieval snapshot'})
    write('structure-resolution-review.csv',unresolved)
    report={'unique_mp_structures':len(entries),'retrieved_manifest_records':manifest.get('retrieved_count',0),'qc_valid':sum(r.get('status')=='valid' for r in qc),'qc_warning':sum(r.get('status')=='warning' for r in qc),'qc_invalid':sum(r.get('status')=='invalid' for r in qc),'unresolved_records':[{'record_id':i['record_id'],'formula':i['formula'],'status':i['structure_status']} for i in manifest.get('entries',[]) if i.get('structure_status')!='retrieved'],'limitations':['Coordination, dimensionality, formation energy, and prototype similarity require additional algorithms/data and are explicitly uncomputed.','MP reference structures do not establish experimental phase identity.']}
    report['feature_coverage']={key:sum(bool(row.get(key) not in ('',None)) for row in features) for key in ('spacegroup_number','density_g_cm3','energy_above_hull_eV_atom','formation_energy_eV_atom','band_gap_eV','is_metal_MP','total_magnetization_muB_MP','coordination_mean','bond_length_mean_angstrom')}
    (out/'deduplication-report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'feature-coverage.json').write_text(json.dumps({'unique_structure_count':len(features),'nonmissing_count_by_feature':report['feature_coverage'],'feature_source_note':'Database values are computed Materials Project properties; geometry values derive from CIF; uncomputed values remain blank.'},indent=2)+'\n')
    # These templates make absent evidence explicit at structure granularity.
    electronic=[]
    for mp,entry in sorted(entries.items()):
        for stage in ('qe_relax','qe_scf','qe_nscf','bader','elf','cohp','wannier','epw'):
            electronic.append({'structure_id':entry['structure_id'],'mp_id':mp,'stage':stage,'status':'blocked','result_file':'','result_sha256':'','software_version':'','settings':'','reason':'External DFT/analysis output not supplied'})
    write('electronic-analysis-results.csv',electronic)
    decision=out/'research-decision-package';decision.mkdir(exist_ok=True)
    (decision/'README.md').write_text('# Research decision package\n\nCurrent run products are linked from the parent artifact: unique-structures.csv, structure-lineage.csv, structure-qc.csv, structure-features.csv, calculation-manifest.csv, and electronic-analysis-results.csv. No first-principles or new experimental calculations are complete. Review records with formula mismatch or unresolved structure mapping before using structures for science. Empirical modeling remains blocked until independent confirmed nonsynthetic families are available.\n')
    (decision/'status.json').write_text(json.dumps({'unique_structures':len(entries),'qc_valid':report['qc_valid'],'qc_warning':report['qc_warning'],'qc_invalid':report['qc_invalid'],'dft_calculations_completed':0,'dft_calculations_blocked':len(entries),'bader_elf_cohp_wannier_completed':0,'empirical_model_status':'blocked_or_insufficient_family_diversity'},indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--harmonized',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.artifact_root,a.harmonized,a.output_dir),indent=2))
