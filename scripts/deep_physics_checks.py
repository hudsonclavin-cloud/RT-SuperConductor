#!/usr/bin/env python3
"""Oxide geometry and ML-potential phonons; no DFT stability certification."""
import argparse,csv,hashlib,json,warnings,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

def oxide_angles(path):
    from pymatgen.core import Structure
    from pymatgen.analysis.local_env import CrystalNN
    from evaluate_3dsc_structure_spin import _ordered_proxy
    s,proxy=_ordered_proxy(Structure.from_file(path));angles=[];buckles=[];cnn=CrystalNN()
    for i,site in enumerate(s):
        if site.specie.symbol!='O':continue
        neighbors=[r['site'] for r in cnn.get_nn_info(s,i) if r['site'].specie.is_metal]
        for a in range(len(neighbors)):
            for b in range(a):
                u=neighbors[a].coords-site.coords;v=neighbors[b].coords-site.coords
                if np.linalg.norm(u)*np.linalg.norm(v)>0:angles.append(float(np.degrees(np.arccos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v),-1,1)))))
        # Local metal-plane oxygen displacement, when at least three non-collinear metals exist.
        ns=[n.coords for n in s.get_neighbors(site,5) if n.specie.is_transition_metal] if sum(n.specie.is_transition_metal for n in neighbors)>=2 else []
        if len(ns)>=3:
            xyz=np.array(ns);center=xyz.mean(0);_,sv,vt=np.linalg.svd(xyz-center,full_matrices=False)
            if sv[1]>1e-5 and sv[-1]/max(sum(sv),1e-9)<.15:buckles.append(float(abs(np.dot(site.coords-center,vt[-1]))))
    return {'oxide_disorder_proxy':proxy,'mom_angle_mean_deg':float(np.mean(angles)) if angles else None,'mom_angle_min_deg':min(angles) if angles else None,'oxide_buckling_proxy_angstrom':float(np.mean(buckles)) if buckles else None}

def mean_d_electrons(formula):
    from pymatgen.core import Composition
    comp=Composition(formula);total=0.
    for el,n in comp.items():
        shells=el.full_electronic_structure;outer=max(k for k,l,v in shells);ds=[(k,v) for k,l,v in shells if l=='d']
        val=ds[-1][1] if ds and (el.is_transition_metal or (ds[-1][0]>=outer-1 and ds[-1][1]<10)) else 0
        total+=float(n)*val
    return total/comp.num_atoms

def cached_oxide(args):
    path,cache=args
    if path is None:return {'oxide_geometry_status':'blocked_missing_cif'}
    try:
        path=Path(path);key=hashlib.sha256(path.read_bytes()).hexdigest();target=Path(cache)/('oxide-v2-'+key+'.json')
        if target.exists():return json.loads(target.read_text())
        data=oxide_angles(path);data.update(oxide_geometry_status='computed_reference_proxy',oxide_cif_sha256=key);target.write_text(json.dumps(data));return data
    except Exception as e:return {'oxide_geometry_status':'blocked_'+type(e).__name__}

def geometry(features,index,output,cache,input_path):
    from rt_superconductor.harmonized import load_harmonized
    from concurrent.futures import ProcessPoolExecutor
    formulas={r.record_id:r.formula for r in load_harmonized(input_path)}
    paths={r['record_id']:r['cif_path'] for r in csv.DictReader(index.open())};rows=list(csv.DictReader(features.open()));cache.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        values=pool.map(cached_oxide,[(paths.get(r['record_id']),str(cache)) for r in rows],chunksize=1)
        for i,(r,d) in enumerate(zip(rows,values)):
            r.update(d);r['is_gga_plus_u']=float('+U' in r.get('run_type_2','').upper()) if r.get('run_type_2','').strip() else None;r['mean_d_electron_count']=mean_d_electrons(formulas[r['record_id']]) if r['record_id'] in formulas else None
            if i%250==0:print('oxide geometry',i,flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    report={'rows':len(rows),'geometry_source':'reference CIF; highest-occupancy proxy used only for descriptor extraction, never candidate phonons','features_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'coverage':{k:sum(r.get(k) not in (None,'') for r in rows) for k in ('mom_angle_mean_deg','mom_angle_min_deg','oxide_buckling_proxy_angstrom','is_gga_plus_u','mean_d_electron_count')},'blocked_cifs':sum(str(r.get('oxide_geometry_status','')).startswith('blocked') for r in rows),'disorder_proxies':sum(r.get('oxide_disorder_proxy') is True for r in rows)}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(report);return report

def prepare_candidate_structures(shortlist,index,output,structure_dir):
    from pymatgen.core import Structure,Composition
    lookup={r['record_id']:Path(r['cif_path']) for r in csv.DictReader(index.open())};rows=list(csv.DictReader(shortlist.open()));structure_dir.mkdir(parents=True,exist_ok=True)
    if not rows:
        if output!=shortlist:output.write_bytes(shortlist.read_bytes())
        return rows
    for row in rows:
        row['candidate_cif_path']='';row['candidate_structure_status']='blocked_missing_parent_structure'
        try:
            parent=lookup.get(row.get('parent_record_id'))
            if parent is None or not parent.is_file():continue
            source=Structure.from_file(parent);parent_formula=row.get('parent_formula')
            if parent_formula and not source.composition.fractional_composition.almost_equals(Composition(parent_formula).fractional_composition,rtol=1e-5,atol=1e-5):
                row['candidate_structure_status']='blocked_parent_composition_mismatch';continue
            old,new=row['substitution'].split('->');species=[]
            for site in source:
                mixed={}
                for el,occupancy in site.species.items():
                    symbol=new if el.symbol==old else el.symbol;mixed[symbol]=mixed.get(symbol,0)+float(occupancy)
                species.append(mixed)
            candidate=Structure(source.lattice,species,source.frac_coords)
            if not candidate.composition.fractional_composition.almost_equals(Composition(row['formula']).fractional_composition,rtol=1e-5,atol=1e-5):
                row['candidate_structure_status']='blocked_substituted_composition_mismatch';continue
            identity=hashlib.sha256((str(row.get('candidate_id'))+row['formula']+str(parent)).encode()).hexdigest()[:24];path=structure_dir/(identity+'.cif');candidate.to(filename=path)
            row.update(candidate_cif_path=str(path),candidate_cif_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),candidate_structure_status='unrelaxed_ordered_parent_substitution' if candidate.is_ordered else 'blocked_occupational_ordering_required',structure_provenance='Geometric substitution hypothesis; parent oxidation states and magnetic properties are not candidate measurements')
        except Exception as e:row['candidate_structure_status']='blocked_'+type(e).__name__
    with output.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    return rows

def phonon_status(minimum_thz,max_force,max_stress=0):
    if max_force>.05 or max_stress>.1:return 'blocked_relaxation_not_converged'
    return 'dynamically_unstable_ml' if minimum_thz<-.5 else 'no_instability_detected_ml'

def phonons(shortlist,raw,output):
    import torch
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import StructOptimizer
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure
    from run_physics_search import _load_raw,generate_substitutions
    torch.set_num_threads(2);warnings.filterwarnings('ignore');started=time.process_time();model=CHGNet.load(use_device='cpu');proposals={r['formula']:r for r in generate_substitutions(_load_raw(raw))};rows=list(csv.DictReader(shortlist.open()));report=[]
    for row in rows:
        result={'candidate_id':row.get('candidate_id'), 'chemical_family':row.get('chemical_family'), 'formula':row['formula'],'method':'CHGNet 0.3.0 relaxation and finite-displacement phonopy; not DFT','status':'blocked_missing_structure'}
        try:
            cand=proposals.get(row['formula']);candidate_path=row.get('candidate_cif_path')
            if candidate_path and Path(candidate_path).is_file():
                s=Structure.from_file(candidate_path)
                if not s.is_ordered:raise ValueError('Occupational ordering required; rounding occupations would change the candidate composition')
                a={'lattice_mat':s.lattice.matrix.tolist(),'coords':s.frac_coords.tolist(),'elements':[site.specie.symbol for site in s],'cartesian':False}
            elif cand and row.get('prototype_jid')==cand.get('prototype_jid') and row.get('substitution')==cand.get('substitution'):
                a=cand['atoms'];s=Structure(a['lattice_mat'],a['elements'],a['coords'],coords_are_cartesian=a.get('cartesian',False))
            else:
                result['reason']=row.get('candidate_structure_status','No defensible candidate structure');report.append(result);continue
            if time.process_time()-started>21600:raise TimeoutError('Six CPU-hour phonon-screen budget reached')
            if len(s)*8>512:raise ValueError('2x2x2 supercell exceeds 512-atom CPU budget')
            opt=StructOptimizer(model=model,use_device='cpu');rel=opt.relax(s,fmax=.05,steps=150,verbose=False);s=rel['final_structure']
            final=model.predict_structure(s,task='efs');maxstress=float(np.max(np.abs(final['s'])));maxforce=float(np.max(np.linalg.norm(final['f'],axis=1)))
            result.update(max_final_force_ev_angstrom=maxforce,residual_stress_max_gpa=maxstress,relaxation_converged=maxforce<=.05 and maxstress<=.1)
            if not result['relaxation_converged']:
                result['status']='blocked_relaxation_not_converged';report.append(result);output.write_text(json.dumps(report,indent=2));print(result,flush=True);continue
            result['input_structure_sha256']=hashlib.sha256(json.dumps(a,sort_keys=True).encode()).hexdigest()
            relaxed_dir=output.parent/'ml-relaxed-structures';relaxed_dir.mkdir(parents=True,exist_ok=True);relaxed_path=relaxed_dir/(result['input_structure_sha256'][:24]+'.cif');s.to(filename=relaxed_path);result['relaxed_cif_path']=str(relaxed_path);result['relaxed_cif_sha256']=hashlib.sha256(relaxed_path.read_bytes()).hexdigest()

            p=Phonopy(PhonopyAtoms(symbols=[site.specie.symbol for site in s],cell=s.lattice.matrix,scaled_positions=s.frac_coords),supercell_matrix=np.diag([2,2,2]));p.generate_displacements(distance=.01);forces=[]
            base=p.supercell;reference=Structure(base.cell,base.symbols,base.scaled_positions);reference_forces=np.asarray(model.predict_structure(reference,task='ef')['f'])
            for cell in p.supercells_with_displacements:
                sc=Structure(cell.cell,cell.symbols,cell.scaled_positions);prediction=model.predict_structure(sc,task='ef');delta=np.asarray(prediction['f'])-reference_forces;delta-=delta.mean(axis=0);forces.append(delta)
                if time.process_time()-started>21600:raise TimeoutError('Six CPU-hour phonon-screen budget reached')
            p.forces=forces;p.produce_force_constants();p.symmetrize_force_constants();p.run_qpoints([[0,0,0]]);gamma=np.array(p.qpoints.frequencies);p.run_mesh([3,3,3]);mesh=np.array(p.mesh.frequencies);minimum=float(min(gamma.min(),mesh.min()))
            # A force threshold must be reached before reporting a usable phonon stability screen.
            final=model.predict_structure(s,task='efs');maxstress=float(np.max(np.abs(final['s'])));maxforce=float(np.max(np.linalg.norm(final['f'],axis=1)))
            result.update(reference_force_subtraction=True,acoustic_sum_rule=True,displacement_angstrom=.01,status=phonon_status(minimum,maxforce,maxstress),minimum_frequency_thz=minimum,gamma_min_thz=float(gamma.min()),coarse_mesh=[3,3,3],supercell=[2,2,2],displacements=len(forces),max_final_force_ev_angstrom=maxforce,relaxation_converged=maxforce<=.05 and maxstress<=.1,residual_stress_max_gpa=maxstress)
            result['cpu_seconds_cumulative']=time.process_time()-started
        except Exception as e:result.update(status='blocked_calculation',reason=type(e).__name__+': '+str(e))
        report.append(result);output.write_text(json.dumps(report,indent=2));print(result,flush=True)
    output.write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['geometry','phonons','prepare']);p.add_argument('--input',type=Path,default=Path('data/training/harmonized_supercon.csv'));p.add_argument('--features',type=Path,default=Path('data/processed/structure_features.csv'));p.add_argument('--index',type=Path,default=Path('data/processed/3dsc-structure-index.csv'));p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,default=Path('data/processed/oxide-cache'));p.add_argument('--raw',type=Path,default=Path('data/raw/jarvis-supercond/jarvis_epc_data_figshare_1058.json.zip'));p.add_argument('--shortlist',type=Path);a=p.parse_args()
    if a.stage=='prepare':prepare_candidate_structures(a.shortlist,a.index,a.output,a.output.parent/'candidate-structures')
    elif a.stage=='geometry':geometry(a.features,a.index,a.output,a.cache,a.input)
    else:phonons(a.shortlist,a.raw,a.output)
