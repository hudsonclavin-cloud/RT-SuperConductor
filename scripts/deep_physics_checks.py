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
    if not np.isfinite([minimum_thz,max_force,max_stress]).all():
        raise ValueError('Non-finite frequency, force or stress cannot establish stability')
    if max_force>.05 or max_stress>.1:return 'blocked_relaxation_not_converged'
    return 'dynamically_unstable_ml' if minimum_thz<-.5 else 'no_instability_detected_ml'

def atomic_screen_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    def numeric(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError('Unsupported checkpoint value: '+type(value).__name__)
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False,default=numeric)+'\n')
    temporary.replace(path)

class ScreenBudget:
    """Persist cumulative CPU use; a higher explicit limit can resume unfinished work."""
    def __init__(self,path,signature,limit):
        self.path,self.signature,self.limit=Path(path),signature,float(limit)
        if self.limit<=0:raise ValueError('CPU budget must be positive')
        old=json.loads(self.path.read_text()) if self.path.exists() else {}
        if old and old.get('signature')!=signature:
            archive=self.path.with_name(self.path.stem+'-'+old['signature'][:16]+'.json')
            if not archive.exists():atomic_screen_json(archive,old)
        self.previous=float(old.get('cpu_seconds',0)) if old.get('signature')==signature else 0.
        self.started=time.process_time()
    @property
    def elapsed(self):return self.previous+time.process_time()-self.started
    def save(self):
        atomic_screen_json(self.path,{'signature':self.signature,'cpu_seconds':self.elapsed,'limit_seconds':self.limit})
    def check(self):
        self.save()
        if self.elapsed>=self.limit:raise TimeoutError('Cumulative phonon-screen CPU budget reached; increase --cpu-budget-hours to resume')

def structure_atoms(s):
    return {'lattice_mat':s.lattice.matrix.tolist(),'coords':s.frac_coords.tolist(),'elements':[site.specie.symbol for site in s],'cartesian':False}

def candidate_structure(row,proposals=None):
    """Require the exact proposed composition and retain occupational ambiguity."""
    from pymatgen.core import Structure,Composition
    candidate_path=row.get('candidate_cif_path')
    if candidate_path and Path(candidate_path).is_file():
        path=Path(candidate_path);actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if row.get('candidate_cif_sha256') and actual!=row['candidate_cif_sha256']:
            raise ValueError('Candidate CIF content hash differs from the prepared input')
        s=Structure.from_file(path)
        if not s.is_ordered:raise ValueError('Occupational ordering required; rounding occupations would change the candidate composition')
    else:
        proposal=(proposals or {}).get(row['formula'])
        if not proposal or row.get('prototype_jid')!=proposal.get('prototype_jid') or row.get('substitution')!=proposal.get('substitution'):
            return None
        atoms=proposal['atoms'];s=Structure(atoms['lattice_mat'],atoms['elements'],atoms['coords'],coords_are_cartesian=atoms.get('cartesian',False))
    if not s.composition.fractional_composition.almost_equals(Composition(row['formula']).fractional_composition,rtol=1e-5,atol=1e-5):
        raise ValueError('Candidate structure composition differs from the shortlist formula')
    return s

def screen_config(relax_steps=150,repeat=3,mesh=3):
    if repeat<1 or mesh<1:raise ValueError("Supercell repeat and mesh must be positive")
    import importlib.metadata
    import chgnet.model.model as model_module
    checkpoint=Path(model_module.__file__).parent/'../pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar'
    versions={name:importlib.metadata.version(name) for name in ('chgnet','torch','phonopy','pymatgen','pymatgen-core','ase','numpy','scipy')}
    return {'protocol':'chgnet-phonopy-screen-v4','model_name':'0.3.0','model_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'software':versions,'relax_steps':int(relax_steps),'optimizer_stopping_rule':'atomic_force_and_stress','force_tolerance_ev_angstrom':.05,'stress_tolerance_gpa':.1,'target_pressure_gpa':0.,'supercell':[repeat]*3,'coarse_mesh':[mesh]*3,'displacement_angstrom':.01,'unstable_below_thz':-.5,'supercell_atom_limit':1024,'reference_force_subtraction':True,'acoustic_sum_rule':True,'nac_applied':False,'fidelity':'ML-potential screen, not DFT or experimental evidence','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reference_control_status':'Al false-positive instability; diagnostic only, not an experimental rejection criterion'}

def finite_prediction(model,structure,task):
    prediction=model.predict_structure(structure,task=task)
    forces=np.asarray(prediction['f'])
    if forces.shape!=(len(structure),3) or not np.isfinite(forces).all():raise ValueError('Invalid ML forces')
    if 's' in task and (np.asarray(prediction['s']).shape!=(3,3) or not np.isfinite(prediction['s']).all()):raise ValueError('Invalid ML stress')
    return prediction

def screen_structure(structure,model,budget,config,cache_dir,calculation_key):
    """Checkpoint a relaxation and each displacement without rounding occupancies."""
    cache_dir.mkdir(parents=True,exist_ok=True);state_path=cache_dir/(calculation_key+'.json');force_path=cache_dir/(calculation_key+'.npz')
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get('result'):return dict(state['result'],reused_calculation=True)
    from ase.optimize import FIRE
    from chgnet.model.dynamics import StructOptimizer
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure
    budget.check();input_sha=hashlib.sha256(json.dumps(structure_atoms(structure),sort_keys=True).encode()).hexdigest()
    result={'method':'CHGNet 0.3.0 relaxation and finite-displacement phonopy; not DFT','calculation_key':calculation_key,'input_structure_sha256':input_sha,'protocol':config,'units':{'force':'eV/angstrom','stress':'GPa','frequency':'THz','displacement':'angstrom'},'status':'blocked_calculation'}
    if len(structure)*8>config['supercell_atom_limit']:
        result.update(status='blocked_supercell_budget',reason='Configured supercell exceeds atom limit');state['result']=result;atomic_screen_json(state_path,state);return result
    if state.get('relaxed_cif_path'):
        path=Path(state['relaxed_cif_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=state['relaxed_cif_sha256']:raise ValueError('Relaxed checkpoint hash mismatch')
        relaxed=Structure.from_dict(state['relaxed_structure']) if state.get('relaxed_structure') else Structure.from_file(path)
    else:
        class BudgetFIRE(FIRE):
            def gradient_converged(self,gradient):
                from ase import units
                budget.check();atoms=self.atoms.atoms if hasattr(self.atoms,'atoms') else self.atoms
                force=np.asarray(atoms.get_forces());stress=np.asarray(atoms.get_stress(voigt=False))/units.GPa
                if not np.isfinite(force).all() or not np.isfinite(stress).all():raise ValueError('Non-finite relaxation force or stress')
                return float(np.max(np.linalg.norm(force,axis=1)))<=.05 and float(np.max(np.abs(stress)))<=.1
            def step(self,*args,**kwargs):
                budget.check();super().step(*args,**kwargs);budget.check()
        optimizer=StructOptimizer(model=model,optimizer_class=BudgetFIRE,use_device='cpu')
        relaxation=optimizer.relax(structure,fmax=.05,steps=config['relax_steps'],relax_cell=True,verbose=False,assign_magmoms=False)
        relaxed=relaxation['final_structure']
        relaxed_dir=cache_dir/'relaxed-structures';relaxed_dir.mkdir(parents=True,exist_ok=True);path=relaxed_dir/(calculation_key+'.cif');relaxed.to(filename=path)
        state.update(relaxed_structure=relaxed.as_dict(),relaxed_cif_path=str(path),relaxed_cif_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),relaxation_steps=len(relaxation['trajectory']))
        atomic_screen_json(state_path,state);budget.check()
    final=finite_prediction(model,relaxed,'efs');max_force=float(np.max(np.linalg.norm(final['f'],axis=1)));max_stress=float(np.max(np.abs(final['s'])))
    result.update(relaxed_cif_path=state['relaxed_cif_path'],relaxed_cif_sha256=state['relaxed_cif_sha256'],relaxation_steps=state.get('relaxation_steps'),max_final_force_ev_angstrom=max_force,residual_stress_max_gpa=max_stress,relaxation_converged=max_force<=.05 and max_stress<=.1)
    if not result['relaxation_converged']:
        result['status']='blocked_relaxation_not_converged';state['result']=result;atomic_screen_json(state_path,state);return result
    phonon=Phonopy(PhonopyAtoms(symbols=[site.specie.symbol for site in relaxed],cell=relaxed.lattice.matrix,scaled_positions=relaxed.frac_coords),supercell_matrix=np.diag(config['supercell']))
    phonon.generate_displacements(distance=config['displacement_angstrom']);forces=[]
    if force_path.exists():
        with np.load(force_path,allow_pickle=False) as saved:
            reference_forces=saved['reference_forces'];forces=list(saved['forces'])
        if reference_forces.shape!=(len(phonon.supercell),3) or len(forces)>len(phonon.supercells_with_displacements):raise ValueError('Force checkpoint shape mismatch')
    else:
        cell=phonon.supercell;reference=Structure(cell.cell,cell.symbols,cell.scaled_positions)
        reference_forces=np.asarray(finite_prediction(model,reference,'ef')['f']);budget.check()
    for cell in phonon.supercells_with_displacements[len(forces):]:
        budget.check();displaced=Structure(cell.cell,cell.symbols,cell.scaled_positions)
        delta=np.asarray(finite_prediction(model,displaced,'ef')['f'])-reference_forces;delta-=delta.mean(axis=0);forces.append(delta)
        temporary=force_path.with_suffix('.npz.tmp')
        with temporary.open('wb') as handle:np.savez_compressed(handle,reference_forces=reference_forces,forces=np.asarray(forces))
        temporary.replace(force_path);budget.check()
    phonon.forces=forces;phonon.produce_force_constants();phonon.symmetrize_force_constants();phonon.run_qpoints([[0,0,0]]);gamma=np.asarray(phonon.qpoints.frequencies);phonon.run_mesh(config['coarse_mesh'],is_gamma_center=True);mesh=np.asarray(phonon.mesh.frequencies)
    if not np.isfinite(gamma).all() or not np.isfinite(mesh).all():raise ValueError('Non-finite phonon frequency')
    minimum=float(min(gamma.min(),mesh.min()))
    result.update(status=phonon_status(minimum,max_force,max_stress),minimum_frequency_thz=minimum,gamma_min_thz=float(gamma.min()),gamma_frequencies_thz=gamma.tolist(),coarse_mesh=config['coarse_mesh'],mesh_irreducible_qpoints=len(mesh),mesh_qpoint_weights_sum=int(np.sum(phonon.mesh.weights)),unit_conversion_factor=phonon.unit_conversion_factor,supercell=config['supercell'],supercell_atoms=len(phonon.supercell),displacements=len(forces),reference_force_subtraction=True,acoustic_sum_rule=True,nac_applied=False,cpu_seconds_cumulative=budget.elapsed)
    state['result']=result;atomic_screen_json(state_path,state);budget.check();return result

def phonons(shortlist,raw,output,*,cpu_budget_hours=6,relax_steps=150,repeat=3,mesh=3):
    import torch
    from chgnet.model.model import CHGNet
    from run_physics_search import _load_raw,generate_substitutions
    torch.set_num_threads(2);warnings.filterwarnings('ignore')
    if relax_steps<1:raise ValueError('Relaxation steps must be positive')
    config=screen_config(relax_steps,repeat,mesh);config_sha=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();shortlist_sha=hashlib.sha256(shortlist.read_bytes()).hexdigest();signature=config_sha
    output.parent.mkdir(parents=True,exist_ok=True);budget=ScreenBudget(output.with_suffix('.cpu.json'),signature,cpu_budget_hours*3600);cache_dir=output.parent/(output.stem+'-cache');rows=list(csv.DictReader(shortlist.open()));report=[];proposals=None;model=None
    old=json.loads(output.read_text()) if output.exists() else []
    if old:
        archive=output.with_name(output.stem+'-snapshot-'+hashlib.sha256(output.read_bytes()).hexdigest()[:16]+'.json')
        if not archive.exists():atomic_screen_json(archive,old)
    for row in rows:
        result={'candidate_id':row.get('candidate_id') or row.get('candidate_jid'),'chemical_family':row.get('chemical_family'),'formula':row['formula'],'prototype_jid':row.get('prototype_jid'),'substitution':row.get('substitution'),'status':'blocked_missing_structure','method':'CHGNet 0.3.0 and phonopy, not DFT','protocol_sha256':config_sha}
        try:
            candidate_path=row.get('candidate_cif_path')
            if candidate_path and Path(candidate_path).is_file():result['source_cif_sha256']=hashlib.sha256(Path(candidate_path).read_bytes()).hexdigest()
            result['candidate_structure_status']=row.get('candidate_structure_status')
            if not row.get('candidate_cif_path') and row.get('prototype_jid'):
                if proposals is None:proposals={r['formula']:r for r in generate_substitutions(_load_raw(raw))}
            structure=candidate_structure(row,proposals)
            if structure is None:result['reason']=row.get('candidate_structure_status') or 'No defensible candidate structure'
            else:
                input_sha=hashlib.sha256(json.dumps(structure_atoms(structure),sort_keys=True).encode()).hexdigest();key=hashlib.sha256((input_sha+config_sha).encode()).hexdigest()
                state_path=cache_dir/(key+'.json');saved=json.loads(state_path.read_text()) if state_path.exists() else {}
                if saved.get('result'):result.update(saved['result'],reused_calculation=True)
                else:
                    budget.check()
                    if model is None:model=CHGNet.load(model_name='0.3.0',use_device='cpu')
                    result.update(screen_structure(structure,model,budget,config,cache_dir,key))
        except TimeoutError as error:result.update(status='blocked_calculation_budget',reason=str(error))
        except Exception as error:
            status='blocked_occupational_ordering_required' if 'Occupational ordering required' in str(error) else 'blocked_input_structure' if isinstance(error,ValueError) else 'blocked_calculation'
            result.update(status=status,reason=type(error).__name__+': '+str(error))
        report.append(result);budget.save();atomic_screen_json(output,report);print(result['formula'],result['status'],result.get('minimum_frequency_thz',result.get('reason','')),flush=True)
    from collections import Counter
    summary={'shortlist_sha256':shortlist_sha,'protocol_sha256':config_sha,'protocol':config,'rows':len(rows),'status_counts':dict(Counter(r['status'] for r in report)),'unique_calculation_keys':len({r.get('calculation_key') for r in report if r.get('calculation_key')}),'cpu_seconds_cumulative':budget.elapsed,'cpu_budget_hours':cpu_budget_hours,'interpretation':'No detected ML instability is not proof of DFT stability, synthesizability, superconductivity, or ambient-pressure operation.'}
    atomic_screen_json(output.with_suffix('.summary.json'),summary);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['geometry','phonons','prepare']);p.add_argument('--input',type=Path,default=Path('data/training/harmonized_supercon.csv'));p.add_argument('--features',type=Path,default=Path('data/processed/structure_features.csv'));p.add_argument('--index',type=Path,default=Path('data/processed/3dsc-structure-index.csv'));p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,default=Path('data/processed/oxide-cache'));p.add_argument('--raw',type=Path,default=Path('data/raw/jarvis-supercond/jarvis_epc_data_figshare_1058.json.zip'));p.add_argument('--shortlist',type=Path);p.add_argument('--cpu-budget-hours',type=float,default=6);p.add_argument('--relax-steps',type=int,default=150);p.add_argument('--repeat',type=int,default=3);p.add_argument('--mesh',type=int,default=3);a=p.parse_args()
    if a.stage=='prepare':prepare_candidate_structures(a.shortlist,a.index,a.output,a.output.parent/'candidate-structures')
    elif a.stage=='geometry':geometry(a.features,a.index,a.output,a.cache,a.input)
    else:phonons(a.shortlist,a.raw,a.output,cpu_budget_hours=a.cpu_budget_hours,relax_steps=a.relax_steps,repeat=a.repeat,mesh=a.mesh)
