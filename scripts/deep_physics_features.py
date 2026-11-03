#!/usr/bin/env python3
"""Audited 3DSC physical features; computed quantities are proxies, never labels."""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ingest_3dsc import _read_rows, _record_id, _cif_candidates
from rt_superconductor.formula import canonicalize_formula

FORBIDDEN = ('synth_doped','totreldiff','formula_similarity','correct_formula_frac','formula_frac','weight','created_at','updated')
NUMERIC = ['band_gap_2','lata_2','latb_2','latc_2','density_2','formation_energy_per_atom_2','efermi_2','total_magnetization_normalized_formula_units_2']
SYSTEMS = ['triclinic','monoclinic','orthorhombic','tetragonal','trigonal','hexagonal','cubic']
ORDERS = ['NM','FM','FiM','AFM']
def number(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (TypeError,ValueError): return None
def assert_safe(names):
    bad=[n for n in names if (any(x in n.lower() for x in FORBIDDEN if x!='weight') or re.search(r'(^|[_\s])weight($|[_\s])',n.lower())) or n in ('tc','tc_kelvin','record_id','run_type_2')]
    if bad: raise ValueError(f'Forbidden model features: {bad}')
def table_features(row):
    d={k:number(row.get(k)) for k in NUMERIC}
    a,b,c=(d[k] for k in ['lata_2','latb_2','latc_2'])
    n=number(row.get('nsites_2'))
    d.update(c_over_a=c/a if a and c else None,b_over_a=b/a if a and b else None)
    v=number(row.get('cell_volume_2'))
    d['volume_per_atom']=v/n if v is not None and n else None
    m=number(row.get('num_magnetic_sites_2'))
    d['magnetic_sites_per_atom']=m/n if m is not None and n else None
    raw=str(row.get('is_magnetic_2','')).lower()
    d['is_magnetic_2']=1. if raw in ('true','1') else 0. if raw in ('false','0') else None
    d.update({f'crystal_system_{s}':float(row.get('crystal_system_2')==s) for s in SYSTEMS})
    order=row.get('ordering_2')
    d.update({f'ordering_{s}':float(order==s) if order in ORDERS else None for s in ORDERS})
    # Point group is one-hot encoded inside training folds, never ordinal.
    d['point_group_2']=row.get('point_group_2','')
    d['run_type_2']=row.get('run_type_2','') # metadata flag; NEVER a predictor
    return d

def cif_features(path):
    from pymatgen.core import Structure
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.analysis.graphs import StructureGraph
    from pymatgen.analysis.dimensionality import get_dimensionality_larsen
    from pymatgen.analysis.ewald import EwaldSummation
    from evaluate_3dsc_structure_spin import _ordered_proxy
    s,proxy=_ordered_proxy(Structure.from_file(path))
    lists=s.get_all_neighbors(12)
    nearest=[min(n.nn_distance for n in ns if n.nn_distance>1e-8) for ns in lists]
    cnn=CrystalNN()
    infos=[cnn.get_nn_info(s,i) for i in range(len(s))]
    cn=np.array([len(x) for x in infos])
    g=StructureGraph.from_empty_graph(s)
    for i,ns in enumerate(infos):
        for x in ns: g.add_edge(i,x['site_index'],to_jimage=x['image'],warn_duplicates=False)
    dim=int(get_dimensionality_larsen(g))
    pairs={'metal_anion':[],'metal_metal':[]}
    # Minimum periodic separations. Anion requires a common negative oxidation state.
    for i,ns in enumerate(lists):
        left=s[i].specie
        for n in ns:
            right=n.specie
            if n.nn_distance<1e-8: continue
            if left.is_metal and right.is_metal: pairs['metal_metal'].append(n.nn_distance)
            if left.is_metal and any(v<0 for v in right.common_oxidation_states): pairs['metal_anion'].append(n.nn_distance)
    d={'nearest_min':float(min(nearest)), 'nearest_mean':float(np.mean(nearest)), 'nearest_std':float(np.std(nearest)),
       'coord_mean':float(cn.mean()),'coord_std':float(cn.std()),'dimensionality':dim,'layer_spacing_proxy_angstrom':None,
       'disorder_proxy':proxy,'ewald_ev_per_atom':None,'ewald_status':'no_common_charge_assignment'}
    d.update({f'coord_fraction_{n}':float(np.mean(cn==n)) for n in (4,6,8,12)})
    for k,v in pairs.items(): d[f'shortest_{k}']=float(min(v)) if v else None
    if dim==2:
        # Largest empty periodic slab along a reciprocal-cell normal; a cell-dependent geometric proxy.
        spacings=[]
        for ax in range(3):
            x=np.sort(s.frac_coords[:,ax]%1); gaps=np.diff(np.r_[x,x[0]+1])
            spacings.append(float(max(gaps)/np.linalg.norm(s.lattice.reciprocal_lattice_crystallographic.matrix[ax])))
        d['layer_spacing_proxy_angstrom']=max(spacings)
    try:
        if any(any(o<0 for o in el.common_oxidation_states) for el in s.composition.elements):
            guesses=s.composition.oxi_state_guesses(max_sites=-1)
            if guesses:
                s.add_oxidation_state_by_element(guesses[0]); d['ewald_ev_per_atom']=float(EwaldSummation(s).total_energy/len(s));d['ewald_status']='formal_oxidation_state_proxy'
    except Exception as e: d['ewald_status']=type(e).__name__
    return d

def cached_cif(job):
    path,cache=map(Path,job)
    warnings.filterwarnings('ignore')
    sha=hashlib.sha256(path.read_bytes()).hexdigest(); target=cache/(sha+'.json')
    if target.exists(): return sha,json.loads(target.read_text())
    try: data=cif_features(path)
    except Exception as e: data={'feature_status':'blocked','error':str(e)}
    cache.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(data,allow_nan=False))
    return sha,data

def build(source,root,output,cache,workers=4,geometry=True):
    rows=_read_rows(source); results=[]; jobs=[]
    for i,row in enumerate(rows,2):
        p=canonicalize_formula(row.get('formula_sc',''));tc=number(row.get('tc'))
        if p.status!='parsed' or tc is None: continue
        d={'record_id':_record_id(row,i,p.canonical,tc),**table_features(row),'magnetic_provenance':'DFT-computed, not measured'}
        path=next((x for x in _cif_candidates(root,row.get('cif','')) if x.is_file()),None)
        if path and geometry: jobs.append((len(results),(str(path),str(cache))))
        results.append(d)
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for count,((idx,_),(sha,features)) in enumerate(zip(jobs,pool.map(cached_cif,[j for _,j in jobs],chunksize=1)),1):
                results[idx].update(features,cif_sha256=sha)
                if count%100==0: print(f'CIFs {count}/{len(jobs)}',flush=True)
    fields=list(dict.fromkeys(k for r in results for k in r)); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
    report={'rows':len(results),'cifs':len(jobs),'blocked':sum(r.get('feature_status')=='blocked' for r in results),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'features_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('data/raw/3dsc/MP/3DSC_MP.csv'));p.add_argument('--root',type=Path,default=Path('data/raw/3dsc/MP/cifs'));p.add_argument('--output',type=Path,default=Path('data/processed/structure_features.csv'));p.add_argument('--cache',type=Path,default=Path('data/processed/cif-feature-cache'));p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();print(build(a.source,a.root,a.output,a.cache,a.workers))
