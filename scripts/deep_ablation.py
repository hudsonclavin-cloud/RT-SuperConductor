#!/usr/bin/env python3
"""Paired feature/encoder evaluations on fixed family and elemental-system folds."""
import argparse,csv,json,sys,warnings,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error,roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rt_superconductor.harmonized import load_harmonized
from train_pressure_aware import build_features,select_training_records,elemental_system_signature
from deep_physics_features import assert_safe,number
FAMILIES=('Ferrites','Cuprates','Other','Oxides')
def groups_for(names):
    electronic=[n for n in names if n in ('band_gap_2','efermi_2','mp_dos_ef_states_ev_cell')]
    lattice=[n for n in names if n.startswith(('crystal_system_','point_group__')) or n in ('lata_2','latb_2','latc_2','c_over_a','b_over_a','volume_per_atom','dimensionality','layer_spacing_proxy_angstrom','mom_angle_mean_deg','mom_angle_min_deg','oxide_buckling_proxy_angstrom')]
    bonding=[n for n in names if n.startswith(('nearest_','coord_','shortest_')) or n in ('density_2','formation_energy_per_atom_2','ewald_ev_per_atom','mp_bulk_modulus_gpa')]
    spin=[n for n in names if n.startswith('ordering_') or n in ('is_magnetic_2','total_magnetization_normalized_formula_units_2','magnetic_sites_per_atom')]
    correlation=[n for n in names if n in ('is_gga_plus_u','mean_d_electron_count')]
    orbital=[n for n in names if n in ('orbital_s_fraction','orbital_p_fraction','orbital_d_fraction','orbital_f_fraction')]
    return dict(electronic=electronic,lattice=lattice,bonding=bonding,spin=spin,correlation=correlation,orbital=orbital)
def auc(y,p):
    if np.isfinite(p).all() and len(set(y))==2: return float(roc_auc_score(y,p))
    return None

def evaluate(X,y,families,systems,names,trees=150,embargo=None,fold_transform=None,checkpoint_dir=None,train_selector=None):
    assert_safe(names)
    out={}; predictions=[]
    scopes=[('cross-family',np.arange(len(y)),families)]
    scopes += [(f,np.where(families==f)[0],systems[families==f]) for f in FAMILIES if sum(families==f)>=100]
    signature=hashlib.sha256(X.tobytes()+y.tobytes()+repr((names,trees,families.tolist(),systems.tolist(),None if embargo is None else embargo.tolist())).encode()).hexdigest()
    if checkpoint_dir is not None:Path(checkpoint_dir).mkdir(parents=True,exist_ok=True)
    for label,idx,group in scopes:
        cache_file=Path(checkpoint_dir)/(label+'.json') if checkpoint_dir is not None else None
        if cache_file is not None and cache_file.exists():
            cached=json.loads(cache_file.read_text())
            if cached.get('signature')==signature:
                out[label]=cached['metrics'];predictions.extend(cached['predictions']);print(label,'resumed cached scope',flush=True);continue
        prediction_start=len(predictions)
        splits=list(GroupKFold(min(5,len(set(group)))).split(X[idx],y[idx],group))
        pred=np.zeros(len(idx));sd=pred.copy();mean=pred.copy();cp={t:np.full(len(idx),np.nan) for t in (0,77)};mean_cp={t:np.full(len(idx),np.nan) for t in (0,77)};imp=np.zeros(X.shape[1]);skips=0
        for fold,(tr,te) in enumerate(splits):
            train,test=idx[tr],idx[te]
            if train_selector is not None:train=train_selector(train,test,label)
            before_purge=len(train)
            if embargo is not None:
                # A reference MP structure must not cross folds, even for distinct doped compositions.
                held={embargo[i] for i in test if embargo[i]}
                train=np.array([i for i in train if not embargo[i] or embargo[i] not in held]);skips+=before_purge-len(train)
            im=SimpleImputer(keep_empty_features=True)
            xa,xb=fold_transform(train,test) if fold_transform else (X[train],X[test])
            xt=im.fit_transform(xa);xv=im.transform(xb)
            rf=RandomForestRegressor(n_estimators=trees,random_state=4201+fold,n_jobs=2).fit(xt,y[train])
            vals=np.array([t.predict(xv) for t in rf.estimators_]);pred[te]=vals.mean(0);sd[te]=vals.std(0);mean[te]=y[train].mean();imp+=rf.feature_importances_/len(splits)
            for threshold in (0,77):
                classes=(y[train]>threshold).astype(int)
                mean_cp[threshold][te]=classes.mean()
                if len(set(classes))==2:
                    cl=RandomForestClassifier(n_estimators=100,random_state=5201+fold+(threshold==77),n_jobs=2).fit(xt,classes)
                    cp[threshold][te]=cl.predict_proba(xv)[:,1]
            predictions.extend(dict(scope=label,index=int(i),actual=float(y[i]),predicted=float(p),sd=float(s),training_max=float(max(y[train]))) for i,p,s in zip(test,pred[te],sd[te]))
        out[label]=dict(rows=len(idx),mae=float(mean_absolute_error(y[idx],pred)),mean_mae=float(mean_absolute_error(y[idx],mean)),spearman=float(spearmanr(y[idx],pred).statistic),auc0=auc(y[idx]>0,cp[0]),auc77=auc(y[idx]>77,cp[77]),mean_auc0=auc(y[idx]>0,mean_cp[0]),mean_auc77=auc(y[idx]>77,mean_cp[77]),mp_overlap_training_rows_purged=skips,top20=sorted(zip(names,imp.tolist()),key=lambda p:-p[1])[:20])
        print(label,out[label]['mae'],flush=True)
        if cache_file is not None:
            tmp=cache_file.with_suffix('.tmp');tmp.write_text(json.dumps({'signature':signature,'metrics':out[label],'predictions':predictions[prediction_start:]}));tmp.replace(cache_file)
    return out,predictions

def load(input_path,features):
    rows=list(csv.DictReader(features.open()));lookup={r['record_id']:r for r in rows}
    records=[r for r in select_training_records(load_harmonized(input_path)) if r.record_id in lookup]
    X,names,backend=build_features(records)
    # Baseline is formula-only Magpie+shell/Lewis; pressure/hull are not formula descriptors.
    keep=[i for i,n in enumerate(names) if n not in ('applied_pressure_gpa','energy_above_hull','energy_above_hull_missing')]
    X=X[:,keep];names=[names[i] for i in keep]
    available=set().union(*(r.keys() for r in rows));gf=groups_for(available)
    point_groups=sorted({r.get('point_group_2','') for r in rows if r.get('point_group_2')})
    gf['lattice'] += ['point_group__'+v for v in point_groups]
    extra={}
    for g,ns in gf.items():
        ns.sort();a=[]
        for r in records:
            d=lookup[r.record_id];a.append([float(d.get('point_group_2')==n.split('__')[1]) if n.startswith('point_group__') else number(d.get(n)) for n in ns])
        array=np.array([[np.nan if v is None else v for v in vs] for vs in a])
        # Do not present an all-missing group as evidence about its physical value.
        if ns and np.isfinite(array).any():extra[g]=(array,ns)
    return records,X,names,extra

def run(input_path,features,output):
    records,X,names,extra=load(input_path,features);y=np.array([r.tc_kelvin for r in records]);fam=np.array([r.chemical_family for r in records]);systems=np.array([elemental_system_signature(r.formula) for r in records]);mp=np.array([r.mp_id for r in records])
    variants={'baseline':(X,names)}
    for g,(a,ns) in extra.items():variants['plus_'+g]=(np.c_[X,a],names+ns)
    variants['all']=(np.column_stack([X]+[a for a,_ in extra.values()]),names+sum([ns for _,ns in extra.values()],[]))
    output.mkdir(parents=True,exist_ok=True);report={}
    (output/'coverage.json').write_text(json.dumps({'rows':len(records),'groups':{g:{'features':ns,'rows_with_any':int(np.isfinite(a).any(axis=1).sum()),'nonmissing_per_feature':dict(zip(ns,np.isfinite(a).sum(axis=0).tolist()))} for g,(a,ns) in extra.items()},'blocked_groups':[g for g in ('electronic','lattice','bonding','spin','correlation','orbital') if g not in extra]},indent=2))
    for variant,(a,ns) in variants.items():
        print(variant,flush=True);result,preds=evaluate(a,y,fam,systems,ns,embargo=mp,checkpoint_dir=output/'scope-cache'/variant);report[variant]=result
        (output/(variant+'-predictions.json')).write_text(json.dumps(preds,allow_nan=False))
        (output/'ablation.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    with (output/'ablation.csv').open('w') as f:
        cols=['variant','scope','rows','mae','mean_mae','spearman','auc0','auc77','mean_auc0','mean_auc77','mp_overlap_training_rows_purged']
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        for variant,scopes in report.items():
            for scope,metrics in scopes.items():writer.writerow({'variant':variant,'scope':scope,**{k:metrics[k] for k in cols[2:]}})
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('data/training/harmonized_supercon.csv'));p.add_argument('--features',type=Path,default=Path('data/processed/structure_features.csv'));p.add_argument('--output',type=Path,default=Path('data/processed/deep-ablation'));a=p.parse_args();run(a.input,a.features,a.output)
