"""Exploratory checks run after rung 4 test T2 failed (docs/rung4-plan.md, result section). Not pre-registered."""
import sys, json, numpy as np
from pathlib import Path; _R = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(_R / 'scripts'), str(_R / 'src')]
import rung4 as R, scratchnet_common as C, run_learning_cycle as L
settings=L.boosting_settings()
train=R.bercx_rows(); X=R.features(train); y=np.array([r['target'] for r in train]); g=np.array([r['system'] for r in train])
folds=np.array([R.system_fold(s) for s in g]); oof=np.full(len(train),np.nan)
for f in range(5):
    m=R.fit_proxy(X,y,g,np.flatnonzero(folds!=f),seed=f,settings=settings); oof[folds==f]=C.boosting_apply(m,X[folds==f])[0]
idx={r['key']:i for i,r in enumerate(train)}
exp=R.experimental_rows(R.load_candidates(), set(idx))
fam={r['key']:C.family_of(set(r['fractions'])) for r in exp}
ind=[r for r in exp if r['in_table_b']]
ye=np.array([r['label'] for r in ind]); po=np.array([oof[idx[r['key']]] for r in ind]); calc=np.array([y[idx[r['key']]] for r in ind])
out={'a_in_domain_rows':len(ind),'a_superconductors':int(ye.sum()),'a_proxy_oof_auc':round(R.auc(ye,po),4),
     'b_calculation_itself_auc':round(R.auc(ye,calc),4),
     'b_share_sc_if_calc_positive':round(float(ye[calc==1].mean()),3),'b_share_sc_if_calc_negative':round(float(ye[calc==0].mean()),3)}
final=R.fit_proxy(X,y,g,np.arange(len(train)),seed=0,settings=settings)
prim=[r for r in exp if not r['in_table_b']]
p=C.boosting_apply(final,R.features(prim))[0]; yp=np.array([r['label'] for r in prim])
conv=np.array([fam[r['key']] in ('Other','Oxides') and 'Cu' not in r['fractions'] for r in prim])
out['c_t2_conventional_lane_rows']=int(conv.sum()); out['c_t2_conventional_lane_sc']=int(yp[conv].sum())
out['c_t2_conventional_lane_auc']=round(R.auc(yp[conv],p[conv]),4)
from collections import Counter
out['t2_family_mix']=dict(Counter(fam[r['key']] for r in prim)); out['t2_sc_by_family']=dict(Counter(fam[r['key']] for r in prim if r['label']))
print(json.dumps(out,indent=1))
from bercx_published import composition_key
cand=R.load_candidates(); info={}
for r in cand:
    if r['theoretical']=='False' and r['is_metal']=='True' and r['known_match']=='exact':
        info.setdefault((composition_key(r['formula']), r['spacegroup_number'].strip()), r)
small=np.array([int(info[r['key']]['num_sites'])<=12 and info[r['key']]['is_magnetic']=='False' for r in prim])
print('d_t2_small_nonmagnetic', int(small.sum()), int(yp[small].sum()), round(R.auc(yp[small],p[small]),4))
print('d_t2_large_or_magnetic', int((~small).sum()), int(yp[~small].sum()), round(R.auc(yp[~small],p[~small]),4))
# in-domain rows scored by the final model (trained on them), for comparison with the OOF figure
pf=C.boosting_apply(final,R.features(ind))[0]; print('e_in_domain_final_model_auc', round(R.auc(ye,pf),4))
