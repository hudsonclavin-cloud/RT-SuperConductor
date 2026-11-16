"""Why rung 4 test T2 failed: element coverage of the calculated set (exploratory; docs/rung4-plan.md)."""
import sys, json, numpy as np, collections
from pathlib import Path; _R = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(_R / 'scripts'), str(_R / 'src')]
import rung4 as R, scratchnet_common as C, run_learning_cycle as L
settings=L.boosting_settings()
train=R.bercx_rows(); X=R.features(train); y=np.array([r['target'] for r in train]); g=np.array([r['system'] for r in train])
el=collections.Counter(e for r in train for e in r['fractions'])
LN=set('La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Ac Th Pa U Np Pu Am'.split())
print('Table B rows with an f-element:', sum(bool(set(r['fractions'])&LN) for r in train), 'of', len(train))
print('rarest elements in Table B:', sorted(el.items(), key=lambda kv: kv[1])[:25])
final=R.fit_proxy(X,y,g,np.arange(len(train)),seed=0,settings=settings)
exp=R.experimental_rows(R.load_candidates(), {r['key'] for r in train})
prim=[r for r in exp if not r['in_table_b']]; ind=[r for r in exp if r['in_table_b']]
p=C.boosting_apply(final,R.features(prim))[0]; yp=np.array([r['label'] for r in prim])
f=np.array([bool(set(r['fractions'])&LN) for r in prim])
cov=np.array([min(el.get(e,0) for e in r['fractions']) for r in prim])
def show(name,mask):
    if mask.sum()>20 and 0<yp[mask].sum()<mask.sum(): print(name, int(mask.sum()), 'SC', int(yp[mask].sum()), 'share', round(float(yp[mask].mean()),3), 'AUC', round(R.auc(yp[mask],p[mask]),3), 'mean proxy', round(float(p[mask].mean()),3))
show('T2 all',np.ones(len(prim),bool)); show('with f-element',f); show('no f-element',~f)
show('rarest element seen <30 times in Table B',cov<30); show('all elements seen >=100 times',cov>=100)
fi=np.array([bool(set(r['fractions'])&LN) for r in ind]); print('in-domain with f-element', int(fi.sum()), 'of', len(ind))
print('mean proxy for SC vs non-SC (T2):', round(float(p[yp==1].mean()),3), round(float(p[yp==0].mean()),3))
# what are the T2 non-superconductors and superconductors that the proxy gets most wrong?
order=np.argsort(-p)
print('top proxy, labelled NOT sc:', [prim[i]['formula'] for i in order if yp[i]==0][:25])
print('bottom proxy, labelled SC:', [prim[i]['formula'] for i in order[::-1] if yp[i]==1][:25])
nel=np.array([len(r['fractions']) for r in prim])
for k in (1,2,3,4): show(f'{k} elements', nel==k)
