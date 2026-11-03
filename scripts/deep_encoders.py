#!/usr/bin/env python3
"""CHGNet frozen and supervised encoder pilot with group-held-out probes."""
import argparse,csv,hashlib,json,sys,time,warnings,os
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from deep_ablation import evaluate,load
from train_pressure_aware import elemental_system_signature
BUDGET_CPU_SECONDS=6*3600

def fold_key(tr,te,y):
    return hashlib.sha256(np.asarray(tr).tobytes()+np.asarray(te).tobytes()+np.asarray(y).tobytes()).hexdigest()[:24]

def atomic_torch_save(data,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if 'net' in data:data['rng']=torch.get_rng_state()
    tmp=path.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(path)

def completed_scopes(directory):
    return {p.stem:json.loads(p.read_text())['metrics'] for p in directory.glob('*.json')}

def budget_start(output):
    ledger=output/'cpu-budget.json';previous=float(os.environ.get('DEEP_PREVIOUS_CPU_SECONDS',0))
    if ledger.exists():previous=max(previous,json.loads(ledger.read_text())['cpu_seconds'])
    return time.process_time()-previous

def check_budget(output,started):
    elapsed=time.process_time()-started;target=output/'cpu-budget.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps({'cpu_seconds':elapsed}));tmp.replace(target)
    if elapsed>BUDGET_CPU_SECONDS:raise TimeoutError('Six cumulative CPU-hour budget reached')

def to_graph(model,path):
    from pymatgen.core import Structure
    from evaluate_3dsc_structure_spin import _ordered_proxy
    s,_=_ordered_proxy(Structure.from_file(path));return model.graph_converter(s)

def encode(model,graphs):
    model.eval();parts=[]
    with torch.no_grad():
        for i in range(0,len(graphs),16): parts.append(model(graphs[i:i+16],task='e',return_crystal_feas=True)['crystal_fea'].cpu().numpy())
    return np.concatenate(parts)

def run(input_path,features,index,output):
    from chgnet.model.model import CHGNet
    torch.set_num_threads(2);warnings.filterwarnings('ignore');started=time.process_time()
    output.mkdir(parents=True,exist_ok=True);started=budget_start(output)
    records,_,_,_=load(input_path,features)
    paths={r['record_id']:r['cif_path'] for r in csv.DictReader(index.open())}
    model=CHGNet.load(model_name='0.3.0',use_device='cpu');graphs=[];valid=[];errors=[]
    ids_path=output/'embedding-record-ids.json'
    if (output/'graphs.pt').exists() and ids_path.exists() and json.loads(ids_path.read_text())==[r.record_id for r in records]:
        graphs=torch.load(output/'graphs.pt',weights_only=False);print('resumed',len(graphs),'cached graphs',flush=True)
    else:
        for i,r in enumerate(records):
            try: graphs.append(to_graph(model,paths[r.record_id]));valid.append(i)
            except Exception as e: errors.append({'record_id':r.record_id,'reason':str(e)})
            if i%250==0:print('graphs',i,flush=True)
        records=[records[i] for i in valid];(output/'embedding-record-ids.json').write_text(json.dumps([r.record_id for r in records]));np.save(output/'frozen-embeddings.npy',encode(model,graphs))
        torch.save(graphs,output/'graphs.pt')
    names=[f'CHGNet_{i}' for i in range(64)];y=np.array([r.tc_kelvin for r in records]);fam=np.array([r.chemical_family for r in records]);sys=np.array([elemental_system_signature(r.formula) for r in records]);mp=np.array([r.mp_id for r in records]);X=np.load(output/'frozen-embeddings.npy')
    cohort_sha=hashlib.sha256(json.dumps([r.record_id for r in records]).encode()).hexdigest()
    report_path=output/'encoder-report.json'
    if report_path.exists() and json.loads(report_path.read_text()).get('cohort_sha256')==cohort_sha:
        report=json.loads(report_path.read_text());print('resumed completed frozen evaluation',flush=True)
    else:
        result,preds=evaluate(X,y,fam,sys,names,embargo=mp)
        report={'cohort_sha256':cohort_sha,'checkpoint_sha256':'d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1','fine_tune_epochs':5,'fine_tune_learning_rate':1e-4,'rows':len(records),'failed_graphs':errors,'frozen':result,'pretraining_caveat':'MP structures may overlap held-out inputs; not an independent structural split of pretraining. Tc labels are never used in the frozen encoder.'}
        (output/'encoder-report.json').write_text(json.dumps(report,indent=2))
        (output/'frozen-predictions.json').write_text(json.dumps(preds))
    def tune(tr,te):
        if time.process_time()-started>BUDGET_CPU_SECONDS: raise TimeoutError('Six CPU-hour encoder budget reached')
        torch.manual_seed(2026);net=CHGNet.load(model_name='0.3.0',use_device='cpu');head=torch.nn.Linear(64,1);opt=torch.optim.Adam(list(net.parameters())+list(head.parameters()),lr=1e-4)
        # Five-epoch pilot, fresh initialization in each outer fold. Target is scaled Tc, not energy.
        key=fold_key(tr,te,y);cp=output/'epoch-checkpoints'/(key+'.pt');first_epoch=0;first_batch=0
        if cp.exists():
            state=torch.load(cp,weights_only=False);net.load_state_dict(state['net']);head.load_state_dict(state['head']);opt.load_state_dict(state['opt']);first_epoch=state['epoch'];first_batch=state['batch'];print('resume fine',key,first_epoch,first_batch,flush=True)
        if cp.exists() and 'rng' in state:torch.set_rng_state(state['rng'])
        net.train()
        for epoch in range(first_epoch,5):
            for start in range(first_batch if epoch==first_epoch else 0,len(tr),16):
                b=tr[start:start+16];z=net([graphs[i] for i in b],task='e',return_crystal_feas=True)['crystal_fea'];p=head(z).view(-1)
                loss=torch.nn.functional.mse_loss(p,torch.tensor(y[b]/100,dtype=torch.float32));opt.zero_grad();loss.backward();opt.step()
                if start%800==0:
                    atomic_torch_save({'net':net.state_dict(),'head':head.state_dict(),'opt':opt.state_dict(),'epoch':epoch,'batch':start+16},cp);check_budget(output,started)
            atomic_torch_save({'net':net.state_dict(),'head':head.state_dict(),'opt':opt.state_dict(),'epoch':epoch+1,'batch':0},cp);check_budget(output,started);print('fine epoch',key,epoch+1,flush=True)
        return encode(net,[graphs[i] for i in tr]),encode(net,[graphs[i] for i in te])
    try:
        fine,preds=evaluate(X,y,fam,sys,names,embargo=mp,fold_transform=tune,checkpoint_dir=output/'fine-scope-checkpoints');report['fine_tuned_five_epoch_pilot']=fine
        (output/'fine-tuned-predictions.json').write_text(json.dumps(preds))
    except Exception as e: report['fine_tuned']={'status':'blocked','reason':str(e),'partial_metrics':completed_scopes(output/'fine-scope-checkpoints')}
    report['cpu_seconds']=time.process_time()-started;(output/'encoder-report.json').write_text(json.dumps(report,indent=2))
    return report

def jarvis(source,output):
    from chgnet.model.model import CHGNet
    from pymatgen.core import Structure
    torch.set_num_threads(2);warnings.filterwarnings('ignore');m=CHGNet.load(model_name='0.3.0',use_device='cpu');import zipfile
    with zipfile.ZipFile(source) as archive: rows=json.load(archive.open('jarvis_epc_data_figshare_1058.json'))
    rows=[r for r in rows if float(r.get('lamb',0))>0 and float(r.get('wlog',0))>0]
    gs=[];valid=[]
    for i,r in enumerate(rows):
        a=r['atoms'];s=Structure(a['lattice_mat'],a['elements'],a['coords'],coords_are_cartesian=a.get('cartesian',False))
        gs.append(m.graph_converter(s));valid.append(r)
        if i%100==0:print('JARVIS graphs',i,flush=True)
    X=encode(m,gs);np.save(output.with_suffix('.embeddings.npy'),X);systems=['-'.join(sorted(set(r['atoms']['elements']))) for r in valid];res={'rows':len(valid)};target_predictions={}
    for target,key in [('lambda','lamb'),('omega_log','wlog')]:
        y=np.array([float(r[key]) for r in valid]);pred=np.zeros(len(y))
        for fold,(tr,te) in enumerate(GroupKFold(5).split(X,y,systems),1):
            rf=RandomForestRegressor(n_estimators=180,min_samples_leaf=2,random_state=(3026 if target=='lambda' else 4026)+fold,n_jobs=2).fit(X[tr],y[tr] if target=='lambda' else np.log(y[tr]));p=rf.predict(X[te]);pred[te]=p if target=='lambda' else np.exp(p)
        res[target+'_mae']=float(mean_absolute_error(y,pred));target_predictions[target]=pred
    from train_physics_targets import allen_dynes_tc_k
    true_tc=np.array([allen_dynes_tc_k(float(r['lamb']),float(r['wlog'])) for r in valid]);pred_tc=np.array([allen_dynes_tc_k(a,b) for a,b in zip(target_predictions['lambda'],target_predictions['omega_log'])]);res['physics_tc_mae_k']=float(mean_absolute_error(true_tc,pred_tc))
    output.write_text(json.dumps(res,indent=2));print(res)
def scratch(input_path,features,index,output):
    from alignn.models.alignn import ALIGNN,ALIGNNConfig
    from alignn.graphs import Graph
    from jarvis.core.atoms import Atoms
    from pymatgen.core import Structure
    from evaluate_3dsc_structure_spin import _ordered_proxy
    import dgl
    torch.set_num_threads(2);warnings.filterwarnings('ignore');output.mkdir(parents=True,exist_ok=True);started=budget_start(output)
    records,_,_,_=load(input_path,features);paths={r['record_id']:r['cif_path'] for r in csv.DictReader(index.open())}
    gs=[];valid=[];errors=[]
    cache=output/'scratch-graphs.pt'
    if cache.exists():
        state=torch.load(cache,weights_only=False)
        if state['ids']!=[r.record_id for r in records]:raise ValueError('Scratch graph cache cohort changed')
        gs,valid,errors=state['graphs'],state['valid'],state['errors'];print('resumed scratch graphs',len(gs),flush=True)
    else:
        for i,r in enumerate(records):
            try:
                s,_=_ordered_proxy(Structure.from_file(paths[r.record_id]));a=Atoms(lattice_mat=s.lattice.matrix,coords=s.frac_coords,elements=[site.specie.symbol for site in s],cartesian=False)
                g,lg=Graph.atom_dgl_multigraph(a,cutoff=8,max_neighbors=12,atom_features='cgcnn',compute_line_graph=True);gs.append((g,lg,torch.tensor(a.lattice_mat,dtype=torch.float32)));valid.append(i)
            except Exception as e:errors.append({'record_id':r.record_id,'reason':str(e)})
            if i%250==0:print('ALIGNN graphs',i,flush=True)
        atomic_torch_save({'ids':[r.record_id for r in records],'graphs':gs,'valid':valid,'errors':errors},cache)
    records=[records[i] for i in valid];y=np.array([r.tc_kelvin for r in records]);fam=np.array([r.chemical_family for r in records]);sys=np.array([elemental_system_signature(r.formula) for r in records]);mp=np.array([r.mp_id for r in records]);X=np.zeros((len(records),64))
    config=ALIGNNConfig(name='alignn',alignn_layers=2,gcn_layers=2,hidden_features=64,embedding_features=32)
    def batch(ids):
        return [dgl.batch([gs[i][0] for i in ids]),dgl.batch([gs[i][1] for i in ids]),torch.stack([gs[i][2] for i in ids])]
    def fit(tr,te):
        torch.manual_seed(2026);net=ALIGNN(config);opt=torch.optim.Adam(net.parameters(),lr=1e-3);key=fold_key(tr,te,y);cp=output/'epoch-checkpoints'/(key+'.pt');first_epoch=0;first_batch=0
        if cp.exists():
            state=torch.load(cp,weights_only=False);net.load_state_dict(state['net']);opt.load_state_dict(state['opt']);first_epoch=state['epoch'];first_batch=state['batch'];print('resume scratch',key,first_epoch,first_batch,flush=True)
        if cp.exists() and 'rng' in state:torch.set_rng_state(state['rng'])
        net.train()
        for epoch in range(first_epoch,5):
            for start in range(first_batch if epoch==first_epoch else 0,len(tr),16):
                b=tr[start:start+16];p=net(batch(b)).view(-1);loss=torch.nn.functional.mse_loss(p,torch.tensor(y[b]/100,dtype=torch.float32));opt.zero_grad();loss.backward();opt.step()
                if start%800==0:
                    atomic_torch_save({'net':net.state_dict(),'opt':opt.state_dict(),'epoch':epoch,'batch':start+16},cp);check_budget(output,started)
            atomic_torch_save({'net':net.state_dict(),'opt':opt.state_dict(),'epoch':epoch+1,'batch':0},cp);check_budget(output,started);print('scratch epoch',key,epoch+1,flush=True)
        net.eval();captured=[];h=net.fc.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().numpy()))
        def emb(ids):
            captured.clear()
            with torch.no_grad():
                for start in range(0,len(ids),16):net(batch(ids[start:start+16]))
            return np.concatenate(captured)
        xa=emb(tr);xb=emb(te);h.remove();return xa,xb
    output.mkdir(parents=True,exist_ok=True)
    try:
        result,preds=evaluate(X,y,fam,sys,[f'ALIGNN_{i}' for i in range(64)],embargo=mp,fold_transform=fit,checkpoint_dir=output/'scratch-scope-checkpoints');report={'rows':len(records),'five_epoch_scratch_pilot':result,'graph_failures':errors}
        (output/'scratch-predictions.json').write_text(json.dumps(preds))
    except Exception as e:report={'rows':len(records),'status':'blocked','reason':type(e).__name__+': '+str(e),'graph_failures':errors,'partial_metrics':completed_scopes(output/'scratch-scope-checkpoints')}
    report['cpu_seconds']=time.process_time()-started;(output/'scratch-report.json').write_text(json.dumps(report,indent=2));return report

def combined_frozen(input_path,features,embeddings,output):
    records,X,names,_=load(input_path,features);graph=np.load(embeddings)
    ids=json.loads((embeddings.parent/'embedding-record-ids.json').read_text())
    if len(records)!=len(graph) or ids!=[r.record_id for r in records]:raise ValueError('Frozen embedding cohort does not match the physical-feature cohort')
    y=np.array([r.tc_kelvin for r in records]);fam=np.array([r.chemical_family for r in records]);systems=np.array([elemental_system_signature(r.formula) for r in records]);mp=np.array([r.mp_id for r in records])
    result,predictions=evaluate(np.c_[X,graph],y,fam,systems,names+[f'CHGNet_{i}' for i in range(graph.shape[1])],embargo=mp,checkpoint_dir=output.parent/'combined-scope-checkpoints')
    output.write_text(json.dumps({'rows':len(records),'combined_frozen':result},indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('data/training/harmonized_supercon.csv'));p.add_argument('--features',type=Path,default=Path('data/processed/structure_features.csv'));p.add_argument('--index',type=Path,default=Path('data/processed/3dsc-structure-index.csv'));p.add_argument('--output',type=Path,default=Path('data/processed/deep-encoders'));p.add_argument('--jarvis',type=Path);p.add_argument('--scratch',action='store_true');p.add_argument('--combined-frozen',type=Path);a=p.parse_args()
    if a.combined_frozen:combined_frozen(a.input,a.features,a.combined_frozen,a.output)
    elif a.scratch:scratch(a.input,a.features,a.index,a.output)
    elif a.jarvis:jarvis(a.jarvis,a.output)
    else:run(a.input,a.features,a.index,a.output)
