#!/usr/bin/env python3
"""Compare global, family-specific, and multi-task models on identical test folds."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from deep_ablation import load,evaluate,auc,FAMILIES
from train_pressure_aware import elemental_system_signature

def global_training_indices(systems,train,test,label):
    if label=='cross-family':return train
    held=set(systems[test]);return np.array([i for i in range(len(systems)) if systems[i] not in held])

def run(input_path,features,output):
    import torch
    torch.set_num_threads(2);records,X,names,extra=load(input_path,features);X=np.column_stack([X]+[v for v,_ in extra.values()]);names+=sum([n for _,n in extra.values()],[])
    y=np.array([r.tc_kelvin for r in records]);families=np.array([r.chemical_family for r in records]);systems=np.array([elemental_system_signature(r.formula) for r in records]);mp=np.array([r.mp_id for r in records]);all_ids=np.arange(len(y));labels=sorted(set(families));family_y=np.array([labels.index(f) for f in families])
    def global_train(train,test,label):
        return global_training_indices(systems,train,test,label)
    output.mkdir(parents=True,exist_ok=True);report={}
    global_scores,_=evaluate(X,y,families,systems,names,embargo=mp,train_selector=global_train);report['global']=global_scores
    (output/'mechanisms.json').write_text(json.dumps(report,indent=2))
    family_scores,_=evaluate(X,y,families,systems,names,embargo=mp);family_scores['cross-family']={'status':'blocked_unseen_family','rows':len(y),'reason':'No family-specific training examples exist for a wholly held-out family.'};report['family_specific']=family_scores
    (output/'mechanisms.json').write_text(json.dumps(report,indent=2));mt={}
    scopes=[('cross-family',all_ids,families)]+[(f,np.where(families==f)[0],systems[families==f]) for f in FAMILIES if sum(families==f)>=100]
    for label,idx,groups in scopes:
        pred=np.zeros(len(idx));binary=np.zeros((len(idx),2));family_pred=np.zeros(len(idx),dtype=int);mean=np.zeros(len(idx));fold_max=np.zeros(len(idx))
        for fold,(tr,te) in enumerate(GroupKFold(min(5,len(set(groups)))).split(X[idx],y[idx],groups)):
            train,test=global_train(idx[tr],idx[te],label),idx[te];held={mp[i] for i in test if mp[i]};train=np.array([i for i in train if not mp[i] or mp[i] not in held])
            im=SimpleImputer(keep_empty_features=True);sc=StandardScaler();xa=sc.fit_transform(im.fit_transform(X[train]));xb=sc.transform(im.transform(X[test]));mu=y[train].mean();scale=max(y[train].std(),1)
            torch.manual_seed(2026+fold);net=torch.nn.Sequential(torch.nn.Linear(X.shape[1],64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1+len(labels)+2));opt=torch.optim.Adam(net.parameters(),lr=.001)
            tx=torch.tensor(xa,dtype=torch.float32);ty=torch.tensor((y[train]-mu)/scale,dtype=torch.float32);tf=torch.tensor(family_y[train]);tb=torch.tensor(np.column_stack([y[train]>0,y[train]>77]),dtype=torch.float32)
            for epoch in range(50):
                for b in torch.randperm(len(train)).split(128):
                    z=net(tx[b]);loss=torch.nn.functional.mse_loss(z[:,0],ty[b])+.1*torch.nn.functional.cross_entropy(z[:,1:1+len(labels)],tf[b])+.1*torch.nn.functional.binary_cross_entropy_with_logits(z[:,-2:],tb[b]);opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():
                z=net(torch.tensor(xb,dtype=torch.float32));pred[te]=z[:,0].numpy()*scale+mu;binary[te]=torch.sigmoid(z[:,-2:]).numpy();family_pred[te]=z[:,1:1+len(labels)].argmax(1).numpy();mean[te]=mu;fold_max[te]=max(y[train])
                for k,threshold in enumerate((0,77)):
                    if len(set(y[train]>threshold))<2:binary[te,k]=np.nan
        mt[label]={'rows':len(idx),'mae':float(mean_absolute_error(y[idx],pred)),'mean_mae':float(mean_absolute_error(y[idx],mean)),'spearman':float(spearmanr(y[idx],pred).statistic),'auc0':auc(y[idx]>0,binary[:,0]),'auc77':auc(y[idx]>77,binary[:,1]),'family_accuracy':float(np.mean(family_pred==family_y[idx])),'above_training_max_count':int(np.sum(pred>fold_max)),'negative_tc_predictions':int(np.sum(pred<0))}
        print('multi-task',label,mt[label],flush=True)
        report['multitask_50_epoch_pilot']=mt;(output/'mechanisms.json').write_text(json.dumps(report,indent=2))
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('data/training/harmonized_supercon.csv'));p.add_argument('--features',type=Path,default=Path('data/processed/structure_features_checks.csv'));p.add_argument('--output',type=Path,default=Path('data/processed/deep-mechanisms'));a=p.parse_args();run(a.input,a.features,a.output)
