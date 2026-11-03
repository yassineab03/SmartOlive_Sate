"""Real multi-source olive benchmark; validation-only tuning, locked holdout tests."""
from pathlib import Path
import argparse
import copy
import hashlib
import html
import json
import platform
import time
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
SEED=42

class SequenceNet(nn.Module):
    def __init__(self, family, features, hidden, outputs):
        super().__init__()
        self.conv=nn.Conv1d(features,features,3,padding=1) if family=='CNN-LSTM' else None
        self.rnn=(nn.GRU if family=='GRU' else nn.LSTM)(features,hidden,batch_first=True)
        self.head=nn.Sequential(nn.Dropout(.15),nn.Linear(hidden,outputs))
    def forward(self,x):
        if self.conv is not None:
            x=torch.relu(self.conv(x.transpose(1,2))).transpose(1,2)
        out,_=self.rnn(x)
        return self.head(out[:,-1])

def metrics(y,p,classification):
    if classification:
        labels=np.arange(p.shape[1])
        pred=p.argmax(axis=1)
        precision,recall,f1,_=precision_recall_fscore_support(y,pred,labels=labels,
            average='macro',zero_division=0)
        # One-vs-rest macro AUC; undefined if any class absent in test.
        auc=np.nan
        if len(np.unique(y))==len(labels):
            auc=roc_auc_score(y,p,multi_class='ovr',average='macro',labels=labels)
        return dict(accuracy=accuracy_score(y,pred),precision_macro=precision,
                    recall_macro=recall,f1_macro=f1,roc_auc_macro=auc)
    return dict(mae=mean_absolute_error(y,p),rmse=mean_squared_error(y,p)**.5,
                r2=r2_score(y,p) if len(y)>1 else np.nan)

def objective(y,p,classification):
    return 1-metrics(y,p,True)['f1_macro'] if classification else mean_squared_error(y,p)**.5

def plot_predictions(y,p,classification,path,title):
    if classification:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        ConfusionMatrixDisplay(confusion_matrix(y,p.argmax(1),labels=np.arange(p.shape[1])),
            display_labels=['0%','25%','50%','100%']).plot(ax=axes[0],colorbar=False)
        for k in range(p.shape[1]):
            binary=(y==k).astype(int)
            if len(np.unique(binary))==2:
                fpr,tpr,_=roc_curve(binary,p[:,k])
                axes[1].plot(fpr,tpr,label=['0%','25%','50%','100%'][k])
        axes[1].plot([0,1],[0,1],'k--',alpha=.4)
        axes[1].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC one-vs-rest')
        axes[1].legend()
    else:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        axes[0].scatter(y,p,s=12,alpha=.5)
        lo,hi=min(np.min(y),np.min(p)),max(np.max(y),np.max(p))
        axes[0].plot([lo,hi],[lo,hi],'k--')
        axes[0].set(xlabel='Observed',ylabel='Predicted')
        # Metadata order is unit/time; plot a bounded first-unit segment, not a
        # fictitious continuous aggregate time series across independent units.
        axes[1].plot(y[:80],label='Observed')
        axes[1].plot(p[:80],label='Predicted')
        axes[1].set(xlabel='Ordered test samples (unit then time)',ylabel='Target')
        axes[1].legend()
    fig.suptitle(title,fontsize=10)
    fig.tight_layout()
    fig.savefig(path,dpi=140)
    plt.close(fig)

def train_dl(family,hidden,lr,cap,X,y,classification,classes):
    torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED)
    model=SequenceNet(family,X['train'].shape[2],hidden,classes if classification else 1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    criterion=nn.CrossEntropyLoss() if classification else nn.MSELoss()
    tx={k:torch.tensor(v,dtype=torch.float32) for k,v in X.items() if k in ['train','validation']}
    ty={k:torch.tensor(y[k],dtype=torch.long if classification else torch.float32)
        for k in ['train','validation']}
    best=float('inf'); state=None; wait=0; history=[]; best_epoch=0
    begin=time.perf_counter()
    for epoch in range(1,cap+1):
        model.train()
        for idx in np.array_split(rng.permutation(len(tx['train'])),max(1,int(np.ceil(len(tx['train'])/128)))):
            optimizer.zero_grad()
            out=model(tx['train'][idx])
            loss=criterion(out if classification else out[:,0],ty['train'][idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            tr=model(tx['train']); va=model(tx['validation'])
            trloss=criterion(tr if classification else tr[:,0],ty['train']).item()
            valoss=criterion(va if classification else va[:,0],ty['validation']).item()
            row=dict(epoch=epoch,train_loss=trloss,validation_loss=valoss)
            if classification:
                row.update(train_accuracy=(tr.argmax(1)==ty['train']).float().mean().item(),
                    validation_accuracy=(va.argmax(1)==ty['validation']).float().mean().item())
        history.append(row)
        if valoss < best-1e-4:
            best=valoss; state=copy.deepcopy(model.state_dict()); wait=0; best_epoch=epoch
        else:
            wait+=1
        if wait>=12:
            break
    model.load_state_dict(state)
    model.eval()
    return model,pd.DataFrame(history),best_epoch,time.perf_counter()-begin

def table(df):
    return df.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='multisource_20261103')
    parser.add_argument('--tasks',nargs='*')
    args=parser.parse_args()
    out=ROOT/'outputs/benchmarks'/args.run
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    manifests=json.loads((ROOT/'data/processed/multidataset/suite_manifest.json').read_text(encoding='utf-8'))
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','torch','xgboost','lightgbm']}
    rows=[]; trial_rows=[]; sections=[]
    for manifest in manifests:
        name=manifest['name']
        if args.tasks and name not in args.tasks:
            continue
        dest=out/name
        if (dest/'completed.json').exists():
            rows.extend(json.loads((dest/'completed.json').read_text()))
            continue
        dest.mkdir(exist_ok=True)
        folder=ROOT/'data/processed/multidataset'/name
        for file,checksum in manifest['files_sha256'].items():
            assert hashlib.sha256((folder/file).read_bytes()).hexdigest()==checksum
        print('START',name,flush=True)
        data=np.load(folder/'samples.npz')
        splits=['train','validation','test','independent']
        raw={s:data['X_'+s] for s in splits}
        y={s:data['y_'+s] for s in splits}
        classification=manifest['kind']=='classification'
        classes=len(np.unique(y['train'])) if classification else 1
        scaler=StandardScaler().fit(raw['train'].reshape(-1,raw['train'].shape[-1]))
        X={s:scaler.transform(v.reshape(-1,v.shape[-1])).reshape(v.shape).astype(np.float32) for s,v in raw.items()}
        ys=StandardScaler().fit(y['train'].reshape(-1,1)) if not classification else None
        yscaled={s:ys.transform(v.reshape(-1,1))[:,0].astype(np.float32) if ys else v.astype(int) for s,v in y.items()}
        flat={s:v.reshape(len(v),-1) for s,v in X.items()}
        joblib.dump(dict(features=scaler,target=ys),dest/'transformations.joblib')
        local=[]
        def evaluate(label,predict,fit_seconds,search_seconds,validation_objective,params,extra=None):
            predictions={s:predict(s) for s in splits}
            # Median warm inference on full test batch; excludes data preparation.
            timings=[]
            for _ in range(5):
                begin=time.perf_counter(); predict('test'); timings.append(time.perf_counter()-begin)
            inference=float(np.median(timings))
            for split in splits:
                result=dict(dataset=name,model=label,split=split,n=len(y[split]),
                    fit_seconds=fit_seconds,search_seconds=search_seconds,
                    inference_ms_per_sample=inference/len(y['test'])*1000,
                    validation_objective=validation_objective,params=json.dumps(params),
                    **metrics(y[split],predictions[split],classification))
                if extra: result.update(extra)
                local.append(result)
            meta=pd.read_csv(folder/'samples.csv')
            for split in ['test','independent']:
                frame=meta[meta.split.eq(split)].copy()
                frame['observed']=y[split]
                p=predictions[split]
                frame['predicted']=p.argmax(1) if classification else p
                if classification:
                    for k in range(classes):frame[f'probability_{k}']=p[:,k]
                frame.to_csv(dest/f'{label}_{split}_predictions.csv',index=False)
                plot_predictions(y[split],p,classification,dest/f'{label}_{split}.png',f'{name} | {label} | {split}')
            print(name,label,'val',round(validation_objective,5),flush=True)

        # Baseline uses same origins and same horizon as every model.
        begin=time.perf_counter()
        if classification:
            baseline=DummyClassifier(strategy='prior').fit(flat['train'],y['train'])
            base_predict=lambda s:baseline.predict_proba(flat[s])
        elif name=='olea_yield':
            baseline=DummyRegressor(strategy='mean').fit(flat['train'],y['train'])
            base_predict=lambda s:baseline.predict(flat[s])
        else:
            index=manifest['features'].index({'csic_moisture':'Hd 25 cm','drought_spad':'Average_SPAD','sentinel_ndvi':'NDVI'}[name])
            base_predict=lambda s:raw[s][:,-1,index]
        elapsed=time.perf_counter()-begin
        evaluate('Baseline',base_predict,elapsed,elapsed,objective(y['validation'],base_predict('validation'),classification),{})
        for family in ['RandomForest','XGBoost','LightGBM']:
            best=None; tuning_start=time.perf_counter()
            for depth,rate in [(3,.03),(5,.05),(8,.1)]:
                cls=dict(RandomForest=RandomForestClassifier,XGBoost=XGBClassifier,LightGBM=LGBMClassifier) if classification else dict(RandomForest=RandomForestRegressor,XGBoost=XGBRegressor,LightGBM=LGBMRegressor)
                params=dict(n_estimators=150,max_depth=depth,random_state=SEED,n_jobs=2)
                if family!='RandomForest':params['learning_rate']=rate
                if family=='LightGBM':params.update(verbosity=-1,min_child_samples=5,num_leaves=min(31,2**depth),deterministic=True,force_col_wise=True)
                if family=='XGBoost':params.update(tree_method='hist',verbosity=0)
                model=cls[family](**params)
                begin=time.perf_counter(); model.fit(flat['train'],y['train']); elapsed=time.perf_counter()-begin
                p=model.predict_proba(flat['validation']) if classification else model.predict(flat['validation'])
                score=objective(y['validation'],p,classification)
                trial_rows.append(dict(dataset=name,model=family,params=json.dumps(params),validation_objective=score,fit_seconds=elapsed))
                if best is None or score<best[0]:best=(score,model,params,elapsed)
            search=time.perf_counter()-tuning_start
            score,model,params,elapsed=best
            joblib.dump(model,dest/f'{family}.joblib')
            predict=lambda s,m=model:m.predict_proba(flat[s]) if classification else m.predict(flat[s])
            evaluate(family,predict,elapsed,search,score,params)
        if manifest['kind']!='cross_sectional_regression':
            for family in ['LSTM','GRU','CNN-LSTM']:
                for cap in [50,100,150]:
                    best=None; tuning_start=time.perf_counter()
                    for hidden,lr in [(16,.001),(32,.003)]:
                        model,history,best_epoch,elapsed=train_dl(family,hidden,lr,cap,X,yscaled,classification,classes)
                        def predict(s,m=model):
                            with torch.no_grad():
                                result=m(torch.tensor(X[s],dtype=torch.float32))
                                if classification:return torch.softmax(result,1).numpy()
                                return ys.inverse_transform(result.numpy())[:,0]
                        score=objective(y['validation'],predict('validation'),classification)
                        params=dict(hidden=hidden,lr=lr,epoch_cap=cap,patience=12,min_delta=.0001,seed=SEED)
                        trial_rows.append(dict(dataset=name,model=family,params=json.dumps(params),validation_objective=score,fit_seconds=elapsed,best_epoch=best_epoch,epochs_run=len(history)))
                        if best is None or score<best[0]:best=(score,model,history,best_epoch,elapsed,params)
                    search=time.perf_counter()-tuning_start
                    score,model,history,best_epoch,elapsed,params=best
                    label=f'{family}_{cap}'
                    torch.save(dict(state_dict=model.state_dict(),params=params,features=manifest['features'],outputs=classes),dest/f'{label}.pt')
                    history.to_csv(dest/f'{label}_history.csv',index=False)
                    fig,axes=plt.subplots(1,2 if classification else 1,figsize=(10,4) if classification else (6,4))
                    axes=np.atleast_1d(axes)
                    history.plot(x='epoch',y=['train_loss','validation_loss'],ax=axes[0])
                    axes[0].axvline(best_epoch,color='gray',ls='--'); axes[0].set_title(label+' | loss')
                    if classification:history.plot(x='epoch',y=['train_accuracy','validation_accuracy'],ax=axes[1])
                    fig.tight_layout();fig.savefig(dest/f'{label}_learning.png',dpi=140);plt.close(fig)
                    def predict(s,m=model):
                        with torch.no_grad():
                            result=m(torch.tensor(X[s],dtype=torch.float32))
                            if classification:return torch.softmax(result,1).numpy()
                            return ys.inverse_transform(result.numpy())[:,0]
                    evaluate(label,predict,elapsed,search,score,params,dict(best_epoch=best_epoch,epochs_run=len(history)))
        (dest/'completed.json').write_text(json.dumps(local,indent=2),encoding='utf-8')
        rows.extend(local)
    results=pd.DataFrame(rows)
    results.to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(trial_rows).to_csv(out/'hyperparameter_trials.csv',index=False)
    protocol=dict(seed=SEED,versions=versions,python=platform.python_version(),platform=platform.platform(),
        epoch_caps=[50,100,150],dl_configs=[dict(hidden=16,lr=.001),dict(hidden=32,lr=.003)],
        selection='Minimum validation RMSE or maximum validation macro F1. Test never used in tuning.',
        timing='Training and search wall times; median of five warm test batch predictions per sample; CPU, 2 threads.',
        limitations='Single seed, small tuning grid; no statistical superiority claim. Data tasks cannot be pooled or ranked by raw metrics.',
        datasets=manifests)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf-8')
    # Select the family + epoch budget using validation only, then show both tests.
    selected=[]
    for name,group in results[results.split.eq('validation')].groupby('dataset'):
        winner=group.loc[group.validation_objective.idxmin()]
        selected.append(dict(dataset=name,selected_model=winner.model,validation_objective=winner.validation_objective))
    pd.DataFrame(selected).to_csv(out/'validation_selected_models.csv',index=False)
    body=['<h1>OliveSmart — comparaison multi-datasets réels</h1>',
        '<p>Chaque tâche est évaluée séparément. Les traitements d’irrigation ne sont pas des diagnostics de stress. '
        'Optimisation sur validation uniquement ; baseline incluse dans la sélection. Une seule graine, grille limitée : résultats exploratoires.</p>']
    for manifest in manifests:
        name=manifest['name']; subset=results[results.dataset.eq(name)]
        if subset.empty:continue
        winner=next(x['selected_model'] for x in selected if x['dataset']==name)
        body += [f'<h2>{html.escape(name)}</h2>',f'<p>{html.escape(manifest["description"])}</p>',
                 f'<p>Modèle retenu par validation : <strong>{winner}</strong>.</p>',
                 '<p>Effectifs : '+html.escape(str(manifest['counts']))+'</p>',
                 '<ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in manifest['limitations'])+'</ul>']
        if name=='olea_yield':body.append('<p>LSTM / GRU / CNN-LSTM : NON APPLICABLES — séries répétées insuffisantes. Split transversal par arbres, pas chronologique.</p>')
        cols=['model','split','n','accuracy','precision_macro','recall_macro','f1_macro','roc_auc_macro'] if name=='drought_irrigation' else ['model','split','n','mae','rmse','r2']
        cols+=['fit_seconds','search_seconds','inference_ms_per_sample']
        body.append(table(subset[subset.split.isin(['test','independent'])][cols]))
        train=subset[subset.split.eq('train')].set_index('model')
        val=subset[subset.split.eq('validation')].set_index('model')
        score='f1_macro' if name=='drought_irrigation' else 'rmse'
        gap=(train[[score]].rename(columns={score:'train_'+score}).join(val[[score]].rename(columns={score:'validation_'+score})))
        gap['validation_minus_train']=gap['validation_'+score]-gap['train_'+score]
        gap.to_csv(out/name/'overfitting.csv')
        body+=['<h3>Surapprentissage / changement de saison</h3>',table(gap.reset_index()),
            '<p>Écart train-validation descriptif : il peut refléter surapprentissage et changement de distribution. '
            'Les pertes DL sont calculées en mode évaluation ; la régression est standardisée sur train.</p>']
        for split in ['test','independent']:
            body.append(f'<img src="{name}/{winner}_{split}.png" alt="Observed predictions or confusion and ROC">')
        if (out/name/f'{winner}_learning.png').exists():body.append(f'<img src="{name}/{winner}_learning.png" alt="Training and validation curves">')
        body.append('<p>Sources : '+', '.join(f'<a href="{s}">{s}</a>' for s in manifest['provenance'])+'</p>')
    body+=['<h2>Interprétation scientifique pour OliveSmart</h2>',
        '<p>Aucun meilleur modèle universel ne peut être déduit de cibles différentes. Utiliser le modèle sélectionné '
        'pour chaque tâche, puis vérifier son gain sur la baseline sur le test réservé et les unités exclues. '
        'Une baseline sélectionnée ou un R² négatif signifie que les modèles complexes ne justifient pas leur coût sur cette tâche. '
        'Les régimes d’irrigation de Fès ne valident pas un diagnostic agronomique. '
        'OLEA-PRED fournit le seul test sur un autre verger dans ce benchmark, mais ses effectifs sont très petits '
        'et sa cible diffère de celles du satellite et du CSIC. Il faut davantage de vergers labellisés et plusieurs années '
        'pour choisir scientifiquement un détecteur de stress utilisable en production.</p>']
    (out/'rapport.html').write_text('<!doctype html><html lang="fr"><meta charset="utf-8"><title>OliveSmart multi-datasets</title>'
        '<style>body{font:15px system-ui;max-width:1400px;margin:35px auto;padding:20px;color:#16352b}table{border-collapse:collapse;font-size:12px}th,td{padding:7px;border-bottom:1px solid #ddd}img{max-width:100%;display:block;margin:25px 0}h2{margin-top:50px}</style>'
        +''.join(body)+'</html>',encoding='utf-8')
    print('DONE',out,flush=True)

if __name__=='__main__':main()
