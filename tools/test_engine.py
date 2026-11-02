"""Tests de fonctionnement sur valeurs artificielles, sans score agronomique."""
from pathlib import Path
import tempfile
import nbformat
import numpy as np
import pandas as pd
import torch
from torch import nn
import random, copy
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

root=Path(__file__).resolve().parents[1]
nb=nbformat.read(root/'analyse_comparative.ipynb',as_version=4)
scope=globals()
SEED=42
CFG={'max_gap_days':15,'target_alert_rate':.05,'event_tolerance_days':10}
for prefix in ['def harmonic(', 'class Autoencoder(', 'def event_metrics(']:
    source=next(c.source for c in nb.cells if c.cell_type=='code' and c.source.startswith(prefix))
    # La cellule de comparaison inclut aussi l'exécution conditionnelle ; on ne charge que la fonction.
    if prefix=='def event_metrics(': source=source.split('\nresults=[]')[0]
    exec(compile(source,str(root/'analyse_comparative.ipynb'),'exec'),scope)

rng=np.random.default_rng(SEED)
dates=pd.date_range('2019-01-01',periods=150,freq='10D')
df=pd.DataFrame({'parcel_id':['a']*150,'date':dates,'label':np.nan,
                 'split':['train']*80+['val']*35+['test']*35,
                 'ndvi':.5+.15*np.sin(np.arange(150)*2*np.pi/36.5)+rng.normal(0,.01,150)})
z,season,scaler=season_fit(df,['ndvi'])
changed=df.copy(); changed.loc[changed.split.eq('test'),'ndvi']=1000
_,season2,scaler2=season_fit(changed,['ndvi'])
np.testing.assert_allclose(scaler.mean_,scaler2.mean_)
np.testing.assert_allclose(season['global'].coef_,season2['global'].coef_)
x,ix=windows(df,z,3)
subset=df.loc[ix]; train=x[subset.split.eq('train')]; val=x[subset.split.eq('val')]
torch.set_num_threads(2)
for method,params in [('zscore',{}),('isolation',dict(n_estimators=10,max_samples=1.,contamination=.05)),('svm',dict(gamma=.1,nu=.05)),('autoencoder',dict(latent=2,lr=.001,weight_decay=.0001))]:
    model,history=fit_method(method,train,val,params)
    s=score(method,model,val,1)
    assert s.shape==(len(val),) and np.isfinite(s).all()
    threshold,_,_=threshold_validation(s,np.full(len(s),np.nan))
    assert np.isfinite(threshold)
    if method=='autoencoder': assert history and all(np.isfinite(h['validation']) for h in history)

assert metrics(np.array([1.,0.,np.nan]),np.array([2.,0.,3.]),1)['f1']==1
assert metrics(np.array([np.nan]),np.array([2.]),1)['precision'] is None
with tempfile.TemporaryDirectory(dir=root) as folder:
    morocco=Path(folder)
    pd.DataFrame([dict(event_id='e1',parcel_id='a',onset='2024-01-20',end='2024-01-25',available_at='2024-02-01',confirmed=1)]).to_csv(morocco/'events.csv',index=False)
    pd.DataFrame([dict(parcel_id='a',start='2024-01-01',end='2024-02-29')]).to_csv(morocco/'surveillance.csv',index=False)
    p=pd.DataFrame({'parcel_id':['a']*4,'date':pd.to_datetime(['2024-01-01','2024-01-21','2024-02-20','2024-02-29']),'alert':[False,True,True,False]})
    m=event_metrics(p)
    assert m['event_recall']==1 and m['event_precision']==.5 and m['mean_delay_days']==1
print('OK : 4 modèles, checkpoint AE, normalisation sans fuite test, seuils et appariement événementiel.')
