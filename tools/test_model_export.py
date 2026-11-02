"""Régression : export d'un test sans colonne bassin ni vérité terrain inventée."""
from pathlib import Path
import tempfile
import json
import time
import copy
import random
import nbformat
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix
from IPython.display import display

ROOT=Path(__file__).resolve().parents[1]
notebook=nbformat.read(ROOT/'02_entrainement_resultats.ipynb',as_version=4)
definition=next(c.source for c in notebook.cells if c.cell_type=='code' and c.source.startswith('class Autoencoder('))
evaluation=next(c.source for c in notebook.cells if c.cell_type=='code' and c.source.startswith('def event_metrics('))
scope=globals();exec(compile(definition,'model_definitions','exec'),scope)
CFG={'max_gap_days':15,'event_tolerance_days':10}
SEED=42
data=pd.DataFrame({'parcel_id':['test_parcel'],'region':['test_region'],
                   'window_start':[pd.Timestamp('2024-12-22')],
                   'date':[pd.Timestamp('2025-01-01')],'label':[np.nan],'split':['test']})
assert 'basin' not in data
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
    OUT=Path(directory).resolve()
    assert OUT.is_relative_to((ROOT/'tmp').resolve())
    morocco=OUT/'maroc';morocco.mkdir()
    pd.DataFrame(columns=['parcel_id','start','end']).to_csv(morocco/'surveillance.csv',index=False)
    pd.DataFrame(columns=['event_id','parcel_id','onset','end','available_at','confirmed']).to_csv(morocco/'events.csv',index=False)
    winners=[dict(name='zscore',ablation='ndvi',features=['ndvi'],params={},length=1,threshold=0.,criterion='test fonctionnel',
                  model=None,x=np.array([[-1.]],dtype=np.float32),ix=np.array([0]))]
    exec(compile(evaluation,'evaluation_export','exec'),scope)
    result=pd.read_csv(OUT/'comparison.csv')
    pred=pd.read_csv(OUT/'predictions_test_ndvi_zscore.csv')
    assert len(result)==len(pred)==1 and 'region' in pred and 'basin' not in pred
    assert result.precision.isna().all() and result.f1.isna().all()
    assert result.mean_delay_days.isna().all() and result.n_labelled.eq(0).all()
    assert result.annual_alert_rate_std.isna().all()
    annual=pd.read_csv(OUT/'annual_test_ndvi_zscore.csv')
    assert annual.year.tolist()==[2024], 'La fenêtre de fin décembre reste dans la campagne 2024'
print('OK : export régional sans bassin, métriques de stress indisponibles conservées nulles.')
