"""Contrôle des fichiers échangés entre les deux notebooks."""
from pathlib import Path
import hashlib
import json
import nbformat
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'data/processed/model_inputs'
manifest=json.loads((PREP/'manifest.json').read_text(encoding='utf-8'))
for name,expected in manifest['files_sha256'].items():
    assert hashlib.sha256((PREP/name).read_bytes()).hexdigest()==expected
data=pd.read_csv(PREP/'observations.csv',parse_dates=['window_start','date'])
splits=pd.read_csv(PREP/'splits.csv',parse_dates=['window_start','date'])
assert len(data)==manifest['rows']==len(splits)
pd.testing.assert_frame_equal(data[['parcel_id','region','window_start','date','split','usable']],splits)
assert not data.duplicated(['parcel_id','date']).any()
assert data.label.isna().all()
cfg=manifest['config']
assert data.loc[data.split.eq('train'),'date'].max()<=pd.Timestamp(cfg['train_end'])
assert data.loc[data.split.eq('val'),'window_start'].min()>=pd.Timestamp(cfg['train_end'])
assert data.loc[data.split.eq('val'),'date'].max()<=pd.Timestamp(cfg['val_end'])
assert data.loc[data.split.eq('test'),'window_start'].min()>=pd.Timestamp(cfg['val_end'])
for variant,array in np.load(PREP/'residuals.npz').items():
    assert array.shape==(len(data),len(manifest['feature_sets'][variant]))
    assert not np.isfinite(array[~data.usable]).all(axis=1).any()
for name in ['01_nettoyage_pretraitement.ipynb','02_entrainement_resultats.ipynb']:
    nb=nbformat.read(ROOT/name,as_version=4)
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
    assert not [o for c in nb.cells for o in c.get('outputs',[]) if o.output_type=='error']
state=json.loads((ROOT/'outputs/experiments/comparaison_01/status.json').read_text(encoding='utf-8'))
assert not state['training_requested'] and state['trained_models']==0
assert state['input_manifest_sha256']==hashlib.sha256((PREP/'manifest.json').read_bytes()).hexdigest()
print('OK : contrat 01 vers 02, ordre des lignes, empreintes, splits, données manquantes et absence d’entraînement.')
