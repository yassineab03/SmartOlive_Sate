"""Verify current 01/02 contract, chronological and spatial isolation."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import nbformat
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'data/processed/model_inputs_sentinel/sentinel_ndvi'
m=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
for f,h in m['files_sha256'].items():assert hashlib.sha256((p/f).read_bytes()).hexdigest()==h
d=pd.read_csv(p/'samples.csv')
a=np.load(p/'samples.npz')
for split in ['train','validation','test','independent']:
    assert len(a['X_'+split])==len(a['y_'+split])==d.split.eq(split).sum()
    assert a['X_'+split].shape[1:]==(6,len(m['features']))
    assert np.isfinite(a['X_'+split]).all()
assert set(d.loc[d.split.eq('independent'),'spatial_group']).isdisjoint(d.loc[~d.split.eq('independent'),'spatial_group'])
for left,right in [('train','validation'),('validation','test')]:
    assert d.loc[d.split.eq(left),'target_time'].max()<d.loc[d.split.eq(right),'input_start'].min()
for name in ['01_nettoyage_pretraitement.ipynb','02_entrainement_resultats.ipynb']:
    n=nbformat.read(ROOT/name,as_version=4)
    cells=[c for c in n.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in cells)
    assert not [o for c in cells for o in c.outputs if o.output_type=='error']
    assert 'data/processed/pilot' not in '\n'.join(c.source for c in cells)
print('OK: current satellite inputs, hashes, 6-month windows, chronology, spatial isolation, both notebooks executed.')
