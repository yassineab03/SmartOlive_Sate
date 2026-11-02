"""Année de campagne = début de fenêtre ; ne change aucun score ni seuil."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import nbformat
ROOT=Path(__file__).resolve().parents[1]
notebook_path=ROOT/'02_entrainement_resultats.ipynb'
notebook=nbformat.read(notebook_path,as_version=4)
evaluation=next(c for c in notebook.cells if c.cell_type=='code' and c.source.startswith('def event_metrics('))
evaluation.source=evaluation.source.replace("['parcel_id','region','date','label']", "['parcel_id','region','window_start','date','label']")
evaluation.source=evaluation.source.replace("pred.assign(year=pred.date.dt.year)","pred.assign(year=pred.window_start.dt.year)")
nbformat.write(notebook,notebook_path)
RUN=ROOT/'outputs/experiments/rapport_exploratoire_20261102'
prepared=pd.read_csv(ROOT/'data/processed/model_inputs/observations.csv')
for col in ['window_start','date']:prepared[col]=pd.to_datetime(prepared[col])
lookup=prepared[['parcel_id','date','window_start']]
comparison=pd.read_csv(RUN/'comparison.csv',float_precision='round_trip')
checks=[]
for i,row in comparison.iterrows():
    tag=row.ablation+'_'+row.model
    path=RUN/f'predictions_{row.split}_{tag}.csv'
    before=pd.read_csv(path,float_precision='round_trip')
    before['date']=pd.to_datetime(before.date)
    score_before=before.score.to_numpy().copy();alert_before=before.alert.to_numpy().copy()
    pred=before.drop(columns=['window_start'],errors='ignore').merge(lookup,on=['parcel_id','date'],validate='one_to_one',how='left',sort=False)
    assert pred.window_start.notna().all()
    np.testing.assert_array_equal(score_before,pred.score.to_numpy())
    np.testing.assert_array_equal(alert_before,pred.alert.to_numpy())
    annual=pred.assign(year=pred.window_start.dt.year).groupby('year').alert.agg(['mean','count'])
    annual.to_csv(RUN/f'annual_{row.split}_{tag}.csv')
    comparison.loc[i,'annual_alert_rate_std']=float(annual['mean'].std()) if len(annual)>1 else np.nan
    temporary=path.with_suffix('.part.csv')
    pred.to_csv(temporary,index=False)
    temporary.replace(path)
    checks.append({'model':row.model,'ablation':row.ablation,'campaign_years':list(map(int,annual.index)),
                   'unchanged_scores':True,'unchanged_alerts':True,'n':len(pred)})
comparison.to_csv(RUN/'comparison.csv',index=False)
assert comparison.annual_alert_rate_std.isna().all()
(RUN/'calendar_statistics_audit.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print('Campagne 2024 vérifiée ; scores et alertes inchangés. Stabilité interannuelle non évaluable.')
