"""Replace notebooks 01/02 with the verified monthly Sentinel-2 workflow."""
from pathlib import Path
import nbformat as nb
from enrich_notebook_explanations import enrich

ROOT=Path(__file__).resolve().parents[1]
def md(s):return nb.v4.new_markdown_cell(s)
def code(s):return nb.v4.new_code_cell(s)
def save(name,cells):
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python (SmartOlive)','language':'python'}})
    nb.write(enrich(book,name),ROOT/name)

prepare='''from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path.cwd()
assert (ROOT/'data').exists(), 'Ouvrir le notebook depuis la racine SmartOlive_Sate.'
DATA=ROOT/'data/processed/sentinel_history_2021_2025'
PREP=ROOT/'data/processed/model_inputs_sentinel'
PREP.mkdir(parents=True,exist_ok=True)
status=json.loads((DATA/'status.json').read_text(encoding='utf-8'))
assert status['complete'] and status['errors']==0
verification=json.loads((ROOT/'outputs/sentinel_history_2021_2025/verification.json').read_text(encoding='utf-8'))
source_hash=hashlib.sha256((DATA/'monthly_history.csv').read_bytes()).hexdigest()
assert source_hash==verification['calendar_sha256']
data=pd.read_csv(DATA/'monthly_history.csv')
data['time']=pd.to_datetime(data.month)
data['acquisition_date']=pd.to_datetime(data.acquisition_date,utc=True,format='ISO8601')
assert not data.duplicated(['parcel_id','month']).any()
assert data.loc[~data.accepted,['ndvi','savi','red','nir']].isna().all().all()
assert data.loc[data.accepted,'ndvi'].between(-1,1).all()
display(data.groupby(data.time.dt.year).accepted.agg(['sum','size']))
'''
windows='''# Réserver environ 20 % des groupes spatiaux dans chaque région.
# Les parcelles fortement superposées restent toujours dans le même groupe.
groups=pd.read_csv(ROOT/'outputs/sentinel_history_2021_2025/spatial_groups.csv')
data=data.merge(groups,on='parcel_id',validate='many_to_one')
holdout=set()
for region,part in data.groupby('region'):
    units=sorted(part.spatial_group.unique(),key=lambda s:hashlib.sha256(s.encode()).hexdigest())
    holdout.update(units[:max(1,round(.2*len(units)))])
data['NDVI']=data.ndvi
data['month_sin']=np.sin(2*np.pi*data.time.dt.month/12)
data['month_cos']=np.cos(2*np.pi*data.time.dt.month/12)
features=['NDVI','savi','red','nir','month_sin','month_cos']
months=sorted(data.month.unique())
assert len(months)==60
a,b=int(.7*len(months)),int(.85*len(months))
records=[]; X={s:[] for s in ['train','validation','test','independent']}; y={s:[] for s in X}
for pid,part in data.groupby('parcel_id'):
    part=part.sort_values('time').reset_index(drop=True)
    external=part.spatial_group.iloc[0] in holdout
    for split,start,stop in [('train',0,a),('validation',a,b),('test',b,len(months))]:
        if external and split!='test':continue
        block=part.iloc[start:stop].reset_index(drop=True)
        for i in range(6,len(block)):
            window=block.iloc[i-6:i]; target=block.iloc[i]
            if not block.iloc[i-6:i+1].accepted.all():continue
            values=window[features].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():continue
            actual='independent' if external else split
            X[actual].append(values);y[actual].append(target.ndvi)
            records.append(dict(unit=pid,region=target.region,spatial_group=target.spatial_group,
                split=actual,input_start=str(window.time.iloc[0]),input_end=str(window.time.iloc[-1]),
                target_time=str(target.time),target_acquisition=str(target.acquisition_date),
                last_input_acquisition=str(window.acquisition_date.iloc[-1])))
arrays={}
for split in X:
    assert len(X[split])>1, split
    arrays['X_'+split]=np.asarray(X[split],dtype=np.float32)
    arrays['y_'+split]=np.asarray(y[split],dtype=np.float32)
metadata=pd.DataFrame(records)
for left,right in [('train','validation'),('validation','test')]:
    assert metadata.loc[metadata.split.eq(left),'target_time'].max()<metadata.loc[metadata.split.eq(right),'input_start'].min()
assert set(metadata.loc[metadata.split.eq('independent'),'spatial_group']).isdisjoint(metadata.loc[~metadata.split.eq('independent'),'spatial_group'])
assert pd.to_datetime(metadata.last_input_acquisition,utc=True,format='ISO8601').lt(pd.to_datetime(metadata.target_acquisition,utc=True,format='ISO8601')).all()
task=PREP/'sentinel_ndvi';task.mkdir(exist_ok=True)
np.savez_compressed(task/'samples.npz',**arrays)
metadata.to_csv(task/'samples.csv',index=False)
manifest=dict(name='sentinel_ndvi',kind='regression',description='NDVI du mois suivant à partir de six mois observés consécutifs; horizon réel variable selon les acquisitions.',
    features=features,counts=metadata.groupby('split').size().to_dict(),
    units=metadata.groupby('split').unit.nunique().to_dict(),
    provenance=['https://doi.org/10.6084/m9.figshare.28486022.v2'],source_csv_sha256=source_hash,
    limitations=['2022 très lacunaire; aucune interpolation.',
        'Culture documentée à la visite seulement; continuité 2021–2025 non confirmée.',
        'Aucun label de stress: tâche de régression, classification non évaluable.',
        'Parcelles superposées regroupées; exploitations/bassins inconnus. Les exemples restent corrélés.',
        '70/15/15 des 60 mois avant filtrage et holdout spatial; proportions finales des exemples différentes.',
        'Train janvier 2021–juin 2024; validation juillet 2024–mars 2025; test avril–décembre 2025.',
        'Groupes spatiaux réservés absents du train/validation; évaluation sur les dates du test seulement.'],
    files_sha256={f:hashlib.sha256((task/f).read_bytes()).hexdigest() for f in ['samples.npz','samples.csv']})
(task/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
(PREP/'suite_manifest.json').write_text(json.dumps([manifest],indent=2,ensure_ascii=False),encoding='utf-8')
display(metadata.groupby('split').agg(fenetres=('unit','size'),parcelles=('unit','nunique')))
print('Préparation terminée. Normalisation ajustée exclusivement sur train dans le notebook 02.')
'''
save('01_nettoyage_pretraitement.ipynb',[
md('# 01 — Préparation du nouvel historique Sentinel-2\n\nSource unique : historique vérifié 2021–2025, 563 polygones éligibles. Prévision du NDVI du mois suivant, six mois d’entrée. Ce notebook remplace l’ancien pilote et ne télécharge rien.'),
code(prepare),md('## Qualité et fenêtres temporelles\n\nPas d’interpolation ni de labels artificiels. Fenêtres entièrement dans chaque bloc. Les polygones fortement superposés sont regroupés pour le holdout spatial. Les statistiques d’entraînement et les normalisations seront ajustées sur train uniquement.'),code(windows),
code("data.groupby('time').accepted.mean().plot(figsize=(12,3),title='Couverture mensuelle après QA',ylabel='Fraction valide'); plt.show()"),
md('Les entrées du notebook 02 sont dans `data/processed/model_inputs_sentinel/`. Les résultats historiques du pilote sont conservés séparément et ne sont plus chargés.')])

engine=(ROOT/'tools/run_multidataset_benchmark.py').read_text(encoding='utf-8')
engine=engine.replace("ROOT=Path(__file__).resolve().parents[1]","ROOT=Path.cwd()")
engine=engine.replace("data/processed/multidataset","data/processed/model_inputs_sentinel")
engine=engine.replace("args=parser.parse_args()","args=parser.parse_args(['--run', RUN_NAME, '--tasks', 'sentinel_ndvi'])")
engine=engine.replace("if __name__=='__main__':main()", "")
# Remove the generic multi-source report interpretation: this is a single satellite task.
start=engine.index("    body+=['<h2>Interprétation scientifique pour OliveSmart</h2>'")
end=engine.index("    (out/'rapport.html').write_text",start)
engine=engine[:start]+'''    body+=['<h2>Interprétation OliveSmart</h2><p>Le modèle est sélectionné uniquement par RMSE de validation, baseline comprise. Comparer ensuite son gain sur le test temporel et les groupes spatiaux réservés. Un seul seed et une petite grille ne démontrent pas une supériorité statistique. Le NDVI ne constitue pas une vérité terrain de stress hydrique.</p>']
'''+engine[end:]
save('02_entrainement_resultats.ipynb',[
md('# 02 — Comparaison des modèles sur le nouvel historique Sentinel-2\n\nExécuter 01 avant 02. Baseline de persistance, Random Forest, XGBoost, LightGBM, LSTM, GRU et CNN-LSTM sur les mêmes exemples. DL : budgets 50/100/150, Early Stopping, deux configurations par budget. ML : trois configurations sélectionnées sur validation. Métriques MAE/RMSE/R², temps, courbes de loss, observé/prédit et écarts train-validation.\n\nAucun label de stress n’est disponible : Accuracy/F1/ROC ne sont pas calculables scientifiquement sur cette cible continue.'),
code("from datetime import datetime\nRUN_TRAINING=False  # True pour lancer la comparaison complète.\nRUN_NAME='sentinel_2021_2025_'+datetime.now().strftime('%Y%m%d_%H%M%S')"),
md('## Implémentation des modèles et de l’évaluation\n\nLe code scientifique est inclus ci-dessous. La normalisation est ajustée sur train uniquement. Les tests temporel et spatial ne servent pas à optimiser les hyperparamètres. Les temps mesurent le CPU ; l’inférence est la médiane de cinq passages, hors prétraitement.'),
code(engine),code("prep=ROOT/'data/processed/model_inputs_sentinel'\nassert (prep/'suite_manifest.json').exists(), 'Exécuter le notebook 01.'\nmanifest=json.loads((prep/'suite_manifest.json').read_text(encoding='utf-8'))[0]\nfor f,h in manifest['files_sha256'].items():\n    assert hashlib.sha256((prep/'sentinel_ndvi'/f).read_bytes()).hexdigest()==h\nprint('Entrées vérifiées :',manifest['counts'])\nprint('Entraînement demandé :',RUN_TRAINING)\nif RUN_TRAINING:\n    main()\nelse:\n    print('Prêt : mettre RUN_TRAINING=True puis exécuter les cellules pour entraîner.')"),
code("if RUN_TRAINING:\n    from IPython.display import display, HTML, Image\n    result=ROOT/'outputs/benchmarks'/RUN_NAME\n    comparison=pd.read_csv(result/'comparison.csv')\n    display(comparison[comparison.split.isin(['test','independent'])])\n    display(HTML((result/'rapport.html').read_text(encoding='utf-8')))\n    for path in sorted((result/'sentinel_ndvi').glob('*learning.png')):\n        display(Image(filename=str(path)))\n    print('Rapport et modèles sauvegardés :',result)")])
print('Replaced notebooks 01 and 02 with the current satellite workflow.')
