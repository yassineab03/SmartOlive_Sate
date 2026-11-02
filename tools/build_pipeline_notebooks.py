"""Construit deux notebooks autonomes ; tout traitement est visible dans leurs cellules."""
from pathlib import Path
import json
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'archives/notebooks'
def old_path(name):
    path=ARCHIVE/name
    return path if path.exists() else ROOT/name
original=nb.read(old_path('analyse_comparative.ipynb'),as_version=4)
imports=nb.read(old_path('import_donnees_oliveraies.ipynb'),as_version=4)
def old_code(prefix):return next(c.source for c in original.cells if c.cell_type=='code' and c.source.startswith(prefix))
def old_md(prefix):return next(c.source for c in original.cells if c.cell_type=='markdown' and c.source.startswith(prefix))
def markdown(text):return nb.v4.new_markdown_cell(text.strip())
def code(text):return nb.v4.new_code_cell(text.strip())
metadata={'kernelspec':{'display_name':'Python (SmartOlive_Sate)','language':'python','name':'python3'}}

pre=[markdown('''# 01 - Nettoyage et prétraitement des données

**À exécuter en premier dans PyCharm, avec le kernel `.venv` du projet.** Ce notebook
regroupe l'import facultatif, les contrôles qualité, la préparation des séries et leur
export. Aucun modèle d'anomalie n'est entraîné ici. La régression saisonnière et les
échelles de normalisation sont des transformations, ajustées sur train uniquement.

Les données du pilote concernent six parcelles avec présence d'oliviers documentée
aux dates de visite. Les événements de stress, l'historique de culture complet et les
identifiants exploitation/bassin ne sont pas inventés. Les lacunes 2022 restent visibles.

**Sortie pour le notebook 02 :** `data/processed/model_inputs/` avec observations,
splits, résidus normalisés, transformations et manifeste d'intégrité. Les données
brutes et les exports du pilote restent conservés. Le notebook 02 repart d'un kernel neuf.
''')]
# Réutiliser le code intégral d'import, en supprimant ses sorties et sa conclusion.
for cell in imports.cells:
    if cell.cell_type=='markdown' and cell.source.startswith('## Limites avant'):continue
    copied=nb.v4.new_markdown_cell(cell.source) if cell.cell_type=='markdown' else nb.v4.new_code_cell(cell.source)
    if copied.cell_type=='markdown' and copied.source.startswith('# OliveSmart'):
        copied.source=copied.source.replace('# OliveSmart - import de données sur des oliveraies documentées','## Sources et import facultatif',1)
    pre.append(copied)
pre.append(markdown('''## 6. Nettoyer et définir les séparations avant apprentissage
Le calendrier conserve les fenêtres sans acquisition et les rejets QA. Les indices
rejetés restent manquants ; aucune interpolation ou moyenne utilisant le futur.
La météo provient de points régionaux ERA5-Land, pas d'une station dans la parcelle.
Les labels de stress sont inconnus : `label=NaN`, jamais `0` par défaut.

Protocole exploratoire proposé : train 2021-2022, validation 2023, test 2024.
Une fenêtre appartient à un split seulement si tout son intervalle y appartient.
Les dates de disponibilité des mesures sont les fins exclusives des fenêtres.
Le test n'est pas utilisé pour ajuster les transformations.

Option `heldout_region` : mettre une région entière à l'écart de train/validation,
puis évaluer uniquement sa période de test. Cette option ne représente ni un bassin
hydrologique certifié ni une vérification des groupes d'exploitation absents.
'''))
pre.append(code('''import hashlib, importlib.metadata, sys, platform
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PREP=ROOT/'data/processed/model_inputs'
PREP.mkdir(parents=True,exist_ok=True)
PREP_REPORT=ROOT/'outputs/preparation'; PREP_REPORT.mkdir(parents=True,exist_ok=True)
SEED=42
CFG=dict(start='2021-01-01',train_end='2023-01-01',val_end='2024-01-01',end='2025-01-01',
         heldout_region=None,min_valid_fraction=.6,max_gap_days=15,step_days=10,
         trials=12,target_alert_rate=.05,event_tolerance_days=10)
FEATURES=['ndvi','ndre','ndwi','savi']
FEATURE_SETS={'ndvi':['ndvi'],'indices':FEATURES,
              'meteo':FEATURES+['temperature_c','soil_moisture_m3m3']}
source_path=ROOT/'data/processed/pilot/sentinel2_10day_with_weather.csv'
data=pd.read_csv(source_path)
for col in ['window_start','date','acquisition_date']:
    data[col]=pd.to_datetime(data[col],utc=True).dt.tz_localize(None)
required=['parcel_id','region','window_start','date','acquisition_date','accepted','qa_valid_fraction']+FEATURE_SETS['meteo']
assert set(required)<=set(data)
assert not data.duplicated(['parcel_id','date']).any()
assert data.groupby('parcel_id').region.nunique().eq(1).all()
assert data.accepted.dtype==bool
assert (data.date>data.window_start).all()
assert data.loc[data.acquisition_date.notna(),'acquisition_date'].lt(data.loc[data.acquisition_date.notna(),'date']).all()
data=data.sort_values(['parcel_id','date']).reset_index(drop=True)
valid=data.accepted & data.qa_valid_fraction.ge(CFG['min_valid_fraction'])
for col in ['ndvi','ndre','ndwi']: valid &= data[col].between(-1,1)
valid &= np.isfinite(data[FEATURES]).all(axis=1)
data.loc[~valid,FEATURES]=np.nan
data['usable']=valid
data['label']=np.nan
# Ne pas mélanger présence d'oliviers et normalité agronomique.
data['split']='excluded'
for name,begin,end in [('train',CFG['start'],CFG['train_end']),
                       ('val',CFG['train_end'],CFG['val_end']),('test',CFG['val_end'],CFG['end'])]:
    mask=data.window_start.ge(begin)&data.date.le(end)
    data.loc[mask,'split']=name
if CFG['heldout_region'] is not None:
    assert CFG['heldout_region'] in set(data.region)
    held=data.region.eq(CFG['heldout_region'])
    data.loc[held,'split']='excluded'
    data.loc[held & data.window_start.ge(CFG['val_end']) & data.date.le(CFG['end']),'split']='spatial_test'
assert {'train','val','test'}<=set(data.split)
assert data.loc[data.split.eq('train'),'date'].max()<=pd.Timestamp(CFG['train_end'])
assert data.loc[data.split.eq('val'),'window_start'].min()>=pd.Timestamp(CFG['train_end'])
assert data.loc[data.split.eq('test'),'window_start'].min()>=pd.Timestamp(CFG['val_end'])
coverage=data.groupby(['split','region']).agg(calendar_rows=('parcel_id','size'),usable_rows=('usable','sum'),
                                            first=('window_start','min'),last=('date','max'))
display(coverage)
coverage.to_csv(PREP_REPORT/'coverage_by_split.csv')
print('Labels de stress confirmés :',int(data.label.notna().sum()))
'''))
prep_functions=old_code('def harmonic(')
pre.append(markdown('''## 7. Saison et normalisation : ajustement sur train uniquement
Régression harmonique annuelle par parcelle lorsque le train couvre au moins un an
et contient suffisamment de mesures ; repli global pour une parcelle inconnue.
StandardScaler sur les résidus train uniquement, pour chacune des trois ablations.
Une modification du test ne doit changer ni les paramètres saisonniers ni l'échelle.
Les points non étiquetés peuvent contenir des anomalies : ce n'est pas une base
certifiée saine. L'entraînement futur restera non supervisé et exploratoire.
'''))
pre.append(code(prep_functions[:prep_functions.index('def windows(')]))
pre.append(code('''residuals={}; transformations={}
for variant,features in FEATURE_SETS.items():
    values,season,scaler=season_fit(data,features)
    residuals[variant]=values
    transformations[variant]={'features':features,'season':season,'scaler':scaler}
    finite=np.isfinite(values).all(axis=1)
    display(data.assign(complete=finite).groupby('split').complete.sum().to_frame(variant))
    # Test de fuite : perturbation du test uniquement, puis comparaison des paramètres.
    perturbed=data.copy()
    perturbed.loc[perturbed.split.isin(['test','spatial_test']),features]=1000.
    _,check_season,check_scaler=season_fit(perturbed,features)
    np.testing.assert_allclose(scaler.mean_,check_scaler.mean_)
    np.testing.assert_allclose(scaler.scale_,check_scaler.scale_)
    assert set(season)==set(check_season)
    for pid in season:
        np.testing.assert_allclose(season[pid].coef_,check_season[pid].coef_)
        np.testing.assert_allclose(season[pid].intercept_,check_season[pid].intercept_)
print('Contrôle : transformations indépendantes des valeurs du test, pour toutes les ablations.')
'''))
pre.append(markdown('''## 8. Exporter le contrat d'entrée du notebook 02
Les résidus conservent exactement l'ordre des observations exportées. Les fenêtres
causales seront construites dans le notebook 02, séparément dans chaque split.
Les empreintes empêchent un chargement silencieux de fichiers modifiés.
Les fichiers joblib proviennent de ce notebook local ; ne pas les remplacer par
des modèles téléchargés sans provenance. Aucune évaluation du test n'est faite ici.
'''))
pre.append(code('''data.to_csv(PREP/'observations.csv',index=False)
data[['parcel_id','region','window_start','date','split','usable']].to_csv(PREP/'splits.csv',index=False)
np.savez_compressed(PREP/'residuals.npz',**residuals)
joblib.dump(transformations,PREP/'transformations.joblib')
files={name:hashlib.sha256((PREP/name).read_bytes()).hexdigest()
       for name in ['observations.csv','splits.csv','residuals.npz','transformations.joblib']}
manifest={'format_version':1,'config':CFG,'feature_sets':FEATURE_SETS,'seed':SEED,
          'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
          'files_sha256':files,'rows':len(data),'usable_rows':int(data.usable.sum()),
          'known_labels':int(data.label.notna().sum()),
          'libraries':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','joblib']},
          'limitations':['Six parcelles pilotes ; culture confirmée aux visites seulement.',
                         'Gros trous en 2022 ; pas de labels de stress.',
                         'Identifiants ferme/bassin absents ; séparation régionale optionnelle seulement.',
                         'Météo régionale ERA5-Land via Open-Meteo, sans pluie importée.']}
(PREP/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
(PREP_REPORT/'status.json').write_text(json.dumps({'preparation_complete':True,'trained_models':0,
    'rows':len(data),'usable_rows':int(data.usable.sum()),'known_labels':int(data.label.notna().sum())},indent=2))
print('Préparation terminée. Ouvrir 02_entrainement_resultats.ipynb dans un kernel neuf.')
'''))

train=[markdown('''# 02 - Entraînement, comparaison et résultats

**Prérequis : exécuter le notebook 01.** Ce notebook recharge les fichiers préparés,
vérifie leurs empreintes et utilise les transformations déjà ajustées. Il ne télécharge
pas les sources et ne réajuste pas le prétraitement sur validation ou test.

L'entraînement est désactivé par défaut (`RUN_TRAINING=False`) : cette séparation
ne lance pas la comparaison. Après vérification du protocole, mettre le drapeau à
`True` puis exécuter le notebook. Les quatre approches et les trois ablations sont
conservées ; le z-score est calculé sur NDVI uniquement.

Sans labels de stress, le critère de validation est un taux d'alerte cible fixé à l'avance,
pas une performance de détection. Précision, rappel, F1, délai et faux positifs agronomiques
restent non évaluables. Les six parcelles ne représentent pas toutes les oliveraies.
'''),code('''from pathlib import Path
import json, hashlib, random, copy, time, sys, platform, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
from torch import nn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
import optuna
from IPython.display import display

ROOT=Path.cwd()
PREP=ROOT/'data/processed/model_inputs'
assert (PREP/'manifest.json').exists(), 'Exécuter d’abord 01_nettoyage_pretraitement.ipynb'
manifest=json.loads((PREP/'manifest.json').read_text(encoding='utf-8'))
assert manifest['format_version']==1
for name,expected in manifest['files_sha256'].items():
    assert hashlib.sha256((PREP/name).read_bytes()).hexdigest()==expected, 'Fichier modifié : '+name
CFG=manifest['config']; FEATURE_SETS=manifest['feature_sets']; SEED=manifest['seed']
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2)
data=pd.read_csv(PREP/'observations.csv')
for name in ['window_start','date','acquisition_date']:data[name]=pd.to_datetime(data[name])
assert len(data)==manifest['rows'] and not data.duplicated(['parcel_id','date']).any()
residual_file=np.load(PREP/'residuals.npz')
residuals={key:residual_file[key] for key in residual_file.files}
transformations=joblib.load(PREP/'transformations.joblib')
for key,features in FEATURE_SETS.items():assert residuals[key].shape==(len(data),len(features))
RUN_TRAINING=False
RUN_EXPERIMENT=RUN_TRAINING
READY=True
RUN_NAME='comparaison_01'
OUT=ROOT/'outputs/experiments'/RUN_NAME
OUT.mkdir(parents=True,exist_ok=True)
META=OUT
morocco=ROOT/'data/raw/maroc'
print('Entrée vérifiée :',len(data),'fenêtres du calendrier,',int(data.usable.sum()),'observations utilisables.')
print('Entraînement activé :',RUN_TRAINING)
display(data.groupby('split').usable.agg(['size','sum']))
'''),markdown('''## 1. Construire des fenêtres causales
Fenêtres 1, 3 ou 6 observations, sans franchir un split ni une lacune supérieure
à 15 jours. Les observations manquantes ne sont pas imputées. Le test final utilise
l'intersection des exemples disponibles pour les modèles sélectionnés.
'''),code(prep_functions[prep_functions.index('def windows('):])]
train.append(markdown(old_md('## Modèle 1')))
train.append(code(old_code('class Autoencoder(')))
train.append(markdown('''## 2. Optimisation et ablations sur validation
Budget : 12 essais TPE par méthode/ablation. Ablations : NDVI ; quatre indices ;
indices + température + humidité du sol. Le seuil est calibré sur validation uniquement.
Le paramètre contamination d'Isolation Forest est conservé, mais un seuil recalibré sur
le score peut rendre son effet redondant : ne pas interpréter ce paramètre comme un taux
de stress connu. En absence de labels, proximité du taux cible ≠ qualité agronomique.
Les périodes train/validation/test et le choix régional sont définis dans le notebook 01.
'''))
optimization=old_code('winners=[]')
optimization=optimization.replace("ablations={'ndvi':['ndvi'],'indices':FEATURES}\n    if {'temperature_c','precipitation_mm'}<=set(data):\n        ablations['meteo']=FEATURES+['temperature_c','precipitation_mm']",'ablations=FEATURE_SETS')
optimization=optimization.replace('residual,season,scaler=season_fit(data,features)',"residual=residuals[ablation]\n        season=transformations[ablation]['season']\n        scaler=transformations[ablation]['scaler']")
optimization=optimization.replace("else: print('Optimisation non exécutée : séries réelles absentes.')","else: print('Entraînement non lancé : RUN_TRAINING=False. Les entrées sont prêtes.')")
optimization=optimization.replace("if READY and RUN_EXPERIMENT:\n", "if READY and RUN_EXPERIMENT:\n    assert not (OUT/'locked_protocol.json').exists(), 'Expérience déjà verrouillée : choisir un nouveau RUN_NAME.'\n",1)
train.append(code(optimization))
train.append(markdown('''## 3. Évaluation et comparaison finale
Les choix sont verrouillés avant scoring du test. Aucun ajustement après consultation
du test. Les métriques sur vérité terrain restent nulles si aucune annotation confirmée
n'est disponible. Les alertes hors périodes de surveillance ne sont pas des faux positifs
connus. La stabilité interannuelle n'est pas mesurable sur une seule année de test.
Les tableaux descriptifs ne permettent pas de désigner un modèle comme meilleur en détection.
'''))
evaluation=old_code('def event_metrics(')
evaluation=evaluation.replace("['parcel_id','farm_id','basin','date','label']", "['parcel_id','region','window_start','date','label']")
evaluation=evaluation.replace("pred.assign(year=pred.date.dt.year)", "pred.assign(year=pred.window_start.dt.year)")
train.append(code(evaluation))
train.append(markdown('''## 4. Courbes, modèles et sauvegarde
Les modèles, seuils, prédictions et courbes d'apprentissage ne sont écrits qu'après un
entraînement réel. Les résidus saisonniers et les scalers proviennent du notebook 01.
Le chronométrage porte sur le scoring en mémoire, hors téléchargement et prétraitement.
'''))
save=old_code('for w in winners:').split('versions={')[0]
train.append(code(save+'''
status={'input_manifest_sha256':hashlib.sha256((PREP/'manifest.json').read_bytes()).hexdigest(),
        'training_requested':RUN_TRAINING,'trained_models':len(winners),
        'known_labels':int(data.label.notna().sum()),'comparison_rows':len(results),
        'detection_metrics_evaluable':bool(data.label.notna().any()),'seed':SEED}
(OUT/'status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf-8')
versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','torch','optuna','joblib']}
(OUT/'environment.json').write_text(json.dumps({'python':sys.version,'packages':versions},indent=2))
display(status)
if not winners: print('Aucun modèle entraîné ; aucune performance ni prédiction produite.')
'''))
train.append(markdown('''## 5. Contrôles et interprétation
Les petits exemples ci-dessous sont des tests de fonctions, pas des résultats sur les
oliveraies. Les données terrain restent indispensables pour mesurer la fiabilité des alertes.
'''))
train.append(code('''assert metrics(np.array([1.,0.,np.nan]),np.array([2.,0.,9.]),1)['f1']==1
assert metrics(np.array([np.nan]),np.array([2.]),1)['precision'] is None
toy=pd.DataFrame({'parcel_id':['a']*6,'split':['train']*3+['test']*3,
                  'date':pd.date_range('2020-01-01',periods=6,freq='10D')})
wx,wi=windows(toy,np.arange(6,dtype=float).reshape(-1,1),3)
assert wi.tolist()==[2,5] and wx.tolist()==[[0.,1.,2.],[3.,4.,5.]]
if CFG['heldout_region'] is not None:
    a=set(data.loc[data.split.isin(['train','val']),'region'])
    b=set(data.loc[data.split.eq('spatial_test'),'region'])
    assert not a&b
print('Contrôles : métriques indisponibles non remplacées par zéro, fenêtres séparées, option régionale vérifiée.')
'''))
for name,cells in [('01_nettoyage_pretraitement.ipynb',pre),('02_entrainement_resultats.ipynb',train)]:
    nb.write(nb.v4.new_notebook(cells=cells,metadata=metadata),ROOT/name)
    print(name,len(cells),'cellules')
