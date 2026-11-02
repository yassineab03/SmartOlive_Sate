"""Rapport PDF fondé uniquement sur l'expérience exploratoire enregistrée."""
from pathlib import Path
import argparse
import json
import math
import html
import hashlib
import pandas as pd
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--run',default='rapport_exploratoire_20261102')
args=parser.parse_args()
RUN_NAME=args.run
assert RUN_NAME and all(c.isalnum() or c in '_-' for c in RUN_NAME), 'Nom d’expérience invalide'
RUN=ROOT/'outputs/experiments'/RUN_NAME
DEST=ROOT/'output/pdf'; DEST.mkdir(parents=True,exist_ok=True)
TMP=ROOT/'tmp/pdfs/resultats'/RUN_NAME; TMP.mkdir(parents=True,exist_ok=True)
source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.iterdir() if p.is_file()}
status=json.loads((RUN/'status.json').read_text(encoding='utf-8'))
assert status['trained_models']==10, 'Attendre la fin de toutes les configurations'
assert status['known_labels']==0 and not status['detection_metrics_evaluable']
protocol=json.loads((RUN/'locked_protocol.json').read_text(encoding='utf-8'))
prep=json.loads((ROOT/'data/processed/model_inputs/manifest.json').read_text(encoding='utf-8'))
assert status['input_manifest_sha256']==hashlib.sha256((ROOT/'data/processed/model_inputs/manifest.json').read_bytes()).hexdigest()
comparison=pd.read_csv(RUN/'comparison.csv')
comparison=comparison[comparison.split.eq('test')].copy()
assert len(comparison)==10
assert comparison.n.nunique()==1 and comparison.n_labelled.eq(0).all()
assert comparison[['precision','recall','f1','ap','auc']].isna().all().all()
assert comparison.annual_alert_rate_std.isna().all(), 'Vérifier le début des fenêtres pour la campagne de test'
coverage=pd.read_csv(ROOT/'outputs/preparation/coverage_by_split.csv')
inventory=pd.read_csv(ROOT/'data/processed/olive_pilot_inventory.csv')
raw_audit=json.loads((ROOT/'data/metadata/morocco_reference_audit.json').read_text(encoding='utf-8'))
pilot_status=json.loads((ROOT/'outputs/pilot/import_status.json').read_text(encoding='utf-8'))
names={'zscore':'Z-score NDVI','isolation':'Isolation Forest','svm':'One-Class SVM','autoencoder':'Autoencoder'}
feature_names={'ndvi':'NDVI','indices':'4 indices','meteo':'Indices + météo'}
order={'zscore':0,'isolation':1,'svm':2,'autoencoder':3}
comparison['order']=comparison.model.map(order)
comparison=comparison.sort_values(['ablation','order'])
choices={(x['name'],x['ablation']):x for x in protocol['choices']}
predictions={}
common=None
for row in comparison.itertuples():
    path=RUN/f'predictions_test_{row.ablation}_{row.model}.csv'
    frame=pd.read_csv(path)
    assert frame.alert.dtype==bool
    keys=set(zip(frame.parcel_id,frame.date))
    if common is None:common=keys
    assert keys==common
    assert len(frame)==row.n
    assert abs(frame.alert.mean()-row.alert_rate)<1e-12
    predictions[row.model,row.ablation]=frame
N=len(common)
trials=pd.concat([pd.read_csv(p).assign(study=p.stem) for p in RUN.glob('trials_*.csv')],ignore_index=True)
assert len(trials)==120

def french(value,digits=3):
    if value is None or pd.isna(value):return 'NE'
    return f'{float(value):.{digits}f}'.replace('.',',')
def pct(value):return french(100*float(value),1)+' %'
def detail(model,variant='ndvi'):
    row=comparison[comparison.model.eq(model)&comparison.ablation.eq(variant)].iloc[0]
    pred=predictions[model,variant]
    return row,pred,choices[model,variant]

assert (TMP/'alert_rates.png').exists() and (TMP/'ae_learning.png').exists(), 'Exécuter plot_report_results.py'

pdfmetrics.registerFont(TTFont('Arial','C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialBold','C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='ArialBold',italic='Arial',boldItalic='ArialBold')
GREEN=colors.HexColor('#3E5B32'); INK=colors.HexColor('#26352B'); PALE=colors.HexColor('#EAF0E3')
styles=getSampleStyleSheet()
for name,font,size,leading,color,after in [
    ('TitleFR','ArialBold',25,29,GREEN,10),('SubtitleFR','Arial',11,15,INK,12),
    ('H1FR','ArialBold',18,23,GREEN,12),('H2FR','ArialBold',11.5,15,GREEN,6),
    ('BodyFR','Arial',10,14,INK,8),('SmallFR','Arial',8.2,11,INK,5),
    ('CellFR','Arial',8.6,11.5,INK,0),('TableHeadFR','ArialBold',8.6,11.5,GREEN,0)]:
    styles.add(ParagraphStyle(name=name,fontName=font,fontSize=size,leading=leading,textColor=color,spaceAfter=after))
story=[]
def para(text,style='BodyFR'):return Paragraph(text,styles[style])
def add(text,style='BodyFR'):story.append(para(text,style))
def heading(text):add(text,'H2FR')
def newpage(title):story.append(PageBreak());add(title,'H1FR')
def table(rows,widths):
    cells=[[para(str(cell),'TableHeadFR' if index==0 else 'CellFR') for cell in row] for index,row in enumerate(rows)]
    t=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LINEBELOW',(0,0),(-1,0),.7,GREEN),('LINEBELOW',(0,1),(-1,-1),.35,colors.HexColor('#D5DDCE'))]))
    story.append(t);story.append(Spacer(1,9))
def picture(path,width=511,max_height=270):
    from PIL import Image as PILImage
    with PILImage.open(path) as image:size=image.size
    height=width*size[1]/size[0]
    if height>max_height:width*=max_height/height;height=max_height
    story.append(Image(str(path),width=width,height=height,hAlign='CENTER'));story.append(Spacer(1,6))
def box(text):
    t=Table([[para(text)]],colWidths=[511]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),('BOX',(0,0),(-1,-1),.5,GREEN),('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
    story.append(t);story.append(Spacer(1,10))

add('OliveSmart','TitleFR')
add(f'Rapport de résultats exploratoires<br/>Analyse satellitaire multi-temporelle des oliveraies marocaines<br/>Expérience : <b>{html.escape(RUN_NAME)}</b><br/>Sujet OLV-4 - Situation au 2 novembre 2026','SubtitleFR')
box(f'<b>Expérience réalisée :</b> quatre approches, dix configurations sélectionnées sur validation et {N} observations parcelle-date communes sur le test 2024. <b>Aucun label de stress confirmé :</b> ce rapport compare les comportements des modèles, pas leur précision agronomique.')
heading('1. Sujet et finalité')
add("OliveSmart est une plateforme de gestion intelligente des oliveraies marocaines. Cette phase IA recherche des baisses ou changements inhabituels de vigueur à l’échelle d’une parcelle, à partir d’observations satellites répétées. Il faut distinguer ces écarts des variations saisonnières normales, des nuages, du sol visible et des pratiques culturales.")
add("L’objectif opérationnel est de signaler les parcelles à vérifier. Une alerte statistique ne permet pas de diagnostiquer automatiquement une maladie, un manque d’eau ou une perte de rendement. L’IA pourra ensuite alimenter les services de la plateforme ; l’intégration microservices n’est pas évaluée dans cette expérience.")
heading('2. Pourquoi combiner statistiques, machine learning et deep learning ?')
table([['Approche','Famille','Pourquoi la comparer ?'],
 ['Z-score NDVI','Statistique','Référence transparente : déficit par rapport au cycle attendu.'],
 ['Isolation Forest','Machine learning','Recherche d’observations multivariées atypiques.'],
 ['One-Class SVM','Machine learning','Apprentissage d’une frontière de normalité.'],
 ['Autoencoder','Deep learning','Reconstruction par réseau de neurones et score d’erreur.']],[112,110,289])
add("Le deep learning est une branche du machine learning. Le projet est donc une <b>comparaison de méthodes de détection d’anomalies</b>, avec une baseline statistique, deux modèles ML et un réseau de neurones. Le mode non supervisé est retenu parce que les anomalies réelles ne sont pas étiquetées ; les données d’apprentissage ne sont pas certifiées saines.")
heading('3. Organisation du travail')
execution_note=("Pour ce rapport, une copie exécutée du notebook 02 est sauvegardée dans un dossier d’expérience distinct ; les notebooks de travail gardent leur configuration initiale."
                if RUN_NAME=='rapport_exploratoire_20261102' else
                f"Ce rapport utilise exclusivement les exports de <b>{html.escape(RUN_NAME)}</b>, l’entraînement exécuté par l’utilisateur dans PyCharm. Aucun entraînement ni réglage n’a été relancé pour générer ce PDF.")
add('<b>Notebook 01 :</b> import, audit, masques, nettoyage, séparations temporelles et normalisation sur train.<br/><b>Notebook 02 :</b> fenêtres causales, optimisation, entraînement, prédictions et comparaison. '+execution_note)

newpage('Données utilisées et périmètre')
table([['Source et type','Données effectivement utilisées','Rôle'],
 ['Terrain marocain publié<br/>Polygones et attributs',f"{raw_audit['olive_labelled_parcels']} parcelles avec champ tree=Olives ; {raw_audit['pure_olive_parcels']} sans culture intercalaire indiquée. Pilote : 6 parcelles valides d’au moins 1 ha.",'Définir des parcelles avec culture observée, pas des labels de stress.'],
 ['Sentinel-2 C1 L2A<br/>Rasters multispectraux', '2021-2024 ; 651 observations parcelle-date, 613 acceptées. Lectures partielles des COG publics, grille 20 m.','Source des quatre indices utilisés par les modèles.'],
 ['Landsat 8/9 Collection 2 L2<br/>Rasters à 30 m','2015-2024 ; 708 observations, 680 acceptées.','Historique indépendant. Non ajouté aux entrées des modèles dans cette expérience.'],
 ['ERA5-Land via Open-Meteo<br/>Séries météo numériques','2021-2024 ; 1 461 jours sur chacun des deux points régionaux, soit 2 922 journées-point.','Température et humidité du sol 0-7 cm. Pas de pluie ERA5-Land importée.']],[120,250,141])
add("Les six parcelles proviennent de la référence terrain version 2, licence CC BY 4.0 [1]. La culture est documentée aux dates de visite ; sa stabilité avant ces dates n’est pas entièrement confirmée. Les séries 2024 fournies par les auteurs servent à l’audit et à une concordance de prétraitement, sans être empilées avec nos extractions comme des exemples indépendants.",'SmallFR')
heading('Indices et sens des mesures')
add('<b>NDVI</b> = (NIR - rouge)/(NIR + rouge) ; indicateur de végétation.<br/><b>NDRE</b> : contraste proche infrarouge/red-edge, information complémentaire sur la végétation.<br/><b>NDWI</b> : ici vert/NIR ; ce n’est pas une mesure directe du déficit hydrique des oliviers.<br/><b>SAVI</b> : correction de l’influence du sol, avec L = 0,5. Les quatre indices sont numériques, pas des images colorisées utilisées comme mesures.')
picture(TMP/'pilot_map.png',max_height=205)
add('Figure 1. Limites des six parcelles issues du terrain publié. Les régions El Haouz et Tadla ne sont pas assimilées à des bassins hydrologiques certifiés. Les identifiants d’exploitation sont absents.','SmallFR')

newpage('Nettoyage et protocole expérimental')
heading('Contrôles et prétraitement')
add("Les archives terrain ont été contrôlées par taille et MD5 fournisseur ; les découpes et fichiers préparés disposent de SHA-256 locaux. Les doublons identiques ont été retirés des séries publiées, les dates contradictoires sans identifiant de scène exclues et consignées, et une géométrie invalide exclue du pilote. Les originaux sont conservés.")
add("Sentinel-2 : échelles/offsets contrôlés, classes SCL 4/5 conservées et moyenne des bandes spectrales sur grille 20 m. Une observation exige au moins 60 % de pixels QA valides et des indices admissibles. Les scènes sont sélectionnées par faible nébulosité globale, un candidat par fenêtre de dix jours : il ne s’agit pas d’un composite de toutes les acquisitions. Les trous ne sont pas interpolés.")
split_rows=[['Ensemble','Période des fenêtres','Calendrier','Mesures utilisables']]
for split,label,period in [('train','Apprentissage','2021-2022'),('val','Validation','2023'),('test','Test futur','2024')]:
    part=coverage[coverage.split.eq(split)]
    split_rows.append([label,period,int(part.calendar_rows.sum()),int(part.usable_rows.sum())])
table(split_rows,[140,155,100,116])
add("La saisonnalité est estimée par régression harmonique annuelle sur train, par parcelle si la couverture le permet, sinon avec repli global. StandardScaler est ajusté sur les résidus train. Les empreintes et l’ordre des observations sont vérifiés lors du passage du notebook 01 au 02. Un test de perturbation des valeurs de test confirme que les transformations train restent identiques.")
add(f"Les fenêtres de longueur 1, 3 ou 6 ne franchissent ni les limites des splits ni une lacune supérieure à 15 jours. Le test final porte sur l’intersection des observations disponibles pour les dix configurations, soit <b>{N} observations communes</b>, sur les 191 mesures initialement utilisables en 2024. Aucune région n’est tenue à l’écart dans ce run : c’est un test temporel, pas une validation spatiale indépendante.")
picture(TMP/'ndvi_series.png',max_height=245)
add('Figure 2. Séries NDVI avant modélisation. Les fortes lacunes de 2022 sont conservées ; elles limitent l’estimation interannuelle et le nombre de fenêtres exploitables.','SmallFR')

newpage('Modèles et optimisation')
heading('Une baseline statistique et trois modèles non supervisés')
for model in names:
    row,pred,choice=detail(model)
    params=choice['params']
    if model=='zscore':
        principle="Le score est l’opposé du dernier résidu NDVI standardisé. Il cible surtout les déficits de végétation par rapport à la saison attendue. C’est une méthode rapide à expliquer ; les hausses inhabituelles ne sont pas son objectif."
        settings=f"Fenêtre sélectionnée : {choice['length']} observation(s)."
    elif model=='isolation':
        principle="Des arbres aléatoires isolent les fenêtres atypiques. Le score multivarié peut réagir à des combinaisons inhabituelles d’indices. La contamination d’apprentissage n’est pas connue ; le seuil final est calibré séparément sur validation."
        settings=f"Cas NDVI : {params['n_estimators']} arbres ; max_samples={params['max_samples']} ; contamination={french(params['contamination'],4)} ; fenêtre={choice['length']}."
    elif model=='svm':
        principle="Le noyau RBF définit une frontière autour des exemples d’apprentissage. Les fenêtres éloignées reçoivent un score d’anomalie élevé. Cette méthode dépend de la normalisation et peut être sensible aux anomalies déjà présentes dans train."
        settings=f"Cas NDVI : gamma={french(params['gamma'],6)} ; nu={french(params['nu'],4)} ; fenêtre={choice['length']}."
    else:
        principle="Un réseau dense reconstruit la fenêtre aplatie : entrée, couche 32/ReLU, latent/ReLU, couche 32/ReLU, sortie. Le score est la MSE de reconstruction. AdamW, mini-lots de 128, maximum 100 époques, patience 10 ; checkpoint choisi sur validation."
        settings=f"Cas NDVI : latent={params['latent']} ; lr={french(params['lr'],6)} ; weight_decay={french(params['weight_decay'],6)} ; fenêtre={choice['length']}."
    heading(names[model])
    add(principle)
    add(settings,'SmallFR')
heading('Budget et critère de sélection')
counts=trials.state.value_counts().to_dict()
add(f"Optuna TPE, graine 42, budget de 12 essais par combinaison méthode/ablation : <b>{len(trials)} essais enregistrés</b> au total ({counts.get('COMPLETE',0)} complets, {counts.get('PRUNED',0)} élagués). Les paramètres et seuils sont choisis sur validation ; aucun choix n’est réajusté après observation du test.")
add("Sans labels, l’objectif est la proximité d’un <b>taux d’alerte de validation cible de 5 %</b>. Il s’agit d’une convention exploratoire fixée à l’avance, pas d’une estimation du vrai taux de stress. Pour l’autoencoder, la validation utilisée pour le checkpoint n’est pas certifiée normale. Trois ablations sont évaluées : NDVI seul ; quatre indices ; indices + température + humidité du sol.")

newpage('Résultats de chaque configuration')
add(f"Les dix configurations ci-dessous sont évaluées sur les <b>mêmes {N} observations de test 2024</b>. Une observation est une parcelle à une date de fenêtre, pas nécessairement un événement indépendant. Les alertes répétées ne sont pas des événements terrain confirmés.")
rows=[['Méthode','Entrées','Fenêtre','Alertes / N','Taux','Scoring (ms)']]
for variant in ['ndvi','indices','meteo']:
    for model in names:
        matches=comparison[comparison.model.eq(model)&comparison.ablation.eq(variant)]
        if matches.empty:continue
        row=matches.iloc[0];choice=choices[model,variant];p=predictions[model,variant]
        rows.append([names[model],feature_names[variant],choice['length'],f'{int(p.alert.sum())} / {len(p)}',pct(row.alert_rate),french(row.inference_seconds*1000,2)])
table(rows,[104,103,48,79,75,102])
add("Le temps est celui d’un appel de scoring sur les fenêtres de test déjà préparées, en mémoire, hors téléchargement, normalisation et construction des fenêtres. Il ne s’agit pas d’un benchmark répété ; de petites différences ne permettent pas un classement fiable du matériel ou des outils.",'SmallFR')
picture(TMP/'alert_rates.png',max_height=205)
add('Figure 3. Comparaison à information identique : NDVI seul. Un taux plus faible ne prouve pas une meilleure détection ; un taux plus élevé ne prouve pas davantage de stress réel.','SmallFR')
box('Les scores bruts et les seuils ont des échelles différentes selon les méthodes. Une erreur de reconstruction faible ou un faible taux d’alertes ne remplace pas une précision ou un F1 mesurés sur vérité terrain.')

newpage('Interprétation et portée des résultats')
for model in names:
    row,pred,choice=detail(model)
    interpretations={
        'zscore':"Aucun déficit ne franchit le seuil sur cette cohorte. Cela ne prouve pas l’absence de stress.",
        'isolation':"Le compte est identique à la baseline pour NDVI, sans prouver l’équivalence des scores. Les ablations changent les alertes, sans démontrer un gain de fiabilité.",
        'svm':"Davantage d’alertes que la baseline sur NDVI seul. Une visite terrain est nécessaire pour savoir si elles sont pertinentes ou fausses.",
        'autoencoder':"Même compte que le SVM sur NDVI, sans garantir les mêmes dates. Les courbes montrent une MSE de validation nettement supérieure à celle de train : la généralisation reste à examiner."}
    add(f"<b>{names[model]} - NDVI seul :</b> {int(pred.alert.sum())} alertes sur {N} observations ({pct(row.alert_rate)}), fenêtre de {choice['length']} observation(s). "+interpretations[model])
picture(TMP/'ae_learning.png',max_height=170)
add("Figure 4. Courbes d’apprentissage de l’autoencoder sélectionné pour chaque ablation. La MSE mesure la reconstruction dans l’espace standardisé. Elle n’est pas comparable à un score d’Isolation Forest et ne prouve pas la capacité de détection de stress.",'SmallFR')
table([['Indicateur','État dans cette expérience'],
 ['Précision, rappel, F1','NE : aucune vérité terrain de stress confirmée.'],
 ['Average precision, ROC-AUC','NE : aucune classe positive/négative confirmée.'],
 ['Délai et faux positifs par parcelle','NE : événements et surveillance exhaustive absents.'],
 ['Stabilité interannuelle du test','NE : le test final porte sur une seule année, 2024.'],
 ['Robustesse spatiale / par exploitation','Non démontrée : mêmes parcelles entre périodes, régions non exclues.']],[175,336])
add("Les changements de score peuvent provenir d’un stress, d’une saison atypique, du sol, de cultures intercalaires non décrites historiquement, de pratiques culturales ou d’un changement de distribution. Ajouter la météo change aussi l’information disponible. Le nombre d’alertes peut donc évoluer sans démontrer une amélioration. La collecte terrain doit trancher.")

newpage('Conclusion : quel est le meilleur outil ?')
box('<b>Aucun meilleur modèle de détection agronomique ne peut être démontré à ce stade.</b> Les quatre approches ont été exécutées, mais la vérité terrain manque pour les classer selon précision, rappel, F1 ou délai.')
heading('Choix provisoire pour continuer le projet')
add("<b>Conserver le z-score NDVI comme référence principale</b> pour expliquer les alertes et faciliter leur vérification terrain. Cette recommandation repose sur sa transparence et sa simplicité, pas sur une supériorité de détection mesurée. Garder Isolation Forest comme candidat multivarié à valider. One-Class SVM et autoencoder restent dans la comparaison ; la complexité du deep learning n’est pas, seule, une preuve de gain.")
add("Pour les outils logiciels, <b>Python dans PyCharm/Jupyter</b> reste l’environnement de travail ; scikit-learn implémente les modèles ML, PyTorch l’autoencoder, Optuna la recherche et Rasterio les traitements raster. L’extraction locale fonctionne sans Earth Engine. Il n’y a pas de comparaison expérimentale entre plateformes logicielles dans ce rapport.")
heading('Ce qu’il faut avant une conclusion opérationnelle')
add("1. Collecter les dates de stress confirmé et des périodes normales réellement surveillées.<br/>2. Vérifier l’historique des plantations et obtenir les groupes exploitation/bassin.<br/>3. Compléter les trous temporels et étendre le nombre de parcelles et de campagnes.<br/>4. Préenregistrer un nouveau protocole, garder un test futur et spatial intact.<br/>5. Évaluer précision, rappel, F1, délai et coûts des alertes, puis choisir la méthode selon un critère métier fixé.")
heading('Livrables et traçabilité')
add(f"<b>01_nettoyage_pretraitement.ipynb</b> : données, splits, transformations.<br/><b>02_entrainement_resultats.ipynb</b> : entraînement et comparaison.<br/><b>outputs/experiments/{html.escape(RUN_NAME)}/</b> : 120 essais, protocole verrouillé, dix configurations, modèles, prédictions, comparaison et environnement.<br/>Les fichiers préparés sont reliés à l’expérience par une empreinte SHA-256. Les exports de l’essai sont lus sans modification pour ce rapport.",'SmallFR')
heading('Références et accès aux données')
refs=[
 ('[1] Ouassanouan et al. - terrain marocain, version 2, CC BY 4.0','https://doi.org/10.6084/m9.figshare.28486022.v2'),
 ('[2] Sentinel-2 COG / collection C1 et documentation radiométrique','https://github.com/Element84/earth-search/blob/main/docs/collections/sentinel-2-l2a.md'),
 ('[3] USGS - Landsat Collection 2','https://www.usgs.gov/landsat-missions/landsat-collection-2'),
 ('[4] Open-Meteo - historique, modèles et variables ERA5-Land','https://open-meteo.com/en/docs/historical-weather-api'),
 ('[5] scikit-learn - détection d’anomalies','https://scikit-learn.org/stable/modules/outlier_detection.html'),
 ('[6] PyTorch - AdamW','https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html')]
for label,url in refs:add(f'<link href="{url}" color="#3E5B32">{html.escape(label)}</link>','SmallFR')

def footer(canvas,doc):
    canvas.setFont('Arial',8);canvas.setFillColor(GREEN)
    canvas.drawString(42,A4[1]-25,'OLIVESMART | OLV-4 | EXPÉRIENCE : '+RUN_NAME)
    canvas.setStrokeColor(colors.HexColor('#D5DDCE'));canvas.line(42,36,A4[0]-42,36)
    canvas.drawString(42,23,'02/11/2026 | Résultats descriptifs, sans validation terrain du stress')
    canvas.drawRightString(A4[0]-42,23,str(doc.page))
target=DEST/('OliveSmart_Rapport_Resultats_IA.pdf' if RUN_NAME=='rapport_exploratoire_20261102' else f'OliveSmart_Rapport_{RUN_NAME}.pdf')
doc=SimpleDocTemplate(str(target),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=45,bottomMargin=49,
                      title='OliveSmart - Rapport de résultats exploratoires IA',author='Équipe OliveSmart')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(target)
assert len(reader.pages)==7,f'Pagination à ajuster : {len(reader.pages)} pages'
assert all(page.extract_text().strip() for page in reader.pages)
assert 'Aucun meilleur modèle' in reader.pages[-1].extract_text()
assert RUN_NAME in reader.pages[0].extract_text()
assert all(hashlib.sha256((RUN/name).read_bytes()).hexdigest()==expected for name,expected in source_hashes.items()), 'Les fichiers de l’essai ont changé pendant la génération'
audit={'pdf':str(target.relative_to(ROOT)),'pages':len(reader.pages),'experiment':str(RUN.relative_to(ROOT)),
       'test_common_observations':N,'configurations':len(comparison),'trials':len(trials),
       'confirmed_stress_labels':0,'source_files_sha256':source_hashes,'source_files_unchanged':True,
       'metrics_not_evaluable':['precision','recall','f1','ap','auc','detection_delay','false_positive_rate'],
       'claim':'Recommandation provisoire du z-score pour transparence ; aucun gagnant agronomique démontré'}
(TMP/'report_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
print(target)
print('Pages :',len(reader.pages),'Test commun :',N)
