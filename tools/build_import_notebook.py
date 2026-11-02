"""Assemble les traitements d'import visibles, sans script scientifique caché."""
from pathlib import Path
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(text):cells.append(nb.v4.new_markdown_cell(text))
def code(text):cells.append(nb.v4.new_code_cell(text))
def source(name):
    text=(ROOT/'tools'/name).read_text(encoding='utf-8')
    return text.replace('ROOT=Path(__file__).resolve().parents[1]','ROOT=Path.cwd()')
def guarded(text,flag):return 'if '+flag+':\n'+''.join('    '+line+'\n' for line in text.splitlines())

md('''# OliveSmart - import de données sur des oliveraies documentées

Ce notebook contient le code intégral d'import et d'audit. Il complète le notebook
de comparaison existant, sans mélanger l'ancienne zone de culture non confirmée et
les nouvelles parcelles terrain. **Aucun modèle n'est entraîné ici.**

Référence terrain : Ouassanouan et al., *Crop and irrigation type dataset for Moroccan
agricultural regions*, [DOI version 2](https://doi.org/10.6084/m9.figshare.28486022.v2),
licence CC BY 4.0. Le champ `tree=Olives` constitue une observation publiée de culture,
pas un label de stress et pas une confirmation de la culture sur toutes les années.
La version a été publiée le 14 juillet 2025 : sélection rétrospective des sites.

Les capteurs restent ceux du sujet : Sentinel-2 L2A et Landsat Collection 2 L2.
Pour automatiser sans authentification Earth Engine, nous utilisons des miroirs COG
publics des mêmes produits : [Sentinel-2 C1 / Earth Search](https://registry.opendata.aws/sentinel-2-l2a-cogs/)
et Landsat / Planetary Computer. Ce ne sont pas des nouveaux jeux de données fusionnés.
L'accès météo est Open-Meteo avec le modèle explicitement fixé à ERA5-Land. Ce n'est
pas un export du catalogue Earth Engine DAILY_AGGR. Température et humidité du sol
seulement ; la pluie proposée par cette API est forcée par ERA5 et n'est pas substituée.
''')
code('''from pathlib import Path
from IPython.display import display, Image
import json
import pandas as pd
ROOT=Path.cwd()
assert (ROOT/'requirements.txt').exists(), 'Lancer le notebook dans SmartOlive_Sate'
RUN_DOWNLOAD=False  # True pour rechercher et importer ; reprise depuis les caches vérifiés.
RUN_REFERENCE_AUDIT=True
RUN_CONSOLIDATION=True
print('Répertoire :',ROOT.name)
''')
md('''## 1. Archives originales et intégrité
Import des polygones et des bandes/NDVI publiés pour 2024 (~68 Mo), comparaison
au MD5 fournisseur, SHA-256 local et contrôle des noms avant extraction. Les photos
de 17 Go ne sont pas téléchargées ni présentées comme inspectées. Les CSV publiés
ne comportent pas de fractions de pixels QA ou de facteurs physiques explicites :
bandes conservées brutes, NDVI fourni conservé sans recalcul d'autres indices.
''')
code(guarded(source('import_morocco_reference.py'),'RUN_DOWNLOAD'))
md('''## 2. Audit des cultures et sélection du pilote
Inventaire complet des lignes `tree=Olives`. La géométrie invalide est conservée dans
l'inventaire et exclue du pilote. Les cultures intercalaires sont distinguées.
Les doublons identiques sont dédupliqués ; toutes les lignes de dates contradictoires
sans identifiant de scène sont exclues du tableau dérivé et consignées. Aucun original
n'est modifié. Les séries absentes restent absentes.

Pilote : trois parcelles à El Haouz et trois au Tadla, choisies sans voir les résultats
satellites parmi les oliveraies sans culture intercalaire indiquée, valides et d'au moins
un hectare. Leur voisinage facilite les découpes. Le plus grand champ de chaque région
sert d'ancre. Aucun identifiant de ferme ou de bassin n'est fabriqué : ces valeurs restent
nulles. Les régions du fournisseur ne sont pas des bassins hydrologiques.
''')
code(guarded(source('audit_morocco_reference.py'),'RUN_REFERENCE_AUDIT'))
md('''## 3. Séries satellitaires sur les mêmes polygones
Sentinel-2 C1 2021-2024 : un candidat par fenêtre de 10 jours. Landsat 8/9 Collection 2
2015-2024 : un candidat par mois pour un historique indépendant. Sélection par nuages
de scène avant calcul des indices ; couverture de l'ensemble du groupe vérifiée.
**Échantillonnage, pas composite de toutes les acquisitions.** Les fenêtres sans candidat
ou de mauvaise qualité restent des trous. Une autre scène peut être utile si le meilleur
candidat de scène est localement nuageux : cette reprise locale n'est pas encore implémentée.

Les fenêtres COG sont lues et conservées en NPZ avec grille, SHA-256 et URLs sources
non signées. Les offsets/échelles des assets C1 sont appliqués ; l'ancien miroir S2
aux offsets contradictoires a été exclu. Grille commune à 20 m, moyennes pour bandes
spectrales et plus proche voisin pour QA. SCL 4/5 conservées ; eau et nuages exclus.
Landsat : QA_PIXEL (bits 0-5) et saturation B4/B5/occlusion terrain (QA_RADSAT).
NDVI, NDRE, NDWI vert/NIR et SAVI L=0,5 pour S2 ; NDVI séparé pour Landsat.
''')
sat=source('import_pilot_satellites.py')
sat='\n'.join(line for line in sat.splitlines() if not line.startswith('parser=') and not line.startswith('args=') and not line.startswith("if __name__"))
sat=sat.replace('def session():','from types import SimpleNamespace\nargs=SimpleNamespace(smoke=False, source="all", workers=4)\ndef session():')
code(sat+'\nif RUN_DOWNLOAD:\n    main()\n')
md('''## 4. Météo ERA5-Land, point de grille régional
Les deux points sont les centroïdes des ensembles de parcelles projetés en UTM,
reconvertis en WGS84. Les coordonnées du point de grille effectivement retourné sont
conservées. Modèle `era5_land`, UTC, altitude non corrigée, maille la plus proche.
Variables : température moyenne (°C) et humidité volumique du sol 0-7 cm (m³/m³),
du 1er janvier 2021 au 31 décembre 2024. Pas de mesure locale d'irrigation.
[Documentation des modèles et variables](https://open-meteo.com/en/docs/historical-weather-api).
''')
code(guarded(source('import_pilot_weather.py'),'RUN_DOWNLOAD'))
md('''## 5. Consolidation, contrôles et résultats
Seules les découpes de version de traitement valide et SHA-256 vérifié entrent dans
le tableau final. Les autres essais restent dans les fichiers bruts mais sont exclus.
La date de chaque fenêtre est sa fin exclusive ; l'acquisition et les jours météo
précèdent cette date. Pas d'interpolation, pas de remplissage depuis le futur.
La météo n'est agrégée que si tous les jours sont disponibles. Les valeurs d'indices
ne sont exposées à l'apprentissage que si fraction QA ≥0,6 et plages admissibles.
Les observations rejetées sont conservées dans le journal d'acquisitions.
''')
final=source('finalize_pilot_data.py')
code(guarded(final,'RUN_CONSOLIDATION'))
code('''status=json.loads((ROOT/'outputs/pilot/import_status.json').read_text(encoding='utf-8'))
display(status)
display(pd.read_csv(ROOT/'outputs/pilot/coverage.csv'))
series=pd.read_csv(ROOT/'data/processed/pilot/sentinel2_10day_with_weather.csv')
display(series.groupby(['region','status']).size().rename('observations').reset_index())
display(Image(filename=str(ROOT/'outputs/pilot/pilot_map.png')))
display(Image(filename=str(ROOT/'outputs/pilot/ndvi_series.png')))
''')
md('''## Limites avant entraînement et évaluation
- La présence d'oliviers est documentée aux dates de visite ; l'historique de culture
  2021-2024 n'est pas entièrement confirmé. Il faut vérifier les changements de plantation.
- Les séries publiques 2024 et les découpes calculées ont des méthodes différentes :
  ne pas les empiler comme des observations indépendantes d'une même parcelle-date.
- Les six parcelles forment un **petit pilote**, pas une population représentative.
- Aucun événement de stress confirmé ni calendrier de surveillance exhaustive n'est
  fourni. Précision, rappel, F1, délai et faux positifs agronomiques restent non évaluables.
- Les identifiants d'exploitation et de bassin manquent. Une séparation par région
  peut être une expérience spatiale explicitement nommée, pas une preuve d'indépendance
  par exploitation ou de robustesse entre bassins.
- Un protocole d'apprentissage doit être fixé avant tout choix sur test. Ces imports
  ne déclenchent pas automatiquement le notebook historique de comparaison.
''')
notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (SmartOlive_Sate)','language':'python','name':'python3'}})
legacy_path=ROOT/'archives/notebooks/import_donnees_oliveraies.ipynb'
legacy_path.parent.mkdir(parents=True,exist_ok=True)
nb.write(notebook,legacy_path)
print('Notebook import créé')
