# OliveSmart — OLV-4

## Import des oliveraies documentées

La référence terrain marocaine et un pilote sur six parcelles sont maintenant disponibles.
Voir [le guide d'import](IMPORT_DONNEES.md). Le notebook principal `analyse_comparative.ipynb`
inclut le code complet du nouvel import ; `import_donnees_oliveraies.ipynb` fournit le
même bloc en version autonome pour la collecte seule.
Le traitement utilise l'environnement `.venv` de ce dossier. L'ancienne zone exploratoire
et ses ZIP restent séparés. Les effectifs vérifiés et les éventuelles lacunes sont dans
`outputs/pilot/import_status.json` après consolidation. Cet import n'entraîne pas les modèles.

Analyse satellitaire multi-temporelle **par parcelle**. Ce dossier contient le sujet satellitaire OLV-4. Le precedent travail OLV-6 reste dans le projet voisin SmartOlive.

## PyCharm

Creer un environnement Python dans ce dossier : py -m venv .venv

1. Choisir l'interpréteur du projet : `SmartOlive_Sate/.venv/Scripts/python.exe` (Python 3.11 ou ultérieur, selon disponibilité des dépendances).
2. Dans le terminal à la racine du projet :

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

3. Ouvrir `analyse_comparative.ipynb`, sélectionner cet interpréteur comme kernel et utiliser **Run All**. Si l'édition de PyCharm ne propose pas les notebooks, installer JupyterLab dans cet environnement et ouvrir le même fichier avec JupyterLab.
4. Le notebook s'exécute sans données : il affiche les blocages et teste les fonctions, sans fabriquer de résultats de modèles.

## Importer les données demandées

### Premier test local sans Earth Engine

Une extraction locale réelle a été effectuée sur le ZIP Sentinel-2A du **1er novembre 2024** et sur `data/raw/maroc/zone_test.geojson`. Cette zone est fournie par l'utilisateur ; ses limites parcellaires et sa culture restent non confirmées. Elle est distincte de la première localisation explorée.

La section **Premier traitement local** du notebook lit directement les ZIP, applique les offsets XML et le masque SCL, aligne B08 de 10 m sur la grille 20 m par moyenne, puis calcule les quatre indices. Elle fonctionne sans Earth Engine. Installer aussi `rasterio` (inclus dans requirements.txt).

Résultats : `data/processed/local_zone_indices.csv`, GeoTIFF à quatre bandes dans `data/processed/`, cartes dans `outputs/`. Ces résultats restent distincts des données d'entraînement : une acquisition ne constitue pas une série temporelle. Ajouter des acquisitions L2A couvrant la même zone permet de compléter ce tableau en relançant le notebook. Il ne faut pas encore entraîner les modèles avec ce seul test. Les modèles nécessitent des parcelles et groupes confirmés, plusieurs campagnes et des splits valides.

Les premiers essais historiques ont utilisé l'interpréteur du projet voisin. Le nouvel import utilise l'environnement `.venv` propre à SmartOlive_Sate ; aucun code du projet voisin n'est importé.

Les URL officielles, usages, unités et limites figurent dans le notebook. Les pages de catalogue sont conservées dans `data/metadata`, distinctes des données scientifiques. Aucun dataset OLV-6 n'est utilisé.

Pour l'extraction réelle : fournir `data/raw/maroc/parcelles.geojson` (FeatureCollection WGS84, polygones réels, propriétés `parcel_id`, `farm_id`, `basin`), activer un projet Earth Engine, puis :

```powershell
.venv/Scripts/earthengine.exe authenticate
$env:EE_PROJECT='identifiant-de-votre-projet'
```

Fixer dates, bassin exclu et qualité dans la première cellule. Passer `RUN_EXTRACTION=True`. Le notebook télécharge par lots annuels les tables Sentinel-2, Landsat 8/9 Collection 2 et ERA5-Land dans `data/raw/<source>`. Les requêtes terminées sont réutilisées si leur SHA-256 est inchangé. Un manifeste conserve source, version, configuration, taille, URL officielle, hash local et contrôle. Le volume réel dépend des parcelles. Copernicus Data Space est l'accès alternatif aux mêmes scènes Sentinel ; il n'est pas additionné comme un autre dataset. Ce notebook utilise la voie Earth Engine prescrite par le protocole, pas un second import redondant via Copernicus.

La collecte Maroc n'a pas d'URL publique dans le sujet. Les CSV vides fournis sont des schémas à compléter, pas des observations. Ne pas inventer des polygones ou des labels. Facultativement, ajouter aux CSV Sentinel `label` (0/1 confirmé, sinon vide) et `label_available_at` ; fournir `events.csv` et des périodes de surveillance exhaustive pour les indicateurs événementiels. Dates ISO, dates de fin inclusives dans surveillance et événements. `confirmed` vaut 1 pour un événement terrain confirmé. Les visites restent une source d'audit : elles ne sont pas automatiquement converties en labels négatifs.

## Expérience

Budget TPE : 12 essais par combinaison méthode/ablation, fenêtres 1/3/6, autoencoder AdamW. Validation temporelle, saisonnalité train uniquement, test futur commun à tous les modèles ; test spatial supplémentaire si un bassin est réservé. Sans ce bassin, la robustesse spatiale reste non évaluée. Sans terrain, précision/rappel/F1 et délai restent indisponibles. Les courbes et modèles ne sont écrits qu'après un entraînement réel.

Le fichier `outputs/locked_protocol.json` interdit de réutiliser silencieusement un test pour ajuster les choix. Pour une nouvelle expérience, changer `OUT` vers un dossier neuf et conserver les résultats précédents. Les scores non supervisés ne démontrent pas une fiabilité agronomique.

## Reproduction

`requirements.txt` définit les dépendances ; `requirements-lock.txt` photographie l'environnement utilisé pour la vérification, et `data/metadata/environment.json` en conserve les versions. L'API Earth Engine est nécessaire seulement pour l'extraction.

```powershell
.venv/Scripts/python.exe tools/execute_notebook.py
```

Tous les traitements scientifiques sont visibles dans le notebook. `tools/build_notebook.py` le reconstruit (écrase les sorties du notebook) ; `execute_notebook.py` l'exécute et sauvegarde les sorties. Les tests intégrés vérifient les métriques, les frontières des fenêtres et les trous temporels. Les assertions sur les données réelles vérifient doublons, groupes, plages et dates.
