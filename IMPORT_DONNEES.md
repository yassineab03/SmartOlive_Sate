# Import des oliveraies documentées

La nouvelle référence ne remplace pas silencieusement l'ancienne zone exploratoire. Les cinq ZIP initiaux ont été supprimés à la demande de l'utilisateur ; leurs résultats historiques restent distincts. Les nouvelles données proviennent des parcelles marocaines publiées par Ouassanouan et al. : https://doi.org/10.6084/m9.figshare.28486022.v2 (CC BY 4.0).

## Données originales et audit

- `data/raw/morocco_reference/` : archives originales (polygones et séries 2024) et fichiers extraits ; MD5 fournisseur vérifiés. Les photos de 17 Go ne sont pas téléchargées.
- `data/raw/maroc/oliveraies_reference.geojson` : toutes les lignes `tree=Olives`, cultures intercalaires et anomalies géométriques signalées.
- `data/raw/maroc/oliveraies_pilote.geojson` : six parcelles valides d'au moins un hectare, sans culture intercalaire indiquée, réparties entre El Haouz et Tadla.
- `data/processed/olive_reference_series_2024.csv` : bandes et NDVI publiés ; lignes du même jour contradictoires exclues, doublons identiques supprimés. Ces séries ne sont pas empilées avec nos extractions comme des exemples indépendants.
- `data/metadata/morocco_reference_audit.json` : couverture, exclusions et limites. Le terrain confirme la culture à la visite, pas l'absence de stress ni la culture sur tout l'historique.

## Satellite et météo

Le code utilise les mêmes capteurs que le sujet, avec des accès publics alternatifs permettant des lectures partielles sans Earth Engine :

- **Sentinel-2 C1 L2A** via Earth Search / miroir COG des produits Copernicus : 2021–2024, un candidat choisi par fenêtre de 10 jours sur le taux de nuages de scène. La collection peut avoir des trous, notamment en 2022. Aucun remplissage artificiel.
- **Landsat 8/9 Collection 2 Level-2 USGS** via Planetary Computer : 2015–2024, un candidat par mois. Historique indépendant à 30 m, pas de fusion radiométrique naïve avec Sentinel.
- **ERA5-Land** via Open-Meteo, modèle `era5_land` fixé : 2021–2024, température et humidité du sol 0–7 cm, deux points de grille régionaux. Ce sont des agrégations de cet accès, pas le fichier Earth Engine DAILY_AGGR. Pas de pluie ERA5 ajoutée sous l'étiquette ERA5-Land.

Les découpes brutes sont conservées sous `data/raw/pilot/` avec grille et SHA-256. Les métadonnées conservent les URLs non signées, produits et facteurs radiométriques. L'ancien miroir S2 avait des métadonnées d'offset contradictoires : ses essais sont exclus du manifeste final. Seule la collection C1 vérifiée est retenue. Le test du 4 juin 2024 concorde à moins de 0,01 de NDVI avec la série publiée pour une même parcelle (contrôle de prétraitement, pas performance de détection).

Le candidat le moins nuageux à l'échelle de la scène n'est pas garanti clair sur le champ. Le filtre local rejette les observations insuffisantes. Il s'agit d'un échantillonnage par fenêtre, pas d'un composite de toutes les acquisitions ; la recherche d'un deuxième candidat en cas de rejet local reste à implémenter.

## Reproduire dans PyCharm

Utiliser l'environnement `.venv` de **SmartOlive_Sate**. Les traitements d'import, nettoyage et prétraitement sont visibles dans `01_nettoyage_pretraitement.ipynb`. Par défaut, `RUN_DOWNLOAD=False` réutilise les données présentes. `02_entrainement_resultats.ipynb` charge les fichiers préparés dans un kernel neuf et conserve l'entraînement désactivé tant que `RUN_TRAINING=False`.

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe tools/import_morocco_reference.py
.venv/Scripts/python.exe tools/audit_morocco_reference.py
.venv/Scripts/python.exe tools/import_pilot_weather.py
.venv/Scripts/python.exe tools/import_pilot_satellites.py --workers 4
.venv/Scripts/python.exe tools/finalize_pilot_data.py
.venv/Scripts/python.exe tools/test_import_integrity.py
.venv/Scripts/python.exe tools/build_pipeline_notebooks.py
.venv/Scripts/python.exe tools/execute_pipeline_notebooks.py
```

La reprise des découpes satellites réutilise uniquement les fichiers de la bonne version de traitement et d'empreinte vérifiée. Les périodes sans données et les échecs réseau sont consignés.

## Résultats et travail restant

Les effectifs réellement importés figurent dans `outputs/pilot/import_status.json` et `outputs/pilot/coverage.csv`. Le tableau avec calendrier, rejets et météo est `data/processed/pilot/sentinel2_10day_with_weather.csv`. Les moyennes NDVI n'indiquent pas, seules, une anomalie.

Les labels de stress, les dates de surveillance exhaustive et les identifiants d'exploitation/bassin hydrologique restent manquants. Ils ne sont pas inventés à partir de la culture, de la météo ou du nom régional. Le pilote n'est pas une validation représentative du Maroc ; les années antérieures à la visite nécessitent une vérification de stabilité de la culture. L'entraînement et le test doivent faire l'objet d'un protocole verrouillé distinct. Aucun score de précision/rappel/F1 agronomique n'est produit par cet import.

Sources techniques : https://registry.opendata.aws/sentinel-2-l2a-cogs/ ; https://github.com/Element84/earth-search/blob/main/docs/collections/sentinel-2-l2a.md ; https://www.usgs.gov/landsat-missions/landsat-collection-2 ; https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2 ; https://open-meteo.com/en/docs/historical-weather-api.
