# Jeu temporel retenu pour OliveSmart

## Recommandation

**CSIC / IRNAS — Dataset and code for ecohydrological regulation of nitrate and potassium dynamics in a Mediterranean olive orchard under contrasting irrigation and fertilization regimes.**

Source officielle : https://digital.csic.es/handle/10261/435580
DOI : https://doi.org/10.20350/digitalCSIC/18491
Auteurs : Francisco Sebastián Benitez, Engracia Madejón, M. Sánchez-Piñero, Laura Lozano de Sosa. Dépôt publié le 12 juin 2026.

Les fichiers ont été téléchargés et inspectés le 3 novembre 2026. L'archive originale est conservée dans `data/raw/benchmark/csic_olive/Repositorio.rar`, avec les CSV et scripts R des auteurs extraits à côté. L'empreinte SHA-256 de première récupération est dans `csic_audit.json` ; il s'agit d'une empreinte locale de traçabilité, pas d'une validation par une empreinte publiée par le fournisseur.

## Volume réellement vérifié

| Station | Observations brutes | Traitement indiqué par le fichier |
|---|---:|---|
| Control 59 | 35 130 | Contrôle |
| Control 60 | 35 130 | Contrôle |
| RDC 49 | 35 090 | Irrigation déficitaire régulée |
| RDC 62 | 35 064 | Irrigation déficitaire régulée |
| Secano 46 | 35 142 | Sec / survie |
| Secano 53 | 35 133 | Sec / survie |
| Total | **210 689** | **6 stations** |

Période mesurée : **1er avril 2025 – 1er avril 2026 inclus**. Intervalle médian vérifié : **15 minutes** par station. Les fichiers ont quelques horodatages dupliqués et des interruptions : les six séries ne sont pas supposées parfaitement régulières.

Variables disponibles : humidité du sol à 5, 15, 25, 35, 45, 55, 65 et 75 cm ; température et humidité de l'air ; température du sol ; dendrométrie ; nitrate et potassium ; irrigation, pluie, conductivité et déficit de pression de vapeur ; type de fertilisation. L'absence fréquente de valeurs dans les colonnes irrigation/pluie ne doit pas être assimilée automatiquement à zéro. Les unités et calibrations de dendrométrie et nutriments doivent être vérifiées avant emploi comme cibles.

## Comparaison LSTM / GRU / CNN-LSTM proposée

La première cible défendable est la **prévision de l'humidité mesurée du sol à 25 cm, à 24 h**, à partir des 24 dernières heures : il s'agit d'une régression, pas d'un diagnostic de stress physiologique.

Une préparation de faisabilité est déjà enregistrée dans `data/processed/benchmark/csic_hourly.csv`. Elle utilise dix variables : les huit profondeurs d'humidité du sol, la température et l'humidité de l'air. Les moyennes horaires sont datées à la fin de l'heure entièrement observée. Aucun remplissage ou interpolation n'a été appliqué. Les valeurs hors de plages physiques élémentaires sont exclues ; cela ne remplace pas une validation complète des capteurs.

Le découpage chronologique est commun à toutes les stations, sur la grille horaire complète : 70 % train, 15 % validation, 15 % test (arrondis à l'heure). Toutes les entrées et la cible d'une fenêtre restent dans leur bloc. Les fenêtres incomplètes sont exclues.

| Bloc | Fenêtres disponibles, six stations réunies |
|---|---:|
| Train | **36 443** |
| Validation | **7 626** |
| Test | **7 476** |

Frontières exactes et effectifs par station : `csic_audit.json` et `csic_window_counts.csv`. Les fenêtres glissantes sont corrélées : ces nombres ne représentent pas autant d'expériences indépendantes.

**Conclusion de faisabilité :** ce volume et cette profondeur temporelle permettent d'entraîner et comparer des architectures compactes LSTM, GRU et CNN-LSTM avec des budgets de 50/100/150 époques et Early Stopping. Ils ne garantissent pas que le DL sera supérieur. Il faut comparer une baseline de persistance, une baseline saisonnière (24 h), Random Forest, XGBoost et LightGBM sur les mêmes cibles et origines de prévision. Les transformations et l'optimisation doivent utiliser le train et la validation exclusivement. Sélection du modèle par validation, test final réservé à l'évaluation.

## Ce que ce jeu ne suffit pas à démontrer

- Les six stations appartiennent à un même verger : une station tenue à l'écart n'est pas un verger indépendant.
- Une seule année ne permet pas de valider une généralisation interannuelle ; le split demandé évalue aussi un changement de saison. Ajouter une validation chronologique glissante dans le développement.
- Les deux stations par régime diffèrent aussi par la fertilisation : une séparation de stations peut confondre ces effets. Les identifiants de station ne doivent pas servir de prédicteurs.
- Les régimes d'irrigation sont des traitements expérimentaux, pas des labels certifiés de stress. Accuracy/F1/ROC ne mesureraient pas automatiquement la détection de stress.
- La dendrométrie n'est pas directement comparable d'une station à l'autre sans contrôle des offsets, unités et dérives.

## Autres candidats examinés

**Goraj, Albanie — Mendeley DOI 10.17632/cnfh4stzzx.1**
https://data.mendeley.com/datasets/cnfh4stzzx/1

Archive de 1,37 Mo téléchargée et vérifiée avec le SHA-256 publié : `34b967284f8447cea5513b3360eb4b12d2a7fa26bdf48844eac73fe1e49d93d0`.
9 435 mesures sur deux unités, du 21 février au 13 mai 2026. Météo, humidité du sol et dendrométrie. Seulement 5 155 valeurs d'humidité volumique du sol valides dans le tableau QC ; nombreux indicateurs de qualité et changement de capteur. Candidat secondaire intéressant pour une évaluation externe de variables communes, après harmonisation des unités, des fréquences et de la profondeur des sondes. Il ne peut pas encore être déclaré compatible avec la cible CSIC à 25 cm.

**Radar et physiologie, oliveraie au Maroc — Zenodo 11299406**
https://zenodo.org/records/11299406

Mesures à 15 minutes, flux de sève, dendrométrie, humidité et météo ; très pertinent pour OliveSmart, mais fichiers actuellement restreints. Non retenu pour une exécution immédiate sans accès accordé.

**Fès — Mendeley DOI 10.17632/22j26tpk63.1** : 1 520 observations / 80 arbres, principalement hebdomadaires. Utile comme complément physiologique ; profondeur temporelle individuelle bien inférieure au CSIC.

**OLEA-PRED — Zenodo 10920137** : les fichiers terrain inspectés n'offrent qu'une ou deux dates SPAD par verger. Utile pour des analyses transversales ; ces seuls fichiers ne constituent pas un benchmark temporel DL.

## Reproduire l'audit

Depuis la racine du projet :

```powershell
.venv/Scripts/python.exe tools/import_benchmark_sources.py
.venv/Scripts/python.exe tools/audit_temporal_dataset.py
```

L'audit prépare les séries et compte les fenêtres ; aucun score d'entraînement n'est inventé et aucun modèle n'a encore été entraîné sur ce nouveau jeu.
