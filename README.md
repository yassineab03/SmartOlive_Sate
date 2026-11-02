# OliveSmart - analyse satellitaire multi-temporelle (OLV-4)

Le travail est maintenant organisé en **deux notebooks**, à exécuter dans cet ordre :

| Notebook | Contenu | Sorties |
|---|---|---|
| [01_nettoyage_pretraitement.ipynb](01_nettoyage_pretraitement.ipynb) | Import facultatif, audit, qualité, nettoyage, splits temporels, saisonnalité et normalisation sur train uniquement | `data/processed/model_inputs/` et `outputs/preparation/` |
| [02_entrainement_resultats.ipynb](02_entrainement_resultats.ipynb) | Chargement des fichiers préparés, z-score NDVI, Isolation Forest, One-Class SVM, autoencoder, optimisation et comparaison | `outputs/experiments/<RUN_NAME>/` |

Chaque notebook contient son code scientifique et démarre dans un kernel neuf. Le second ne dépend d'aucune variable laissée en mémoire par le premier.

## Lancer dans PyCharm

1. Ouvrir le dossier `SmartOlive_Sate` et choisir `.venv/Scripts/python.exe` comme interpréteur et kernel Jupyter.
2. Si l'environnement n'existe pas : `py -m venv .venv`.
3. Installer les dépendances dans le terminal PowerShell :

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

4. Ouvrir **01**, puis **Run All**. `RUN_DOWNLOAD=False` conserve l'import réseau désactivé et utilise les données présentes. Les paramètres de split sont fixés dans ce notebook avant la normalisation.
5. Ouvrir **02** dans un kernel neuf, puis **Run All**. Par défaut, **`RUN_TRAINING=False`** : les entrées sont vérifiées et les méthodes chargées, mais aucun modèle n'est entraîné.
6. Quand le protocole sera validé, passer `RUN_TRAINING=True` dans **02** pour lancer l'entraînement et les résultats. Un fichier `locked_protocol.json` empêche la réutilisation silencieuse d'une expérience ; choisir un nouveau `RUN_NAME` pour une nouvelle expérience.

Pour vérifier les deux notebooks sans interface PyCharm :

```powershell
.venv/Scripts/python.exe tools/execute_pipeline_notebooks.py
.venv/Scripts/python.exe tools/test_notebook_handoff.py
```

Le vérificateur respecte la valeur de `RUN_TRAINING` sauvegardée dans le notebook. Il ne la change pas automatiquement.

## Données et protocole actuel

Le pilote comprend six parcelles avec oliviers documentés aux dates de visite, dans les régions El Haouz et Tadla. Les sources sont Sentinel-2 C1 L2A (2021-2024), Landsat Collection 2 L2 (2015-2024) et ERA5-Land via Open-Meteo (température et humidité du sol, 2021-2024).

Le notebook 01 propose train 2021-2022, validation 2023, test 2024. Les fenêtres ne chevauchent pas les séparations, les trous sont conservés et les transformations sont ajustées sur train uniquement. La séparation régionale est facultative et ne représente pas un bassin hydrologique certifié.

**Limites :** gros trous Sentinel-2 en 2022, culture documentée aux visites seulement, identifiants exploitation/bassin absents, aucun label de stress ni surveillance exhaustive. La comparaison restera exploratoire : sans terrain, précision, rappel, F1 et délai agronomique sont non évaluables. Un taux d'alerte cible n'est pas un score de fiabilité.

Le détail des sources, contrôles et licences est dans [IMPORT_DONNEES.md](IMPORT_DONNEES.md).

## Fichiers échangés entre 01 et 02

```text
data/processed/model_inputs/
  observations.csv         # calendrier, mesures, labels inconnus et split
  splits.csv               # identifiants et frontières temporelles
  residuals.npz            # résidus standardisés, ordre identique au CSV
  transformations.joblib   # modèles saisonniers et scalers ajustés sur train
  manifest.json            # configuration, colonnes, versions et SHA-256
```

Le second notebook vérifie toutes les empreintes avant chargement. Après modification des données, de la période ou des caractéristiques, réexécuter le notebook 01 avant le 02.

## Archives et outils

Les anciens notebooks sont conservés dans `archives/notebooks/`, avec leurs sorties historiques. Les cinq ZIP de l'ancienne zone ont été supprimés à la demande de l'utilisateur ; les données du pilote documenté sont conservées.

`tools/build_pipeline_notebooks.py` reconstruit les deux notebooks actuels à partir des versions de référence archivées. Il remplace leurs cellules et leurs sorties : exécuter ensuite le vérificateur. Les traitements sont visibles dans les notebooks, pas cachés derrière des imports de scripts locaux.

`requirements-lock.txt` photographie l'environnement de vérification. Les données brutes, `.venv`, les secrets et les gros modèles ne doivent pas être poussés dans GitHub.
