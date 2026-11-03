# OliveSmart — historique satellitaire 2021–2025

Exécuter les deux notebooks depuis la racine du projet, avec `.venv/Scripts/python.exe` comme kernel.

1. **01_nettoyage_pretraitement.ipynb** : contrôle du nouvel historique Sentinel-2, fenêtres de six mois pour prévoir le NDVI du mois suivant, séparation chronologique 70/15/15 des mois et groupes spatiaux réservés. Sorties : `data/processed/model_inputs_sentinel/`.
2. **02_entrainement_resultats.ipynb** : baseline de persistance, Random Forest, XGBoost, LightGBM, LSTM, GRU et CNN-LSTM. Par défaut `RUN_TRAINING=False` vérifie les entrées et charge les implémentations. Passer à `True` pour entraîner ; un nom horodaté évite l'ancien verrouillage d'expérience.

DL : budgets de 50/100/150 époques, Early Stopping, deux configurations par budget. ML : trois configurations par famille. Normalisation sur train uniquement, sélection sur validation uniquement, test temporel et spatial séparés. Sorties : `outputs/benchmarks/<RUN_NAME>/` (modèles, métriques, prédictions, temps, graphiques et rapport HTML).

Les données ne contiennent pas de labels de stress : la tâche est une régression NDVI avec MAE, RMSE et R², pas une classification agronomique. Les lacunes de 2022 restent explicites ; aucune interpolation. La stabilité de la culture sur les cinq années n'est pas confirmée. Des groupes spatiaux distincts ne prouvent pas l'indépendance des exploitations. Les proportions d'exemples après filtrage diffèrent des proportions des mois.

Les anciens jeux de données ont été supprimés. Seuls l'historique Sentinel-2 2021–2025, ses découpes/provenance et les entrées des modèles actuels restent dans `data/`. Les anciens rapports et résultats restent distincts dans les dossiers de sorties.

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe tools/execute_pipeline_notebooks.py
.venv/Scripts/python.exe tools/test_notebook_handoff.py
```

Pour reconstruire les notebooks actuels (réinitialise leurs sorties) : `tools/build_current_satellite_notebooks.py`. L'ancien nom `tools/build_pipeline_notebooks.py` redirige vers ce générateur.
