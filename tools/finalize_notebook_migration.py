"""Update active documentation and retire the previous notebook generator/test."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'README.md').write_text('''# OliveSmart — historique satellitaire 2021–2025

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
''',encoding='utf-8')
(ROOT/'tools/build_pipeline_notebooks.py').write_text('''"""Compatibility entry point: build the CURRENT satellite notebooks."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('build_current_satellite_notebooks.py')),run_name='__main__')
''',encoding='utf-8')
(ROOT/'tools/test_notebook_handoff.py').write_text('''"""Verify current 01/02 contract, chronological and spatial isolation."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import nbformat
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'data/processed/model_inputs_sentinel/sentinel_ndvi'
m=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
for f,h in m['files_sha256'].items():assert hashlib.sha256((p/f).read_bytes()).hexdigest()==h
d=pd.read_csv(p/'samples.csv')
a=np.load(p/'samples.npz')
for split in ['train','validation','test','independent']:
    assert len(a['X_'+split])==len(a['y_'+split])==d.split.eq(split).sum()
    assert a['X_'+split].shape[1:]==(6,len(m['features']))
    assert np.isfinite(a['X_'+split]).all()
assert set(d.loc[d.split.eq('independent'),'spatial_group']).isdisjoint(d.loc[~d.split.eq('independent'),'spatial_group'])
for left,right in [('train','validation'),('validation','test')]:
    assert d.loc[d.split.eq(left),'target_time'].max()<d.loc[d.split.eq(right),'input_start'].min()
for name in ['01_nettoyage_pretraitement.ipynb','02_entrainement_resultats.ipynb']:
    n=nbformat.read(ROOT/name,as_version=4)
    cells=[c for c in n.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in cells)
    assert not [o for c in cells for o in c.outputs if o.output_type=='error']
    assert 'data/processed/pilot' not in '\\n'.join(c.source for c in cells)
print('OK: current satellite inputs, hashes, 6-month windows, chronology, spatial isolation, both notebooks executed.')
''',encoding='utf-8')
print('Active documentation, generator and handoff test migrated.')
