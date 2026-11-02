# Notebooks historiques

Ces versions exécutées sont conservées pour la traçabilité. Les deux points d'entrée
actuels sont à la racine :

1. `01_nettoyage_pretraitement.ipynb`
2. `02_entrainement_resultats.ipynb`

Les fichiers historiques peuvent contenir des résultats de l'ancienne zone de culture
non confirmée. Les cinq ZIP initiaux ont été supprimés à la demande de l'utilisateur ;
leurs résultats historiques ne doivent pas être confondus avec le pilote documenté.

Les anciens générateurs `tools/build_notebook.py` et `tools/build_import_notebook.py`
écrivent désormais ici. `tools/build_pipeline_notebooks.py` reconstruit les deux
notebooks actuels à partir de ces versions de référence. Les générateurs remplacent
les cellules et leurs sorties ; relancer ensuite le vérificateur de pipeline.
