"""Exécute le notebook d'import avec l'interpréteur du projet courant."""
from pathlib import Path
import os
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]
for name in ['IPYTHONDIR','JUPYTER_RUNTIME_DIR','JUPYTER_CONFIG_DIR','JUPYTER_DATA_DIR']:
    path=ROOT/'.runtime'/name.lower();path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
path=ROOT/'archives/notebooks/import_donnees_oliveraies.ipynb'
nb=nbformat.read(path,as_version=4)
km=KernelManager(kernel_name='python3')
km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client=NotebookClient(nb,timeout=900,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
client.km=km
try:client.execute()
finally:nbformat.write(nb,path)
print('Notebook import exécuté, sorties conservées ; kernel :',sys.executable)
