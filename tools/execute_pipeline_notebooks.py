"""Exécute 01 puis 02, chacun dans un nouveau kernel du projet."""
from pathlib import Path
import argparse
import os
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--only',choices=['01','02'])
args=parser.parse_args()
for name in ['IPYTHONDIR','JUPYTER_RUNTIME_DIR','JUPYTER_CONFIG_DIR','JUPYTER_DATA_DIR']:
    folder=ROOT/'.runtime'/name.lower();folder.mkdir(parents=True,exist_ok=True);os.environ[name]=str(folder)
names=['01_nettoyage_pretraitement.ipynb','02_entrainement_resultats.ipynb']
for name in names:
    if args.only and not name.startswith(args.only):continue
    path=ROOT/name
    notebook=nbformat.read(path,as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type=='code':compile(cell.source,name,'exec')
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(notebook,timeout=900,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
    client.km=km
    print('Exécution en kernel neuf :',name,flush=True)
    try:client.execute()
    finally:nbformat.write(notebook,path)
    print('Terminé, sorties conservées :',name,flush=True)
