from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parents[1]
for key in ['IPYTHONDIR','JUPYTER_RUNTIME_DIR','JUPYTER_CONFIG_DIR','JUPYTER_DATA_DIR']:
    folder=root/'.runtime'/key.lower()
    folder.mkdir(parents=True,exist_ok=True)
    os.environ[key]=str(folder)
path=root/'analyse_comparative.ipynb'
nb=nbformat.read(path,as_version=4)
client=NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(root)}})
try:
    client.execute()
finally:
    nbformat.write(nb,path)
print('Notebook exécuté et sorties conservées :',path)
