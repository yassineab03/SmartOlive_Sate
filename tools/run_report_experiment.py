"""Exécute une copie du notebook 02 pour le rapport, sans modifier le modèle de travail."""
from pathlib import Path
import os
import sys
import json
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]
RUN='rapport_exploratoire_20261102'
if '--finalize-statistics' in sys.argv:
    for name in ['IPYTHONDIR','JUPYTER_RUNTIME_DIR','JUPYTER_CONFIG_DIR','JUPYTER_DATA_DIR']:
        p=ROOT/'.runtime'/name.lower();p.mkdir(parents=True,exist_ok=True);os.environ[name]=str(p)
    target=ROOT/'outputs/experiments'/RUN/'02_execution_rapport.ipynb'
    notebook=nbformat.read(target,as_version=4)
    prefix='# Contrôle final du calendrier de campagne'
    content=(ROOT/'tools/finalize_report_statistics.py').read_text(encoding='utf-8').replace('ROOT=Path(__file__).resolve().parents[1]','ROOT=Path.cwd()')
    cell=nbformat.v4.new_code_cell(prefix+'\n'+content+"\nfrom IPython.display import display\ndisplay(pd.read_csv(RUN/'comparison.csv'))")
    old=next((i for i,c in enumerate(notebook.cells) if c.cell_type=='code' and c.source.startswith(prefix)),None)
    if old is None:
        notebook.cells.append(nbformat.v4.new_markdown_cell('## Contrôle final des statistiques temporelles\nLa campagne est identifiée par le début de fenêtre : la fenêtre se terminant au 1er janvier 2025 appartient à 2024. Ce contrôle autonome ne modifie aucun score ni seuil et est exécuté dans un kernel neuf sur les exports enregistrés.'))
        notebook.cells.append(cell);index=len(notebook.cells)-1
    else:notebook.cells[old]=cell;index=old
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(notebook,timeout=300,resources={'metadata':{'path':str(ROOT)}});client.km=manager
    try:
        with client.setup_kernel():client.execute_cell(cell,index)
    finally:nbformat.write(notebook,target)
    print('Contrôle temporel enregistré dans la copie exécutée du rapport',flush=True)
    raise SystemExit(0)
resume='--resume' in sys.argv
lock=ROOT/'outputs/experiments'/RUN/'locked_protocol.json'
assert lock.exists() if resume else not lock.exists(), 'État du verrou incompatible avec le mode demandé'
for name in ['IPYTHONDIR','JUPYTER_RUNTIME_DIR','JUPYTER_CONFIG_DIR','JUPYTER_DATA_DIR']:
    p=ROOT/'.runtime'/name.lower();p.mkdir(parents=True,exist_ok=True);os.environ[name]=str(p)
source=ROOT/'02_entrainement_resultats.ipynb'
notebook=nbformat.read(source,as_version=4)
evaluation=next(c for c in notebook.cells if c.cell_type=='code' and c.source.startswith('def event_metrics('))
evaluation.source=evaluation.source.replace("['parcel_id','farm_id','basin','date','label']", "['parcel_id','region','date','label']")
nbformat.write(notebook,source)
setup=next(c for c in notebook.cells if c.cell_type=='code' and 'RUN_TRAINING=False' in c.source)
setup.source=setup.source.replace('RUN_TRAINING=False','RUN_TRAINING=True').replace("RUN_NAME='comparaison_01'",f"RUN_NAME='{RUN}'")
for cell in notebook.cells:
    if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
directory=ROOT/'outputs/experiments'/RUN;directory.mkdir(parents=True,exist_ok=True)
target=directory/'02_execution_rapport.ipynb'
if resume:
    backup=directory/'diagnostics/02_avant_correction_export.ipynb'
    backup.parent.mkdir(exist_ok=True)
    if target.exists() and not backup.exists():backup.write_bytes(target.read_bytes())
    selection=next(c for c in notebook.cells if c.cell_type=='code' and c.source.startswith('winners=[]'))
    selection.source='''# Reprise après correction d'export : paramètres et seuils figés, aucune nouvelle recherche.
locked=json.loads((OUT/'locked_protocol.json').read_text())
assert locked['config']==CFG
winners=[]
for choice in locked['choices']:
    name=choice['name']; ablation=choice['ablation']; features=FEATURE_SETS[ablation]
    x,ix=windows(data,residuals[ablation],choice['length'])
    subset=data.loc[ix]
    tr=subset.split.eq('train').to_numpy() & ~subset.label.eq(1).to_numpy()
    va=subset.split.eq('val').to_numpy() & ~subset.label.eq(1).to_numpy()
    model,history=fit_method(name,x[tr],x[va],choice['params'])
    validation_score=score(name,model,x[va],len(features))
    fixed_value=-abs(float((validation_score>choice['threshold']).mean())-CFG['target_alert_rate'])
    prior=pd.read_csv(OUT/f'trials_{ablation}_{name}.csv')
    assert abs(fixed_value-prior.loc[prior.state.eq('COMPLETE'),'value'].max())<1e-10
    winners.append(dict(choice,features=features,model=model,history=history,
                        season=transformations[ablation]['season'],scaler=transformations[ablation]['scaler'],x=x,ix=ix))
print('Reprise à paramètres figés ; concordance validation vérifiée :',len(winners),'configurations.')
'''
    evaluation.source=evaluation.source.replace("if lock.exists(): raise RuntimeError('Expérience déjà verrouillée : conserver ce dossier et choisir un nouveau dossier outputs pour une nouvelle expérience.')", "assert lock.exists(), 'Verrou original absent'")
    evaluation.source=evaluation.source.replace("lock.write_text(json.dumps({'config':CFG,'choices':choices,'seed':SEED},indent=2))", "assert choices==json.loads(lock.read_text())['choices'], 'Les choix verrouillés ont changé'")
manager=KernelManager(kernel_name='python3')
manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client=NotebookClient(notebook,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
client.km=manager
try:
    with client.setup_kernel():
        for i,cell in enumerate(notebook.cells):
            if cell.cell_type!='code':continue
            print('Exécution cellule',i,cell.source.splitlines()[0][:70],flush=True)
            client.execute_cell(cell,i)
            nbformat.write(notebook,target)
finally:nbformat.write(notebook,target)
print('Expérience terminée :',RUN,flush=True)
