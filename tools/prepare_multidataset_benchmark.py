"""Build separate, auditable tasks from real olive datasets (no synthetic labels)."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/processed/multidataset'

def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()

def save_task(name, records, features, kind, description, provenance, limits):
    folder = OUT/name
    folder.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame([{k:v for k,v in r.items() if k not in ('x','y')} for r in records])
    if not len(meta):
        raise ValueError(f'No samples for {name}')
    arrays = {}
    for split in ['train','validation','test','independent']:
        idx = np.flatnonzero(meta.split.eq(split))
        arrays['X_'+split] = np.array([records[i]['x'] for i in idx], dtype=np.float32)
        arrays['y_'+split] = np.array([records[i]['y'] for i in idx], dtype=np.float32)
        if not len(idx):
            raise ValueError(f'Empty {name}/{split}')
    np.savez_compressed(folder/'samples.npz', **arrays)
    meta.to_csv(folder/'samples.csv', index=False)
    counts = meta.groupby('split').size().to_dict()
    # Assert unit holdout and strict chronological separation of all temporal windows.
    assert set(meta.loc[meta.split.eq('independent'),'unit']).isdisjoint(
        meta.loc[~meta.split.eq('independent'),'unit'])
    if kind != 'cross_sectional_regression':
        for left,right in [('train','validation'),('validation','test')]:
            assert meta.loc[meta.split.eq(left),'target_time'].max() < meta.loc[meta.split.eq(right),'input_start'].min()
    manifest = dict(name=name, kind=kind, description=description, features=features,
        counts=counts, units=meta.groupby('split').unit.nunique().to_dict(),
        provenance=provenance, limitations=limits,
        files_sha256={f:hashlib.sha256((folder/f).read_bytes()).hexdigest()
                      for f in ['samples.csv','samples.npz']})
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(name,counts,flush=True)
    return manifest

def temporal(name, df, features, target, length, horizon, kind, description,
             source, limits, period=None, stride=1, max_gap_days=None, holdout=None):
    df = df.copy().sort_values(['time','unit'])
    if df.duplicated(['unit','time']).any():
        raise ValueError(f'Duplicate unit/time: {name}')
    calendar = np.sort(df.time.unique()) if period is None else period
    a,b = int(.7*len(calendar)),int(.85*len(calendar))
    t1,t2 = pd.Timestamp(calendar[a]),pd.Timestamp(calendar[b])
    if holdout is None:
        units = sorted(df.unit.unique(),key=digest)
        holdout = set(units[:max(1,int(.2*len(units)))])
    records = []
    for unit,block in df.groupby('unit'):
        for split,mask in [('train',block.time.lt(t1)),('validation',block.time.ge(t1)&block.time.lt(t2)),
                           ('test',block.time.ge(t2))]:
            if unit in holdout and split != 'test':
                continue
            g = block.loc[mask].sort_values('time').reset_index(drop=True)
            for i in range(length-1,len(g)-horizon,stride):
                window = g.iloc[i-length+1:i+1]
                future = g.iloc[i+horizon]
                x = window[features].to_numpy(dtype=float)
                y = future[target]
                if not np.isfinite(x).all() or not np.isfinite(y):
                    continue
                if max_gap_days is not None:
                    dates = g.iloc[i-length+1:i+horizon+1].time
                    if dates.diff().dt.total_seconds().max() > max_gap_days*86400:
                        continue
                records.append(dict(x=x,y=y,unit=unit,split='independent' if unit in holdout else split,
                    input_start=str(window.time.iloc[0]),input_end=str(window.time.iloc[-1]),
                    target_time=str(future.time)))
    return save_task(name,records,features,kind,description,source,
                     limits+[f'70/15/15 calendar boundaries: {t1}, {t2}.',
                     f'Window length {length}, horizon {horizon} observations, stride {stride}.',
                     'Windows wholly inside one split; no interpolation. Holdout units absent from train/validation.'])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifests = []
    # CSIC hourly series from prior audited real raw data. NaNs remain NaNs.
    d = pd.read_csv(ROOT/'data/processed/benchmark/csic_hourly.csv')
    d['time'] = pd.to_datetime(d.hour_end)
    d['unit'] = d.station
    features = ['Tª Ambiente','·Hd Ambiente']+[f'Hd {n:02} cm' for n in range(5,76,10)]
    manifests.append(temporal('csic_moisture',d,features,'Hd 25 cm',24,24,'regression',
        'Soil moisture at 25 cm, forecast 24 hours ahead from previous 24 hours; daily forecast origins.',
        ['https://doi.org/10.20350/digitalCSIC/18491'],
        ['One orchard and one annual cycle; not an independent-orchard validation.',
         'Held-out mineral stations vs compost training stations confound station transfer and fertilisation.',
         'Window stride 24 hours limits overlapping origins; statistical independence is not implied.'],
        stride=24,holdout={'Control 60','RDC 62','Secano 46'}))
    # Fes: retain timestamp chronology; week grouping prevents splitting one campaign.
    d = pd.read_excel(ROOT/'data/raw/benchmark/drought/Growth.xlsx')
    d['time'] = pd.to_datetime(d.Date).dt.to_period('W').dt.start_time
    d['unit'] = d.Olive_ID
    d['regime'] = d['Irrigation Regime'].astype(str).str.rstrip('%').astype(float)
    d['class'] = d.regime.map({0.:0,25.:1,50.:2,100.:3})
    holdout = set()
    for _,g in d.groupby('regime'):
        units = sorted(g.unit.unique(),key=digest)
        holdout.update(units[:4])
    features = ['Height','Trunk Diameter','Canopy Cover ','Average_SPAD','Temperature','Branches']
    common = ['Weekly sequences are short (19 campaigns per tree); DL results exploratory.',
              'Independent trees share the same farm; not independent parcels.',
              'Tree/group identifiers and irrigation values excluded from predictors.',
              'SPAD 1/2/3 excluded to avoid redundant direct reconstruction of Average_SPAD.']
    for name,target,kind,desc in [
        ('drought_spad','Average_SPAD','regression','Next weekly SPAD from two past campaigns.'),
        ('drought_irrigation','class','classification','Experimental irrigation regime (0/25/50/100 percent); NOT a physiological stress diagnosis.')]:
        manifests.append(temporal(name,d,features,target,2,1,kind,desc,
            ['https://doi.org/10.17632/22j26tpk63.1'],common,max_gap_days=8,holdout=holdout))
    # Sentinel: all geometrically valid, non-intercropped, documented olive parcels.
    inv = pd.read_csv(ROOT/'data/processed/olive_parcels_inventory.csv')
    ids = inv.loc[inv.geometry_valid.eq(True)&inv.mixed_crop.eq(False),'parcel_id']
    d = pd.read_csv(ROOT/'data/processed/olive_reference_series_2024.csv')
    d = d[d.parcel_id.isin(ids)].copy()
    d['time'] = pd.to_datetime(d.date)
    d['unit'] = d.parcel_id
    # Published bands are on a common provider scale; train-only scaling in runner.
    d['gap_days'] = d.sort_values('time').groupby('unit').time.diff().dt.total_seconds()/86400
    d['day_sin'] = np.sin(2*np.pi*d.time.dt.dayofyear/365.25)
    d['day_cos'] = np.cos(2*np.pi*d.time.dt.dayofyear/365.25)
    features=['NDVI','B4','B8','B11','B12','gap_days','day_sin','day_cos']
    manifests.append(temporal('sentinel_ndvi',d,features,'NDVI',3,1,'regression',
        'NDVI at next available acquisition from three previous acquisitions; irregular horizon recorded in samples.csv.',
        ['https://doi.org/10.6084/m9.figshare.28486022.v2'],
        ['Published Sentinel-2 series lack local pixel cloud QA; temporal prediction is not stress detection.',
         'Crop documented at visit, not every acquisition; one year only.',
         'Held-out parcels may share farms/scenes; no verified farm/basin independence.'],max_gap_days=35))
    # OLEA-PRED has no sufficient repeated temporal measurements: honest ML-only task.
    p=ROOT/'data/raw/benchmark/olea_pred/Field_DATA/SPAD_&_Yield.xlsx'
    records=[]
    for sheet in pd.ExcelFile(p).sheet_names:
        g=pd.read_excel(p,sheet_name=sheet)
        spad=next(c for c in g if 'SPAD' in c)
        g['target']=pd.to_numeric(g['Yield (kg)'],errors='coerce')
        g=g.dropna(subset=['target',spad]).copy()
        units=sorted(g.Tree.astype(str),key=digest)
        a,b=int(len(units)*.7),int(len(units)*.85)
        splits={u:('train' if i<a else 'validation' if i<b else 'test') for i,u in enumerate(units)}
        for _,row in g.iterrows():
            records.append(dict(x=[[float(row[spad])]],y=float(row.target),unit=sheet+'_'+str(row.Tree),
                split='independent' if sheet=='FBS' else splits[str(row.Tree)],
                input_start='2023-10-02' if sheet=='Settat' else '2023-10-05',
                input_end='2023-10-02' if sheet=='Settat' else '2023-10-05',target_time='harvest_2023_date_unspecified'))
    manifests.append(save_task('olea_yield',records,['first_october_SPAD'],'cross_sectional_regression',
        'Yield kg vs first October SPAD: association benchmark; harvest date unspecified, not validated forecasting.',
        ['https://doi.org/10.5281/zenodo.10920137'],
        ['Small single-campaign table, only one predictor common to both orchards.',
         'Settat tree holdout 70/15/15 is cross-sectional, NOT chronological.',
         'FBS is a separate orchard external test; different yield distribution.',
         'LSTM/GRU/CNN-LSTM inapplicable: no adequate repeated series; report N/A.',
         'Missing yield x excluded, never converted to zero; no unverified soil/tree joins.']))
    (OUT/'suite_manifest.json').write_text(json.dumps(manifests,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':
    main()
