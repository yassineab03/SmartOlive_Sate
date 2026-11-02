from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
for source in ['sentinel2','landsat']:
    f=next((ROOT/f'data/raw/pilot/{source}/Tadla').glob('*.npz'))
    print(f.name)
    for k,a in np.load(f).items():
        if a.ndim==2:
            print(k,a.dtype,a.shape,'min/max/mean',np.nanmin(a),np.nanmax(a),np.nanmean(a))
            if k.startswith('qa_'):print('unique',np.unique(a,return_counts=True))
p=ROOT/'data/raw/morocco_reference/Sentinel2 bands for all surveyed parcels/Trees/Tadla_Trees_20240224_002_S2_timeseries.csv'
d=pd.read_csv(p)
print(d[d.date.between('2024-06-01','2024-06-30')][['date','B4','B8','NDVI']].to_string(index=False))
