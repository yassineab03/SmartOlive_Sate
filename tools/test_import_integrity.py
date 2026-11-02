"""Tests de risques : provenance, physique, calendrier et absence de faux labels."""
from pathlib import Path
import hashlib
import json
import geopandas as gpd
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
pilot=gpd.read_file(ROOT/'data/raw/maroc/oliveraies_pilote.geojson')
assert len(pilot)==6 and pilot.parcel_id.is_unique and pilot.geometry.is_valid.all()
assert pilot.tree.eq('Olives').all() and pilot.crop.eq('Trees').all()
assert pilot.farm_id.isna().all() and pilot.basin_id.isna().all() and pilot.stress_label.isna().all()
manifest=json.loads((ROOT/'data/metadata/morocco_reference_manifest.json').read_text(encoding='utf-8'))
for item in manifest:
    path=ROOT/item['file']
    assert hashlib.md5(path.read_bytes()).hexdigest()==item['md5']
data=pd.read_csv(ROOT/'data/processed/pilot/satellite_acquisitions_verified.csv')
assert not data.duplicated(['source','parcel_id','date']).any()
assert data.valid_fraction.between(0,1).all()
assert data.stress_label.isna().all()
series=pd.read_csv(ROOT/'data/processed/pilot/sentinel2_10day_with_weather.csv')
assert len(series)==6*4*37
assert not series.duplicated(['parcel_id','date']).any()
accepted=series[series.accepted]
assert accepted.qa_valid_fraction.ge(.6).all()
for index in ['ndvi','ndre','ndwi']:assert accepted[index].between(-1,1).all()
assert series.loc[~series.accepted,['ndvi','ndre','ndwi','savi']].isna().all().all()
assert pd.to_datetime(accepted.acquisition_date,utc=True).lt(pd.to_datetime(accepted.date,utc=True)).all()
assert series[['farm_id','basin_id','stress_label']].isna().all().all()
weather=pd.read_csv(ROOT/'data/processed/pilot/era5_land_daily_2021_2024.csv')
assert len(weather)==2*1461
assert not weather.duplicated(['weather_region','date']).any()
assert weather.model.eq('ERA5-Land').all()
assert 'precipitation_mm' not in weather,'Ne pas remplacer silencieusement la pluie ERA5-Land'
# Concordance de prétraitement, pas performance agronomique : même parcelle et date.
source=ROOT/'data/raw/morocco_reference/Sentinel2 bands for all surveyed parcels/Trees/Tadla_Trees_20240224_002_S2_timeseries.csv'
reference=pd.read_csv(source)
ref=reference.loc[reference.date.eq('2024-06-04'),'NDVI'].iloc[0]
sample=pd.read_csv(ROOT/'data/processed/pilot/satellite_acquisitions_smoke_all.csv')
actual=sample.loc[sample.source.eq('sentinel2') & sample.parcel_id.eq('Tadla_Trees_20240224_002'),'ndvi'].iloc[0]
assert abs(ref-actual)<.01, 'Écart radiométrique anormal : contrôler les offsets'
print('OK : intégrité, parcelles, calendriers, QA, causalité, absence de faux labels et concordance NDVI.')
