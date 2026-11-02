"""ERA5-Land via Open-Meteo, modèle explicitement fixé, sans mélange de pluie ERA5."""
from pathlib import Path
import json
import hashlib
from datetime import datetime,timezone
import requests
import geopandas as gpd
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/pilot/era5_land_openmeteo';RAW.mkdir(parents=True,exist_ok=True)
META=ROOT/'data/metadata/pilot';META.mkdir(parents=True,exist_ok=True)
OUT=ROOT/'data/processed/pilot';OUT.mkdir(parents=True,exist_ok=True)
pilot=gpd.read_file(ROOT/'data/raw/maroc/oliveraies_pilote.geojson')
records=[];manifest=[]
for region,group in pilot.groupby('zone'):
    center=group.to_crs(32629).geometry.union_all().centroid
    point=gpd.GeoSeries([center],crs=32629).to_crs(4326).iloc[0]
    path=RAW/(region.replace(' ','_')+'_2021_2024.json')
    params=dict(latitude=point.y,longitude=point.x,start_date='2021-01-01',end_date='2024-12-31',
                models='era5_land',daily='temperature_2m_mean,soil_moisture_0_to_7cm_mean',
                timezone='UTC',elevation='nan',cell_selection='nearest')
    if path.exists():data=json.loads(path.read_text())
    else:
        response=requests.get('https://archive-api.open-meteo.com/v1/archive',params=params,timeout=(20,120))
        if response.status_code!=200:raise ValueError(response.text[:300])
        data=response.json();path.write_text(json.dumps(data,indent=2))
    assert data['daily_units']['temperature_2m_mean']=='°C'
    assert data['daily_units']['soil_moisture_0_to_7cm_mean']=='m³/m³'
    frame=pd.DataFrame(data['daily']).rename(columns={'time':'date','temperature_2m_mean':'temperature_c','soil_moisture_0_to_7cm_mean':'soil_moisture_m3m3'})
    frame['date']=pd.to_datetime(frame.date)
    assert len(frame)==1461 and frame.date.is_unique
    assert frame.temperature_c.notna().all() and frame.soil_moisture_m3m3.notna().all()
    frame['weather_region']=region;frame['model']='ERA5-Land';frame['access_provider']='Open-Meteo'
    frame['grid_latitude']=data['latitude'];frame['grid_longitude']=data['longitude']
    records.append(frame)
    manifest.append(dict(file=str(path.relative_to(ROOT)),bytes=path.stat().st_size,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(),published_checksum=None,
                         source='Copernicus ERA5-Land, accès et agrégations Open-Meteo',
                         url='https://archive-api.open-meteo.com/v1/archive',request=params,
                         response_units=data['daily_units'],response_grid={'latitude':data['latitude'],'longitude':data['longitude']},
                         integrity='JSON, unités, calendrier complet et valeurs non nulles vérifiés',
                         retrieved_utc=datetime.now(timezone.utc).isoformat(),
                         caveat='Point de grille régional, non une station dans la parcelle. Pas le produit Earth Engine DAILY_AGGR ; pas de pluie ERA5 ajoutée silencieusement.'))
    print(region,len(frame),'jours ERA5-Land importés',flush=True)
pd.concat(records,ignore_index=True).to_csv(OUT/'era5_land_daily_2021_2024.csv',index=False)
(META/'weather_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
