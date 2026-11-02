"""Consolide uniquement les découpes vérifiées, conserve les trous et la provenance."""
from pathlib import Path
import hashlib
import json
from datetime import timedelta
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/pilot'; META=ROOT/'data/metadata/pilot';OUT=ROOT/'data/processed/pilot'
REPORT=ROOT/'outputs/pilot';REPORT.mkdir(parents=True,exist_ok=True)
pilot=gpd.read_file(ROOT/'data/raw/maroc/oliveraies_pilote.geojson')
rows=[];manifest=[]
full_run=OUT/'satellite_acquisitions_all.csv'
assert full_run.exists(),'Attendre la fin du téléchargement complet avant consolidation'
selected=pd.read_csv(full_run)
selected_ids=set(zip(selected.source,selected.item_id))
for source in ['sentinel2','landsat']:
    for path in (RAW/source).rglob('*.json'):
        data=json.loads(path.read_text())
        if data.get('config',{}).get('processing_version')!=2:continue
        if (source,data['item_id']) not in selected_ids:continue
        # Exclure tous les essais de l'ancien miroir aux offsets contradictoires.
        if source=='sentinel2' and not data['item_id'].startswith(('S2A_T','S2B_T','S2C_T')):continue
        target=ROOT/data['file']
        assert hashlib.sha256(target.read_bytes()).hexdigest()==data['sha256']
        rows.extend(data['rows'])
        manifest.append({k:data[k] for k in ['source','item_id','date','file','sha256','retrieved_utc']})
frame=pd.DataFrame(rows)
assert len(frame),'Aucune découpe vérifiée'
frame['date']=pd.to_datetime(frame.date,utc=True)
assert not frame.duplicated(['parcel_id','source','date']).any()
frame=frame.sort_values(['source','parcel_id','date'])
frame.to_csv(OUT/'satellite_acquisitions_verified.csv',index=False)
weather=pd.read_csv(OUT/'era5_land_daily_2021_2024.csv',parse_dates=['date'])
weather['date']=pd.to_datetime(weather.date,utc=True)
calendar=[]
for parcel in pilot.itertuples():
    part=frame[frame.parcel_id.eq(parcel.parcel_id)&frame.source.eq('sentinel2')]
    meteo=weather[weather.weather_region.eq(parcel.zone)]
    for year in range(2021,2025):
        begin=pd.Timestamp(f'{year}-01-01',tz='UTC');finish=pd.Timestamp(f'{year+1}-01-01',tz='UTC')
        for start in pd.date_range(begin,finish,freq='10D',inclusive='left'):
            end=min(pd.Timestamp(start.to_pydatetime()+timedelta(days=10)),finish)
            candidates=part[part.date.ge(start)&part.date.lt(end)]
            assert len(candidates)<=1,'Plusieurs scènes retenues dans une même fenêtre'
            row=dict(parcel_id=parcel.parcel_id,region=parcel.zone,window_start=start,date=end,
                     acquisition_date=None,accepted=False,qa_valid_fraction=np.nan,
                     status='acquisition_absente',crop_observed_at=str(parcel.crop_observed_at)[:10],
                     crop_evidence_release='2025-07-14',stress_label=np.nan,
                     farm_id=None,basin_id=None)
            for index in ['ndvi','ndre','ndwi','savi']:row[index]=np.nan
            if len(candidates):
                item=candidates.iloc[0]
                range_ok=all(pd.notna(item[k]) and -1<=item[k]<=1 for k in ['ndvi','ndre','ndwi'])
                accepted=bool(item.accepted and range_ok)
                row.update(acquisition_date=item.date,item_id=item.item_id,qa_valid_fraction=float(item.valid_fraction),
                           accepted=accepted,status='valide' if accepted else 'rejet_qualite_ou_plage')
                if accepted:
                    for index in ['ndvi','ndre','ndwi','savi']:row[index]=item[index]
            m=meteo[meteo.date.ge(start)&meteo.date.lt(end)]
            complete=len(m)==(end-start).days and m[['temperature_c','soil_moisture_m3m3']].notna().all().all()
            row['n_weather_days']=len(m)
            row['temperature_c']=float(m.temperature_c.mean()) if complete else np.nan
            row['soil_moisture_m3m3']=float(m.soil_moisture_m3m3.mean()) if complete else np.nan
            row['weather_model']='ERA5-Land via Open-Meteo ; point de grille régional'
            calendar.append(row)
series=pd.DataFrame(calendar).sort_values(['parcel_id','date'])
assert not series.duplicated(['parcel_id','date']).any()
assert series.loc[series.acquisition_date.notna(),'acquisition_date'].lt(series.loc[series.acquisition_date.notna(),'date']).all()
series.to_csv(OUT/'sentinel2_10day_with_weather.csv',index=False)
# Landsat historique indépendant, aucune fusion radiométrique avec Sentinel.
landsat=frame[frame.source.eq('landsat')].copy()
landsat.to_csv(OUT/'landsat_history.csv',index=False)
summary=frame.groupby(['source','region']).agg(observations=('parcel_id','size'),
    first=('date','min'),last=('date','max'),accepted=('accepted','sum')).reset_index()
summary.to_csv(REPORT/'coverage.csv',index=False)
manifest.extend(json.loads((META/'weather_manifest.json').read_text(encoding='utf-8')))
(META/'verified_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
for ax,(region,group) in zip(axes,pilot.groupby('zone')):
    for parcel in group.parcel_id:
        values=series[series.parcel_id.eq(parcel)]
        ax.plot(values.date,values.ndvi,marker='.',markersize=2,linewidth=.7,label=parcel)
    ax.set_title(region+' - NDVI Sentinel-2, trous conservés');ax.set_ylim(-.2,1);ax.set_ylabel('NDVI');ax.legend(fontsize=6)
fig.tight_layout();fig.savefig(REPORT/'ndvi_series.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,5))
for ax,(region,group) in zip(axes,pilot.groupby('zone')):
    group.plot(ax=ax,facecolor='#91ae70',edgecolor='#35552e')
    for parcel in group.itertuples():
        point=parcel.geometry.representative_point()
        ax.annotate(parcel.parcel_id.rsplit('_',1)[-1],(point.x,point.y),fontsize=8)
    ax.set_title(region+' - polygones terrain publiés');ax.set_xlabel('Longitude');ax.set_ylabel('Latitude')
fig.tight_layout();fig.savefig(REPORT/'pilot_map.png',dpi=160);plt.close(fig)
status=dict(confirmed_crop_source='Terrain publié : DOI 10.6084/m9.figshare.28486022.v2 ; culture au jour de visite',
            pilot_parcels=len(pilot),source_counts=frame.source.value_counts().to_dict(),
            calendar_rows=len(series),accepted_sentinel_rows=int(series.accepted.sum()),
            weather_days=int(len(weather)),trained_models=0,
            pending=['Événements de stress et surveillance exhaustive non disponibles publiquement dans cette référence.',
                     'Identifiants exploitation et bassin hydrologique absents ; régions distinctes ne prouvent pas l’indépendance par exploitation.',
                     'Stabilité de la culture sur les années antérieures aux visites non confirmée.',
                     'Découpes sélectionnées par fenêtre ; pas un composite de toutes les acquisitions.',
                     'Météo via accès Open-Meteo, pas un export du catalogue Earth Engine demandé. Pluie ERA5-Land non importée.'])
(REPORT/'import_status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf-8')
print(summary.to_string(index=False));print(json.dumps(status,indent=2,ensure_ascii=False))
