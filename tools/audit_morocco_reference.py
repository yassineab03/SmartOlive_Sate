"""Audit du terrain publié et sélection d'un petit pilote sans étiquettes inventées."""
from pathlib import Path
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/morocco_reference'
META=ROOT/'data/metadata'; OUT=ROOT/'data/processed'; OUT.mkdir(exist_ok=True)
fields=gpd.read_file(next(RAW.rglob('*.shp')))
assert fields.crs.to_epsg()==4326
assert fields.parcel_id.is_unique
olive=fields[fields.tree.eq('Olives')].copy()
olive['area_ha']=olive.to_crs(6933).area/10000
olive['geometry_valid']=olive.geometry.is_valid
olive['crop_observed_at']=pd.to_datetime(olive.date).dt.strftime('%Y-%m-%d')
olive['crop_evidence']='terrain publié : tree=Olives ; DOI 10.6084/m9.figshare.28486022.v2'
olive['stress_label']=None
olive['farm_id']=None
olive['basin_id']=None
olive['source_doi']='10.6084/m9.figshare.28486022.v2'
olive['source_release_date']='2025-07-14'
olive['mixed_crop']=~olive.crop.eq('Trees')
olive['date']=olive.crop_observed_at
dest=ROOT/'data/raw/maroc/oliveraies_reference.geojson'
olive.to_file(dest,driver='GeoJSON')
olive.drop(columns='geometry').to_csv(OUT/'olive_parcels_inventory.csv',index=False)
files={p.stem.removesuffix('_S2_timeseries'):p for p in RAW.rglob('*_S2_timeseries.csv') if '.ipynb_checkpoints' not in p.parts}
records=[]; missing=[]; duplicate_audit=[]
for row in olive.itertuples():
    path=files.get(row.parcel_id)
    if path is None: missing.append(row.parcel_id); continue
    df=pd.read_csv(path)
    assert {'NDVI','date'}<=set(df),str(path)
    df['date']=pd.to_datetime(df.date,errors='raise')
    n_exact=int(df.duplicated().sum())
    df=df.drop_duplicates().copy()
    conflicting=df.date.duplicated(keep=False)
    if n_exact or conflicting.any():
        duplicate_audit.append(dict(parcel_id=row.parcel_id,exact_duplicates_removed=n_exact,
                                    ambiguous_rows_excluded=int(conflicting.sum())))
    # Sans identifiant d'acquisition, ne pas choisir arbitrairement entre deux valeurs.
    df=df.loc[~conflicting].copy()
    df['parcel_id']=row.parcel_id
    df['region']=row.zone
    df['source_file']=str(path.relative_to(ROOT))
    df['qa_coverage_available']=False
    df['stress_label']=None
    records.append(df)
series=pd.concat(records,ignore_index=True).sort_values(['parcel_id','date'])
series.to_csv(OUT/'olive_reference_series_2024.csv',index=False)
pd.DataFrame(duplicate_audit).to_csv(META/'reference_series_duplicate_audit.csv',index=False)
eligible=olive[olive.geometry_valid & olive.crop.eq('Trees') & olive.area_ha.ge(1)].copy()
# Deux régions publiées ; choisir les parcelles les plus proches de la plus grande
# de chaque région. Aucun résultat satellite ne participe à cette sélection.
pilots=[]
for region in ['el haouz','Tadla']:
    candidates=eligible[eligible.zone.eq(region)].copy()
    projected=candidates.to_crs(32629)
    anchor=projected.loc[candidates.area_ha.idxmax()].geometry.centroid
    candidates['distance_to_anchor_m']=projected.geometry.centroid.distance(anchor)
    pilots.append(candidates.sort_values(['distance_to_anchor_m','parcel_id']).head(3))
pilot=gpd.GeoDataFrame(pd.concat(pilots,ignore_index=True),crs=4326)
pilot.to_file(ROOT/'data/raw/maroc/oliveraies_pilote.geojson',driver='GeoJSON')
pilot.drop(columns='geometry').to_csv(OUT/'olive_pilot_inventory.csv',index=False)
summary=dict(source_doi='10.6084/m9.figshare.28486022.v2',license='CC BY 4.0',
             all_parcels=len(fields),olive_labelled_parcels=len(olive),
             pure_olive_parcels=int(olive.crop.eq('Trees').sum()),invalid_olive_geometries=int((~olive.geometry_valid).sum()),
             parcels_with_published_series=int(series.parcel_id.nunique()),series_rows=len(series),
             first_date=str(series.date.min()),last_date=str(series.date.max()),
             missing_series=missing,regions=olive.zone.value_counts().to_dict(),
             exact_duplicate_rows_removed=sum(r['exact_duplicates_removed'] for r in duplicate_audit),
             ambiguous_rows_excluded=sum(r['ambiguous_rows_excluded'] for r in duplicate_audit),
             pilot_parcels=pilot.parcel_id.tolist(),
             caveats=['Culture observée au jour de visite, pas confirmation rétroactive sur toutes les années.',
                      'Aucun label de stress ; aucun identifiant exploitation ou bassin hydrologique fourni.',
                      'Bandes des CSV publiés conservées brutes : facteur/offset et taux de pixels QA non fournis dans les fichiers.',
                      'NDVI fourni par les auteurs conservé sans recalcul ; indices physiques à extraire des rasters audités.',
                      'Les régions source ne sont pas des bassins hydrologiques.',
                      'La géométrie invalide reste dans inventaire et est exclue du pilote sans correction silencieuse.'])
(META/'morocco_reference_audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(summary,indent=2,ensure_ascii=False))
print(pilot[['parcel_id','zone','area_ha','distance_to_anchor_m']].to_string(index=False))
print('BBox pilote :',pilot.total_bounds)
