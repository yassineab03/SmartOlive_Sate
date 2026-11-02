"""Import public de découpes Sentinel-2/Landsat sur six parcelles terrain.

Les COG sont des miroirs des produits ESA/USGS, pas de nouveaux capteurs.
Un candidat au maximum par fenêtre de 10 jours (S2) ou mois (Landsat), choisi
sur nuages de scène ; les trous et rejets locaux restent explicites.
"""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import math
import time
import threading
from urllib.parse import urlparse
from datetime import datetime, timezone
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from shapely.geometry import shape

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/pilot'; META=ROOT/'data/metadata/pilot'; OUT=ROOT/'data/processed/pilot'
for p in [RAW,META,OUT]:p.mkdir(parents=True,exist_ok=True)
PILOT=gpd.read_file(ROOT/'data/raw/maroc/oliveraies_pilote.geojson')
CONFIG={'processing_version':2,'sentinel_start':'2021-01-01','landsat_start':'2015-01-01','end':'2024-12-31',
        'step_days':10,'max_scene_cloud':40,'min_valid_fraction':.6,
        'selection':'minimum eo:cloud_cover parmi les scènes couvrant le groupe, par fenêtre, sans accès aux résultats',
        'qa_sentinel_classes':[4,5], 'note':'eau exclue du pilote oléicole ; échantillonnage temporel, pas composite de toutes les scènes'}
parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); parser.add_argument('--source',choices=['all','sentinel2','landsat'],default='all'); parser.add_argument('--workers',type=int,default=3)
args=parser.parse_args()
SAS_CACHE={}
SAS_LOCK=threading.Lock()
def session():
    s=requests.Session(); s.mount('https://',HTTPAdapter(max_retries=Retry(total=2,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
    return s

def search(region,group,source):
    start='2024-06-01' if args.smoke else CONFIG['sentinel_start' if source=='sentinel2' else 'landsat_start']
    end='2024-06-30' if args.smoke else CONFIG['end']
    col='sentinel-2-c1-l2a' if source=='sentinel2' else 'landsat-c2-l2'
    api='https://earth-search.aws.element84.com/v1/search' if source=='sentinel2' else 'https://planetarycomputer.microsoft.com/api/stac/v1/search'
    slug=region.replace(' ','_')
    dest=META/f'{col}_{slug}_{start}_{end}_catalogue.json'
    if dest.exists(): items=json.loads(dest.read_text())['features']
    else:
        items=[]; client=session()
        for year in range(int(start[:4]),int(end[:4])+1):
            begin=max(start,f'{year}-01-01');finish=min(end,f'{year}-12-31')
            yearfile=META/f'{col}_{slug}_{begin}_{finish}_part.json'
            if yearfile.exists():items.extend(json.loads(yearfile.read_text())['features']);continue
            yearitems=[];url=api
            params={'collections':col,'bbox':','.join(map(str,group.total_bounds)),
                    'datetime':begin+'T00:00:00Z/'+finish+'T23:59:59Z','limit':100}
            while url:
                r=client.get(url,params=params,timeout=(20,90));r.raise_for_status();data=r.json()
                yearitems.extend(data['features'])
                link=next((l for l in data.get('links',[]) if l['rel']=='next'),None)
                url=link['href'] if link else None;params=None
            yearfile.write_text(json.dumps({'type':'FeatureCollection','features':yearitems}))
            items.extend(yearitems)
        dest.write_text(json.dumps({'type':'FeatureCollection','features':items},indent=2))
    union=group.geometry.union_all()
    candidates={}
    for item in items:
        props=item['properties']; cloud=props.get('eo:cloud_cover',100)
        if cloud is None or cloud>CONFIG['max_scene_cloud']:continue
        if not shape(item['geometry']).covers(union):continue
        if source=='landsat' and props.get('platform') not in ['landsat-8','landsat-9']:continue
        dt=pd.Timestamp(props['datetime'])
        # Origine annuelle fixe, la dernière fenêtre de l'année peut être courte.
        key=(dt.year,(dt.dayofyear-1)//10) if source=='sentinel2' else (dt.year,dt.month)
        rank=(cloud,props['datetime'],item['id'])
        if key not in candidates or rank<candidates[key][0]:candidates[key]=(rank,item)
    selected=[v[1] for _,v in sorted(candidates.items())]
    if args.smoke:selected=selected[:1]
    print(source,region,':',len(items),'catalogue,',len(selected),'candidats retenus',flush=True)
    return selected

def raster_crop(item,key,source,grid,client):
    asset=item['assets'][key]; url=asset['href']
    if source=='landsat':
        parsed=urlparse(url)
        account=parsed.hostname.split('.')[0];container=parsed.path.split('/')[1]
        cache_key=(account,container)
        with SAS_LOCK:
            if cache_key not in SAS_CACHE:
                sign=client.get(f'https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}',timeout=30)
                sign.raise_for_status();SAS_CACHE[cache_key]=sign.json()['token']
            token=SAS_CACHE[cache_key]
        url=url+'?'+token
    qa=key in ['scl','qa_pixel','qa_radsat']
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif,.TIF',
                      GDAL_HTTP_CONNECTTIMEOUT=20,GDAL_HTTP_TIMEOUT=60,GDAL_HTTP_MAX_RETRY=2,GDAL_HTTP_RETRY_DELAY=1,
                      GDAL_HTTP_IPRESOLVE='V4',VSI_CACHE=True):
        with rasterio.open(url) as src:
            with WarpedVRT(src,crs='EPSG:32629',transform=grid['transform'],width=grid['width'],height=grid['height'],
                           src_nodata=src.nodata,nodata=0,resampling=Resampling.nearest if qa else Resampling.average,
                           dtype='uint16' if qa else 'float32') as warped:
                array=warped.read(1)
    metadata=asset.get('raster:bands',[{}])[0]
    if not qa:
        if 'scale' not in metadata:raise ValueError('Facteur physique manquant dans STAC : '+item['id']+' '+key)
        valid=array!=metadata.get('nodata',0)
        array=array*metadata['scale']+metadata.get('offset',0)
        array[~valid]=np.nan
    return array

def process(region,group,source,item):
    slug=region.replace(' ','_'); folder=RAW/source/slug;folder.mkdir(parents=True,exist_ok=True)
    dest=folder/(item['id']+'.npz'); stats=folder/(item['id']+'.json')
    if dest.exists() and stats.exists():
        prior=json.loads(stats.read_text())
        if prior.get('config',{}).get('processing_version')==CONFIG['processing_version'] and hashlib.sha256(dest.read_bytes()).hexdigest()==prior['sha256']:return prior['rows']
    projected=group.to_crs(32629)
    resolution=20 if source=='sentinel2' else 30
    xmin,ymin,xmax,ymax=projected.total_bounds
    xmin=math.floor(xmin/resolution)*resolution;ymax=math.ceil(ymax/resolution)*resolution
    grid=dict(transform=from_origin(xmin,ymax,resolution,resolution),width=math.ceil((xmax-xmin)/resolution),height=math.ceil((ymax-ymin)/resolution))
    client=session()
    keys=['scl','green','red','rededge1','nir','nir08'] if source=='sentinel2' else ['qa_pixel','qa_radsat','red','nir08']
    arrays={key:raster_crop(item,key,source,grid,client) for key in keys}
    if source=='sentinel2':valid=np.isin(arrays['scl'],CONFIG['qa_sentinel_classes'])
    else:
        valid=(arrays['qa_pixel'] & 63)==0
        # L8/L9 : saturation des seules bandes utilisées B4/B5 + occlusion terrain.
        valid &= (arrays['qa_radsat'] & ((1<<3)|(1<<4)|(1<<11)))==0
    for key in keys:
        if key not in ['scl','qa_pixel','qa_radsat']:valid &= np.isfinite(arrays[key])
    def ratio(a,b):
        den=a+b
        return np.divide(a-b,den,out=np.full_like(den,np.nan),where=np.abs(den)>1e-6)
    nir=arrays['nir'] if source=='sentinel2' else arrays['nir08'];red=arrays['red']
    indices={'ndvi':ratio(nir,red)}
    if source=='sentinel2':
        indices['ndre']=ratio(arrays['nir08'],arrays['rededge1'])
        indices['ndwi']=ratio(arrays['green'],nir)
        den=nir+red+.5
        indices['savi']=np.divide(1.5*(nir-red),den,out=np.full_like(den,np.nan),where=np.abs(den)>1e-6)
    for values in indices.values():valid &= np.isfinite(values)
    rows=[]
    for row in projected.itertuples():
        inside=geometry_mask([row.geometry.__geo_interface__],out_shape=valid.shape,transform=grid['transform'],invert=True)
        good=inside & valid;n=int(inside.sum());ng=int(good.sum())
        fraction=ng/n if n else 0
        result=dict(parcel_id=row.parcel_id,region=region,date=item['properties']['datetime'],source=source,
                    item_id=item['id'],scene_cloud_percent=item['properties'].get('eo:cloud_cover'),
                    n_pixels=n,n_valid_pixels=ng,valid_fraction=fraction,accepted=fraction>=CONFIG['min_valid_fraction'],
                    crop_observed_at=str(row.crop_observed_at)[:10],source_release_date='2025-07-14',stress_label=None)
        for name,array in indices.items():
            result[name]=float(np.mean(array[good])) if ng else None
        rows.append(result)
    temp=dest.with_suffix('.part.npz')
    np.savez_compressed(temp,**arrays,valid_mask=valid,transform=np.array(tuple(grid['transform'])),crs='EPSG:32629')
    temp.replace(dest)
    metadata={'item_id':item['id'],'source':source,'date':item['properties']['datetime'],
              'unsigned_assets':{k:item['assets'][k] for k in keys},'config':CONFIG,
              'file':str(dest.relative_to(ROOT)),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
              'grid':{'crs':'EPSG:32629','transform':list(grid['transform']),'width':grid['width'],'height':grid['height']},
              'scale_policy':'raster:bands scale et offset appliqués ; aucun offset supplémentaire fondé sur le nom',
              'rows':rows,'retrieved_utc':datetime.now(timezone.utc).isoformat()}
    stats.write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    print(source,region,item['id'],'OK',flush=True)
    return rows

def main():
    (META/'import_config.json').write_text(json.dumps(CONFIG,indent=2))
    tasks=[]
    for region,group in PILOT.groupby('zone'):
        for source in (['sentinel2','landsat'] if args.source=='all' else [args.source]):
            for item in search(region,group,source):tasks.append((region,group,source,item))
    errors=[];rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(process,*task):task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            task=futures[future]
            try:rows.extend(future.result())
            except Exception as exc:
                if isinstance(exc,TypeError):
                    import traceback
                    traceback.print_exc(limit=8)
                # Ne pas stocker de lien SAS dans le diagnostic.
                error=dict(source=task[2],region=task[0],item_id=task[3]['id'],error_type=type(exc).__name__)
                errors.append(error);print('ERREUR',error,flush=True)
    suffix='_smoke' if args.smoke else ''
    pd.DataFrame(rows).to_csv(OUT/f'satellite_acquisitions{suffix}_{args.source}.csv',index=False)
    (META/f'import_errors{suffix}_{args.source}.json').write_text(json.dumps(errors,indent=2))
    print('Terminé :',len(rows),'observations parcelle-date ;',len(errors),'erreurs',flush=True)
    if errors:raise SystemExit(2)
if __name__=='__main__':main()
