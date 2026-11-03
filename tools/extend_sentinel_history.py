"""Monthly polygon Sentinel-2 history over all eligible documented olive parcels.

Public Earth Search C1 L2A; explicit local SCL QA, native B4/B8 reflectance,
monthly sampling (not temporal compositing). Resumable item caches and hashes.
No Earth Engine account, fabricated labels or filling of unavailable months.
"""
from pathlib import Path
import argparse
import concurrent.futures as cf
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import time
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import rasterio
from rasterio.windows import from_bounds, Window, transform as window_transform
from rasterio.features import geometry_mask
from rasterio.enums import Resampling
from shapely.geometry import shape

ROOT=Path(__file__).resolve().parents[1]
COLLECTION='sentinel-2-c1-l2a'
API='https://earth-search.aws.element84.com/v1/search'

def atomic_replace(temp,path):
    """Keep the old file intact while retrying transient Windows sharing locks."""
    for attempt in range(20):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt==19:
                raise
            time.sleep(min(.1*2**attempt,1.))

def atomic_json(path,value):
    temp=path.with_suffix('.part.json')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    atomic_replace(temp,path)

def client():
    s=requests.Session()
    s.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=.8,
            status_forcelist=[429,500,502,503,504])))
    return s

def catalogue(region,group,year,cache):
    dest=cache/f'{region.replace(" ","_")}_{year}.json.gz'
    if dest.exists():
        return json.loads(gzip.decompress(dest.read_bytes()))['features']
    params=dict(collections=COLLECTION,bbox=','.join(map(str,group.total_bounds)),
        datetime=f'{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z',limit=100)
    url=API;items=[];session=client();seen=set()
    while url:
        response=session.get(url,params=params,timeout=(20,90))
        response.raise_for_status();data=response.json()
        for item in data['features']:
            if item['id'] not in seen:
                # Retain provider band calibration and native raster grids.
                item['assets']={k:item['assets'][k] for k in ['scl','red','nir']}
                items.append(item);seen.add(item['id'])
        link=next((x for x in data.get('links',[]) if x['rel']=='next'),None)
        url=link['href'] if link else None;params=None
    packed=gzip.compress(json.dumps(dict(features=items)).encode())
    temp=dest.with_suffix('.part');temp.write_bytes(packed);atomic_replace(temp,dest)
    print('CATALOGUE',region,year,len(items),flush=True)
    return items

def extract(item,parcels,raw,config):
    name=item['id'];folder=raw/name[:12];folder.mkdir(exist_ok=True)
    metadata=folder/f'{name}.json';patches=folder/f'{name}.npz'
    retained_rows=[];retained_arrays={}
    if metadata.exists() and patches.exists():
        prior=json.loads(metadata.read_text(encoding='utf-8'))
        if prior.get('processing_hash')==config['processing_hash']:
            if hashlib.sha256(patches.read_bytes()).hexdigest()!=prior['sha256']:
                raise ValueError(f'Corrupt patch cache {name}')
            old={r['parcel_id']:r for r in prior['rows']}
            if set(parcels.parcel_id).issubset(old):
                return [old[p] for p in parcels.parcel_id]
            retained_rows=prior['rows']
            with np.load(patches) as saved:
                retained_arrays={k:saved[k] for k in saved.files}
    arrays={};rows=[]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif,.TIF',GDAL_HTTP_CONNECTTIMEOUT=20,
            GDAL_HTTP_TIMEOUT=60,GDAL_HTTP_MAX_RETRY=2,GDAL_HTTP_RETRY_DELAY=1,
            GDAL_HTTP_IPRESOLVE='V4',VSI_CACHE=True,GDAL_CACHEMAX=64_000_000):
        # Open SCL first: spectral reads are unnecessary for cloud-rejected parcels.
        with rasterio.open(item['assets']['scl']['href']) as qa:
            projected=parcels.to_crs(qa.crs)
            red=None;nir=None
            try:
                for parcel in projected.itertuples():
                    # B4/B8 are native 10m in the same UTM grid as SCL20m.
                    xmin,ymin,xmax,ymax=parcel.geometry.bounds
                    base=item['assets']['red']['proj:transform']
                    from affine import Affine
                    grid=Affine(*base)
                    w=from_bounds(xmin,ymin,xmax,ymax,grid)
                    x0,y0=math.floor(w.col_off),math.floor(w.row_off)
                    width,height=math.ceil(w.col_off+w.width)-x0,math.ceil(w.row_off+w.height)-y0
                    window=Window(x0,y0,width,height)
                    transform=window_transform(window,grid)
                    inside=geometry_mask([parcel.geometry.__geo_interface__],
                        out_shape=(height,width),transform=transform,invert=True)
                    n=int(inside.sum())
                    qwindow=from_bounds(*rasterio.windows.bounds(window,grid),qa.transform)
                    qa_outside=(qwindow.col_off<0 or qwindow.row_off<0 or
                        qwindow.col_off+qwindow.width>qa.width or qwindow.row_off+qwindow.height>qa.height)
                    scl=qa.read(1,window=qwindow,out_shape=(height,width),resampling=Resampling.nearest,
                                boundless=qa_outside,fill_value=0)
                    good=inside & np.isin(scl,[4,5])
                    fraction=float(good.sum()/n) if n else 0.
                    accepted=n>=config['min_parcel_pixels'] and fraction>=config['min_valid_fraction']
                    prefix='p'+str(parcel.Index)
                    arrays[prefix+'_scl']=scl
                    arrays[prefix+'_inside']=inside
                    arrays[prefix+'_transform']=np.array(tuple(transform))
                    row=dict(parcel_id=parcel.parcel_id,region=parcel.zone,
                        month=pd.Timestamp(item['properties']['datetime']).strftime('%Y-%m'),
                        acquisition_date=item['properties']['datetime'],item_id=name,
                        scene_cloud_percent=item['properties'].get('eo:cloud_cover'),
                        n_pixels=n,n_valid_pixels=int(good.sum()),valid_fraction=fraction,
                        accepted=False,status='too_few_pixels' if n<config['min_parcel_pixels'] else 'local_cloud_rejected',
                        red=np.nan,nir=np.nan,ndvi=np.nan,savi=np.nan,
                        crop_observed_at=str(parcel.crop_observed_at)[:10],
                        crop_stability_verified=False,stress_label=None,
                        patch_prefix=prefix,patch_file=str(patches.relative_to(ROOT)))
                    if accepted:
                        if red is None:
                            red=rasterio.open(item['assets']['red']['href'])
                            nir=rasterio.open(item['assets']['nir']['href'])
                        bands={}
                        for key,src in [('red',red),('nir',nir)]:
                            outside=(window.col_off<0 or window.row_off<0 or
                                window.col_off+window.width>src.width or window.row_off+window.height>src.height)
                            values=src.read(1,window=window,boundless=outside,fill_value=0)
                            calibration=item['assets'][key]['raster:bands'][0]
                            physical=values.astype(np.float32)*calibration['scale']+calibration.get('offset',0)
                            good &= (values!=calibration.get('nodata',0)) & np.isfinite(physical) & (physical>=0)
                            bands[key]=physical
                            arrays[prefix+'_'+key]=values
                        den=bands['nir']+bands['red']
                        good &= den>1e-6
                        fraction=float(good.sum()/n) if n else 0
                        row.update(n_valid_pixels=int(good.sum()),valid_fraction=fraction)
                        if fraction>=config['min_valid_fraction']:
                            index=(bands['nir'][good]-bands['red'][good])/den[good]
                            savi=1.5*(bands['nir'][good]-bands['red'][good])/(den[good]+.5)
                            row.update(accepted=True,status='valid',ndvi=float(index.mean()),
                                savi=float(savi.mean()),red=float(bands['red'][good].mean()),nir=float(bands['nir'][good].mean()))
                        else:
                            row['status']='reflectance_rejected'
                    arrays[prefix+'_valid']=good
                    rows.append(row)
            finally:
                if red is not None:red.close()
                if nir is not None:nir.close()
    retained_arrays.update(arrays)
    current={r['parcel_id']:r for r in retained_rows}
    current.update({r['parcel_id']:r for r in rows})
    temp=patches.with_suffix('.part.npz');np.savez_compressed(temp,**retained_arrays);atomic_replace(temp,patches)
    atomic_json(metadata,dict(item_id=name,processing_hash=config['processing_hash'],
        sha256=hashlib.sha256(patches.read_bytes()).hexdigest(),
        retrieved_utc=datetime.now(timezone.utc).isoformat(),crs=str(projected.crs),
        assets=item['assets'],rows=list(current.values())))
    return rows

def consolidate(parcels,months,attempts,out,config,errors,planned,done,complete):
    failures={(p,e['month']) for e in errors for p in e.get('parcels',[])}
    attempts=list({(r['parcel_id'],r['item_id']):r for r in attempts}.values())
    best={}
    for row in attempts:
        key=(row['parcel_id'],row['month'])
        rank=(not row['accepted'],row['scene_cloud_percent'],row['acquisition_date'],row['item_id'])
        if key not in best or rank<best[key][0]:best[key]=(rank,row)
    records=[]
    for p in parcels.itertuples():
        for month in months:
            key=(p.parcel_id,month)
            row=dict(parcel_id=p.parcel_id,region=p.zone,month=month,
                acquisition_date=None,item_id=None,accepted=False,
                status='no_catalogue_candidate' if complete else 'pending_or_no_candidate',
                ndvi=np.nan,savi=np.nan,red=np.nan,nir=np.nan,
                crop_observed_at=str(p.crop_observed_at)[:10],crop_stability_verified=False,stress_label=None)
            if key in best:row.update(best[key][1])
            row['download_error']=key in failures
            if key in failures and key not in best:row['status']='download_failed'
            records.append(row)
    frame=pd.DataFrame(records).sort_values(['parcel_id','month'])
    frame.loc[~frame.accepted,['ndvi','savi','red','nir']]=np.nan
    temp=out/'monthly_history.part.csv';frame.to_csv(temp,index=False);atomic_replace(temp,out/'monthly_history.csv')
    counts=frame.assign(year=frame.month.str[:4]).groupby(['region','year']).agg(
        parcel_months=('parcel_id','size'),accepted=('accepted','sum')).reset_index()
    counts.to_csv(out/'coverage_by_region_year.csv',index=False)
    status=dict(complete=complete,parcels=len(parcels),months=len(months),
        calendar_rows=len(frame),accepted=int(frame.accepted.sum()),
        attempted_parcel_scenes=len(attempts),planned_item_jobs=planned,completed_item_jobs=done,
        errors=len(errors),first=frame.loc[frame.accepted,'acquisition_date'].min() if frame.accepted.any() else None,
        last=frame.loc[frame.accepted,'acquisition_date'].max() if frame.accepted.any() else None,
        config=config,updated_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(out/'status.json',status)
    atomic_json(out/'errors.json',errors)
    return frame,status

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--start-year',type=int,default=2021)
    parser.add_argument('--end-year',type=int,default=2025)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--executor',choices=['threads','processes'],default='processes')
    parser.add_argument('--catalogue-only',action='store_true')
    parser.add_argument('--max-items',type=int)
    parser.add_argument('--candidates',type=int,default=2)
    args=parser.parse_args()
    root=ROOT/f'data/raw/sentinel_history_{args.start_year}_{args.end_year}'
    cache=root/'catalogues';raw=root/'patches'
    out=ROOT/f'data/processed/sentinel_history_{args.start_year}_{args.end_year}'
    for p in [cache,raw,out]:p.mkdir(parents=True,exist_ok=True)
    input_file=ROOT/'data/raw/sentinel_history_2021_2025/provenance/oliveraies_reference.geojson'
    parcels=gpd.read_file(input_file)
    parcels=parcels[parcels.geometry_valid.eq(True)&parcels.mixed_crop.eq(False)].copy().reset_index(drop=True)
    assert parcels.geometry.is_valid.all() and parcels.parcel_id.is_unique
    config=dict(version=1,collection=COLLECTION,start_year=args.start_year,end_year=args.end_year,
        frequency='monthly',max_scene_cloud=40,min_valid_fraction=.6,min_parcel_pixels=9,
        candidates_per_parcel_month=args.candidates,qa_classes=[4,5],bands=['B4','B8'],resolution_m=10,
        indices=['NDVI=(B8-B4)/(B8+B4)','SAVI=1.5*(B8-B4)/(B8+B4+0.5)'],
        selection='Minimum scene cloud covering complete parcel, local QA fallback to next ranked scene; no temporal composite.',
        parcels_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest(),
        crop_evidence='DOI 10.6084/m9.figshare.28486022.v2; crop only confirmed at field visit.',
        limitations=['Pre-visit and post-visit olive continuity not verified.','No physiological stress labels.',
                    'Monthly scenes represent different days; sampling, not a monthly average.',
                    'Catalogue gaps kept; failed downloads distinct from missing acquisitions.'])
    config['processing_hash']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    atomic_json(out/'config.json',config)
    parcels.to_file(out/'parcels.geojson',driver='GeoJSON')
    months=[f'{y}-{m:02}' for y in range(args.start_year,args.end_year+1) for m in range(1,13)]
    jobs=[(region,group,year) for region,group in parcels.groupby('zone')
          for year in range(args.start_year,args.end_year+1)]
    items={};catalogue_errors=[]
    with cf.ThreadPoolExecutor(max_workers=min(4,args.workers)) as pool:
        futures={pool.submit(catalogue,r,g,y,cache):(r,y) for r,g,y in jobs}
        for f in cf.as_completed(futures):
            try:
                for item in f.result():items[item['id']]=item
            except Exception as exc:
                catalogue_errors.append(dict(region=futures[f][0],year=futures[f][1],error=type(exc).__name__,stage='catalogue'))
    atomic_json(out/'catalogue_errors.json',catalogue_errors)
    if catalogue_errors:
        raise RuntimeError(f'{len(catalogue_errors)} catalogue jobs failed; rerun to resume')
    usable=[]
    for item in items.values():
        if item['properties'].get('eo:cloud_cover',100) is not None and item['properties'].get('eo:cloud_cover',100)<=40:
            usable.append(item)
    bymonth={m:[] for m in months}
    for item in usable:
        month=pd.Timestamp(item['properties']['datetime']).strftime('%Y-%m')
        bymonth[month].append(item)
    rounds=[{} for _ in range(args.candidates)]
    for month,candidates in bymonth.items():
        candidates.sort(key=lambda it:(it['properties']['eo:cloud_cover'],it['properties']['datetime'],it['id']))
        footprints={it['id']:shape(it['geometry']) for it in candidates}
        for p in parcels.itertuples():
            selected=[it for it in candidates if footprints[it['id']].covers(p.geometry)][:args.candidates]
            for rank,it in enumerate(selected):rounds[rank].setdefault(it['id'],[]).append(p.Index)
    plan=[dict(rank=i+1,item_id=k,parcels=len(v)) for i,round_ in enumerate(rounds) for k,v in round_.items()]
    pd.DataFrame(plan).to_csv(out/'extraction_plan.csv',index=False)
    print('PLAN',len(parcels),'parcels',len(months),'months',len(plan),'item jobs',len(items),'catalogue scenes',flush=True)
    if args.catalogue_only:return
    attempts=[];errors=[];accepted=set();done=0;planned=len(plan);limited=False
    # Restore all verified observations before scheduling, so a restart never
    # temporarily drops earlier results or redownloads an already accepted month.
    for metadata in raw.rglob('*.json'):
        saved=json.loads(metadata.read_text(encoding='utf-8'))
        if saved.get('processing_hash')!=config['processing_hash']:continue
        patch=metadata.with_suffix('.npz')
        if hashlib.sha256(patch.read_bytes()).hexdigest()!=saved['sha256']:
            raise ValueError(f'Corrupt patch cache: {patch}')
        attempts.extend(saved['rows'])
    accepted.update((r['parcel_id'],r['month']) for r in attempts if r['accepted'])
    if (out/'errors.json').exists():
        previous=json.loads((out/'errors.json').read_text(encoding='utf-8'))
        if previous:
            history=json.loads((out/'error_history.json').read_text(encoding='utf-8')) if (out/'error_history.json').exists() else []
            history.append(dict(retried_utc=datetime.now(timezone.utc).isoformat(),errors=previous))
            atomic_json(out/'error_history.json',history)
    print('RESUME',len(accepted),'verified accepted parcel-months',flush=True)
    consolidate(parcels,months,attempts,out,config,errors,planned,done,False)
    for rank,round_ in enumerate(rounds):
        tasks=[]
        for item_id,ids in round_.items():
            month=pd.Timestamp(items[item_id]['properties']['datetime']).strftime('%Y-%m')
            ids=[i for i in ids if (parcels.iloc[i].parcel_id,month) not in accepted]
            if ids:tasks.append((items[item_id],parcels.iloc[ids]))
        if args.max_items is not None and len(tasks)>max(0,args.max_items-done):
            tasks=tasks[:max(0,args.max_items-done)];limited=True
        executor=cf.ProcessPoolExecutor if args.executor=='processes' else cf.ThreadPoolExecutor
        with executor(max_workers=args.workers) as pool:
            futures={pool.submit(extract,it,g,raw,config):(it,g) for it,g in tasks}
            for f in cf.as_completed(futures):
                item,group=futures[f]
                try:
                    result=f.result();attempts.extend(result)
                    accepted.update((r['parcel_id'],r['month']) for r in result if r['accepted'])
                except Exception as exc:
                    errors.append(dict(item_id=item['id'],error=type(exc).__name__,message=str(exc).split('?')[0][:200],stage='raster',
                        month=pd.Timestamp(item['properties']['datetime']).strftime('%Y-%m'),parcels=group.parcel_id.tolist()))
                done+=1
                if done%10==0:
                    consolidate(parcels,months,attempts,out,config,errors,planned,done,False)
                print('PROGRESS',done,'/',planned,'accepted',len(accepted),'errors',len(errors),flush=True)
    frame,status=consolidate(parcels,months,attempts,out,config,errors,planned,done,not limited and not errors)
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)
    if errors:raise SystemExit(2)

if __name__=='__main__':main()
