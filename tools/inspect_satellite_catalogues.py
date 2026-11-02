from pathlib import Path
import json
import requests
import geopandas as gpd
ROOT=Path(__file__).resolve().parents[1]
p=gpd.read_file(ROOT/'data/raw/maroc/oliveraies_pilote.geojson')
bbox=p[p.zone.eq('el haouz')].total_bounds.tolist()
targets={
 'sentinel2':('https://earth-search.aws.element84.com/v1/search','sentinel-2-l2a'),
 'landsat':('https://planetarycomputer.microsoft.com/api/stac/v1/search','landsat-c2-l2')}
for name,(url,col) in targets.items():
    r=requests.get(url,params={'collections':col,'bbox':','.join(map(str,bbox)),
                             'datetime':'2024-06-01T00:00:00Z/2024-06-30T23:59:59Z','limit':2},timeout=45)
    r.raise_for_status(); d=r.json()
    (ROOT/f'data/metadata/{name}_public_sample.json').write_text(json.dumps(d,indent=2))
    item=d['features'][0]
    print(name,item['id'],flush=True)
    print(json.dumps({k:v for k,v in item['properties'].items() if any(s in k for s in ['baseline','offset','cloud','datetime','reflectance'])},indent=2),flush=True)
    print(json.dumps({k:{field:v[field] for field in ['href','raster:bands','bands'] if field in v} for k,v in item['assets'].items() if k in ['green','red','rededge1','nir','nir08','nir09','scl','qa_pixel','qa_radsat','B03','B04','B05','B08','B8A']},indent=2),flush=True)
