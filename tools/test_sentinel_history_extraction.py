"""Regression checks on controlled rasters, never added to scientific datasets."""
from pathlib import Path
import tempfile
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from extend_sentinel_history import extract, atomic_replace, ROOT
from unittest.mock import patch

def main():
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='qa_sentinel_') as temp:
        folder=Path(temp);assets={}
        source=folder/'new.part';target=folder/'current.csv'
        source.write_text('new');target.write_text('old')
        original=Path.replace
        calls=[]
        def transient_lock(self,destination):
            calls.append(1)
            if len(calls)<4:
                assert target.read_text()=='old'
                raise PermissionError('Simulated transient Windows sharing lock')
            return original(self,destination)
        with patch.object(Path,'replace',transient_lock):
            atomic_replace(source,target)
        assert len(calls)==4 and target.read_text()=='new' and not source.exists()
        for key,value,resolution,size in [('scl',4,20,10),('red',2000,10,20),('nir',5000,10,20)]:
            path=folder/f'{key}.tif'
            transform=from_origin(500000,3500000,resolution,resolution)
            with rasterio.open(path,'w',driver='GTiff',width=size,height=size,count=1,
                    dtype='uint16',crs='EPSG:32629',transform=transform,nodata=0) as ds:
                ds.write(np.full((size,size),value,dtype=np.uint16),1)
            calibration=dict(nodata=0,scale=.0001,offset=-.1) if key!='scl' else dict(nodata=0)
            assets[key]={'href':str(path),'raster:bands':[calibration],
                         'proj:transform':list(transform)[:6]}
        item=dict(id='S2A_TEST_CONTROLLED',assets=assets,
                  properties={'datetime':'2021-01-15T11:00:00Z','eo:cloud_cover':0})
        parcels=gpd.GeoDataFrame(dict(parcel_id=['interior','edge','outside'],zone=['test']*3,
            crop_observed_at=['2024-01-01']*3),
            geometry=[box(500020,3499920,500080,3499980),
                      box(500160,3499770,500220,3499830),
                      box(500250,3499750,500300,3499800)],crs=32629)
        raw=folder/'patches';raw.mkdir()
        cfg=dict(processing_hash='controlled_test',min_parcel_pixels=9,min_valid_fraction=.6)
        rows=extract(item,parcels,raw,cfg)
        assert rows[0]['accepted'] and abs(rows[0]['ndvi']-.6)<1e-6,rows[0]
        assert not rows[1]['accepted'] and not rows[2]['accepted']
        assert rows[1]['valid_fraction']<.6
        again=extract(item,parcels,raw,cfg)
        assert [r['ndvi'] for r in again[:1]]==[r['ndvi'] for r in rows[:1]]
        # A changed raw cache must never be silently reused.
        path=next(raw.rglob('*.npz'))
        with path.open('ab') as stream:stream.write(b'corrupt')
        try:extract(item,parcels,raw,cfg)
        except ValueError:pass
        else:raise AssertionError('Corrupted cache was reused')
    print('OK: Windows sharing-lock retries preserve old file; scale/offset, edge nodata, verified resume, corrupt cache rejection.')

if __name__=='__main__':main()
