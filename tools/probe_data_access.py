"""Diagnostic borné des accès publics ; ne lit aucun identifiant utilisateur."""
import concurrent.futures
import requests
import socket
import urllib3.util.connection
urllib3.util.connection.allowed_gai_family=lambda:socket.AF_INET
TARGETS={
 'figshare_api_download':'https://api.figshare.com/v2/file/download/56267117',
 'figshare_file':'https://api.figshare.com/v2/articles/28486022/files/56267117',
 'earthsearch':'https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a',
 'planetary':'https://planetarycomputer.microsoft.com/api/stac/v1/collections/landsat-c2-l2',
 'cdse':'https://stac.dataspace.copernicus.eu/v1/collections/sentinel-2-l2a',
 'era5land_api':'https://archive-api.open-meteo.com/v1/archive?latitude=31.8&longitude=-7.4&start_date=2024-01-01&end_date=2024-01-02&daily=temperature_2m_mean&models=era5_land&timezone=UTC'
}
def probe(item):
    name,url=item
    try:
        r=requests.get(url,timeout=(8,20),allow_redirects=False)
        info=[name,r.status_code,r.headers.get('Content-Type'),len(r.content)]
        if r.status_code==200 and 'json' in r.headers.get('Content-Type',''):
            info.append(list(r.json())[:10])
        return info
    except Exception as e: return [name,type(e).__name__,str(e)[:130]]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    for result in pool.map(probe,TARGETS.items()):print(result,flush=True)
