"""Import reproductible de la référence terrain publiée (culture, pas stress)."""
from pathlib import Path
import hashlib
import json
import requests
import subprocess
import socket
import urllib3.util.connection
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/morocco_reference'; RAW.mkdir(parents=True,exist_ok=True)
META=ROOT/'data/metadata'; META.mkdir(parents=True,exist_ok=True)
URL='https://api.figshare.com/v2/articles/28486022/versions/2'
urllib3.util.connection.allowed_gai_family=lambda: socket.AF_INET
session=requests.Session()
session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
response=session.get(URL,timeout=60); response.raise_for_status(); article=response.json()
(META/'morocco_reference_figshare_v2.json').write_text(json.dumps(article,indent=2),encoding='utf-8')
manifest=[]
for f in sorted(article['files'],key=lambda f:f['size']):
    if f['size']>100_000_000: continue  # Photos 17 Go : pas nécessaires à cet import des polygones.
    target=RAW/('parcels.rar' if f['id']==56267117 else 'sentinel2_2024.rar')
    part=target.with_suffix('.part')
    if not target.exists():
        offset=part.stat().st_size if part.exists() else 0
        download_url=f"https://api.figshare.com/v2/file/download/{f['id']}"
        with session.get(download_url,stream=True,headers={'Range':f'bytes={offset}-'} if offset else {},timeout=(15,90)) as r:
            r.raise_for_status()
            mode='ab' if offset and r.status_code==206 else 'wb'
            with part.open(mode) as stream:
                for block in r.iter_content(1024*1024): stream.write(block)
        assert part.stat().st_size==f['size'],'Taille incorrecte'
        part.replace(target)
    assert target.stat().st_size==f['size']
    md5=hashlib.md5(target.read_bytes()).hexdigest()
    assert md5==f['computed_md5'],'MD5 fournisseur incorrect'
    print(target.name,target.stat().st_size,'MD5 fournisseur OK',flush=True)
    manifest.append(dict(source=article['doi'],version=2,url=f['download_url'],original_name=f['name'],file=str(target.relative_to(ROOT)),bytes=target.stat().st_size,md5=md5,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),license=article['license'],integrity='Taille et MD5 fournisseur vérifiés',retrieved_utc=datetime.now(timezone.utc).isoformat()))
    listing=subprocess.run(['tar','-tf',str(target)],capture_output=True,text=True,check=True)
    print(listing.stdout[:2500],flush=True)
    # Contrôle des noms avant extraction, aucune traversée hors destination.
    for name in listing.stdout.splitlines():
        dest=(RAW/name).resolve()
        assert dest.is_relative_to(RAW.resolve()),'Chemin archive non sûr'
    subprocess.run(['tar','-xf',str(target),'-C',str(RAW)],check=True)
(META/'morocco_reference_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print('Import référence terminé',flush=True)
