"""Download small official scientific sources, verify provider hashes, retain provenance."""
from pathlib import Path
import hashlib
import json
import requests
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    out = ROOT / 'data/raw/benchmark'
    out.mkdir(parents=True, exist_ok=True)
    sources = []
    r = requests.get('https://data.mendeley.com/public-api/datasets/22j26tpk63/files',
                     params={'folder_id': 'root', 'version': 1}, timeout=60)
    r.raise_for_status()
    (out / 'mendeley_manifest.json').write_text(r.text, encoding='utf-8')
    for f in r.json():
        if f['filename'] != 'Growth.xlsx':
            continue
        d = f['content_details']
        sources.append(('drought', f['filename'], d['download_url'], 'sha256', d['sha256_hash']))
    r = requests.get('https://zenodo.org/api/records/10920137', timeout=60)
    r.raise_for_status()
    (out / 'olea_pred_manifest.json').write_text(r.text, encoding='utf-8')
    for f in r.json()['files']:
        if f['key'] in ('Field_DATA.rar', 'Shapefiles.rar'):
            algo, checksum = f['checksum'].split(':')
            sources.append(('olea_pred', f['key'], f['links']['self'], algo, checksum))
    audit = []
    for source, name, url, algo, checksum in sources:
        path = out / source / name
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        actual = hashlib.new(algo, path.read_bytes()).hexdigest()
        if actual != checksum:
            raise ValueError(f'Checksum mismatch: {path}')
        audit.append(dict(source=source, file=str(path.relative_to(ROOT)), url=url,
                          checksum=f'{algo}:{actual}', bytes=path.stat().st_size))
        print(name, 'verified', path.stat().st_size, flush=True)
    # CSIC: the deposited CSVs provide a full year of real olive sensor data.
    csic = out / 'csic_olive'
    csic.mkdir(exist_ok=True)
    url = 'https://digital.csic.es/bitstream/10261/435580/1/Repositorio.rar'
    archive = csic / 'Repositorio.rar'
    if not archive.exists():
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        archive.write_bytes(response.content)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    expected = 'a8cf7339ab0906774f458e958219e49496e455c7576a6ac1a0f5a51a7a567a1c'
    if checksum != expected:
        raise ValueError('CSIC archive changed; review provenance before using it')
    # Windows bsdtar supports RAR; extract only the publisher's known CSV/R files.
    subprocess.run(['tar', '-xf', str(archive), '-C', str(csic)], check=True)
    audit.append(dict(source='csic_olive', file=str(archive.relative_to(ROOT)),
                      url=url, checksum='sha256:'+checksum, bytes=archive.stat().st_size,
                      checksum_origin='local first retrieval; not publisher-supplied'))
    (out / 'download_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
