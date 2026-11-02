"""Audit du produit natif sans extraction et sans inventer de parcelle."""
from pathlib import Path
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import rasterio
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'data/metadata'
META.mkdir(parents=True, exist_ok=True)
report = []
for path in sorted((ROOT / 'data/raw/sentinel2').glob('*.zip')):
    print('Vérification :', path.name, flush=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f'CRC ZIP incorrect : {corrupt}')
        names = archive.namelist()
        metadata = next(n for n in names if n.endswith('/MTD_MSIL2A.xml'))
        xml = ET.fromstring(archive.read(metadata))
        fields = {}
        offsets = []
        for node in xml.iter():
            tag = node.tag.split('}')[-1]
            if tag in {'PRODUCT_START_TIME', 'PRODUCT_STOP_TIME', 'PROCESSING_LEVEL',
                       'PROCESSING_BASELINE', 'BOA_QUANTIFICATION_VALUE', 'SPACECRAFT_NAME'}:
                fields[tag] = node.text
            if tag == 'BOA_ADD_OFFSET':
                offsets.append({'band_id': node.attrib.get('band_id'), 'offset': float(node.text)})
        bands = []
        # Grille à 20 m commune pour les quatre indices et le masque SCL.
        for suffix in ['B03_20m.jp2', 'B04_20m.jp2', 'B05_20m.jp2',
                       'B08_10m.jp2', 'B8A_20m.jp2', 'SCL_20m.jp2']:
            matches = [n for n in names if n.endswith(suffix)]
            if not matches:
                raise ValueError(f'Bande requise absente : {suffix}')
            for name in matches:
                uri = '/vsizip/' + path.resolve().as_posix() + '/' + name
                with rasterio.open(uri) as src:
                    # Lecture effective d'une fenêtre, pas uniquement des en-têtes.
                    block = src.read(1, window=rasterio.windows.Window(0, 0, 64, 64))
                    bands.append(dict(file=name, shape=[src.height, src.width],
                                      dtype=src.dtypes[0], crs=str(src.crs),
                                      resolution=list(src.res), nodata=src.nodata,
                                      bounds_wgs84=list(transform_bounds(src.crs, 'EPSG:4326', *src.bounds)),
                                      sample_read_ok=block.size == 4096))
        item = dict(file=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                    sha256=digest.hexdigest(), published_checksum=None,
                    integrity='ZIP : tous les CRC valides ; XML lisible ; bandes requises lisibles',
                    archive_members=len(names), metadata=fields, boa_add_offsets=offsets,
                    bands=bands, parcel_geojson_available=bool(list((ROOT/'data/raw/maroc').glob('*.geojson'))),
                    note='Empreinte locale, non comparée à une empreinte fournisseur. Couverture = tuile, pas parcelle confirmée.')
        report.append(item)
        manifest_path = META / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
        manifest = [r for r in manifest if r.get('file') != item['file']]
        manifest.append({k: item[k] for k in ['file','bytes','sha256','published_checksum','integrity']} | {
            'source': 'Copernicus Data Space / Sentinel-2 L2A natif',
            'version': fields.get('PROCESSING_BASELINE'), 'url': 'https://dataspace.copernicus.eu/',
            'retrieved_utc': None, 'audited_utc': datetime.now(timezone.utc).isoformat()})
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({k: item[k] for k in ['bytes','integrity','metadata','parcel_geojson_available']}, ensure_ascii=False), flush=True)
(META/'sentinel_zip_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
print('Audit enregistré :', META/'sentinel_zip_audit.json')
