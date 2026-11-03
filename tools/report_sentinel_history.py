"""Verify extracted history, quantify real coverage and generate scientific QA plots."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/sentinel_history_2021_2025'
RAW=ROOT/'data/raw/sentinel_history_2021_2025'
OUT=ROOT/'outputs/sentinel_history_2021_2025'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    status=json.loads((DATA/'status.json').read_text(encoding='utf-8'))
    cfg=status['config']
    d=pd.read_csv(DATA/'monthly_history.csv')
    assert not d.duplicated(['parcel_id','month']).any()
    assert len(d)==status['parcels']*status['months']
    good=d[d.accepted.eq(True)].copy()
    assert len(good)==status['accepted']
    assert good.ndvi.between(-1,1).all()
    assert good[['ndvi','savi','red','nir']].notna().all().all()
    assert good.valid_fraction.ge(cfg['min_valid_fraction']).all()
    assert good.n_pixels.ge(cfg['min_parcel_pixels']).all()
    assert d.loc[~d.accepted,['ndvi','savi','red','nir']].isna().all().all()
    assert d.stress_label.isna().all()
    assert not d.crop_stability_verified.any()
    assert pd.to_datetime(good.acquisition_date,utc=True,format='ISO8601').dt.strftime('%Y-%m').eq(good.month).all()
    files=[];item_rows={};item_meta={}
    for meta in (RAW/'patches').rglob('*.json'):
        info=json.loads(meta.read_text(encoding='utf-8'))
        if info['processing_hash']!=cfg['processing_hash']:continue
        npz=meta.with_suffix('.npz')
        digest=hashlib.sha256(npz.read_bytes()).hexdigest()
        assert digest==info['sha256'],str(npz)
        item_rows[info['item_id']]={r['parcel_id']:r for r in info['rows']}
        item_meta[info['item_id']]=info
        files.append(dict(file=str(npz.relative_to(ROOT)),sha256=digest,
                          item_id=info['item_id'],retrieved_utc=info['retrieved_utc']))
    for row in good.itertuples():
        original=item_rows[row.item_id][row.parcel_id]
        assert original['accepted'] and abs(original['ndvi']-row.ndvi)<1e-6
    # Recompute the physical pixel index from retained DNs, scale/offset and QA.
    sample=good.sample(n=min(32,len(good)),random_state=42)
    for row in sample.itertuples():
        meta=item_meta[row.item_id];prefix=row.patch_prefix
        with np.load(ROOT/row.patch_file) as patch:
            bands={}
            for key in ['red','nir']:
                calibration=meta['assets'][key]['raster:bands'][0]
                bands[key]=patch[prefix+'_'+key].astype(np.float32)*calibration['scale']+calibration.get('offset',0)
            mask=patch[prefix+'_valid']
            recomputed=((bands['nir'][mask]-bands['red'][mask])/(bands['nir'][mask]+bands['red'][mask])).mean()
            assert abs(float(recomputed)-row.ndvi)<1e-6
    parcels=gpd.read_file(DATA/'parcels.geojson')
    assert len(parcels)==status['parcels'] and parcels.geometry.is_valid.all()
    assert not parcels.mixed_crop.any()
    assert not good.empty
    region_sizes=parcels.groupby('zone').size().sort_values(ascending=False)
    projected=parcels.to_crs(32629)
    pairs=projected.sindex.query(projected.geometry,predicate='intersects')
    overlaps=[]
    parent=list(range(len(projected)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,j in zip(*pairs):
        if i>=j:continue
        first,second=projected.geometry.iloc[i],projected.geometry.iloc[j]
        shared=first.intersection(second).area
        if shared<=0:continue
        union=first.area+second.area-shared
        iou=shared/union
        containment=shared/min(first.area,second.area)
        if iou>=.8 or containment>=.8:
            overlaps.append(dict(first=parcels.parcel_id.iloc[i],second=parcels.parcel_id.iloc[j],
                                 intersection_over_union=iou,smaller_polygon_overlap=containment))
            parent[find(i)]=find(j)
    pd.DataFrame(overlaps,columns=['first','second','intersection_over_union','smaller_polygon_overlap']).to_csv(OUT/'strong_polygon_overlaps.csv',index=False)
    spatial_groups=pd.DataFrame(dict(parcel_id=parcels.parcel_id,
        spatial_group=[parcels.parcel_id.iloc[find(i)] for i in range(len(parcels))]))
    spatial_groups.to_csv(OUT/'spatial_groups.csv',index=False)
    d['year']=d.month.str[:4]
    summary=d.groupby('year').agg(calendar=('parcel_id','size'),valid=('accepted','sum'))
    summary['coverage_percent']=100*summary.valid/summary.calendar
    summary.to_csv(OUT/'coverage_by_year.csv')
    byparcel=d.groupby(['parcel_id','region']).agg(valid=('accepted','sum'),calendar=('month','size')).reset_index()
    byparcel['coverage_percent']=100*byparcel.valid/byparcel.calendar
    byparcel.to_csv(OUT/'coverage_by_parcel.csv',index=False)
    matrix=d.groupby(['region','year']).accepted.mean().unstack()*100
    fig,ax=plt.subplots(figsize=(9,4))
    im=ax.imshow(matrix,vmin=0,vmax=100,cmap='YlGn')
    ax.set_xticks(range(len(matrix.columns)),matrix.columns)
    ax.set_yticks(range(len(matrix.index)),matrix.index)
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):ax.text(j,i,f'{matrix.iloc[i,j]:.0f}%',ha='center',va='center')
    ax.set_title('Sentinel-2: actual monthly coverage after local cloud QA')
    fig.colorbar(im,ax=ax,label='Valid parcel-months (%)')
    fig.tight_layout();fig.savefig(OUT/'coverage.png',dpi=160);plt.close(fig)
    top=byparcel.sort_values('valid',ascending=False).groupby('region').head(1).parcel_id
    fig,axes=plt.subplots(3,2,figsize=(13,9),sharex=True,sharey=True)
    for ax,pid in zip(axes.flat,top):
        g=d[d.parcel_id.eq(pid)].sort_values('month')
        ax.plot(pd.to_datetime(g.month),g.ndvi,marker='.',linewidth=1)
        ax.set_title(str(g.region.iloc[0])+' | '+pid.rsplit('_',1)[-1])
        ax.set_ylim(-.1,1);ax.set_ylabel('NDVI');ax.grid(alpha=.2)
        ax.set_xlim(pd.Timestamp('2021-01-01'),pd.Timestamp('2025-12-31'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.suptitle('Illustrative best-covered parcel per region; gaps kept, no interpolation')
    fig.tight_layout();fig.savefig(OUT/'ndvi_examples.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    parcels.plot(ax=ax,column='zone',legend=True,cmap='tab10',markersize=3)
    ax.set(title='Documented olive polygons — 563 eligible parcels',xlabel='Longitude',ylabel='Latitude')
    fig.tight_layout();fig.savefig(OUT/'parcels_map.png',dpi=160);plt.close(fig)
    # Feasibility for strictly-contained windows: six input months -> next month.
    months=sorted(d.month.unique());a,b=int(.7*len(months)),int(.85*len(months))
    counts=[]
    for pid,g in d.groupby('parcel_id'):
        g=g.sort_values('month').reset_index(drop=True)
        for split,start,end in [('train',0,a),('validation',a,b),('test',b,len(months))]:
            valid=g.iloc[start:end].accepted.to_numpy(dtype=bool)
            n=sum(valid[i-6:i+1].all() for i in range(6,len(valid)))
            counts.append(dict(parcel_id=pid,split=split,valid_windows=int(n)))
    counts=pd.DataFrame(counts)
    counts.to_csv(OUT/'six_month_windows.csv',index=False)
    pd.DataFrame(files).to_csv(OUT/'verified_patch_manifest.csv',index=False)
    reference=ROOT/'data/processed/olive_reference_series_2024.csv'
    if reference.exists():
        published=pd.read_csv(reference)
        published['day']=pd.to_datetime(published.date).dt.strftime('%Y-%m-%d')
        extracted=good.copy()
        extracted['day']=pd.to_datetime(extracted.acquisition_date,utc=True,format='ISO8601').dt.strftime('%Y-%m-%d')
        matched=extracted.merge(published[['parcel_id','day','NDVI']],on=['parcel_id','day'],validate='one_to_one')
    else:
        # Retain the completed processing audit after deleting the old dataset.
        matched=pd.read_csv(OUT/'published_2024_consistency.csv')
    matched['absolute_ndvi_difference']=(matched.ndvi-matched.NDVI).abs()
    matched[['parcel_id','day','ndvi','NDVI','absolute_ndvi_difference']].to_csv(OUT/'published_2024_consistency.csv',index=False)
    consistency=dict(matched_parcel_dates=len(matched),
        mean_absolute_difference=float(matched.absolute_ndvi_difference.mean()),
        median_absolute_difference=float(matched.absolute_ndvi_difference.median()),
        percentile95_difference=float(matched.absolute_ndvi_difference.quantile(.95)),
        note='Processing consistency at same parcel/date; not stress detection accuracy or independent ground truth.')
    manifest=dict(status=status,calendar_sha256=hashlib.sha256((DATA/'monthly_history.csv').read_bytes()).hexdigest(),
        verified_patch_files=len(files),parcels_with_valid_observations=int(good.parcel_id.nunique()),
        six_month_windows=counts.groupby('split').valid_windows.sum().to_dict(),
        split_months=dict(train=months[:a],validation=months[a:b],test=months[b:]),
        recomputed_pixel_checks=len(sample),
        strong_overlap_pairs=len(overlaps),spatial_groups=int(spatial_groups.spatial_group.nunique()),
        published_2024_consistency=consistency,
        note='Window counts precede any independent-parcel holdout; all seven months must be observed.')
    (OUT/'verification.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    def md_table(frame):
        frame=frame.reset_index()
        return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None))
    text=f'''# Historique Sentinel-2 étendu des oliveraies

État : **{'extraction terminée' if status['complete'] else 'extraction partielle / en cours'}**.

Période demandée : janvier 2021 – décembre 2025. **{status['parcels']} parcelles**, {status['months']} mois, {len(d):,} emplacements calendaires. **{len(good):,} observations acceptées** sur **{good.parcel_id.nunique()} parcelles** après contrôle local. Les emplacements calendaires ne sont pas des mesures. {len(files)} fichiers de découpes vérifiés par SHA-256.

## Couverture réelle

{md_table(summary.round(2))}

![Couverture réelle](coverage.png)

![Exemples NDVI](ndvi_examples.png)

## Méthode

Polygones complets d'oliveraies documentées ; géométries valides, absence de culture intercalaire déclarée. Source terrain : https://doi.org/10.6084/m9.figshare.28486022.v2 .

Sentinel-2 Collection 1 L2A via Earth Search v1, collection `{cfg['collection']}`. Pour chaque parcelle et mois, scènes couvrant le polygone complet et au plus 40 % de nuages de scène, classées par nuages puis date et identifiant. Jusqu'à deux candidats : le second est essayé si le premier échoue au contrôle local ou à la lecture. Ce sont des observations mensuelles sélectionnées, pas des composites ni des moyennes mensuelles de toutes les scènes.

Bandes B4 et B8 natives à 10 m, conversion en réflectance par les `scale`/`offset` STAC. SCL à 20 m rééchantillonnée au plus proche à 10 m. Seules les classes 4 et 5 sont retenues. Au moins neuf centres de pixels dans le polygone et au moins 60 % de pixels localement valides après masque de nuages et contrôle de réflectance. NDVI et SAVI sont calculés par pixel, puis moyennés sur le polygone. Les petits polygones et bords restent sensibles aux pixels mixtes.

Les dates d'acquisition, produits, fractions valides, raisons de rejet et références des découpes restent dans le CSV. Les mois sans mesure valide restent NaN. Aucune interpolation, aucun label de stress créé. Les erreurs de la dernière exécution figurent dans `errors.json`, celles d'exécutions reprises dans `error_history.json` ; elles ne doivent pas être confondues avec une absence de scène.

Contrôle de cohérence avec les séries publiées en 2024, sur exactement les mêmes parcelles et jours : **{len(matched)} correspondances**, écart absolu moyen de NDVI **{consistency['mean_absolute_difference']:.5f}**, médiane **{consistency['median_absolute_difference']:.5f}**, 95e percentile **{consistency['percentile95_difference']:.5f}**. Les deux traitements ne sont pas nécessairement identiques (grille, QA et agrégation). Ce contrôle vérifie la cohérence de traitement ; il n'évalue pas la détection de stress et ne constitue pas une validation terrain indépendante.

## Utilisation pour le sujet

Ce jeu sert directement à l'analyse satellitaire multi-temporelle : trajectoires NDVI, saisonnalité, prévision du prochain NDVI et changements de végétation. Il remplace la profondeur d'une seule année par une grille de cinq années pour de nombreuses parcelles.

Pour une comparaison séquentielle stricte : six mois d'entrée, prévision du mois suivant ; mêmes origines pour Random Forest, XGBoost, LightGBM, LSTM, GRU et CNN-LSTM. Le split 70/15/15 de la grille donne train janvier 2021–juin 2024, validation juillet 2024–mars 2025, test avril–décembre 2025. Fenêtres entièrement dans chaque bloc, sans combler les trous. Disponibilité brute avant exclusion de parcelles indépendantes : **{manifest['six_month_windows']}**. Une fenêtre sur douze mois ne tient pas entièrement dans les blocs validation/test de neuf mois ; utiliser six mois ou définir explicitement une évaluation avec contexte passé causal.

Les parcelles réservées à l'évaluation spatiale doivent être exclues du train et de la validation ; leurs résultats doivent être présentés à part. Les limites d'exploitation et bassins ne sont pas connus, donc des parcelles distinctes ne garantissent pas une indépendance agronomique.

Audit géométrique : **{len(overlaps)} paires de polygones fortement superposés** (IoU ou fraction de recouvrement du petit polygone ≥ 80 %), regroupées en **{spatial_groups.spatial_group.nunique()} groupes spatiaux**. Pour les splits, conserver les membres d'un même groupe ensemble et ne pas compter des polygones superposés comme observations indépendantes. Voir `strong_polygon_overlaps.csv` et `spatial_groups.csv`. Ce regroupement n'identifie pas les exploitations.

## Limites

La culture a été observée aux dates de visite indiquées dans les métadonnées, pas continuellement entre 2021 et 2025. Vérifier plantations, arrachages et changements de culture avant d'interpréter tout l'historique comme une oliveraie stable. Les trous de catalogue, notamment en 2022, restent visibles. Un changement de NDVI ne prouve pas un stress hydrique ; cette attribution demande des observations terrain sur les mêmes parcelles et dates. Le NDVI est un indice de végétation et non une mesure directe de santé. Les séries publiées de 2024 ne sont pas empilées comme exemples indépendants avec ces nouvelles extractions.

L'échantillon est déséquilibré : **{int(region_sizes.iloc[0])}/{len(parcels)} polygones dans {region_sizes.index[0]}**. Il ne représente pas toutes les oliveraies marocaines. Rapporter aussi les performances par région et par groupe spatial, sans assimiler des polygones voisins à des exploitations indépendantes.

## Fichiers et reproduction

- `data/processed/sentinel_history_2021_2025/monthly_history.csv` : historique et calendrier.
- `data/processed/sentinel_history_2021_2025/parcels.geojson` : polygones retenus.
- `data/processed/sentinel_history_2021_2025/status.json` : état réel de l'import.
- `data/raw/sentinel_history_2021_2025/` : catalogues, découpes et calibration/provenance.
- `outputs/sentinel_history_2021_2025/` : couverture, graphiques, fenêtres et manifeste vérifié.

```powershell
.venv/Scripts/python.exe tools/extend_sentinel_history.py --workers 8
.venv/Scripts/python.exe tools/report_sentinel_history.py
```

L'import reprend uniquement les découpes de la bonne version avec empreinte correcte. Les notebooks 01/02 et leurs résultats antérieurs ne sont pas remplacés. Aucun entraînement n'est réalisé par cette extension.
'''
    (OUT/'RAPPORT.md').write_text(text,encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['verified_patch_files','parcels_with_valid_observations','six_month_windows']},indent=2))
    print(summary.to_string())

if __name__=='__main__':main()
