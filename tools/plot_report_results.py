from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--run',default='rapport_exploratoire_20261102')
args=parser.parse_args()
assert args.run and all(c.isalnum() or c in '_-' for c in args.run), 'Nom d’expérience invalide'
RUN=ROOT/'outputs/experiments'/args.run
TMP=ROOT/'tmp/pdfs/resultats'/args.run;TMP.mkdir(parents=True,exist_ok=True)
comparison=pd.read_csv(RUN/'comparison.csv')
names={'zscore':'Z-score NDVI','isolation':'Isolation Forest','svm':'One-Class SVM','autoencoder':'Autoencoder'}
feature_names={'ndvi':'NDVI','indices':'4 indices','meteo':'Indices + météo'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
plt.style.use('default')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
fig,ax=plt.subplots(figsize=(9,3.2))
rates=[100*comparison.loc[comparison.model.eq(model)&comparison.ablation.eq('ndvi'),'alert_rate'].iloc[0] for model in names]
ax.barh(list(names.values()),rates,color=['#839b62','#426e55','#6e83a4','#ad8658'])
for i,value in enumerate(rates):ax.text(value+.3,i,f'{value:.1f} %'.replace('.',','),va='center')
ax.set_xlim(0,max(5,max(rates)*1.2+1));ax.set_xlabel('Observations de test avec alerte (%) - NDVI seul')
ax.invert_yaxis();fig.tight_layout();fig.savefig(TMP/'alert_rates.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(10,2.65))
for ax,variant in zip(axes,['ndvi','indices','meteo']):
    history=pd.read_csv(RUN/f'{variant}_autoencoder_learning.csv')
    ax.plot(history.epoch+1,history.train,label='Train',color='#426e55')
    ax.plot(history.epoch+1,history.validation,label='Validation',color='#ad8658')
    ax.set_title(feature_names[variant]);ax.set_xlabel('Époque');ax.set_ylabel('MSE de reconstruction');ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(TMP/'ae_learning.png',dpi=180);plt.close(fig)
pilot=gpd.read_file(ROOT/'data/raw/maroc/oliveraies_pilote.geojson')
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,(region,group) in zip(axes,pilot.groupby('zone')):
    group.plot(ax=ax,facecolor='#91ae70',edgecolor='#35552e')
    for parcel in group.itertuples():
        point=parcel.geometry.representative_point()
        ax.annotate(parcel.parcel_id.rsplit('_',1)[-1],(point.x,point.y),fontsize=9)
    ax.set_title(region+' - parcelles publiées');ax.set_xlabel('Longitude');ax.set_ylabel('Latitude')
    ax.tick_params(labelsize=8)
fig.tight_layout();fig.savefig(TMP/'pilot_map.png',dpi=180);plt.close(fig)
series=pd.read_csv(ROOT/'data/processed/pilot/sentinel2_10day_with_weather.csv',parse_dates=['date'])
fig,axes=plt.subplots(2,1,figsize=(11,5),sharex=True)
for ax,(region,group) in zip(axes,pilot.groupby('zone')):
    for parcel in group.parcel_id:
        values=series[series.parcel_id.eq(parcel)]
        ax.plot(values.date,values.ndvi,marker='.',markersize=2,linewidth=.8,label=parcel.rsplit('_',1)[-1])
    ax.set_title(region+' - NDVI, trous conservés');ax.set_ylabel('NDVI');ax.set_ylim(-.1,1)
    ax.legend(title='Parcelle',fontsize=8,title_fontsize=8,loc='upper right')
axes[-1].set_xlabel('Date de fin de fenêtre (exclusive)')
fig.tight_layout();fig.savefig(TMP/'ndvi_series.png',dpi=180);plt.close(fig)
print('Graphiques du rapport exportés')
