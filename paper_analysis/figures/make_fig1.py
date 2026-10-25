import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path
plt.rcParams.update({'font.family':'sans-serif','font.size':9.5,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
OUT=Path(__file__).resolve().parent/'out'; OUT.mkdir(exist_ok=True)
import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]   # repository root
DATASET = Path(os.environ.get('AOD_DATASET', REPO / 'data' / 'dataset.csv'))   # download dataset.csv from the Hugging Face dataset
df=pd.read_csv(DATASET)
refl=pd.read_csv(REPO/'paper_analysis/outputs/site_reflectance.csv').set_index('site')
g=df.groupby('AERONET_Site').agg(lat=('Site_Latitude(Degrees)','first'),lon=('Site_Longitude(Degrees)','first'),N=('AOD_550nm','size'),aod=('AOD_550nm','mean')).reset_index()
pretty=lambda s:s.replace('_',' ')
fig=plt.figure(figsize=(7.2,8.6))
gs_top=gridspec.GridSpec(1,1,figure=fig,left=0.01,right=0.97,top=0.985,bottom=0.64)
gs_bot=gridspec.GridSpec(2,2,figure=fig,height_ratios=[3.4,1.9],hspace=0.42,wspace=0.1,left=0.21,right=0.985,top=0.585,bottom=0.055)
# ---- (a) map with satellite-style basemap
proj=ccrs.PlateCarree()
ax=fig.add_subplot(gs_top[0,0],projection=proj); ax.set_extent([24.5,61,20,38.6],crs=proj)
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
# Natural Earth I (50m, shaded relief): https://www.naturalearthdata.com/downloads/50m-raster-data/50m-natural-earth-1/
_im=Image.open(Path(os.environ.get('NE1_TIF', REPO/'paper_analysis/basemap/NE1_50M_SR_W.tif')))
_x0,_x1,_y0,_y1=24.5,61.0,20.0,38.6
_box=(int((_x0+180)/360*10800),int((90-_y1)/180*5400),int((_x1+180)/360*10800),int((90-_y0)/180*5400))
ax.imshow(np.asarray(_im.crop(_box)),extent=[_x0,_x1,_y0,_y1],transform=proj,origin='upper',zorder=0)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.5,edgecolor='#333333')
ax.add_feature(cfeature.BORDERS.with_scale('50m'),lw=0.4,edgecolor='#555555',ls='--')
gl=ax.gridlines(crs=proj,draw_labels=True,lw=0.2,color='white',alpha=0.7,ls='--'); gl.top_labels=False; gl.right_labels=False
gl.xlabel_style={'size':9}; gl.ylabel_style={'size':9}
sc=ax.scatter(g.lon,g.lat,s=g.N/4.5,c=g.aod,cmap='YlOrRd',vmin=0.05,vmax=0.37,edgecolors='k',lw=0.6,zorder=5,transform=proj)
SHORT={'DEWA_ResearchCentre':'DEWA','Masdar_Institute':'Masdar','Mezaira':'Mezaira','IMS-METU-ERDEMLI':'IMS-METU','KAUST_Campus':'KAUST','SEDE_BOKER':'Sede Boker','Migal':'Migal','Technion_Haifa_IL':'Technion','Weizmann_Institute':'Weizmann','Eilat':'Eilat','Kuwait_University':'Kuwait U.','Cairo_EMA_2':'Cairo','Qena_SVU':'Qena','El_Farafra':'El Farafra','Shagaya_Park':'Shagaya Park','University_of_Nizwa':'Nizwa'}
OFF={'Eilat':(2.2,-1.5,'left'),'SEDE_BOKER':(-3.2,-0.6,'right'),'Weizmann_Institute':(-3.2,0.9,'right'),'Technion_Haifa_IL':(2.4,1.6,'left'),'Migal':(2.6,0.2,'left'),
 'Cairo_EMA_2':(-2.6,0.9,'right'),'Qena_SVU':(2.2,-1.4,'left'),'El_Farafra':(2.2,-1.0,'left'),'IMS-METU-ERDEMLI':(2.2,1.3,'left'),'Kuwait_University':(1.5,2.4,'left'),
 'Shagaya_Park':(-2.0,-2.0,'right'),'KAUST_Campus':(2.4,-1.4,'left'),'DEWA_ResearchCentre':(1.8,2.0,'left'),'Masdar_Institute':(-1.5,-2.4,'right'),'Mezaira':(-2.6,0.6,'right'),'University_of_Nizwa':(1.6,-1.8,'left')}
T=proj._as_mpl_transform(ax)
cy=g[g.AERONET_Site.isin(['CUT-TEPAK','AgiaMarina_Xyliatou','Nicosia','Troodos_CAO'])]
for _,r in g.iterrows():
    n=r.AERONET_Site
    if n in SHORT:
        dx,dy,ha=OFF[n]; ax.annotate(SHORT[n],xy=(r.lon,r.lat),xytext=(r.lon+dx,r.lat+dy),fontsize=9,ha=ha,va='center',xycoords=T,textcoords=T,
            arrowprops=dict(arrowstyle='-',color='#333333',lw=0.5),bbox=dict(boxstyle='round,pad=0.12',fc='white',alpha=0.8,ec='none'),zorder=8)
ax.annotate('Cyprus (4 sites)',xy=(cy.lon.mean(),cy.lat.mean()),xytext=(cy.lon.mean()-3.8,cy.lat.mean()+1.6),fontsize=9,ha='right',va='center',xycoords=T,textcoords=T,
    arrowprops=dict(arrowstyle='-',color='#333333',lw=0.5),bbox=dict(boxstyle='round,pad=0.12',fc='white',alpha=0.8,ec='none'),zorder=8)
cb=fig.colorbar(sc,ax=ax,orientation='vertical',fraction=0.025,pad=0.015); cb.set_label('Mean AOD at 550 nm',fontsize=9.5); cb.ax.tick_params(labelsize=9)
for n,l in [(100,'100'),(400,'400'),(800,'800')]:
    ax.scatter([],[],s=n/4.5,c='#cccccc',edgecolors='k',lw=0.6,label=l)
ax.legend(title='Samples',fontsize=9,title_fontsize=9,loc='lower left',framealpha=0.9,labelspacing=1.0,borderpad=0.7)
ax.set_title('(a)',loc='left',fontsize=11,fontweight='bold',pad=3)
# ---- (b) samples per station, (c) surface brightness
o=g.sort_values('N',ascending=False).reset_index(drop=True); y=np.arange(len(o))
axb=fig.add_subplot(gs_bot[0,0]); axb.barh(y,o.N,color='#4878a8',height=0.72)
axb.set_yticks(y); axb.set_yticklabels([pretty(s) for s in o.AERONET_Site],fontsize=9); axb.invert_yaxis(); axb.set_xlabel('Matched samples',fontsize=9.5)
axb.tick_params(axis='x',labelsize=9); axb.set_title('(b)',loc='left',fontsize=11,fontweight='bold',pad=3)
axc=fig.add_subplot(gs_bot[0,1],sharey=axb)
for band,col,lab in [('B02','#2b6cb0','B02 (blue)'),('B04','#c53030','B04 (red)'),('B11','#2d3748','B11 (SWIR 1)')]:
    axc.scatter([refl.loc[s,band] for s in o.AERONET_Site],y,s=22,c=col,label=lab,zorder=3,edgecolors='white',lw=0.3)
for i in y: axc.axhline(i,color='#eeeeee',lw=0.6,zorder=0)
axc.set_xlabel('Median TOA reflectance',fontsize=9.5); axc.tick_params(axis='x',labelsize=9); plt.setp(axc.get_yticklabels(),visible=False)
axc.legend(fontsize=9,loc='upper right',frameon=True,framealpha=0.9,borderpad=0.4); axc.set_title('(c)',loc='left',fontsize=11,fontweight='bold',pad=3)
# ---- (d) label distribution, (e) per year
axd=fig.add_subplot(gs_bot[1,0]); axd.hist(df.AOD_550nm,bins=np.arange(0,3.05,0.04),color='#c9784b'); axd.set_yscale('log')
axd.axvline(df.AOD_550nm.mean(),color='k',ls='--',lw=1); axd.text(df.AOD_550nm.mean()+0.05,1500,f'mean = {df.AOD_550nm.mean():.3f}',fontsize=9)
axd.set_xlabel('AOD at 550 nm',fontsize=9.5); axd.set_ylabel('Count (log scale)',fontsize=9.5); axd.tick_params(labelsize=9); axd.set_title('(d)',loc='left',fontsize=11,fontweight='bold',pad=3)
year=pd.to_datetime(df['Date(dd:mm:yyyy)'],format='%d:%m:%Y').dt.year.value_counts().sort_index()
axe=fig.add_subplot(gs_bot[1,1]); axe.bar(year.index,year.values,color='#6a9b5a'); axe.set_xticks(year.index); axe.set_xticklabels([str(i)[2:] for i in year.index],fontsize=9)
axe.set_xlabel('Year (20xx)',fontsize=9.5); axe.set_ylabel('Matched samples',fontsize=9.5); axe.tick_params(axis='y',labelsize=9); axe.set_title('(e)',loc='left',fontsize=11,fontweight='bold',pad=3)
fig.savefig(OUT/'figure1.pdf',bbox_inches='tight'); print('saved')
