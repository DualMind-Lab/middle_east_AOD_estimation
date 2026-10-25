import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
plt.rcParams.update({'font.family':'sans-serif','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
OUT=Path(__file__).resolve().parent/'out'; OUT.mkdir(exist_ok=True)
import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]   # repository root
DATASET = Path(os.environ.get('AOD_DATASET', REPO / 'data' / 'dataset.csv'))   # download dataset.csv from the Hugging Face dataset
cs=pd.read_csv(REPO/'feature_engineering/features_analysis/categories/category_summary.csv',index_col=0)
NAMES={'meta':'Scene metadata','advanced':'Advanced band interactions','band_stats':'Per-band statistics',
 'aod_physics':'AOD-physics indices','band_ratios':'Band differences and ratios','veg_idx2':'Vegetation/soil indices II',
 'bloom':'Bloom/dust indices','veg_idx1':'Vegetation indices I','rededge':'Red-edge indices','glcm_b02':'Blue-band texture (GLCM)'}
GROUP={'meta':'#C0845A','advanced':'#4A90C4','band_stats':'#4A90C4','aod_physics':'#4A90C4','band_ratios':'#4A90C4',
 'veg_idx2':'#7AAF7A','bloom':'#7AAF7A','veg_idx1':'#7AAF7A','rededge':'#7AAF7A','glcm_b02':'#B0B0B0'}
cs=cs.sort_values('median_abs_r',ascending=False)
assert cs.n_features.sum()==3700
labels=[NAMES[i] for i in cs.index]; y=np.arange(len(cs))
# ---------------- Figure 2: three panels
fig,axes=plt.subplots(1,3,figsize=(7.2,3.5),sharey=True,gridspec_kw={'wspace':0.12,'width_ratios':[1,1,1]})
for ax,col,xl,c in zip(axes,['median_abs_r','n_features','compression_ratio'],
    ['Median |Spearman $\\rho$|\nwith AOD','Number of features','Compression ratio\n(features / PCs at 95%)'],
    ['steelblue','darkorange','seagreen']):
    ax.barh(y,cs[col].values,color=c,height=0.7); ax.set_xlabel(xl,fontsize=8.5); ax.tick_params(labelsize=8)
    ax.grid(axis='x',color='#dddddd',lw=0.5); ax.set_axisbelow(True)
axes[0].set_yticks(y); axes[0].set_yticklabels(labels,fontsize=8); axes[0].invert_yaxis()
axes[1].set_xscale('log'); axes[1].set_xlim(3,3000)
fig.savefig(OUT/'figure2.pdf',bbox_inches='tight'); plt.close()
# ---------------- Figure 11: (a) category importance + (b) metadata MI
D=DATASET
from sklearn.feature_selection import mutual_info_regression
df=pd.read_csv(D)
cols={'Latitude':'Site_Latitude(Degrees)','Longitude':'Site_Longitude(Degrees)','Elevation':'Site_Elevation(m)','Day of year':'Day_of_Year','Solar zenith':'Solar_Zenith_Angle(Degrees)'}
X=df[list(cols.values())].values; mi=mutual_info_regression(X,df['AOD_550nm'].values,n_neighbors=3,random_state=42)
print('MI recomputed:',dict(zip(cols,np.round(mi,3))),' (paper: lat .206 lon .188 elev .185 doy .159 sza .087)')
mi_paper={'Latitude':0.206,'Longitude':0.188,'Elevation':0.185,'Day of year':0.159,'Solar zenith':0.087}
mv=sorted(mi_paper.items(),key=lambda t:-t[1])
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.2,3.6),gridspec_kw={'width_ratios':[1.5,1],'wspace':0.9})
a1.barh(y,cs.median_abs_r.values,color=[GROUP[i] for i in cs.index],edgecolor='#333333',lw=0.5,height=0.68,zorder=3)
for i,(v,n) in enumerate(zip(cs.median_abs_r.values,cs.n_features.values)):
    a1.text(v+0.006,i,f'$n$ = {n:,}',va='center',ha='left',fontsize=7.5,color='#333333')
a1.set_yticks(y); a1.set_yticklabels(labels,fontsize=8.5); a1.invert_yaxis(); a1.set_xlim(0,0.42)
a1.set_xlabel('Median |Spearman $\\rho$| with AOD$_{550}$',fontsize=9); a1.tick_params(axis='x',labelsize=8.5)
a1.grid(axis='x',color='#dddddd',lw=0.5,zorder=0); a1.set_axisbelow(True)
a1.set_title('(a)',loc='left',fontsize=10,fontweight='bold')
a1.legend(handles=[Patch(fc='#C0845A',ec='#333',lw=.5,label='Geographic / seasonal'),Patch(fc='#4A90C4',ec='#333',lw=.5,label='Aerosol-sensitive spectral'),
  Patch(fc='#7AAF7A',ec='#333',lw=.5,label='Surface-sensitive spectral'),Patch(fc='#B0B0B0',ec='#333',lw=.5,label='Texture')],fontsize=7.5,loc='upper center',bbox_to_anchor=(0.35,-0.2),ncol=2,frameon=False)
y2=np.arange(len(mv)); a2.barh(y2,[v for _,v in mv],color='#C0845A',edgecolor='#333333',lw=0.5,height=0.6,zorder=3)
for i,(k,v) in enumerate(mv): a2.text(v+0.004,i,f'{v:.3f}',va='center',fontsize=8.5)
a2.set_yticks(y2); a2.set_yticklabels([k for k,_ in mv],fontsize=8.5); a2.invert_yaxis(); a2.set_xlim(0,0.29)
a2.set_xlabel('Mutual information with AOD$_{550}$',fontsize=9); a2.tick_params(axis='x',labelsize=8.5)
a2.grid(axis='x',color='#dddddd',lw=0.5,zorder=0); a2.set_axisbelow(True); a2.set_title('(b)',loc='left',fontsize=10,fontweight='bold')
fig.savefig(OUT/'figure11.pdf',bbox_inches='tight'); plt.close()
print('ok')
