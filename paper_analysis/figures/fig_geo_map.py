"""
Publication-quality geographic map of LOSO per-site MAE.
Includes a Cyprus zoom inset to handle the 4-site cluster.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype']=42
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]   # repository root
DATASET = Path(os.environ.get('AOD_DATASET', REPO / 'data' / 'dataset.csv'))   # download dataset.csv from the Hugging Face dataset
LOSO_DIR = REPO / 'modeling/results/loso'
OUT_DIR  = Path(__file__).resolve().parent / 'out'
OUT_DIR.mkdir(exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────
site_df = pd.read_csv(LOSO_DIR / 'per_site_metrics.csv')
mi300   = site_df[
    (site_df['model']     == 'tabpfn_mi300') &
    (site_df['transform'] == 'log1p')
][['site', 'MAE', 'MBE', 'N', 'pct_within_EE']].copy()

dataset     = pd.read_csv(DATASET)
site_coords = dataset.groupby('AERONET_Site')[[
    'Site_Latitude(Degrees)', 'Site_Longitude(Degrees)'
]].first().reset_index().rename(columns={
    'Site_Latitude(Degrees)':  'lat',
    'Site_Longitude(Degrees)': 'lon',
})

df = mi300.merge(site_coords, left_on='site', right_on='AERONET_Site', how='left')

CYPRUS_SITES = {'CUT-TEPAK', 'AgiaMarina_Xyliatou', 'Nicosia', 'Troodos_CAO'}
df_main    = df[~df['site'].isin(CYPRUS_SITES)]
df_cyprus  = df[ df['site'].isin(CYPRUS_SITES)]

# ── colour / norm ─────────────────────────────────────────────────────────
norm = mcolors.Normalize(vmin=0.02, vmax=0.40)
cmap = plt.cm.RdYlGn_r

FEATURES = dict(
    ocean     = dict(facecolor='#cfe2f3', zorder=0),
    land      = dict(facecolor='#f2ede6', zorder=1),
    coastline = dict(linewidth=0.6, edgecolor='#444444', zorder=2),
    borders   = dict(linewidth=0.35, edgecolor='#888888', linestyle='--', zorder=2),
    lakes     = dict(facecolor='#cfe2f3', zorder=2),
)

def add_features(axes):
    axes.add_feature(cfeature.OCEAN.with_scale('50m'),     **FEATURES['ocean'])
    axes.add_feature(cfeature.LAND.with_scale('50m'),      **FEATURES['land'])
    axes.add_feature(cfeature.COASTLINE.with_scale('50m'), **FEATURES['coastline'])
    axes.add_feature(cfeature.BORDERS.with_scale('50m'),   **FEATURES['borders'])
    axes.add_feature(cfeature.LAKES.with_scale('50m'),     **FEATURES['lakes'])

# ── main figure ───────────────────────────────────────────────────────────
proj = ccrs.PlateCarree()
fig  = plt.figure(figsize=(7.6, 4.7))
ax   = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_extent([19, 64.5, 15.5, 39.5], crs=proj)
add_features(ax)

gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.25,
                  color='grey', alpha=0.5, linestyle='--')
gl.top_labels = False; gl.right_labels = False
gl.xlabel_style = {'size': 10.8}; gl.ylabel_style = {'size': 10.8}

# plot non-Cyprus sites
sc = ax.scatter(
    df_main['lon'], df_main['lat'],
    c=df_main['MAE'].clip(upper=0.40),
    s=df_main['N'] / 7,
    cmap=cmap, norm=norm,
    alpha=0.92, edgecolors='#222222', linewidths=0.5,
    transform=proj, zorder=5,
)

# plot Cyprus sites on main map (small, no labels — see inset)
ax.scatter(
    df_cyprus['lon'], df_cyprus['lat'],
    c=df_cyprus['MAE'].clip(upper=0.40),
    s=df_cyprus['N'] / 7,
    cmap=cmap, norm=norm,
    alpha=0.92, edgecolors='#222222', linewidths=0.5,
    transform=proj, zorder=5,
)

# ── labels for non-Cyprus sites ───────────────────────────────────────────
SHORT = {
    'DEWA_ResearchCentre': 'DEWA',
    'Masdar_Institute':    'Masdar',
    'Mezaira':             'Mezaira',
    'IMS-METU-ERDEMLI':   'IMS-METU',
    'KAUST_Campus':        'KAUST',
    'SEDE_BOKER':          'Sede Boker',
    'Migal':               'Migal',
    'Technion_Haifa_IL':   'Technion',
    'Weizmann_Institute':  'Weizmann',
    'Eilat':               'Eilat',
    'Kuwait_University':   'Kuwait U.',
    'Cairo_EMA_2':         'Cairo',
    'Qena_SVU':            'Qena',
    'El_Farafra':          'El Farafra',
    'Shagaya_Park':        'Shagaya Park',
    'University_of_Nizwa': 'Nizwa',
}

# (delta_lon, delta_lat, ha)  — arrows must not cross each other
OFFSETS = {
    # Israel / Levant column — fan left and right alternately
    'Eilat':               ( 3.0, -1.5, 'left'),
    'SEDE_BOKER':          (-5.0,  0.0, 'right'),
    'Weizmann_Institute':  (-5.5,  1.5, 'right'),
    'Technion_Haifa_IL':   ( 3.0,  1.5, 'left'),
    'Migal':               ( 3.5,  0.0, 'left'),
    # Egypt
    'Cairo_EMA_2':         ( 3.0,  2.0, 'left'),
    'Qena_SVU':            ( 3.0, -2.0, 'left'),
    'El_Farafra':          ( 2.5,  1.3, 'left'),
    # Turkey
    'IMS-METU-ERDEMLI':   ( 3.0,  2.0, 'left'),
    # Kuwait
    'Kuwait_University':   ( 3.0,  2.5, 'left'),
    'Shagaya_Park':        ( 3.0, -2.5, 'left'),
    # Saudi
    'KAUST_Campus':        ( 3.0, -2.5, 'left'),
    # UAE / Oman — fan: DEWA northeast, Masdar south, Mezaira west, Nizwa east
    'DEWA_ResearchCentre': ( 3.0,  2.0, 'left'),
    'Masdar_Institute':    (-5.0, -1.5, 'right'),
    'Mezaira':             (-5.0,  0.0, 'right'),
    'University_of_Nizwa': ( 3.0, -2.0, 'left'),
}

TRANSFORM = proj._as_mpl_transform(ax)

for _, row in df_main.iterrows():
    name = row['site']
    label = SHORT.get(name, name)
    dlon, dlat, ha = OFFSETS.get(name, (2.5, 1.5, 'left'))
    ax.annotate(
        label,
        xy=(row['lon'], row['lat']),
        xytext=(row['lon'] + dlon, row['lat'] + dlat),
        fontsize=8.5,
        color='#111111',
        ha=ha, va='center',
        xycoords=TRANSFORM,
        textcoords=TRANSFORM,
        arrowprops=dict(arrowstyle='-', color='#666666', lw=0.5),
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                  alpha=0.75, edgecolor='none'),
        zorder=8,
    )

# ── draw zoom-box around Cyprus on main map ───────────────────────────────
CY_LON0, CY_LON1 = 32.55, 33.65
CY_LAT0, CY_LAT1 = 34.45, 35.35
box_lons = [CY_LON0, CY_LON1, CY_LON1, CY_LON0, CY_LON0]
box_lats = [CY_LAT0, CY_LAT0, CY_LAT1, CY_LAT1, CY_LAT0]
ax.plot(box_lons, box_lats, transform=proj,
        color='#333333', linewidth=1.2, zorder=9, linestyle='-')
ax.text(CY_LON0 - 0.3, (CY_LAT0 + CY_LAT1) / 2 + 1.0,
        'see\ninset', fontsize=9.5, color='#333333',
        ha='right', va='center', transform=proj,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  alpha=0.8, edgecolor='#aaaaaa'))

# ── Cyprus inset ──────────────────────────────────────────────────────────
# Position in figure coordinates [left, bottom, width, height]
ax_cy = fig.add_axes([0.03, 0.04, 0.27, 0.33], projection=proj)
ax_cy.set_extent([CY_LON0, CY_LON1, CY_LAT0, CY_LAT1], crs=proj)
add_features(ax_cy)

ax_cy.add_feature(cfeature.COASTLINE.with_scale('10m'),
                  linewidth=0.8, edgecolor='#333333', zorder=3)

gl_cy = ax_cy.gridlines(crs=proj, draw_labels=True, linewidth=0.2,
                         color='grey', alpha=0.4, linestyle='--')
gl_cy.top_labels = False; gl_cy.right_labels = False
gl_cy.bottom_labels = True; gl_cy.left_labels = True
import matplotlib.ticker as mticker
gl_cy.xlocator = mticker.FixedLocator([32.8, 33.2]); gl_cy.ylocator = mticker.FixedLocator([34.6, 35.0])
gl_cy.xlabel_style = {'size': 8.8}; gl_cy.ylabel_style = {'size': 8.8}

CY_SHORT = {
    'CUT-TEPAK':           'CUT-TEPAK',
    'AgiaMarina_Xyliatou': 'Agia Marina',
    'Nicosia':             'Nicosia',
    'Troodos_CAO':         'Troodos',
}
# Larger bubbles in inset for visibility
sc_cy = ax_cy.scatter(
    df_cyprus['lon'], df_cyprus['lat'],
    c=df_cyprus['MAE'].clip(upper=0.40),
    s=df_cyprus['N'] / 2.5,          # bigger in inset
    cmap=cmap, norm=norm,
    alpha=0.92, edgecolors='#222222', linewidths=0.7,
    transform=proj, zorder=5,
)

# Cyprus label offsets within inset (degrees)
CY_OFFSETS = {
    'CUT-TEPAK':           ( 0.15,  0.08, 'left'),
    'AgiaMarina_Xyliatou': (-0.15,  0.12, 'right'),
    'Nicosia':             ( 0.15,  0.08, 'left'),
    'Troodos_CAO':         (-0.18, -0.08, 'right'),
}

TRANSFORM_CY = proj._as_mpl_transform(ax_cy)
for _, row in df_cyprus.iterrows():
    name  = row['site']
    label = CY_SHORT.get(name, name)
    mae   = row['MAE']
    dlon, dlat, ha = CY_OFFSETS.get(name, (0.12, 0.08, 'left'))
    ax_cy.annotate(
        f'{label}\n(MAE={mae:.3f})',
        xy=(row['lon'], row['lat']),
        xytext=(row['lon'] + dlon, row['lat'] + dlat),
        fontsize=7.6,
        color='#111111',
        ha=ha, va='center',
        xycoords=TRANSFORM_CY,
        textcoords=TRANSFORM_CY,
        arrowprops=dict(arrowstyle='-', color='#555555', lw=0.5),
        bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                  alpha=0.8, edgecolor='none'),
        zorder=8,
    )

ax_cy.set_title('Cyprus (inset)', fontsize=8.5, fontweight='bold', pad=3)

# frame the inset
for spine in ax_cy.spines.values():
    spine.set_edgecolor('#333333')
    spine.set_linewidth(1.2)

# ── colourbar ─────────────────────────────────────────────────────────────
cbar = plt.colorbar(sc, ax=ax, shrink=0.60, pad=0.03, extend='max')
cbar.set_label('LOSO MAE (TabPFN-MI300)', fontsize=9.5)
cbar.ax.tick_params(labelsize=10.8)
cbar.set_ticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])

# ── bubble size legend ────────────────────────────────────────────────────
for n_raw, lbl in [(56, '~400 obs'), (112, '~800 obs'), (224, '~1600 obs')]:
    ax.scatter([], [], c='#aaaaaa', s=n_raw, alpha=0.8,
               edgecolors='#333333', linewidths=0.5,
               label=lbl, transform=proj)
ax.legend(title='Site N', fontsize=8.5, title_fontsize=9,
          loc='upper right', framealpha=0.9, edgecolor='#cccccc')



plt.tight_layout(rect=[0, 0.03, 1, 1])
out_path = OUT_DIR / 'fig_geo_map.pdf'
fig.savefig(out_path, bbox_inches='tight')
plt.close()
print(f'Saved: {out_path}')
