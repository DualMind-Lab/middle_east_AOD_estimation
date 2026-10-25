"""
Generate all result figures for results_discussion/.

Figures produced:
  fig1_model_ci_mae.png       — Forest plot: per-model MAE + 95% CI, 3 splits
  fig2_model_ci_rmse.png      — Same for RMSE
  fig3_model_ci_ee.png        — Same for pct_within_EE
  fig4_perbin_mae.png         — Per-bin MAE profiles, 3 splits
  fig5_perbin_mbe.png         — Per-bin MBE profiles, 3 splits
  fig6_diff_forest_strat.png  — Pairwise diff CI vs tabpfn_mi300, stratified
  fig7_diff_forest_hold.png   — Pairwise diff CI vs tabpfn_mi300, holdout_time
  fig8_diff_forest_loso.png   — Pairwise diff CI (all pairs), LOSO
  fig9_site_map.png           — Geographic bubble map of LOSO per-site MAE
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]   # repository root
DATASET = Path(os.environ.get('AOD_DATASET', REPO / 'data' / 'dataset.csv'))   # download dataset.csv from the Hugging Face dataset
RES     = REPO / 'modeling/results'
OUT_DIR = REPO / 'paper_analysis/outputs'      # cluster_bootstrap_ci.csv, pairwise_diff_ci.csv
FIG_OUT = Path(__file__).resolve().parent / 'out'
FIG_OUT.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
ci_df   = pd.read_csv(OUT_DIR / 'cluster_bootstrap_ci.csv')
diff_df = pd.read_csv(OUT_DIR / 'pairwise_diff_ci.csv')
bin_df  = pd.read_csv(RES / 'per_bin/per_bin_mae_mbe.csv')
site_df = pd.read_csv(RES / 'loso/per_site_metrics.csv')
dataset = pd.read_csv(DATASET)
metrics = pd.read_csv(RES / 'metrics/all_metrics.csv')

# Best transform per model per split
best_tf = (
    metrics.sort_values('RMSE')
    .groupby(['split', 'model']).first()
    .reset_index()[['split', 'model', 'transform']]
)

# Per-bin: filter to best transform per (split, model)
bin_rows = []
for _, row in best_tf.iterrows():
    sub = bin_df[
        (bin_df['split'] == row['split']) &
        (bin_df['model'] == row['model']) &
        (bin_df['transform'] == row['transform'])
    ]
    bin_rows.append(sub)
bin_best = pd.concat(bin_rows, ignore_index=True)

# Site geography
site_coords = (
    dataset.groupby('AERONET_Site')
    .agg(lat=('Site_Latitude(Degrees)', 'mean'),
         lon=('Site_Longitude(Degrees)', 'mean'),
         elev=('Site_Elevation(m)', 'mean'),
         N=('AOD_550nm', 'count'),
         mean_aod=('AOD_550nm', 'mean'))
    .reset_index()
)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'pdf.fonttype': 42,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.dpi': 150,
})

# Colour palette: colourblind-safe (Wong 2011)
COLORS = {
    'tabpfn_mi300':  '#009E73',   # green
    'tabpfn_pca150': '#56B4E9',   # sky blue
    'tabpfn_mi200':  '#0072B2',   # blue
    'tabpfn_mi100':  '#F0E442',   # yellow
    'svr':           '#CC79A7',   # pink
    'catboost':      '#E69F00',   # orange
    'xgboost':       '#D55E00',   # vermilion
    'lgbm':          '#8B4513',   # brown
    'gbm':           '#555555',   # dark grey
    'extratrees':    '#AAAAAA',   # light grey
    'rf':            '#888888',   # medium grey
    'ridge':         '#44AA99',   # teal
    'lasso':         '#AA4499',   # purple
}

SPLIT_LABELS = {
    'stratified':   'Stratified\n(n = 1,035)',
    'holdout_time': 'Holdout 2024–2025\n(n = 834)',
    'loso':         'LOSO\n(n = 5,134)',
}

MODEL_DISPLAY = {
    'tabpfn_mi300':  'TabPFN-MI300',
    'tabpfn_pca150': 'TabPFN-PCA150',
    'tabpfn_mi200':  'TabPFN-MI200',
    'tabpfn_mi100':  'TabPFN-MI100',
    'svr':           'SVR',
    'catboost':      'CatBoost',
    'xgboost':       'XGBoost',
    'lgbm':          'LightGBM',
    'gbm':           'GBM',
    'extratrees':    'ExtraTrees',
    'rf':            'RandomForest',
    'ridge':         'Ridge',
    'lasso':         'Lasso',
}


# ── Helper: forest plot per metric ────────────────────────────────────────────

def forest_plot(metric, ci_lo_col, ci_hi_col, xlabel, fname):
    """Three-panel horizontal CI forest plot, one per split."""
    splits = ['stratified', 'holdout_time', 'loso']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.4), sharey=False)

    for ax, split in zip(axes, splits):
        sub = ci_df[ci_df['split'] == split].copy()
        sub = sub.sort_values(metric)
        models = sub['model'].tolist()
        y = np.arange(len(models))

        for i, (_, row) in enumerate(sub.iterrows()):
            color = COLORS.get(row['model'], '#333333')
            ax.errorbar(
                row[metric], y[i],
                xerr=[[row[metric] - row[ci_lo_col]],
                      [row[ci_hi_col] - row[metric]]],
                fmt='o', color=color,
                markersize=5, linewidth=1.2, capsize=2.5,
                elinewidth=1.0
            )

        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_DISPLAY.get(m, m) for m in models], fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(SPLIT_LABELS[split], fontsize=9, fontweight='bold', pad=5)
        ax.axvline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.3)
        ax.grid(axis='x', linewidth=0.4, alpha=0.4)

    axes[0].set_ylabel('Model', fontsize=9)
    plt.tight_layout()
    path = FIG_OUT / fname.replace('.png', '.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


forest_plot('MAE', 'MAE_ci_lo', 'MAE_ci_hi', 'MAE (AOD units)', 'fig1_model_ci_mae.png')
forest_plot('RMSE', 'RMSE_ci_lo', 'RMSE_ci_hi', 'RMSE (AOD units)', 'fig2_model_ci_rmse.png')
forest_plot('pct_within_EE', 'pct_within_EE_ci_lo', 'pct_within_EE_ci_hi',
            'Within EE (%)', 'fig3_model_ci_ee.png')


# ── Per-bin MAE and MBE ───────────────────────────────────────────────────────

# Select representative models: best, mid, worst in each split
HIGHLIGHT_MODELS = ['tabpfn_mi300', 'tabpfn_pca150', 'svr', 'gbm', 'rf', 'lasso']

def perbin_plot(metric, ylabel, fname, yline=0):
    splits = ['stratified', 'holdout_time', 'loso']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3), sharey=False)

    for ax, split in zip(axes, splits):
        sub = bin_best[bin_best['split'] == split]
        models_in = [m for m in HIGHLIGHT_MODELS if m in sub['model'].unique()]

        for model in models_in:
            m_sub = sub[sub['model'] == model].sort_values('bin_idx')
            if len(m_sub) == 0:
                continue
            # x-axis: AOD bin midpoint
            x = m_sub['aod_mid'].values
            y = m_sub[metric].values
            color = COLORS.get(model, '#333333')
            ax.plot(x, y, 'o-', color=color, linewidth=1.4, markersize=4,
                    label=MODEL_DISPLAY.get(model, model))

        if yline is not None:
            ax.axhline(yline, color='k', linewidth=0.6, linestyle='--', alpha=0.4)

        ax.set_xlabel('AOD bin midpoint', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(SPLIT_LABELS[split], fontsize=9, fontweight='bold', pad=5)
        ax.grid(linewidth=0.3, alpha=0.4)

    # Add legend to last panel
    handles = [mpatches.Patch(color=COLORS.get(m, '#333333'),
                               label=MODEL_DISPLAY.get(m, m))
               for m in models_in]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.07))

    plt.tight_layout()
    path = FIG_OUT / fname.replace('.png', '.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


perbin_plot('MAE', 'MAE (AOD units)', 'fig4_perbin_mae.png', yline=None)
perbin_plot('MBE', 'MBE (AOD units)', 'fig5_perbin_mbe.png', yline=0)


# ── Pairwise difference forest plots ─────────────────────────────────────────

def diff_vs_reference(split, reference, metric, fname, title):
    """Plot metric diff (all models - reference) with CI."""
    sub = diff_df[diff_df['split'] == split].copy()
    diff_col = f'{metric}_diff'
    lo_col   = f'{metric}_ci_lo'
    hi_col   = f'{metric}_ci_hi'
    sig_col  = f'{metric}_sig'

    rows = []
    for _, r in sub.iterrows():
        if r['model_a'] == reference:
            rows.append({'model': r['model_b'],
                         'diff':  -r[diff_col],
                         'lo':    -r[hi_col],
                         'hi':    -r[lo_col],
                         'sig':   r[sig_col]})
        elif r['model_b'] == reference:
            rows.append({'model': r['model_a'],
                         'diff':  r[diff_col],
                         'lo':    r[lo_col],
                         'hi':    r[hi_col],
                         'sig':   r[sig_col]})

    if not rows:
        return
    plot_df = pd.DataFrame(rows).sort_values('diff')
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for i, (_, row) in enumerate(plot_df.iterrows()):
        color = '#D55E00' if row['sig'] else '#888888'
        ax.errorbar(
            row['diff'], y[i],
            xerr=[[row['diff'] - row['lo']],
                  [row['hi'] - row['diff']]],
            fmt='o', color=color, markersize=5,
            linewidth=1.2, capsize=2.5, elinewidth=1.0
        )

    ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_DISPLAY.get(m, m) for m in plot_df['model']], fontsize=9)
    ax.set_xlabel(f'{metric} difference (model − {MODEL_DISPLAY.get(reference, reference)})',
                  fontsize=9)

    sig_patch   = mpatches.Patch(color='#D55E00', label='Significant (CI excludes 0)')
    nosig_patch = mpatches.Patch(color='#888888', label='Not significant')
    ax.legend(handles=[sig_patch, nosig_patch], fontsize=8, loc='upper left',
              framealpha=0.9, edgecolor='#cccccc')
    ax.grid(axis='x', linewidth=0.3, alpha=0.4)

    plt.tight_layout()
    path = FIG_OUT / fname.replace('.png', '.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


diff_vs_reference('stratified', 'tabpfn_mi300', 'MAE',
                  'fig6_diff_forest_strat.png',
                  'MAE difference vs TabPFN-MI300 — Stratified')
diff_vs_reference('holdout_time', 'tabpfn_mi300', 'MAE',
                  'fig7_diff_forest_hold.png',
                  'MAE difference vs TabPFN-MI300 — Holdout 2024–2025')


def diff_all_loso(metric, fname):
    """All significant LOSO pairs, full difference matrix style."""
    sub = diff_df[diff_df['split'] == 'loso'].copy()
    diff_col = f'{metric}_diff'
    lo_col   = f'{metric}_ci_lo'
    hi_col   = f'{metric}_ci_hi'
    sig_col  = f'{metric}_sig'

    sub['label'] = sub.apply(
        lambda r: f"{MODEL_DISPLAY.get(r['model_a'], r['model_a'])} − "
                  f"{MODEL_DISPLAY.get(r['model_b'], r['model_b'])}",
        axis=1
    )
    sub = sub.sort_values(diff_col)
    y = np.arange(len(sub))

    fig, ax = plt.subplots(figsize=(6.6, 8.0))
    for i, (_, row) in enumerate(sub.iterrows()):
        color = '#009E73' if row[sig_col] else '#AAAAAA'
        lw = 1.4 if row[sig_col] else 0.8
        ax.errorbar(
            row[diff_col], y[i],
            xerr=[[row[diff_col] - row[lo_col]],
                  [row[hi_col] - row[diff_col]]],
            fmt='o', color=color, markersize=4,
            linewidth=lw, capsize=2, elinewidth=lw
        )

    ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_yticks(y)
    ax.set_yticklabels(sub['label'], fontsize=8.5)
    ax.set_xlabel(f'{metric} difference (model_a − model_b)', fontsize=9)
    ax.grid(axis='x', linewidth=0.3, alpha=0.4)

    plt.tight_layout()
    path = FIG_OUT / fname.replace('.png', '.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


diff_all_loso('MAE', 'fig8_diff_forest_loso.png')