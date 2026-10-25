"""
Cluster bootstrap analysis for all three evaluation splits.

Cluster unit: AERONET site (identical to Uendes et al. 2026 who used
participant-level resampling because observations within a cluster are
correlated and should not be treated as independent).

For each split the script does two things:

(A) Per-model 95% CI
    Bootstrap resample sites → pool observations → compute metric.
    Gives uncertainty on each model's absolute performance.

(B) Pairwise difference 95% CI (paired cluster bootstrap)
    Same site resample used for BOTH models in a pair.  This is the
    correct paired test: the difference CI excludes 0 iff the models
    are distinguishable at the cluster level.

Splits:
  stratified   — chronological 80/20 split within each site; 20 sites; site labels recovered
                 by matching y_true → AOD_550nm in dataset.csv.
  holdout_time — temporal holdout (test = 2024-2025); 11 sites; site
                 labels recovered by matching to 2024+ rows only.
  loso         — leave-one-site-out; site labels already in CSV.

Site label recovery note:
  40 AOD_550nm values appear in more than one dataset row (coincidental
  rounding collisions, ~0.6% of observations). For those cases we take
  the first matching row. Impact on cluster bootstrap is negligible.

Outputs (paper_analysis/outputs/):
  cluster_bootstrap_ci.csv   — per (split, model, metric): point + CI
  pairwise_diff_ci.csv       — per (split, metric, pair): diff + CI + sig

N_BOOT = 2000 (Uendes et al. 2026 used 2000 participant-level resamples)
SEED   = 42
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from itertools import combinations

import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]   # repository root
DATASET = Path(os.environ.get('AOD_DATASET', REPO / 'data' / 'dataset.csv'))   # download dataset.csv from the Hugging Face dataset
PRED_DIR = REPO / 'modeling/results/predictions'
LOSO_DIR = REPO / 'modeling/results/loso'
DATA_CSV = DATASET
METRICS_CSV = REPO / 'modeling/results/metrics/all_metrics.csv'
OUT_DIR  = REPO / 'paper_analysis/outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 2000
SEED   = 42
rng    = np.random.default_rng(SEED)

METRICS = ['RMSE', 'MAE', 'MBE', 'Pearson_r', 'Spearman_r', 'pct_within_EE']


# ── Metric computation ────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    n = len(yt)
    if n < 3:
        return {k: np.nan for k in METRICS}
    res  = yp - yt
    rmse = float(np.sqrt(np.mean(res ** 2)))
    mae  = float(np.mean(np.abs(res)))
    mbe  = float(np.mean(res))
    ee   = 0.05 + 0.15 * yt
    pct  = float(np.mean(np.abs(res) <= ee) * 100)
    try:
        r = float(stats.pearsonr(yt, yp)[0])
    except Exception:
        r = np.nan
    try:
        rho = float(stats.spearmanr(yt, yp).statistic)
    except Exception:
        rho = np.nan
    return {'RMSE': rmse, 'MAE': mae, 'MBE': mbe,
            'Pearson_r': r, 'Spearman_r': rho, 'pct_within_EE': pct}


# ── Cluster bootstrap kernel ──────────────────────────────────────────────────

def cluster_bootstrap(df, sites, models, n_boot, rng):
    """
    Paired cluster bootstrap.

    df      : DataFrame with columns [site, y_true, <model1>, <model2>, ...]
              Each row is one observation. Model columns contain y_pred.
    sites   : list of unique site names in df
    models  : list of model column names to evaluate
    n_boot  : number of bootstrap resamples
    rng     : numpy random Generator

    Returns
    -------
    boot_metrics : dict  model -> metric -> array(n_boot)
    boot_diffs   : dict  (model_a, model_b) -> metric -> array(n_boot)
    """
    site_data = {s: df[df['site'] == s] for s in sites}
    G = len(sites)

    boot_metrics = {m: {k: np.empty(n_boot) for k in METRICS} for m in models}
    pairs = list(combinations(models, 2))
    boot_diffs = {p: {k: np.empty(n_boot) for k in METRICS} for p in pairs}

    for i in range(n_boot):
        sampled = rng.choice(sites, size=G, replace=True)
        pool = pd.concat([site_data[s] for s in sampled], ignore_index=True)
        yt = pool['y_true'].values
        # per-model metrics
        for mdl in models:
            yp = pool[mdl].values
            m = compute_metrics(yt, yp)
            for k in METRICS:
                boot_metrics[mdl][k][i] = m[k]
        # pairwise diffs on the SAME pool (paired)
        for ma, mb in pairs:
            m_a = compute_metrics(yt, pool[ma].values)
            m_b = compute_metrics(yt, pool[mb].values)
            for k in METRICS:
                boot_diffs[(ma, mb)][k][i] = m_a[k] - m_b[k]

    return boot_metrics, boot_diffs


def ci_from_boot(arr):
    arr = arr[np.isfinite(arr)]
    if len(arr) < 10:
        return np.nan, np.nan
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ── Load best-transform lookup ────────────────────────────────────────────────

all_metrics = pd.read_csv(METRICS_CSV)
best_transforms = (
    all_metrics.sort_values('RMSE')
    .groupby(['split', 'model'])
    .first()
    .reset_index()[['split', 'model', 'transform']]
)

def best_tf(split, model):
    row = best_transforms[
        (best_transforms['split'] == split) &
        (best_transforms['model'] == model)
    ]
    if len(row) == 0:
        return None
    return row.iloc[0]['transform']


# ── Site label recovery for stratified / holdout_time ────────────────────────

dataset = pd.read_csv(DATA_CSV)
dataset['date_parsed'] = pd.to_datetime(
    dataset['Date(dd:mm:yyyy)'], format='%d:%m:%Y', errors='coerce')
dataset['year'] = dataset['date_parsed'].dt.year


def add_site_labels(pred_df, split):
    """
    pred_df : rows from ONE model's prediction CSV; has y_true column.
    Returns same dataframe with a 'site' column added.
    """
    pred_df = pred_df.copy().reset_index(drop=True)
    pred_df['_row_id'] = pred_df.index

    if split == 'holdout_time':
        # Restrict to 2024+ rows to reduce ambiguity
        ref = dataset[dataset['year'] >= 2024][['AOD_550nm', 'AERONET_Site']].copy()
    else:
        ref = dataset[['AOD_550nm', 'AERONET_Site']].copy()

    merged = pred_df.merge(ref, left_on='y_true', right_on='AOD_550nm', how='left')
    # Drop duplicate rows created by coincidental AOD collisions; keep first match
    merged = merged.drop_duplicates(subset='_row_id', keep='first')
    merged = merged.drop(columns=['AOD_550nm', '_row_id'])
    merged = merged.rename(columns={'AERONET_Site': 'site'})

    n_missing = merged['site'].isna().sum()
    if n_missing > 0:
        print(f'  WARNING: {n_missing} rows in {split} could not be assigned a site')
    return merged


# ── Build wide prediction matrices ───────────────────────────────────────────
# One dict per split:  { split: DataFrame(site, y_true, model1, model2, ...) }

split_dfs = {}

for split in ['stratified', 'holdout_time']:
    split_dir = PRED_DIR / split
    models_in_split = best_transforms[
        best_transforms['split'] == split]['model'].tolist()

    frames = {}
    for model in models_in_split:
        tf = best_tf(split, model)
        csv = split_dir / f'{model}_{tf}.csv'
        if not csv.exists():
            print(f'  MISSING: {csv}')
            continue
        df = pd.read_csv(csv)[['y_true', 'y_pred']]
        frames[model] = df

    if not frames:
        continue

    # All y_true are row-aligned; use any model to get y_true and site
    ref_model = list(frames.keys())[0]
    base = add_site_labels(frames[ref_model][['y_true']].copy(), split)
    base = base.rename(columns={'y_true': 'y_true'})

    for model, df in frames.items():
        base[model] = df['y_pred'].values

    split_dfs[split] = base
    sites_found = base['site'].dropna().nunique()
    print(f'[{split}]  {len(base)} obs  |  {sites_found} sites  |  '
          f'{len(frames)} models')
    print(f'  Sites: {sorted(base["site"].dropna().unique())}')

# LOSO: site column already present
loso_preds = pd.read_csv(LOSO_DIR / 'loso_predictions.csv')
loso_models = best_transforms[best_transforms['split'] == 'loso']['model'].tolist()
loso_frames = {}
for model in loso_models:
    tf = best_tf('loso', model)
    grp = loso_preds[
        (loso_preds['model'] == model) &
        (loso_preds['transform'] == tf)
    ][['site', 'y_true', 'y_pred']].copy()
    loso_frames[model] = grp

# Build wide LOSO DataFrame aligned on (site, y_true)
loso_base = loso_frames[loso_models[0]][['site', 'y_true']].copy().reset_index(drop=True)
for model in loso_models:
    tf = best_tf('loso', model)
    grp = loso_preds[
        (loso_preds['model'] == model) &
        (loso_preds['transform'] == tf)
    ].reset_index(drop=True)
    loso_base[model] = grp['y_pred'].values

split_dfs['loso'] = loso_base
loso_sites = sorted(loso_base['site'].unique())
print(f'[loso]  {len(loso_base)} obs  |  {len(loso_sites)} sites  |  '
      f'{len(loso_models)} models')
print(f'  Sites: {loso_sites}')


# ── Run cluster bootstrap for all splits ─────────────────────────────────────

ci_rows   = []
diff_rows = []

for split, df in split_dfs.items():
    df_clean = df.dropna(subset=['site']).copy()
    sites = sorted(df_clean['site'].unique())
    models = [c for c in df_clean.columns if c not in ('site', 'y_true')]
    G = len(sites)

    print(f'\n[{split}] cluster bootstrap — G={G} sites, '
          f'{len(models)} models, {N_BOOT} resamples ...')

    boot_metrics, boot_diffs = cluster_bootstrap(
        df_clean, sites, models, N_BOOT, rng)

    # (A) Per-model CIs
    for model in models:
        # Point estimate on full data
        yt = df_clean['y_true'].values
        yp = df_clean[model].values
        point = compute_metrics(yt, yp)

        row = {'split': split, 'model': model,
               'N': int(df_clean.shape[0]), 'N_sites': G,
               'bootstrap_method': 'cluster_site'}
        for k in METRICS:
            row[k] = point[k]
            lo, hi = ci_from_boot(boot_metrics[model][k])
            row[f'{k}_ci_lo'] = lo
            row[f'{k}_ci_hi'] = hi
        ci_rows.append(row)

    # (B) Pairwise difference CIs
    pairs = list(combinations(models, 2))
    for (ma, mb) in pairs:
        yt = df_clean['y_true'].values
        point_a = compute_metrics(yt, df_clean[ma].values)
        point_b = compute_metrics(yt, df_clean[mb].values)

        row = {'split': split, 'model_a': ma, 'model_b': mb,
               'N': int(df_clean.shape[0]), 'N_sites': G}
        for k in METRICS:
            diff_point = point_a[k] - point_b[k]
            lo, hi = ci_from_boot(boot_diffs[(ma, mb)][k])
            row[f'{k}_diff'] = diff_point
            row[f'{k}_ci_lo'] = lo
            row[f'{k}_ci_hi'] = hi
            # Significant if CI excludes 0
            sig = (lo > 0 or hi < 0) if (np.isfinite(lo) and np.isfinite(hi)) else False
            row[f'{k}_sig'] = sig
        diff_rows.append(row)

    # Quick summary for this split
    sig_pairs = {}
    for k in ['MAE', 'RMSE']:
        n_sig = sum(
            r[f'{k}_sig']
            for r in diff_rows if r['split'] == split
        )
        sig_pairs[k] = n_sig
    print(f'  Significant pairs: MAE={sig_pairs["MAE"]}  RMSE={sig_pairs["RMSE"]}  '
          f'(out of {len(pairs)})')


# ── Save outputs ──────────────────────────────────────────────────────────────

ci_df = pd.DataFrame(ci_rows)
diff_df = pd.DataFrame(diff_rows)

ci_path   = OUT_DIR / 'cluster_bootstrap_ci.csv'
diff_path = OUT_DIR / 'pairwise_diff_ci.csv'
ci_df.to_csv(ci_path, index=False)
diff_df.to_csv(diff_path, index=False)

print(f'\nSaved: {ci_path}  ({len(ci_df)} rows)')
print(f'Saved: {diff_path}  ({len(diff_df)} rows)')

# ── Summary table ─────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('SUMMARY — significant pairwise differences (CI excludes 0)')
print('='*70)
for split in ['stratified', 'holdout_time', 'loso']:
    sub = diff_df[diff_df['split'] == split]
    print(f'\n[{split}]')
    for metric in ['MAE', 'RMSE', 'pct_within_EE']:
        sig = sub[sub[f'{metric}_sig']]
        if len(sig) == 0:
            print(f'  {metric}: no significant pairs')
        else:
            print(f'  {metric}: {len(sig)} significant pairs')
            for _, r in sig.iterrows():
                print(f'    {r.model_a} vs {r.model_b}: '
                      f'diff={r[f"{metric}_diff"]:+.4f}  '
                      f'CI=[{r[f"{metric}_ci_lo"]:+.4f}, {r[f"{metric}_ci_hi"]:+.4f}]')

print('\nDone.')
