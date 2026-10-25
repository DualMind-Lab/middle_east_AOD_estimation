#!/usr/bin/env python3
"""
Friedman omnibus test on per-site MAE for ALL THREE evaluation splits
(stratified, holdout_time, loso).

Rationale
---------
The manuscript reports the Friedman test for every split so the temporal vs.
spatial contrast is symmetric: the test rejects equal model performance under
the two temporal protocols but not under LOSO. This script reproduces those
numbers.

Method
------
* Per-site MAE is the block-level statistic; sites are the Friedman blocks.
* One transform per model is selected by lowest pooled RMSE (the same rule used
  for the performance tables and the cluster bootstrap).
* For LOSO, per-site MAE is read directly from results/loso/per_site_metrics.csv
  (each site is its own held-out fold, so site labels are intrinsic).
* For stratified / holdout_time, the prediction CSVs carry no site column, so
  site labels are recovered by matching y_true to AOD_550nm in data/dataset.csv
  (holdout restricted to year >= 2024). ~40 AOD values are non-unique; ties are
  resolved keep-first. This is the same site-recovery used by
  results_discussion/01_cluster_bootstrap.py.

Outputs
-------
Prints chi-square, p-value, n_sites and n_models per split, and writes
friedman_all_splits.csv next to this script.

Reference values (this dataset):
    stratified    chi2=110.143  p=5.6e-18  sites=20  models=13
    holdout_time  chi2= 53.287  p=3.7e-07  sites=11  models=13
    loso          chi2=  7.936  p=0.635    sites=20  models=11
"""
import warnings
warnings.filterwarnings('ignore')

import csv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]   # repository root
DATASET = Path(os.environ.get('AOD_DATASET', REPO / 'data' / 'dataset.csv'))   # download dataset.csv from the Hugging Face dataset
PRED = REPO / 'modeling/results/predictions'
DATA = DATASET
ALLM = REPO / 'modeling/results/metrics/all_metrics.csv'
LOSO_SITE = REPO / 'modeling/results/loso/per_site_metrics.csv'
OUT = REPO / 'paper_analysis/outputs/friedman_all_splits.csv'

all_metrics = pd.read_csv(ALLM)
ds = pd.read_csv(DATA)
ds['date_parsed'] = pd.to_datetime(ds['Date(dd:mm:yyyy)'], format='%d:%m:%Y',
                                   errors='coerce')
ds['year'] = ds['date_parsed'].dt.year


def best_transform(split):
    """Lowest-RMSE transform per model, matching the tables/bootstrap."""
    s = (all_metrics[all_metrics['split'] == split]
         .sort_values('RMSE').groupby('model').first().reset_index())
    return dict(zip(s['model'], s['transform']))


def add_site(pred, split):
    """Recover an AERONET_Site column by matching y_true -> AOD_550nm."""
    pred = pred.copy().reset_index(drop=True)
    pred['_rid'] = pred.index
    ref = (ds[ds['year'] >= 2024] if split == 'holdout_time' else ds)
    ref = ref[['AOD_550nm', 'AERONET_Site']].copy()
    m = (pred.merge(ref, left_on='y_true', right_on='AOD_550nm', how='left')
             .drop_duplicates(subset='_rid', keep='first'))
    return m.rename(columns={'AERONET_Site': 'site'})


def per_site_mae(split):
    """Return a (sites x models) DataFrame of per-site MAE."""
    tf = best_transform(split)
    if split == 'loso':
        # site labels are intrinsic to the LOSO folds; read per-site MAE directly.
        # NB: use sd['transform'] (bracket) — sd.transform is the DataFrame method.
        sd = pd.read_csv(LOSO_SITE)
        sites = sorted(sd['site'].unique())
        cols = {}
        for m in sorted(tf):
            sub = (sd[(sd['model'] == m) & (sd['transform'] == tf[m])]
                   .drop_duplicates('site').set_index('site').reindex(sites))
            cols[m] = sub['MAE']
        return pd.DataFrame(cols, index=sites)
    cols = {}
    for mdl in sorted(tf):
        p = pd.read_csv(PRED / split / f'{mdl}_{tf[mdl]}.csv')
        m = add_site(p, split)
        m['ae'] = (m['y_true'] - m['y_pred']).abs()
        cols[mdl] = m.groupby('site')['ae'].mean()
    return pd.DataFrame(cols)


rows = []
for split in ['stratified', 'holdout_time', 'loso']:
    M = per_site_mae(split).dropna(axis=0, how='any')
    chi2, p = stats.friedmanchisquare(*[M[c].values for c in M.columns])
    print(f'{split:14s}  chi2={chi2:8.3f}  p={p:.3e}  '
          f'sites={M.shape[0]}  models={M.shape[1]}')
    rows.append({'split': split, 'chi2': chi2, 'p_value': p,
                 'n_sites': M.shape[0], 'n_models': M.shape[1],
                 'significant_05': p < 0.05})

with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f'\nWrote {OUT}')
