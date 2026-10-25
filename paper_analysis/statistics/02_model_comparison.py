"""
Correct statistical model comparison for LOSO evaluation.

The unit of analysis is the SITE (fold), not the observation.
The 20 LOSO sites function as 20 generalisation experiments, but they are
not fully spatially independent: Cyprus (4 sites within 60 km), UAE (2 sites
84 km apart), and the Levant/Israel cluster share surface and aerosol
characteristics. The effective number of independent blocks is < 20.
Tests should be reported with this caveat.

Three tests are run:

1. Friedman test (omnibus)
   Tests whether any model systematically ranks better across sites.
   Input: 20 x N_models matrix of per-site MAE.
   Cite: Demsar (2006), JMLR 7:1-30.

2. Pairwise Wilcoxon signed-rank test
   For each model pair: test whether per-site MAE differences are
   consistently in one direction. Applied to all pairs.
   Holm-Bonferroni correction for multiple comparisons.
   Cite: Demsar (2006); Benavoli et al. (2016), JMLR 17:1-10.
   NOTE: Do NOT use Nemenyi post-hoc — it is pool-dependent (Benavoli 2016).

3. Sign-flip permutation test on per-site MAE
   Non-parametric permutation equivalent. Shuffles signs of the 20
   paired differences 10,000 times. Comparable to the RMSE version
   already in loso_site_permutation.csv.

IMPORTANT CAVEAT:
   With N=20 sites, statistical power is severely limited. For the
   sign-flip permutation test, the minimum achievable two-tailed
   p-value (all 20 differences pointing the same direction) is
   2/2^20 ≈ 1.9e-6. For the Wilcoxon signed-rank test, the minimum
   two-tailed p with n=20 is ~0.000002 (exact tables). Power to
   detect small-to-moderate effect sizes at α=0.05 is low (~15-25%
   for d=0.3). Failure to reject H0 does not mean models are
   equivalent — it means 20 sites is insufficient to confirm a
   consistent direction with confidence. Always report effect sizes
   (median per-site MAE difference) alongside p-values.

   NOTE ON POOLED vs PER-SITE RANKINGS: The Friedman test weights
   all sites equally regardless of sample size. Pooled MAE weights
   observations, so large sites (CUT-TEPAK n=816) dominate. This
   is why tabpfn_mi300 has the best pooled MAE but not the best
   Friedman rank — the two metrics answer different questions and
   the paper must not present them as equivalent.

Outputs (in additional_stat/):
  friedman_test.csv           — omnibus test result for MAE and RMSE
  pairwise_wilcoxon_mae.csv   — all pairwise Wilcoxon results, MAE
  pairwise_wilcoxon_rmse.csv  — all pairwise Wilcoxon results, RMSE
  sign_flip_permutation_mae.csv — sign-flip permutation on MAE
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from itertools import combinations
from statsmodels.stats.multitest import multipletests

import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]   # repository root
DATASET = Path(os.environ.get('AOD_DATASET', REPO / 'data' / 'dataset.csv'))   # download dataset.csv from the Hugging Face dataset
LOSO_DIR = REPO / 'modeling/results/loso'
METRICS_PATH = REPO / 'modeling/results/metrics/all_metrics.csv'
OUT_DIR  = REPO / 'paper_analysis/outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERM = 10_000
SEED   = 42
rng    = np.random.default_rng(SEED)

# ── Load per-site metrics ────────────────────────────────────────────────────
site_df = pd.read_csv(LOSO_DIR / 'per_site_metrics.csv')

# Best transform per model (lowest RMSE, matching all_metrics.csv convention)
all_metrics = pd.read_csv(METRICS_PATH)
loso_best = (
    all_metrics[all_metrics['split'] == 'loso']
    .sort_values('RMSE')
    .groupby('model')
    .first()
    .reset_index()[['model', 'transform']]
)

# Filter site_df to best transform per model
site_filtered = []
for _, row in loso_best.iterrows():
    sub = site_df[(site_df['model'] == row['model']) &
                  (site_df['transform'] == row['transform'])]
    site_filtered.append(sub)
site_filtered = pd.concat(site_filtered, ignore_index=True)

models_list = sorted(loso_best['model'].tolist())
sites_list  = sorted(site_df['site'].unique().tolist())
n_models = len(models_list)
n_sites  = len(sites_list)
print(f'Models: {n_models}  |  Sites: {n_sites}')
print(f'Models: {models_list}')


def get_site_vector(metric, model):
    """Return array of per-site metric values aligned to sites_list."""
    row = loso_best[loso_best['model'] == model].iloc[0]
    sub = (site_df[(site_df['model'] == model) &
                   (site_df['transform'] == row['transform'])]
           .set_index('site')
           .reindex(sites_list))
    return sub[metric].values


# ── Build 20 x N_models matrices ────────────────────────────────────────────
mae_matrix  = np.column_stack([get_site_vector('MAE', m) for m in models_list])
rmse_matrix = np.column_stack([get_site_vector('RMSE', m) for m in models_list])
# Shape: (20, n_models)  — rows=sites, cols=models


# ── 1. Friedman test ─────────────────────────────────────────────────────────
def run_friedman(matrix, metric_name, models):
    """Run Friedman test on (n_sites x n_models) matrix.
    Drops rows with any NaN before testing."""
    valid = ~np.any(np.isnan(matrix), axis=1)
    mat = matrix[valid, :]
    n_valid = mat.shape[0]
    chi2, p = stats.friedmanchisquare(*[mat[:, j] for j in range(mat.shape[1])])
    avg_ranks = np.mean(np.apply_along_axis(
        lambda row: stats.rankdata(row), 1, mat), axis=0)
    print(f'\n[Friedman — {metric_name}]')
    print(f'  Sites used: {n_valid}/{len(sites_list)} (dropped rows with NaN)')
    print(f'  chi2={chi2:.3f}  df={mat.shape[1]-1}  p={p:.4f}')
    print('  Average ranks (lower = better):')
    for m, r in sorted(zip(models, avg_ranks), key=lambda x: x[1]):
        print(f'    {m:20s}  rank={r:.2f}')
    return {
        'metric': metric_name,
        'n_sites_used': n_valid,
        'n_models': mat.shape[1],
        'chi2': chi2,
        'df': mat.shape[1] - 1,
        'p_value': p,
        'significant_05': p < 0.05,
        'avg_ranks': dict(zip(models, avg_ranks.round(3)))
    }

friedman_rows = []
for mat, name in [(mae_matrix, 'MAE'), (rmse_matrix, 'RMSE')]:
    res = run_friedman(mat, name, models_list)
    # Flatten avg_ranks into columns
    r = {k: v for k, v in res.items() if k != 'avg_ranks'}
    for m, rank in res['avg_ranks'].items():
        r[f'avg_rank_{m}'] = rank
    friedman_rows.append(r)

friedman_df = pd.DataFrame(friedman_rows)
friedman_df.to_csv(OUT_DIR / 'friedman_test.csv', index=False)
print(f'\nSaved: {OUT_DIR}/friedman_test.csv')


# ── 2. Pairwise Wilcoxon signed-rank test ────────────────────────────────────
def run_pairwise_wilcoxon(matrix, metric_name, models):
    """All pairwise Wilcoxon signed-rank tests with Holm correction.
    Uses per-site metric vectors (one value per site per model)."""
    pairs = list(combinations(range(len(models)), 2))
    results = []
    for i, j in pairs:
        a = matrix[:, i]
        b = matrix[:, j]
        # Drop sites where either model has NaN
        valid = np.isfinite(a) & np.isfinite(b)
        av, bv = a[valid], b[valid]
        n_valid = valid.sum()
        diff = av - bv
        # Skip if all differences are zero (Wilcoxon undefined)
        if np.all(diff == 0) or n_valid < 5:
            results.append({
                'model_a': models[i], 'model_b': models[j],
                'n_sites': n_valid,
                'median_diff': 0.0,
                'w_statistic': np.nan, 'p_value': np.nan,
                'p_holm': np.nan, 'significant_holm_05': False
            })
            continue
        try:
            w, p = stats.wilcoxon(av, bv, alternative='two-sided')
        except Exception as e:
            w, p = np.nan, np.nan
        results.append({
            'model_a': models[i], 'model_b': models[j],
            'n_sites': int(n_valid),
            'median_diff': float(np.median(diff)),
            'w_statistic': w, 'p_value': p,
            'p_holm': np.nan,
            'significant_holm_05': False
        })

    df = pd.DataFrame(results)
    # Holm-Bonferroni correction
    valid_p = df['p_value'].notna()
    if valid_p.sum() > 0:
        _, p_corrected, _, _ = multipletests(
            df.loc[valid_p, 'p_value'], method='holm')
        df.loc[valid_p, 'p_holm'] = p_corrected
        df.loc[valid_p, 'significant_holm_05'] = p_corrected < 0.05

    print(f'\n[Pairwise Wilcoxon — {metric_name}]  {len(pairs)} pairs, Holm corrected')
    sig = df[df['significant_holm_05']]
    print(f'  Significant pairs (Holm p<0.05): {len(sig)}')
    for _, r in sig.iterrows():
        print(f'    {r.model_a} vs {r.model_b}: p_holm={r.p_holm:.4f}  '
              f'median_diff={r.median_diff:+.4f}')
    if len(sig) == 0:
        print('  None — no model pair significantly different at the site level')
    return df

for mat, name in [(mae_matrix, 'mae'), (rmse_matrix, 'rmse')]:
    wdf = run_pairwise_wilcoxon(mat, name.upper(), models_list)
    wdf.to_csv(OUT_DIR / f'pairwise_wilcoxon_{name}.csv', index=False)
    print(f'Saved: {OUT_DIR}/pairwise_wilcoxon_{name}.csv')


# ── 3. Sign-flip permutation test on per-site MAE ───────────────────────────
# Same method as loso_site_permutation.csv (which used RMSE).
# For each pair: flip signs of the 20 paired MAE differences 10,000 times.
def run_sign_flip(matrix, metric_name, models, n_perm, rng):
    pairs = list(combinations(range(len(models)), 2))
    results = []
    print(f'\n[Sign-flip permutation — {metric_name}]  {len(pairs)} pairs, {n_perm:,} permutations')
    for i, j in pairs:
        a = matrix[:, i]
        b = matrix[:, j]
        valid = np.isfinite(a) & np.isfinite(b)
        av, bv = a[valid], b[valid]
        n_valid = int(valid.sum())
        if n_valid < 3:
            results.append({'model_a': models[i], 'model_b': models[j],
                             'n_sites': n_valid, 'mean_diff': np.nan,
                             'p_value': np.nan, 'significant_05': False})
            continue
        diff = av - bv
        obs_stat = np.mean(diff)
        perm_stats = np.empty(n_perm)
        for k in range(n_perm):
            signs = rng.choice([-1, 1], size=n_valid)
            perm_stats[k] = np.mean(diff * signs)
        p = float(np.mean(np.abs(perm_stats) >= np.abs(obs_stat)))
        results.append({
            'model_a': models[i], 'model_b': models[j],
            'n_sites': n_valid,
            'mean_diff': float(obs_stat),
            'p_value': p,
            'significant_05': p < 0.05
        })
    df = pd.DataFrame(results)
    sig = df[df['significant_05']]
    print(f'  Significant pairs (p<0.05): {len(sig)}')
    for _, r in sig.iterrows():
        print(f'    {r.model_a} vs {r.model_b}: p={r.p_value:.4f}  '
              f'mean_diff={r.mean_diff:+.5f}')
    if len(sig) == 0:
        print('  None')
    return df

perm_df = run_sign_flip(mae_matrix, 'MAE', models_list, N_PERM, rng)
perm_df.to_csv(OUT_DIR / 'sign_flip_permutation_mae.csv', index=False)
print(f'Saved: {OUT_DIR}/sign_flip_permutation_mae.csv')


# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f'N sites: {n_sites}  |  N models: {n_models}')
print(f'N pairwise comparisons: {len(list(combinations(models_list,2)))}')
print()
print('IMPORTANT: With N=20 sites, the Wilcoxon test has limited power.')
print('Non-significant results reflect small sample size, not model equivalence.')
print('Report effect sizes (median per-site MAE difference) alongside p-values.')
print()
print('Files written:')
for f in ['friedman_test.csv', 'pairwise_wilcoxon_mae.csv',
          'pairwise_wilcoxon_rmse.csv', 'sign_flip_permutation_mae.csv']:
    print(f'  {OUT_DIR}/{f}')
