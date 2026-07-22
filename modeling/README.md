# Modeling

Model training, evaluation, and leave-one-site-out (LOSO) cross-validation for AOD estimation.

## Directory Structure

```
modeling/
├── src/
│   ├── 01_train.ipynb       ← Train 9 base models + TabPFN study on a given split (set SPLIT variable)
│   ├── 02_evaluate.ipynb    ← Full metric panel, bootstrap CIs, permutation tests, per-bin analysis
│   ├── 03_loso.ipynb        ← Leave-one-site-out CV across 20 AERONET sites
│   └── 04_evaluate_loso.ipynb ← LOSO site-level analyses: cluster bootstrap CIs, site-paired permutation, forest plot
├── models/
│   ├── stratified/          ← {model}_{transform}.joblib
│   └── holdout_time/
├── results/
│   ├── predictions/
│   │   ├── stratified/      ← {model}_{transform}.csv  (y_true, y_pred, split, model, target_transform)
│   │   ├── holdout_time/
│   │   └── loso/            ← {model}_{transform}.csv  (same schema; auto-picked up by 02_evaluate)
│   ├── metrics/             ← all_metrics.csv  (point estimates + bootstrap 95% CIs, all 3 splits)
│   ├── significance/        ← pairwise_permutation.csv
│   ├── per_bin/             ← per_bin_mae_mbe.csv
│   ├── plots/               ← scatter_{split}.png, rmse_comparison.png, per_bin_mae.png
│   └── loso/                ← per_site_metrics.csv, loso_predictions.csv, loso_cluster_bootstrap_ci.csv,
│                               loso_site_permutation.csv, plots
```

Reads feature CSVs from `feature_engineering/features/{split}/{train,val}.csv` for train/val splits,
and from `feature_engineering/features/loso/{site}/{train,val}.csv` for LOSO folds.

---

## Execution Order

Run each notebook via **Kernel → Restart & Run All**. Run `01_train.ipynb` twice (once per split).

| Step | Notebook | SPLIT | Prerequisite |
|------|----------|-------|-------------|
| 1a | `01_train.ipynb` | `stratified` | features extracted |
| 1b | `01_train.ipynb` | `holdout_time` | features extracted |
| 2 | `02_evaluate.ipynb` | — reads all predictions (stratified + holdout_time + loso) | steps 1a + 1b + 3 |
| 3 | `03_loso.ipynb` | — reads from `features/loso/{site}/` | LOSO features extracted (`02b_features_loso.ipynb`) |
| 4 | `04_evaluate_loso.ipynb` | — reads LOSO results | step 3 |

---

## Models (10 total)

| Model | Feature input | Hardware |
|---|---|---|
| LightGBM | Raw ~3,700 features | GPU |
| XGBoost | Raw ~3,700 features | GPU (CUDA) |
| CatBoost | Raw ~3,700 features | GPU |
| Random Forest | Raw ~3,700 features | CPU (parallel) |
| Extra Trees | Raw ~3,700 features | CPU (parallel) |
| Gradient Boosting | Raw ~3,700 features | CPU |
| Ridge | PCA-150 (99% var) | CPU |
| Lasso | PCA-150 | CPU |
| SVR-RBF | PCA-150 | CPU |
| TabPFN v2.6 | PCA-150 or MI top-N (see study below) | GPU (CUDA) |

All models use default / near-default parameters. No Optuna tuning — described as "default baseline comparison" in the paper.

**Target transforms**: `raw` (AOD as-is) and `log1p`. Both are evaluated; the paper reports whichever minimises RMSE on each split.

### TabPFN Feature Study (`01_train.ipynb`)

TabPFN is benchmarked with 4 feature configurations × 2 transforms = 8 runs per split:

| Config | Features |
|--------|----------|
| `tabpfn_pca150` | PCA-150 components (99% variance) |
| `tabpfn_mi100` | Top-100 by mutual information |
| `tabpfn_mi200` | Top-200 by mutual information |
| `tabpfn_mi300` | Top-300 by mutual information |

Results from the stratified split: `tabpfn_mi300` achieved the best RMSE on both transforms.
In LOSO, only `tabpfn_pca150` and `tabpfn_mi300` are evaluated (the two best configs).

---

## Evaluation Protocol

**Canonical AOD-retrieval metric panel** (Ichoku et al. 2002 / MODIS Dark Target convention):

| Metric | Description |
|---|---|
| RMSE | Root mean squared error |
| MAE | Mean absolute error |
| MBE | Mean bias error (signed) |
| RMB | Relative mean bias: mean(ŷ) / mean(y) |
| Pearson r | Primary correlation (field standard — AOD magnitude matters) |
| Spearman ρ | Secondary rank-based check |
| R² | Full-range only; not reported in narrow bins |
| slope, intercept | Linear fit ŷ vs y |
| %within EE | Fraction within EE = ±(0.05 + 0.20·τ) — Ichoku envelope |
| %above EE, %below EE | Over/under-retrieval fractions |

All point metrics reported with **95% bootstrap CIs** (1,000 resamples, paired indices, percentile method).

**Per-bin**: MAE and MBE in 5 AOD quintile bins on y_true (avoids unstable R² in narrow bins).

**Pairwise significance**: all model pairs, paired permutation test on |error| vectors (10,000 permutations, two-sided).

---

## LOSO Cross-Validation (`03_loso.ipynb` + `04_evaluate_loso.ipynb`)

Evaluates spatial generalisation: for each of the 20 AERONET sites, train on the remaining 19 and predict on the held-out site.

- Reads **pre-extracted LOSO features** from `feature_engineering/features/loso/{site}/train.csv` and `val.csv`. These were produced by `02b_features_loso.ipynb` with the QT fit exclusively on each fold's 19-site training set — no normalisation leakage.
- PCA (150 components) is refit per fold on the 19-site training features. MI top-300 selection is also refit per fold for `tabpfn_mi300`.
- **Prerequisite:** run `feature_engineering/src/02b_features_loso.ipynb` first (~7 hours).
- `03_loso.ipynb` outputs: per-site metrics, RMSE heatmap, per-site boxplot, and per-(model, transform) prediction CSVs to `predictions/loso/` (auto-picked up by `02_evaluate.ipynb`).
- `04_evaluate_loso.ipynb` outputs: cluster-bootstrap CIs (resampling over sites), site-paired permutation tests, per-site forest plot.

