# Note on the expected-error (EE) columns in this folder

`per_site_metrics.csv` and `pooled_metrics.csv` report `pct_within_EE` computed with the envelope ±(0.05 + 0.20 τ), which is not the
envelope used in the paper. The paper uses ±(0.05 + 0.15 τ) (Levy et al., 2013). RMSE, MAE, MBE, correlations, and R² in these two files
are unaffected. For the paper's EE values use `../metrics/all_metrics.csv` (pooled) and `../../../paper_analysis/outputs/cluster_bootstrap_ci.csv`
(with site-clustered confidence intervals).

Example: Nicosia, Lasso (log1p): 87.45 % within ±(0.05 + 0.20 τ) as stored here, 78.24 % within ±(0.05 + 0.15 τ).
TabPFN-MI300 (log1p), pooled LOSO: 77.95 % here, 72.83 % in `all_metrics.csv`.
