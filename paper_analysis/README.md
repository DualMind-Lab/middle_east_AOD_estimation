# Paper analysis

Scripts and small outputs behind the statistics, figures, and Supporting Information of the manuscript
*Machine Learning and Tabular Foundation Models for Aerosol Optical Depth Estimation From Sentinel-2 L1C Imagery Over the Middle East:
Within-Region Skill and the Spatial Generalization Gap* (Earth and Space Science, manuscript 2026EA005451).

All scripts are repository-relative (they find the repository root from their own location). Inputs that are not tracked here:

| Input | How to get it | Environment variable |
|---|---|---|
| `dataset.csv` | Download from the [Hugging Face dataset](https://huggingface.co/datasets/BaraaLazkani/Middle_East_AOD_Dataset) | `AOD_DATASET` (default `data/dataset.csv`) |
| Sentinel-2 images (only `revision_2026/reflectance.py`) | Same Hugging Face dataset; the folder containing `Data_Collection/Dataset/images/` | `AOD_IMAGES_ROOT` |
| Feature CSVs (only `revision_2026/mi_top300*.py`) | Regenerate with `feature_engineering/src/02_features.ipynb` and `02b_features_loso.ipynb` | `AOD_FEATURES` (default `feature_engineering/features`) |
| Natural Earth I basemap (only `figures/make_fig1.py`) | [50m Natural Earth I with shaded relief](https://www.naturalearthdata.com/downloads/50m-raster-data/50m-natural-earth-1/) | `NE1_TIF` |

Model predictions and metrics are read from `modeling/results/`.

## Order of use

1. `statistics/01_cluster_bootstrap.py` — site-clustered bootstrap CIs and paired differences for all three designs (`outputs/cluster_bootstrap_ci.csv`, `outputs/pairwise_diff_ci.csv`).
2. `statistics/02_model_comparison.py`, `03_friedman_all_splits.py` — Friedman omnibus tests, Wilcoxon and sign-flip tests (`outputs/friedman_*.csv`, `pairwise_wilcoxon_*.csv`, `sign_flip_permutation_mae.csv`, `bootstrap_cis.csv`).
3. `revision_2026/rel_metrics.py` — RMSE, MAE, MBE as a percentage of the mean observed AOD (Table 7; `outputs/rel_metrics.csv`).
4. `revision_2026/reflectance.py` — median top-of-atmosphere reflectance of the clearest scenes per station (Figure 1c; `outputs/site_reflectance.csv`).
5. `revision_2026/mi_top300.py`, `mi_top300_b.py` — composition of the mutual-information top-300 feature set (Section 5.2; `outputs/mi_top300*.json`). `mi_top300_all_folds.py` repeats the check for the stratified, hold-out, and all 20 LOSO training sets (`outputs/mi_top300_all_folds.json`; requires the LOSO feature CSVs).
6. `revision_2026/feature_families.py`, `extract_index_defs.py`, `build_si.py` — feature list (Data Set S1, `outputs/ds01_feature_list.csv`), spectral-index definitions, and the LaTeX source of the Supporting Information (`si/si.tex`; compile with AGU's `agutexSI2019.cls` and `apacite`).
7. `figures/make_figures.py`, `make_fig1.py`, `make_fig2_fig11.py`, `fig_geo_map.py` — Figures 1–11 as vector PDFs (written to `figures/out/`). The forest and bin plots (Figures 3–7, 9, 10) come from `make_figures.py`; the map (Figure 8) from `fig_geo_map.py`.

## Notes

* Station labels for the stratified and hold-out designs are recovered by matching `y_true` to `AOD_550nm` in `dataset.csv`; 40 values occur in more than one row (about 0.6 % of observations) and the first match is used. The effect on the cluster bootstrap is negligible.
* `outputs/ds01_feature_list.csv` is Data Set S1 of the Supporting Information: the name, category code, and category name (as in Table S2 and Figure 2) of each of the 3,700 retained features.
* `revision_2026/build_si.py` notes that the `mbi` index as implemented in the feature code is constant by construction (its ratio equals 1), so most of its statistics were removed by the variance filter.
