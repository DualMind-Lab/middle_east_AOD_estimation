# Feature Engineering

This directory contains the full feature engineering pipeline for AOD estimation from Sentinel-2 L1C imagery.

## Directory Structure

```
feature_engineering/
├── src/
│   ├── 01_split.ipynb         ← Step 1: Split dataset.csv — stratified, holdout_time, and 20 LOSO folds
│   ├── 02_features.ipynb      ← Step 2a: Extract features for stratified / holdout_time (set SPLIT)
│   ├── 02b_features_loso.ipynb← Step 2b: Extract features for all 20 LOSO folds in one run
│   ├── 03_pca.ipynb                          ← Step 3a: PCA — dimensionality and variance analysis
│   ├── 04_correlation.ipynb                  ← Step 3b: Correlation with AOD and multicollinearity
│   ├── 05_feature_categories.ipynb           ← Step 3c: Per-category contribution analysis
│   └── 06_downsampling_sensitivity_analysis.ipynb  ← Step 3d: 128×128 resolution sensitivity analysis
├── splits/
│   ├── stratified_train.csv
│   ├── stratified_val.csv
│   ├── holdout_time_train.csv
│   └── holdout_time_val.csv
├── features/                  ← Extracted features; input to modeling/
│   ├── raw_all.csv             ← All 6,183 samples, raw (pre-QT) features (Phase 1 of 02b)
│   ├── stratified/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── transformers.joblib
│   ├── holdout_time/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── transformers.joblib
│   └── loso/                  ← One directory per site (output of 02b_features_loso.ipynb)
│       ├── AgiaMarina_Xyliatou/
│       │   ├── train.csv       ← 19-site training features, QT fit here
│       │   ├── val.csv         ← held-out site features, QT applied from train
│       │   └── transformers.joblib
│       └── ... (one directory per site)
└── features_analysis/         ← PCA, correlation, feature importance notebooks
```

Model training and evaluation live in `modeling/` (separate directory), which reads feature CSVs from `feature_engineering/features/`.

> **Note:** the `features/` directory itself (extracted feature CSVs and
> `transformers.joblib` files, ~8.5 GB total) is not included in this repository — it
> is far too large for GitHub and is fully regenerable. Run `01_split.ipynb` →
> `02_features.ipynb` / `02b_features_loso.ipynb` against `Data_Collection/Dataset/dataset.csv`
> to reproduce it locally.

---

## Step 1 — Data Splitting (`01_split.ipynb`)

Splits `Data_Collection/Dataset/dataset.csv` (6,183 samples, 20 sites) into train/val pairs for two strategies. Splitting is done **before** feature extraction to prevent data leakage.

LOSO folds are derived inside `02b_features_loso.ipynb` by partitioning `dataset.csv` directly on `AERONET_Site` — no pre-computed LOSO split CSVs are needed.

Columns dropped before splitting: `Time(hh:mm:ss)` (AERONET measurement time, not satellite overpass) and `Day_of_Year(Fraction)` (redundant with `Day_of_Year`).

### Split 1 — Stratified by Site

For each of the 20 sites, samples are sorted chronologically and split 80/20. Every site appears in both train and val.

| Set   | Samples | % |
|-------|---------|---|
| Train | 4,940   | 79.9% |
| Val   | 1,243   | 20.1% |

**Tests:** Does the model generalise across time within known sites?

### Split 2 — Hold-out Time

Samples from 2024–2025 are held out. All 20 sites appear in training.

| Set   | Samples | % | Period |
|-------|---------|---|--------|
| Train | 5,166   | 83.6% | 2015–2023 |
| Val   | 1,017   | 16.4% | 2024–2025 |

**Tests:** Does the model generalise to future acquisitions?

---

## Step 2 — Feature Extraction

### Step 2a — Stratified & Hold-out Time (`02_features.ipynb`)

Reads each split's train/val CSVs and the corresponding GeoTIFFs, extracts features, applies all transformers fit on train only, and writes processed feature CSVs ready for modeling.

Change `SPLIT` in cell 1 to switch between splits; run **Kernel → Restart & Run All** each time.

### Image Loading

Each GeoTIFF is resized from its native ~500 × 500 pixels (10 m, ~5 km × 5 km) to **128 × 128** using `cv2.INTER_AREA` (area-averaging, geometrically correct for downsampling). Two filters are applied before accepting an image:

- **Coverage filter** (`min_coverage = 0.90`): drop if any band has < 90 % valid pixels (DN in (0, 10 000])
- **Cirrus filter**: drop if B10 (SWIR-cirrus) 95th-percentile DN > 300

Remaining bad pixels (nodata + cloud, ≤ 10 % by construction) are replaced with the per-band scene mean of valid pixels. Two arrays are returned per image: `bands_sqrt` (√DN, for statistics) and `bands_raw` (DN / 10 000 reflectance, for physical indices).

Sample counts after filtering:

| Split | Train kept | Val kept |
|---|---|---|
| stratified | 4,099 / 4,940 | 1,035 / 1,243 |
| holdout_time | 4,300 / 5,166 | 834 / 1,017 |
| LOSO (all samples pooled in raw_all.csv) | varies ~4,700–5,100 per fold | varies ~40–700 per fold |

### Feature Groups

All features are extracted per image (spatial statistics over the 128 × 128 patch):

| Group | Description | Features |
|---|---|---|
| Band statistics | Mean, variance, median, kurtosis, skew, std, p10, p25, p75, p90 for each of 13 bands | 130 |
| Band ratios | Normalised difference `(a−b)/(a+b)` and raw ratio `a/b` for all 78 band pairs, 10 stats each | 1,560 |
| Advanced band features | Geometric mean, harmonic mean, normalised ratio for 78 pairs; per-band Shannon entropy; double-product and cross-ratio for all 286 triple-band combinations | 516 |
| GLCM texture (B02 raw) | Contrast, dissimilarity, homogeneity, energy, correlation — 4-direction averaged on fixed [0, 0.5] reflectance scale | 5 |
| Vegetation indices (set 1) | 20 indices (NDVI, EVI, SAVI, BSI, NDWI, …) — 10 stats + 5 GLCM each | 300 |
| Vegetation indices (set 2) | 25 indices (BSI variants, DBSI, NDSI, …) — same 15 features each | 375 |
| Bloom / dust indices | 6 indices (NDGI, NDPI, PEBI, …) — 15 features each | 90 |
| Red-edge indices | 18 indices (NDVIre, NDRE, CIre, MCARI, TCARI, MTCI) — 15 features each | 270 |
| AOD-physics indices | 14 indices designed around aerosol scattering physics (B01/B11, B02/B12, spectral curvature, AFRI, Ångström α) — 10 stats each | 140 |
| Metadata | Day of year, lat, lon, elevation, SZA, SAA, airmass, year, month | 9 |

Raw feature count per sample: **~3,709**.

### Post-processing (fit on train, applied to both)

Applied in this order, transformers fit on train data only:

1. **Column alignment** — assert val has no columns absent from train
2. **Inf replacement** — ±inf → NaN
3. **Outlier clip** — mask `|value| > 1e4` (catches EPS-singularity blow-ups in signed-denominator indices such as SIPI, MNDVI, MTCI where denominator → ±EPS produces values up to ~1/EPS = 10⁶)
4. **NaN drop** — drop columns where > 80 % of train values are NaN
5. **NaN imputation** — fill remaining NaN with per-column train mean
6. **Variance filter** (image features only) — drop columns with train std < 1e-5; ~3,692 / 3,700 image features kept
7. **QuantileTransformer** (`output_distribution='normal'`) — fit on all image features, applied to train and val
8. **QuantileTransformer (meta)** — separate QT fit on the 9 metadata features

Final feature count per split: **~3,701** (1 AOD target + 9 meta + ~3,691 image).

### Outputs

Each split produces three files under `features/{split}/`:

| File | Contents |
|---|---|
| `train.csv` | (N_train, ~3,701) — AOD_550nm + all features, QT-normalised |
| `val.csv` | (N_val, ~3,701) — same schema, transformers applied from train fit |
| `transformers.joblib` | `qt`, `qt_meta`, `keep_cols`, `train_means`, `kept_img_cols`, `meta_feat_cols`, `final_cols` |

### Step 2b — LOSO Feature Extraction (`02b_features_loso.ipynb`)

Two-phase design so heavy image loading runs **once** rather than 20 times:

**Phase 1** (run once, ~30 min): Loads all 6,183 images, applies coverage + cirrus filters, extracts the same ~3,700 raw features as Step 2a. Saves `features/raw_all.csv` — inf-replaced and outlier-clipped, but **no QT and no imputation** (those are fold-dependent and must not see the held-out site). Set `SKIP_PHASE1 = True` to skip if `raw_all.csv` already exists.

**Phase 2** (20× fast, ~5 min total): Loops over all 20 sites. For each fold:
1. Splits `raw_all.csv` by `AERONET_Site` into 19-site train and 1-site val
2. Fits QT, variance filter, NaN imputation, and meta QT **exclusively on the 19-site train set**
3. Applies transformers to the held-out site's val set
4. Saves to `features/loso/{site}/train.csv`, `val.csv`, `transformers.joblib`

The QT is refit per fold — there is no normalisation leakage from the held-out site. This makes the LOSO in `modeling/src/03_loso.ipynb` a clean spatial generalisation test.

Expected runtime: Phase 1 ≈ 30 min | Phase 2 ≈ 5 min total. Completed folds are skipped automatically (`RESUME = True`).

### Known Gaps — Step 2

- **Mean imputation before GLCM**: bad pixels are replaced with the scene mean before GLCM computation, which creates artificial intensity boundaries. Severity is bounded by the 90 % coverage filter (≤ 10 % pixels ever imputed) and the 1st–99th percentile normalisation in `to_uint8_glcm()`, which places imputed values near the patch median rather than at extreme contrast. See rationale comment in `02_features.ipynb`.
- **128 × 128 downsampling**: driven by RAM constraints (~50 GB would be required at native resolution). INTER_AREA averaging is radiometrically correct and one GLCM pixel at 128 × 128 represents ~39 m, still meaningful for land-cover texture at the AOD footprint scale (~5 km). Sensitivity analysis in `06_downsampling_sensitivity_analysis.ipynb` confirms that 128 × 128 sits in the flat region of the resolution curve (median feature r vs native = 0.999); increasing to 160 × 160 improves median r by only 0.0002.
- **Feature-to-sample ratio**: ~3,700 features against ~4,000–5,100 training samples. PCA analysis (`03_pca.ipynb`) shows the effective dimensionality is 13–150 components (not 3,700), giving a true ratio of 0.003–0.037. Distance-based models (SVR-RBF) should still be treated with caution.
- **Physical interpretability**: which features drive predictions is not assessed here. SHAP values and permutation importance are deferred to `modeling/` once a best model is identified.

---

## Step 3 — Feature Analysis (`03` – `06` notebooks)

All four notebooks read `features/stratified/train.csv` (4,099 samples, ~3,700 features). Outputs are written to `features_analysis/`.

---

### Step 3a — PCA (`03_pca.ipynb`)

Fits full PCA on the QT-normalised training features to characterise effective dimensionality.

**Results:**

| Variance threshold | Components needed | % of 3,700 features | Compression |
|---|---|---|---|
| 90% | 13 | 0.35% | 285× |
| 95% | 29 | 0.78% | 128× |
| 99% | 150 | 4.05% | 25× |

The feature space is highly redundant. Despite ~3,700 raw features, 99% of variance is captured by 150 components. The true feature-to-sample ratio for the training set is therefore 0.003–0.037, not the nominal 0.86–0.93.

**Outputs:** `features_analysis/pca/cumulative_variance.png`, `pca_scatter_aod.png`, `pca_summary.csv`

---

### Step 3b — Correlation & Multicollinearity (`04_correlation.ipynb`)

Ranks all 3,700 features by Spearman correlation and mutual information with AOD 550nm. Examines inter-feature collinearity among the top 40.

**Results:**

- Strongest Spearman correlations: `cratio_1_8_9` (|r| = 0.567), `cratio_0_8_9` (|r| = 0.567), `rdiff_9_12_mean` (|r| = 0.563). The top features are cross-ratio triplets and band differences involving B09 (water vapour), which co-varies with aerosol loading in the Middle East.
- Strongest mutual information: `rdiff_0_1_std` (MI = 0.353), `r04_std` (MI = 0.352). Variance/dispersion statistics of blue–coastal-aerosol differences dominate.
- Maximum individual |r| across all features: 0.567. No single feature is strongly predictive on its own — the task requires combining many weak signals, motivating ensemble/tree models.
- 120 highly collinear pairs (|r| ≥ 0.95) among the top 40 features. Most are mathematically equivalent pairs (e.g., `rdiff` vs `rdiv` for the same band pair), confirming the redundancy shown by PCA.

**Outputs:** `features_analysis/correlation/top30_spearman.png`, `top30_mi.png`, `inter_feature_correlation.png`, `spearman_all.csv`, `mutual_info_all.csv`, `high_corr_pairs.csv`

---

### Step 3c — Feature Category Analysis (`05_feature_categories.ipynb`)

Groups all 3,700 features into 10 categories and measures per-category AOD signal (median |Spearman r|) and internal redundancy (PCA compression ratio at 95% variance).

**Results:**

| Category | Features | Median |r| | PCA compression |
|---|---|---|---|
| meta | 9 | 0.291 | 1.5× |
| advanced | 819 | 0.248 | 137× |
| band_stats | 130 | 0.227 | 13× |
| aod_physics | 150 | 0.180 | 13× |
| band_ratios | 1,560 | 0.178 | 82× |
| veg_idx2 | 368 | 0.156 | 14× |
| bloom | 90 | 0.146 | 8× |
| veg_idx1 | 300 | 0.140 | 10× |
| rededge | 269 | 0.121 | 11× |
| glcm_b02 | 5 | 0.025 | 1.7× |

Key findings: (1) Metadata (site latitude, SZA, airmass) carries the strongest per-feature signal — geographic location is a strong AOD prior in the Middle East. (2) The advanced category (cross-ratios, entropy) ranks second with 819 features that collapse to just 6 independent dimensions, indicating extreme redundancy alongside high signal. (3) The aod_physics category ranks 4th, above all vegetation index categories. (4) GLCM texture (median |r| = 0.025) contributes negligible linear signal; its utility, if any, is through non-linear interactions captured by tree models.

**Outputs:** `features_analysis/categories/category_summary.png`, `category_summary.csv`

---

### Step 3d — Downsampling Sensitivity Analysis (`06_downsampling_sensitivity_analysis.ipynb`)

Tests whether the 128 × 128 px downsampling choice materially affects the features. A stratified sample of 100 images (25 per AOD quartile) was loaded at native resolution (~500 × 500 px) and features extracted at 64, 96, 128, and 160 px using `cv2.INTER_AREA`. Features: 13 band means, 13 band stds, NDVI mean/std, B01/B11 ND ratio mean/std, 5 GLCM properties (32 total).

**Results:**

| Resolution | Median r vs native | % features r > 0.99 | Mean \|ΔAOD-r\| |
|---|---|---|---|
| 64 × 64 | 0.9965 | 74% | 0.0139 |
| 96 × 96 | 0.9982 | 77% | 0.0100 |
| **128 × 128** | **0.9989** | **77%** | **0.0076** |
| 160 × 160 | 0.9991 | 80% | 0.0073 |

At 128 × 128, the median feature correlation with native resolution is 0.999. Increasing to 160 × 160 improves this by 0.0002 and reduces the mean AOD-correlation error by 0.0003 — both within statistical noise. The 128 × 128 choice sits in the flat region of the sensitivity curve where further resolution adds no meaningful information, making the RAM trade-off (3.5 GB vs ~50 GB at native) fully justified.

**Outputs:** `features_analysis/sensitivity_analysis_downsampling/sensitivity_analysis.png`, `resolution_summary.csv`, `per_feature_results.csv`


