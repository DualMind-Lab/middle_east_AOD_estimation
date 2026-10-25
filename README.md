# Regional-Scale AOD Estimation

Estimating Aerosol Optical Depth (AOD) at 550 nm over the Middle East directly from
Sentinel-2 L1C satellite imagery, using AERONET ground station measurements as labels.
The goal is a model that predicts AOD from a raw Sentinel-2 image patch alone — no
atmospheric pre-correction, no auxiliary retrieval product.

## Overview

| | |
|---|---|
| Input | Sentinel-2 L1C, 13 bands, ~5×5 km patch centred on an AERONET station |
| Label | AOD at 550 nm, interpolated from AERONET AOD_500nm / AOD_675nm via the Ångström exponent |
| Samples | 6,183 labelled images |
| Sites | 20 AERONET stations — Cyprus, Egypt, Israel, Kuwait, Oman, Saudi Arabia, Turkey, UAE |
| Date range | 2015-08-20 → 2025-11-30 |

**Why L1C, not L2A:** raw top-of-atmosphere reflectance is used intentionally, so the
model learns to handle atmospheric effects itself rather than relying on a
pre-correction algorithm as an intermediate step.

## Repository Structure

```
Regional-Scale-AOD-Estimation/
├── Data_Collection/         ← AERONET + Sentinel-2 acquisition pipeline (scripts only; see note below)
├── feature_engineering/     ← Train/val/LOSO splitting, feature extraction, feature analysis
├── modeling/                ← Model training, evaluation, LOSO cross-validation, results
├── paper_analysis/          ← Statistics, figures, and Supporting Information of the paper (scripts + small outputs)
└── requirements.txt
```

Each directory has its own `README.md` with full pipeline documentation:
[`Data_Collection/README.md`](Data_Collection/README.md) ·
[`feature_engineering/README.md`](feature_engineering/README.md) ·
[`modeling/README.md`](modeling/README.md) ·
[`paper_analysis/README.md`](paper_analysis/README.md)

## Data Availability

Large binary artifacts are not tracked in this repository (GitHub is not a data host,
and several exceed its 100 MB per-file limit):

| Excluded | Size | How to get it |
|---|---|---|
| Raw Sentinel-2 triplets, merged 13-band images, `dataset.csv` | tens of GB | Available on [Hugging Face](https://huggingface.co/datasets/BaraaLazkani/Middle_East_AOD_Dataset) |
| `feature_engineering/features/` (extracted feature CSVs, transformers) | ~8.5 GB | Regenerate locally by running the `feature_engineering/src/` notebooks against `dataset.csv` |
| `modeling/models/` (trained model weights) | several GB | Regenerate locally by running `modeling/src/01_train.ipynb` |

Everything needed to reproduce the above from source — every processing script and
every notebook — is included in this repo. Evaluation outputs (`modeling/results/`) are
committed, so results can be inspected without re-running the full pipeline.

## Reproducing the Pipeline

```bash
python -m venv venv
venv/Scripts/activate        # or source venv/bin/activate on Linux/macOS
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu132
```

Copy `.env.example` to `.env` and fill in Copernicus Data Space Ecosystem (CDSE)
credentials if re-running the acquisition scripts.

Execution order:

1. `Data_Collection/src/` — AERONET filtering → Sentinel-2 download → merge → `dataset.csv` (see its README for the full 8-step pipeline)
2. `feature_engineering/src/01_split.ipynb` → `02_features.ipynb` / `02b_features_loso.ipynb` → `03`–`06` analysis notebooks
3. `modeling/src/01_train.ipynb` → `02_evaluate.ipynb` → `03_loso.ipynb` → `04_evaluate_loso.ipynb`
4. `paper_analysis/` — cluster-bootstrap statistics, figures, and Supporting Information tables used in the paper (see its README for the order)

## Authors

**Baraa Lazkani** · **Modar Ibrahim**
[DualMind-Lab](https://github.com/DualMind-Lab)

## Citation

Paper currently under peer review — citation details to follow upon publication.

## License

Released under the [MIT License](LICENSE).
