# Data Collection

The pipeline is split into three phases:

1. **AERONET Measurements** — download and clean the ground-truth AOD labels
2. **Sentinel-2 Imagery** — download, clean, and merge the satellite images
3. **Final Dataset** — join labels to images and produce `dataset.csv`

---

## Phase 1: AERONET Measurements

### Step 1: Download Global AERONET Data

All AOD measurements were downloaded from the [AERONET (Aerosol Robotic Network)](https://aeronet.gsfc.nasa.gov/) portal as **Version 3, Level 2.0** (cloud-cleared and quality-assured) for all available sites and time periods globally. The downloaded archive is:

```
Data_Collection/All_Sites_Times_All_Points_AOD20.zip
```

### Step 2: Filter to Middle East Sites

The raw global dataset contained **55,461,878 observations** across all AERONET stations worldwide. A Python script filtered it to only Middle East stations, verified against the official AERONET site registry (`aeronet_locations_v3.txt`, fetched 2026-04-19).

**Script:** `Data_Collection/src/filter_middle_east.py`  
**Output:** `Data_Collection/temp_csv_files/Middle_East_AOD.csv`

| | Count |
|---|---|
| Total rows in raw file | 55,461,878 |
| Rows kept (Middle East) | 3,212,072 |
| Sites | 56 |

Countries represented: **Bahrain, Cyprus, Egypt, Iran, Iraq, Israel, Jordan, Kuwait, Oman, Saudi Arabia, Turkey, UAE**

> No AERONET stations exist for Lebanon, Syria, Qatar, Yemen, or Palestine in this dataset.

### Step 3: Filter to Sentinel-2 Era (2015 onwards)

Sentinel-2 launched in 2015, so all AERONET rows before 2015 are unusable — no satellite imagery exists for them.

**Script:** `Data_Collection/src/filter_2015.py`  
**Output:** `Data_Collection/temp_csv_files/Middle_East_AOD_2015.csv`

| | Count |
|---|---|
| Rows before filtering | 3,212,072 |
| Rows kept (2015 onwards) | 2,061,772 |
| Sites | 29 |
| Unique (site, date) combinations | 33,418 |

> 27 of the original 56 Middle East sites had no measurements from 2015 onwards and were dropped entirely by this filter.

### Step 4: Compute AOD at 550nm and Clean Columns

AOD at 550nm is not directly measured by AERONET. It is interpolated from the two nearest measured wavelengths (500nm and 675nm) using the Ångström exponent formula. Only rows where **both** AOD_500nm and AOD_675nm are valid (> 0) are kept — rows missing either wavelength cannot produce a reliable 550nm estimate and are dropped. All non-essential columns are removed, leaving only the 10 columns needed for model training.

```
α        = −ln(AOD_500 / AOD_675) / ln(500 / 675)
AOD(550) =  AOD_500 × (550 / 500) ^ (−α)
```

**Script:** `Data_Collection/src/compute_aod550.py`  
**Output:** `Data_Collection/temp_csv_files/Middle_East_AOD_550nm.csv`

| | Count |
|---|---|
| Rows before | 2,061,772 |
| Rows dropped (missing AOD_500nm or AOD_675nm) | 112,194 |
| Rows kept | 1,949,578 |
| Sites | 28 |

> IASBS (Iran) lost all of its rows at this step — every one of its measurements was missing at least one of the two required wavelengths, dropping the site count from 29 to 28.

#### Columns

| # | Column |
|---|---|
| 1 | AERONET_Site |
| 2 | Date(dd:mm:yyyy) |
| 3 | Time(hh:mm:ss) |
| 4 | Day_of_Year |
| 5 | Day_of_Year(Fraction) |
| 6 | AOD_550nm *(computed)* |
| 7 | Site_Latitude(Degrees) |
| 8 | Site_Longitude(Degrees) |
| 9 | Site_Elevation(m) |
| 10 | Solar_Zenith_Angle(Degrees) |

---

## Phase 2: Sentinel-2 Imagery

### Step 5: Download Sentinel-2 L1C Images

For each unique (site, date) combination in `temp_csv_files/Middle_East_AOD_2015.csv`, the Sentinel-2 catalog is queried for that day. Any overpass whose timestamp falls within **±30 minutes** of an AERONET measurement at that site is downloaded. The ±30-minute window ensures the atmospheric state seen by the satellite matches the ground measurement. Images are downloaded at all 13 native bands across 3 resolution groups.

**Script:** `Data_Collection/src/download_sentinel2_L1C.py`  
**Output:** `Data_Collection/Sentinel2_L1C/{site_name}/{yyyymmdd_HHMMSS}_{resolution}m.tif`

#### Specifications

| Parameter | Value |
|---|---|
| Spatial extent | 5×5 km centered on site lat/lon |
| Time match window | ±30 minutes from AERONET measurement |
| Bands (10m) | B02, B03, B04, B08 |
| Bands (20m) | B05, B06, B07, B8A, B11, B12 |
| Bands (60m) | B01, B09, B10 |
| Total bands | 13 |
| Pixel values | UINT16, scaled ×10000 (typical range 0–10000; values >10000 are valid and represent bright surfaces or clouds in L1C TOA data) |
| Source | Copernicus Data Space Ecosystem (CDSE) |
| Timestamps | Both AERONET and Sentinel-2 timestamps are UTC |

#### Output Structure

```
Data_Collection/Sentinel2_L1C/
└── {site_name}/
    ├── {yyyymmdd_HHMMSS}_10m.tif   ← B02, B03, B04, B08
    ├── {yyyymmdd_HHMMSS}_20m.tif   ← B05, B06, B07, B8A, B11, B12
    └── {yyyymmdd_HHMMSS}_60m.tif   ← B01, B09, B10
```

#### Per-Site Image Count (before cleanup)

| Site | Country | Triplets |
|---|---|---|
| AgiaMarina_Xyliatou | Cyprus | 770 |
| CUT-TEPAK | Cyprus | 932 |
| Cairo_EMA_2 | Egypt | 411 |
| DEWA_ResearchCentre | UAE | 70 |
| Eilat | Israel | 165 |
| El_Farafra | Egypt | 344 |
| IASBS | Iran | 32 |
| IMS-METU-ERDEMLI | Turkey | 634 |
| KAUST_Campus | Saudi Arabia | 718 |
| KITcube_Masada | Israel | 1 |
| KITcube_WadiRum | Jordan | 2 |
| Kuwait_University | Kuwait | 474 |
| Masdar_Institute | UAE | 426 |
| Mezaira | UAE | 229 |
| Migal | Israel | 356 |
| Nes_Ziona | Israel | 1 |
| Nicosia | Cyprus | 609 |
| Pafos | Cyprus | 6 |
| Qena_SVU | Egypt | 134 |
| SEDE_BOKER | Israel | 421 |
| Shagaya_Park | Kuwait | 266 |
| TUBITAK_UZAY_Ankara | Turkey | 1 |
| Technion_Haifa_IL | Israel | 174 |
| Troodos_CAO | Cyprus | 54 |
| Tuz_Golu_3 | Turkey | 8 |
| University_of_Nizwa | Oman | 101 |
| Weizmann_Institute | Israel | 319 |
| **Total** | **27 sites** | **7,658** |

> The 2015 CSV contained 29 sites, but 2 of them — Cairo_EMA and Kuwait_Inst_Sci_Res — never had a Sentinel-2 overpass within ±30 minutes of any AERONET measurement and produced zero downloaded images.

### Step 6: Remove Low-Coverage Sites and Clean Up Orphan Images

With image triplet counts now known, sites contributing fewer than **10 triplets** are removed from `Middle_East_AOD_550nm.csv`. A site with fewer than 10 samples is too sparse for the model to learn any meaningful pattern from, and the samples are insufficient even as general training examples given the noise inherent in AOD data.

After the CSV is updated, the image folder is scanned for **orphan triplets** — images that have no surviving AERONET row in `Middle_East_AOD_550nm.csv` within ±30 minutes of the image timestamp. Orphans arise from two sources:

1. Their site was removed above (low coverage, or IASBS which lost all CSV rows in Step 4)
2. Their only matching AERONET measurement was dropped in Step 4 due to missing wavelength data, and no other valid measurement for that site falls within the ±30-minute window

**Scripts:** one-off pandas filter applied directly to `Middle_East_AOD_550nm.csv` (CSV removal) · `Data_Collection/src/cleanup_images.py` (image deletion)

#### Sites Removed from CSV

| Site removed | Country | Triplets | Reason |
|---|---|---|---|
| KITcube_Masada | Israel | 1 | Insufficient coverage |
| Nes_Ziona | Israel | 1 | Insufficient coverage |
| TUBITAK_UZAY_Ankara | Turkey | 1 | Insufficient coverage |
| KITcube_WadiRum | Jordan | 2 | Insufficient coverage |
| Pafos | Cyprus | 6 | Insufficient coverage |
| Tuz_Golu_3 | Turkey | 8 | Insufficient coverage |

| | Count |
|---|---|
| Sites before | 28 |
| Sites removed (low coverage) | 6 |
| Sites after | 22 |
| CSV rows removed | 9,171 |
| CSV rows remaining | 1,940,407 |

#### Images Deleted

| | Count |
|---|---|
| Triplets before cleanup | 7,658 |
| Triplets deleted (orphaned) | 427 |
| Files deleted | 1,281 |
| **Triplets kept** | **7,231** |
| **Sites with images** | **20** |

> IASBS had 32 image triplets but 0 surviving CSV rows (all dropped in Step 4), so its images were deleted here as orphans. The 6 low-coverage sites contributed a combined 19 triplets (1+1+1+2+6+8). Together these 7 sites account for 32 + 19 = 51 of the 427 deleted triplets. The remaining 376 are individual acquisitions from otherwise-valid sites whose only matching AERONET measurement happened to be missing both wavelengths in Step 4, leaving that specific image with no label.

### Step 7: Upsample and Merge Image Triplets

Each acquisition's three resolution GeoTIFFs are merged into a single 13-band GeoTIFF at 10m resolution. The 20m bands are upsampled ×2 and the 60m bands are upsampled ×6 using **nearest-neighbour interpolation** (no new values are invented — each upsampled pixel is a copy of its source pixel). The 10m image defines the exact output dimensions.

**Script:** `Data_Collection/src/upsample_and_merge.py`  
**Output:** `Data_Collection/Dataset/images/{site_name}/{yyyymmdd_HHMMSS}.tif`

#### Output Band Order (wavelength order)

| Index | Band | Native res | Description |
|---|---|---|---|
| 0 | B01 | 60m | Coastal aerosol |
| 1 | B02 | 10m | Blue |
| 2 | B03 | 10m | Green |
| 3 | B04 | 10m | Red |
| 4 | B05 | 20m | Red-edge 1 |
| 5 | B06 | 20m | Red-edge 2 |
| 6 | B07 | 20m | Red-edge 3 |
| 7 | B08 | 10m | NIR |
| 8 | B8A | 20m | Narrow NIR |
| 9 | B09 | 60m | Water vapour |
| 10 | B10 | 60m | SWIR cirrus |
| 11 | B11 | 20m | SWIR 1 |
| 12 | B12 | 20m | SWIR 2 |

| | Count |
|---|---|
| Triplets merged | 7,231 |
| Sites | 20 |
| Output size per image | ~6.25 MB (~500×500×13 px, UINT16) |

---

## Phase 3: Final Dataset

### Step 8: Build Final Dataset CSV

Each row in `temp_csv_files/Middle_East_AOD_550nm.csv` is matched to its closest merged image (within ±30 minutes) using binary search on the image timestamps. Rows with no matching image are dropped — these are AERONET measurements for days when Sentinel-2 was not overhead.

**Script:** `Data_Collection/src/build_dataset_csv.py`  
**Output:** `Data_Collection/Dataset/dataset.csv`

#### Temporal Aggregation

Multiple AERONET measurements within the ±30-minute window of the same Sentinel-2 overpass are **averaged into a single mean AOD value** per image. Images with only **1 matching measurement are discarded** — a minimum of 2 measurements is required to produce a statistically representative label. Non-AOD metadata (site, coordinates, solar zenith, day-of-year) is taken from the measurement closest in time to the satellite overpass.

This follows the standard matchup protocol used in the remote sensing community:

- **NASA Dark Target Validation Protocol** averages all AERONET Level 2 measurements within ±30 min and requires ≥2 for a valid matchup — the gold standard for MODIS AOD global validation. ([darktarget.gsfc.nasa.gov/validation](https://darktarget.gsfc.nasa.gov/validation))
- **Sayer et al. (2020)** quantify AERONET temporal variability: σ ≈ 0.019 at 30 min, confirming the mean over the window is representative and within retrieval uncertainty. ([AGU Earth and Space Science, doi:10.1029/2020EA001290](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020EA001290))
- **Virtanen et al. (2018)** analyse collocation mismatch uncertainties and confirm ±30 min mean averaging as standard practice. ([AMT, doi:10.5194/amt-11-925-2018](https://amt.copernicus.org/articles/11/925/2018/))

| | Count |
|---|---|
| Input AERONET rows | 1,940,407 |
| Rows matched to an image | 33,572 |
| Rows dropped (no overpass) | 1,906,835 |
| Images dropped (<2 measurements) | 643 |
| **Final rows in dataset.csv** | **6,183** |
| **Sites** | **20** |

#### Final Columns

| # | Column |
|---|---|
| 1 | AERONET_Site |
| 2 | Date(dd:mm:yyyy) |
| 3 | Time(hh:mm:ss) |
| 4 | Day_of_Year |
| 5 | Day_of_Year(Fraction) |
| 6 | AOD_550nm *(mean of ≥2 measurements)* |
| 7 | Site_Latitude(Degrees) |
| 8 | Site_Longitude(Degrees) |
| 9 | Site_Elevation(m) |
| 10 | Solar_Zenith_Angle(Degrees) |
| 11 | image_path |

---

## Final Dataset Summary

| Property | Value |
|---|---|
| Total samples | 6,183 |
| Unique images | 6,183 |
| Sites | 20 |
| Date range | 2015-08-20 → 2025-11-30 |
| AOD label | Mean of ≥2 AERONET measurements per image |

### Sites in Final Dataset

| Site | Country | Samples |
|---|---|---|
| AgiaMarina_Xyliatou | Cyprus | 465 |
| CUT-TEPAK | Cyprus | 831 |
| Nicosia | Cyprus | 493 |
| Troodos_CAO | Cyprus | 47 |
| Cairo_EMA_2 | Egypt | 317 |
| El_Farafra | Egypt | 89 |
| Qena_SVU | Egypt | 109 |
| Eilat | Israel | 155 |
| Migal | Israel | 338 |
| SEDE_BOKER | Israel | 383 |
| Technion_Haifa_IL | Israel | 155 |
| Weizmann_Institute | Israel | 297 |
| Kuwait_University | Kuwait | 424 |
| Shagaya_Park | Kuwait | 240 |
| University_of_Nizwa | Oman | 99 |
| KAUST_Campus | Saudi Arabia | 560 |
| IMS-METU-ERDEMLI | Turkey | 515 |
| DEWA_ResearchCentre | UAE | 69 |
| Masdar_Institute | UAE | 376 |
| Mezaira | UAE | 221 |
| **Total** | | **6,183** |

### Samples per Year

| Year | Samples |
|---|---|
| 2015 | 37 |
| 2016 | 202 |
| 2017 | 492 |
| 2018 | 670 |
| 2019 | 702 |
| 2020 | 908 |
| 2021 | 691 |
| 2022 | 771 |
| 2023 | 693 |
| 2024 | 583 |
| 2025 | 434 |
| **Total** | **6,183** |

> 2015 is low because Sentinel-2 L1C became available in June 2015. 2025 covers only up to November 2025 (end of available data at time of download).

### Directory Structure

```
Data_Collection/
├── src/                          ← all processing scripts
│   ├── filter_middle_east.py
│   ├── filter_2015.py
│   ├── download_sentinel2_L1C.py
│   ├── compute_aod550.py
│   ├── cleanup_images.py
│   ├── upsample_and_merge.py
│   └── build_dataset_csv.py
├── temp_csv_files/               ← intermediate CSVs (not used for training)
│   ├── Middle_East_AOD.csv
│   ├── Middle_East_AOD_2015.csv
│   └── Middle_East_AOD_550nm.csv
├── Sentinel2_L1C/                ← raw triplets (10m/20m/60m per acquisition)
│   └── {site_name}/
│       └── {yyyymmdd_HHMMSS}_{10|20|60}m.tif
├── Dataset/                      ← final training dataset
│   ├── dataset.csv               ← 6,183 labelled samples with image paths
│   └── images/
│       └── {site_name}/
│           └── {yyyymmdd_HHMMSS}.tif   ← 13-band merged GeoTIFF at 10m
└── README.md
```

> **Data availability:** `temp_csv_files/`, `Sentinel2_L1C/`, and `Dataset/` (including
> `dataset.csv` and all 6,183 merged images) are not included in this repository due to
> size. The dataset is available at
> [huggingface.co/datasets/BaraaLazkani/Middle_East_AOD_Dataset](https://huggingface.co/datasets/BaraaLazkani/Middle_East_AOD_Dataset).
> All processing scripts in `src/` are provided so the pipeline can also be reproduced
> from source AERONET/Sentinel-2 data.
