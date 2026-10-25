"""Assign each of the 3,700 retained features to one of the ten categories used in the paper
(same rule as feature_engineering/src/05_feature_categories.ipynb) and write the feature list (Data Set S1)."""
import re
import pandas as pd
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
META = ['Day_of_Year', 'Site_Latitude(Degrees)', 'Site_Longitude(Degrees)', 'Site_Elevation(m)', 'Solar_Zenith_Angle(Degrees)',
        'Solar_Azimuth_Angle(Degrees)', 'Airmass', 'Year', 'Month']
STAT_SUFFIX = r'_(mean|var|median|kurtosis|skew|std|p10|p25|p75|p90|GLCM_\w+)$'

def cat(col):
    if col in META: return 'meta'
    if col.startswith('aod_'): return 'aod_physics'
    if col.startswith(('NDVIre', 'NDRE', 'CIre', 'MCARI', 'TCARI', 'MTCI')): return 'rededge'
    if col.startswith(('NDGI', 'DYI', 'NDPI', 'PEBI', 'NDYI', 'YEBI')): return 'bloom'
    if col.startswith(('ndvi_r', 'chl', 'nmdi', 'wbi', 'msi', 'bsi1', 'bsi2', 'ndsi1', 'ndsi2', 'bi2', 'bi', 'si_', 'dbsi', 'mbi', 'mresr',
                       'psri', 'tvi', 'arvi', 'exg', 'aci', 'r01', 'r02', 'r03', 'r04', 'mi')): return 'veg_idx2'
    if col.startswith(('ndvi', 'gli', 'cvi', 'sipi', 'ndwi', 'ccci', 'hue', 'rendvi', 'reci', 'evi', 'bsi', 'npcri', 'mndvi',
                       'ndsi', 'ndmi', 'ndbi', 'ndci', 'gndvi', 'savi')): return 'veg_idx1'
    if col.startswith('B02raw_GLCM_'): return 'glcm_b02'
    if col.startswith(('geomean_', 'harmmean_', 'normratio_', 'dprod_', 'cratio_')) or col.endswith('_entropy'): return 'advanced'
    if col.startswith(('rdiff_', 'rdiv_')): return 'band_ratios'
    if col.startswith(('B0', 'B1', 'B8')): return 'band_stats'
    return 'other'

if __name__ == '__main__':
    names = pd.read_csv(REPO / 'feature_engineering/features_analysis/correlation/spearman_all.csv')['feature'].tolist()
    df = pd.DataFrame({'feature': names, 'category': [cat(c) for c in names]})
    assert len(df) == 3700 and 'other' not in set(df.category)
    out = REPO / 'paper_analysis/outputs/ds01_feature_list.csv'
    df.to_csv(out, index=False)
    print(df.category.value_counts().to_string()); print('wrote', out)
