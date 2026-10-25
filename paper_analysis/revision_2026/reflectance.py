"""Per-site median TOA reflectance of the cleanest-atmosphere scenes (lowest AOD quintile per site).
Reads images read-only from the external drive. Output: site_reflectance.csv"""
import numpy as np, pandas as pd, tifffile, sys
from multiprocessing import Pool
import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]   # repository root
DATASET = Path(os.environ.get('AOD_DATASET', REPO / 'data' / 'dataset.csv'))   # download dataset.csv from the Hugging Face dataset
ROOT = os.environ.get('AOD_IMAGES_ROOT', str(REPO)) + '/'   # folder that contains Data_Collection/Dataset/images/ (image_path in dataset.csv is relative to it)
DS = DATASET
BANDS = {'B01':0,'B02':1,'B03':2,'B04':3,'B08':7,'B11':11,'B12':12}
MAXN = 12
df = pd.read_csv(DS)
jobs = []
for site, g in df.groupby('AERONET_Site'):
    g = g.sort_values('AOD_550nm').head(max(MAXN*2, int(0.2*len(g))))   # candidate pool: low-AOD quintile
    g = g.head(MAXN*2)
    jobs.append((site, g[['image_path','AOD_550nm']].values.tolist()))

def one(path):
    a = tifffile.imread(ROOT + path.replace('Data_Collection','Data_Collection',1))   # (H,W,13)
    if a.ndim != 3: return None
    a = a[::4, ::4, :].astype(np.float32)
    valid = (a > 0) & (a <= 10000)
    if (valid.mean(axis=(0,1)) < 0.9).any(): return None                      # coverage screen
    if np.percentile(a[..., 10][valid[..., 10]], 95) > 300: return None        # cirrus screen
    return {b: float(np.median(a[..., i][valid[..., i]]) / 1e4) for b, i in BANDS.items()}

def site_job(job):
    site, items = job
    rows = []
    for path, aod in items:
        try:
            r = one(path)
        except Exception as e:
            r = None
        if r: r['aod'] = aod; rows.append(r)
        if len(rows) >= MAXN: break
    if not rows: return {'site': site, 'n': 0}
    d = pd.DataFrame(rows)
    out = {'site': site, 'n': len(d), 'aod_median': float(d.aod.median())}
    for b in BANDS: out[b] = float(d[b].median())
    return out

if __name__ == '__main__':
    with Pool(4) as p:
        res = p.map(site_job, jobs)
    pd.DataFrame(res).to_csv(REPO/'paper_analysis/outputs/site_reflectance.csv', index=False)
    print(pd.DataFrame(res).round(3).to_string())
