"""Composition of the MI top-300 feature set in every training set used by the paper:
stratified, hold-out, and all 20 LOSO folds (same recipe as modeling/src/03_loso.ipynb:
mutual information with the raw AOD target, random_state=42; computed here in parallel feature chunks).
Output: paper_analysis/outputs/mi_top300_all_folds.json"""
import os, sys, json, re
import numpy as np, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_regression
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_families import cat, META
REPO = Path(__file__).resolve().parents[2]
FEAT = Path(os.environ.get('AOD_FEATURES', REPO / 'feature_engineering/features'))
SUF = re.compile(r'_(mean|var|median|kurtosis|skew|std|p10|p25|p75|p90|GLCM_\w+)$')
jobs = {'stratified': FEAT / 'stratified/train.csv', 'holdout_time': FEAT / 'holdout_time/train.csv'}
for d in sorted((FEAT / 'loso').iterdir()):
    if (d / 'train.csv').exists(): jobs['loso_' + d.name] = d / 'train.csv'
out = {}
outp = REPO / 'paper_analysis/outputs/mi_top300_all_folds.json'
for name, p in jobs.items():
    df = pd.read_csv(p); cols = [c for c in df.columns if c != 'AOD_550nm']
    X = df[cols].values.astype(np.float32); y = df['AOD_550nm'].values
    chunks = np.array_split(np.arange(X.shape[1]), 40)
    mi = np.concatenate(Parallel(n_jobs=4)(delayed(mutual_info_regression)(X[:, c], y, random_state=42) for c in chunks))
    order = np.argsort(mi)[::-1]; names = [cols[i] for i in order]
    top = names[:300]
    suf = [SUF.search(c).group(1) if SUF.search(c) else 'other' for c in top]
    out[name] = {'n_features': len(cols),
                 'meta_in_top300': [c for c in top if c in META],
                 'best_meta_rank': min(names.index(c) + 1 for c in META if c in names),
                 'n_std_or_var_in_top300': int(sum(s in ('std', 'var') for s in suf)),
                 'top300_by_category': pd.Series([cat(c) for c in top]).value_counts().to_dict()}
    print(name, out[name]['meta_in_top300'], out[name]['best_meta_rank'], out[name]['n_std_or_var_in_top300'], flush=True)
    json.dump(out, open(outp, 'w'), indent=1)
