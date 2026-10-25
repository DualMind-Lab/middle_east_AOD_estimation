"""Relative (percentage) errors: metric / mean observed AOD of the evaluation set."""
import pandas as pd, numpy as np
import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]   # repository root
DATASET = Path(os.environ.get('AOD_DATASET', REPO / 'data' / 'dataset.csv'))   # download dataset.csv from the Hugging Face dataset
ci=pd.read_csv(REPO/'paper_analysis/outputs/cluster_bootstrap_ci.csv')
rows=[]
for split in ['stratified','holdout_time','loso']:
    p=pd.read_csv(REPO/f'modeling/results/predictions/{split}/tabpfn_mi300_raw.csv') if split!='loso' else pd.read_csv(REPO/'modeling/results/predictions/loso/tabpfn_mi300_log1p.csv')
    mean_aod=p.y_true.mean()
    sub=ci[ci.split==split].copy()
    for m in ['RMSE','MAE','MBE']:
        sub['r'+m]=100*sub[m]/mean_aod
    sub['mean_aod']=mean_aod
    rows.append(sub[['split','model','N','mean_aod','RMSE','MAE','MBE','rRMSE','rMAE','rMBE']])
out=pd.concat(rows); out.to_csv(REPO/'paper_analysis/outputs/rel_metrics.csv',index=False)
print(out.groupby('split').mean_aod.first())
print(out.round(3).to_string())
