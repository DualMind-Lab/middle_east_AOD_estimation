"""Extract the definitions of all 84 spectral indices from feature_engineering/src/02_features.ipynb (for Supporting Information Tables S3-S5)."""
import json, re
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
nb = json.load(open(REPO / 'feature_engineering/src/02_features.ipynb'))
src = next(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code' and 'def calculate_vegetation_indices' in ''.join(c['source']))
def block(fn):
    i = src.index('def ' + fn); j = src.find('\ndef ', i + 5); return src[i:j if j > 0 else None]
def parse_dict(b):
    body = re.search(r'indices = \{(.*?)\n    \}', b, flags=re.S).group(1)
    parts = re.split(r"\n\s*'(\w+)'\s*:\s*", '\n' + body)
    return [(parts[k], re.sub(r'\s+', ' ', parts[k + 1].strip().rstrip(','))) for k in range(1, len(parts), 2)]
def clean(e):
    e = e.replace(' + EPS', '').replace('+EPS', '')
    e = e.replace('**2', '^2').replace('np.arctan2', 'arctan2').replace('np.log', 'ln').replace('np.deg2rad', 'rad').replace('np.cos', 'cos')
    return re.sub(r'\s*\*\s*', ' * ', e)
fam = {'veg1': parse_dict(block('calculate_vegetation_indices(')), 'veg2': parse_dict(block('calculate_vegetation_indices2')),
       'rededge': parse_dict(block('calculate_rededge_indices')), 'aod': parse_dict(block('calculate_aod_indices'))}
b = block('calculate_bloom_indices')
fam['bloom'] = [(m.group(1), re.sub(r'\s+', ' ', m.group(2).strip())) for m in re.finditer(r'^\s*(NDGI|DYI|NDPI|NDYI|PEBI|YEBI)\s*=\s*(.*)$', b, flags=re.M)]
json.dump({k: [(n, clean(e)) for n, e in v] for k, v in fam.items()}, open(REPO / 'paper_analysis/outputs/index_defs.json', 'w'), indent=1)
print({k: len(v) for k, v in fam.items()})
