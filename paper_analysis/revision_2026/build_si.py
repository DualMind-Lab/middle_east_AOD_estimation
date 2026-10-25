import json,re,pandas as pd,collections
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
OUTD = REPO/'paper_analysis/outputs'
defs=json.load(open(OUTD/'index_defs.json'))
fl=pd.read_csv(OUTD/'ds01_feature_list.csv')
S=r'_(mean|var|median|kurtosis|skew|std|p10|p25|p75|p90|GLCM_\w+)$'
fl['base']=[re.sub(S,'',c) for c in fl.feature]
cnt=fl.groupby('base').size().to_dict()
def clean(e):
    e=re.sub(r'(?<![A-Za-z])\((B\w+)\)',r'\1',e)
    e=e.replace('(1.0 / cos(rad(sza_deg)))[:, None]','m')
    e=e.replace('_','\\_')
    return e
def tt(e): return '\\texttt{'+e.replace('%','\\%').replace('&','\\&').replace('^','\\^{}')+'}'
def rows(fam,aodfam=False):
    out=[]
    for n,e in defs[fam]:
        c=cnt.get(n,0); note='' if (c in (15,10)) else f' ({c} of 15 retained)'
        out.append(f"{tt(n.replace('_','\\_'))} & {tt(clean(e))}{note} \\\\")
    return out
bands=[('0','B01','443','60','Coastal aerosol'),('1','B02','490','10','Blue'),('2','B03','560','10','Green'),('3','B04','665','10','Red'),('4','B05','705','20','Red edge 1'),('5','B06','740','20','Red edge 2'),('6','B07','783','20','Red edge 3'),('7','B08','842','10','Near infrared'),('8','B8A','865','20','Narrow near infrared'),('9','B09','945','60','Water vapour'),('10','B10','1375','60','Cirrus'),('11','B11','1610','20','Shortwave infrared 1'),('12','B12','2190','20','Shortwave infrared 2')]
tab1="\n".join(f"{i} & {b} & {w} & {r} & {d} \\\\" for i,b,w,r,d in bands)
cat=fl.groupby(lambda i: 0)  # placeholder
ncat=fl.category.value_counts().to_dict()
fams=[
('Scene metadata','meta','Day of year, latitude, longitude, elevation, solar zenith angle, solar azimuth angle (from the overpass time), airmass $=1/\\cos(\\mathrm{SZA})$, year, month.','\\texttt{Day\\_of\\_Year}, \\texttt{Airmass}'),
('Per-band statistics','band_stats','Ten statistics (mean, variance, standard deviation, median, skewness, kurtosis, 10th, 25th, 75th, 90th percentile) of each of the 13 bands, computed over the 128$\\times$128 pixels on square-root-scaled digital numbers.','\\texttt{B02\\_std}, \\texttt{B8A\\_p90}'),
('Band differences and ratios','band_ratios','For each of the 78 band pairs $i<j$ (band indices of Table S1), the normalized difference $(B_i-B_j)/(B_i+B_j)$ and the ratio $B_i/B_j$ per pixel, each summarized by the ten statistics (78$\\times$2$\\times$10).','\\texttt{rdiff\\_0\\_1\\_std}, \\texttt{rdiv\\_4\\_5\\_p90}'),
('Advanced band interactions','advanced','Per band pair: geometric mean $\\sqrt{B_iB_j}$, harmonic mean, and normalized ratio $(B_i-B_j)/\\sqrt{B_i^2+B_j^2}$ (78$\\times$3, scene means); per-band Shannon entropy of the 256-bin histogram (13); per band triple $i<j<k$: product difference $B_iB_j-B_iB_k$ and cross ratio $B_iB_j/B_k$ (286$\\times$2, scene means). Square-root-scaled digital numbers.','\\texttt{geomean\\_1\\_3}, \\texttt{B02\\_entropy}, \\texttt{cratio\\_1\\_8\\_9}'),
('Blue-band texture','glcm_b02','Gray-level co-occurrence features (contrast, dissimilarity, homogeneity, energy, correlation) of the blue band B02, computed on a fixed reflectance scale of 0--0.5 mapped to 256 gray levels, pixel distance 1, averaged over four directions.','\\texttt{B02raw\\_GLCM\\_contrast}'),
('Vegetation indices I','veg_idx1','20 indices (Table S3), each with the ten statistics plus five GLCM features (15 per index).','\\texttt{ndvi\\_mean}, \\texttt{evi\\_GLCM\\_energy}'),
('Vegetation and soil indices II','veg_idx2','25 further vegetation, soil, moisture, and band-ratio indices (Table S3), 15 features per index; seven features were removed by the filtering steps.','\\texttt{msi\\_p25}, \\texttt{r04\\_std}'),
('Bloom and dust indices','bloom','Six indices built from B02, B03, B04, and B08 (Table S4), 15 features per index.','\\texttt{NDYI\\_mean}'),
('Red-edge indices','rededge','18 indices using B05--B07 (Table S4), 15 features per index; one feature was removed by the filtering steps.','\\texttt{NDRE1\\_p10}'),
('Aerosol-physics indices','aod_physics','15 indices built around blue/SWIR contrast, spectral curvature, dust, and Angstrom-like band slopes (Table S5), ten statistics per index (no texture features), computed on reflectance.','\\texttt{aod\\_B01\\_over\\_B02\\_std}'),
]
tab2="\n".join(f"{n} & {ncat[k]:,} & {d} & {e} \\\\" for n,k,d,e in fams)
assert sum(ncat[k] for _,k,_,_ in fams)==3700
vegrows=rows('veg1')+['\\multicolumn{2}{l}{\\textit{Vegetation and soil indices II}} \\\\']+rows('veg2')
bloomrows=rows('bloom')+['\\multicolumn{2}{l}{\\textit{Red-edge indices}} \\\\']+rows('rededge')
aodrows=rows('aod')+[f"{tt('aod\\_angstrom\\_alpha')} & slope of $\\ln\\rho$ against $\\ln\\lambda$ over B01--B04 (443, 490, 560, 665~nm), least-squares fit per pixel \\\\"]
aodrows=[r.replace('aod\\_B02\\_airmass','aod\\_B02\\_airmass') for r in aodrows]
tex=r"""\documentclass[draft,jgrga]{agutexSI2019}
\usepackage{graphicx}
\setkeys{Gin}{draft=false}
\begin{document}
\title{Supporting Information for ``Machine Learning and Tabular Foundation Models for Aerosol Optical Depth Estimation From Sentinel-2 L1C Imagery Over the Middle East: Within-Region Skill and the Spatial Generalization Gap''}
\authors{Baraa Lazkani\affil{1}, Modar Ibrahim\affil{1}}
\affiliation{1}{Department of Robotics and Intelligent Systems Engineering, Manara University, Latakia, Syria}
\begin{article}
\noindent\textbf{Contents of this file}
\begin{enumerate}
\item Text S1
\item Tables S1 to S5
\end{enumerate}
\noindent\textbf{Additional Supporting Information (Files uploaded separately)}
\begin{enumerate}
\item Caption for Data Set S1
\end{enumerate}

\noindent\textbf{Introduction}

This file documents the 3,700 image-derived and metadata features used in the main article. Text S1 describes how the features are computed and named, Table S1 gives the band numbering used in feature names, Table S2 describes the ten feature categories shown in Figure 2 and Figure 11 of the main article, and Tables S3 to S5 list the definition of every spectral index. Data Set S1 lists the name and category of every feature. The processing code that generates the features is available in the code repository cited in the Open Research section of the main article.

\noindent\textbf{Text S1. Feature construction and naming.}

Each Sentinel-2 L1C image has 13 bands, which are numbered 0 to 12 in feature names (Table S1). All images are reduced from about 500$\times$500 to 128$\times$128 pixels by area averaging before feature extraction. Pixels with digital numbers outside $(0,10^{4}]$ are treated as invalid; an image is rejected if more than 10\% of the pixels of any band are invalid or if the 95th percentile of band B10 exceeds 300, and the remaining invalid pixels are replaced by the band mean. Two arrays are used: square-root-scaled digital numbers (for the band statistics, band differences and ratios, and advanced band interactions) and reflectance, equal to the digital number divided by $10^{4}$ (for the spectral indices and the blue-band texture). A small constant of $10^{-6}$ is added to denominators to avoid division by zero; it is omitted from the definitions below.

The ten distribution statistics used throughout are the mean (\texttt{mean}), variance (\texttt{var}), median (\texttt{median}), kurtosis (\texttt{kurtosis}), skewness (\texttt{skew}), standard deviation (\texttt{std}), and the 10th, 25th, 75th, and 90th percentiles (\texttt{p10}, \texttt{p25}, \texttt{p75}, \texttt{p90}), each computed over the pixels of one image. A feature name is the name of the quantity followed by the statistic, for example \texttt{rdiff\_0\_1\_std} is the standard deviation over the image of the normalized difference between bands 0 (B01) and 1 (B02). Texture features are named with the suffix \texttt{\_GLCM\_} followed by the texture property; for the spectral indices, the gray-level co-occurrence matrix is computed at pixel distance 1 and direction 0, after scaling the index image between its 1st and 99th percentile to 256 levels.

The raw feature matrix has 3,708 columns. Features are then filtered with the training data only: values with an absolute magnitude above $10^{4}$ are set to missing, features with more than 80\% missing training values are removed, missing values are replaced by the training mean, and image-derived features with a training standard deviation below $10^{-5}$ are removed. Eight features were removed in this way, which leaves 3,700 features (Table S2). A quantile transformation to a normal distribution is then applied, separately to the image-derived features and to the nine metadata features. The names and categories of all 3,700 retained features are listed in Data Set S1.

\noindent\textbf{Data Set S1.} \texttt{ds01\_feature\_list.csv}: the 3,700 features used in the study, with one row per feature and the columns \texttt{feature} (feature name as used in the code) and \texttt{category} (one of the ten categories of Table S2). The file is also deposited in the code repository cited in the main article.

\end{article}
\clearpage

\begin{table}
\settablenum{S1}
\caption{Band numbering used in feature names (Sentinel-2 L1C, nominal central wavelengths).}
\centering
\begin{tabular}{c l r r l}
\hline
Index & Band & Wavelength (nm) & Native resolution (m) & Description \\
\hline
"""+tab1+r"""
\hline
\end{tabular}
\end{table}

\begin{table}
\settablenum{S2}
\caption{Feature categories (Figure 2 and Figure 11 of the main article). $n$ is the number of features retained after filtering.}
\centering
\small\baselineskip=11pt\normalbaselineskip=11pt
\begin{tabular}{p{0.15\textwidth} r p{0.45\textwidth} p{0.2\textwidth}}
\hline
Category & $n$ & Content & Example names \\
\hline
"""+tab2+r"""
\hline
Total & 3,700 & & \\
\hline
\end{tabular}
\end{table}

\begin{table}
\settablenum{S3}
\caption{Definition of the vegetation, soil, and moisture indices (top: indices I; bottom: indices II). Each index is computed per pixel from reflectance bands B01--B12, as implemented in the code; names follow the code and may differ from the standard index of the same name. Each index is summarized by ten statistics and five GLCM features.}
\centering
\scriptsize\baselineskip=9pt\normalbaselineskip=9pt
\begin{tabular}{l p{0.72\textwidth}}
\hline
Index & Definition (per pixel) \\
\hline
\multicolumn{2}{l}{\textit{Vegetation and soil indices I}} \\
"""+"\n".join(vegrows)+r"""
\hline
\end{tabular}
\end{table}

\begin{table}
\settablenum{S4}
\caption{Definition of the bloom and dust indices and the red-edge indices (per pixel, from reflectance). Each index is summarized by ten statistics and five GLCM features.}
\centering
\scriptsize\baselineskip=9pt\normalbaselineskip=9pt
\begin{tabular}{l p{0.72\textwidth}}
\hline
Index & Definition (per pixel) \\
\hline
\multicolumn{2}{l}{\textit{Bloom and dust indices}} \\
"""+"\n".join(bloomrows)+r"""
\hline
\end{tabular}
\end{table}

\begin{table}
\settablenum{S5}
\caption{Definition of the aerosol-physics-motivated indices (per pixel, from reflectance). $m=1/\cos(\mathrm{SZA})$ is the airmass. Each index is summarized by ten statistics (no texture features).}
\centering
\scriptsize\baselineskip=9pt\normalbaselineskip=9pt
\begin{tabular}{l p{0.62\textwidth}}
\hline
Index & Definition (per pixel) \\
\hline
"""+"\n".join(aodrows)+r"""
\hline
\end{tabular}
\end{table}
\end{document}
"""
(REPO/'paper_analysis/si').mkdir(exist_ok=True)
open(REPO/'paper_analysis/si/si.tex','w').write(tex)   # compile with AGU's agutexSI2019.cls + apacite
# data set S1
print(len(fl), 'features in SI')
