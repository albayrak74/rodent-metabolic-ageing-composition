"""Sample QC: within each dataset x tissue, Pearson r of each sample to the tissue median profile (expressed genes).
Flag samples with r < 0.90 (gross library failure or tissue mislabel); a stricter MAD rule removed whole age groups and was rejected. Flagged samples are removed from the *_meta.csv files."""
import numpy as np, pandas as pd, os
W = "/home/claude/g12/work/"
sets = [("rat_logcpm.parquet", "rat_meta.csv"), ("mouse_logcpm.parquet", "mouse_meta.csv"),
        ("rat_gastroc_logcpm.parquet", "rat_gastroc_meta.csv"), ("mouse_gastroc_logcpm.parquet", "mouse_gastroc_meta.csv"),
        ("tms_logcpm.parquet", "tms_meta.csv"), ("bodymap_log.parquet", "bodymap_meta.csv")]
log = []
for xf, mf in sets:
    raw = W + mf.replace(".csv", "_preQC.csv")
    if not os.path.exists(raw): os.rename(W + mf, raw)
    M = pd.read_csv(raw, index_col=0); M.index = M.index.astype(str)
    X = pd.read_parquet(W + xf); X.columns = X.columns.astype(str); X = X.groupby(level=0).mean()
    keep = []
    for tis, mm in M.groupby("tissue"):
        x = X[mm.index]; x = x[x.mean(axis=1) > 1]
        med = x.median(axis=1); r = x.corrwith(med)
        mad = np.median(np.abs(r - r.median())) + 1e-6; rz = (r - r.median()) / mad
        badmask = (r < 0.90); thr = 0.90
        bad = r[badmask]
        for s, v in bad.items(): log.append(dict(dataset=xf.split("_log")[0], tissue=tis, sample=s, age=mm.loc[s, "age_m"], r=v, threshold=thr))
        keep += list(r[~badmask].index)
    M.loc[keep].to_csv(W + mf)
L = pd.DataFrame(log); L.to_csv(W + "qc_excluded.csv", index=False)
pd.set_option("display.width", 200); print(L.round(3).to_string(index=False)); print("excluded:", len(L))
