"""G12 step 5: composition-adjusted senescence vs metabolic lead-lag (T50) in atlas (rat, mouse) and TMS (mouse, 10 ages)."""
import sys; sys.path.insert(0, "/home/claude/g12/scripts")
import numpy as np, pandas as pd
from lib import *
S = gene_sets()

def datasets():
    for sp in ["rat", "mouse"]:
        X = pd.read_parquet(W + f"{sp}_logcpm.parquet"); X = X.groupby(level=0).mean()
        M = pd.read_csv(W + f"{sp}_meta.csv", index_col=0); M.index = M.index.astype(str)
        M = M[M.tissue.isin(["gWAT", "liver", "left ventricle", "kidney"])]
        yield f"atlas_{sp}", X, M[["tissue", "Sex", "age_m"]]
        G = pd.read_parquet(W + f"{sp}_gastroc_logcpm.parquet"); GM = pd.read_csv(W + f"{sp}_gastroc_meta.csv", index_col=0)
        yield f"atlas_{sp}", G, GM
    T = pd.read_parquet(W + "tms_logcpm.parquet"); TM = pd.read_csv(W + "tms_meta.csv", index_col=0)
    TM = TM[(TM.age_m >= 3) & (TM.tissue != "Pancreas")]  # adults only; pancreas dropped (43% of libraries failed QC)
    yield "TMS_mouse", T, TM

rows = []; allscores = []
for name, X, M in datasets():
    for (tis, sex), mm in M.groupby(["tissue", "Sex"]):
        z = zscore_tissue(X[mm.index])
        P, n = program_scores(z, S)
        C, nc = program_scores(z, CELL)
        P["Senescence_adj"] = residualise(P["Senescence"], C)
        age = mm.age_m.values.astype(float)
        sc = pd.concat([P, C], axis=1); sc["age"] = age; sc["dataset"] = name; sc["tissue"] = tis; sc["Sex"] = sex
        allscores.append(sc)
        r = dict(dataset=name, tissue=tis, sex=sex, n=len(mm), n_ages=len(np.unique(age)))
        for k in ["Senescence", "Senescence_adj", "Metabolic", "Macrophage", "Plasma"]:
            r[f"rho_{k}"], r[f"p_{k}"] = trend(age, P[k].values if k in P else C[k].values)
        for sen in ["Senescence", "Senescence_adj"]:
            ta, tb, d, lo, hi, p = lead_lag(age, P[sen].values, P["Metabolic"].values, nboot=1000)
            r.update({f"T50_{sen}": ta, "T50_Metabolic": tb, f"lag_{sen}": d, f"lagLo_{sen}": lo, f"lagHi_{sen}": hi, f"lagP_{sen}": p})
        rows.append(r); print(name, tis, sex, "done", flush=True)
R = pd.DataFrame(rows); R.to_csv(W + "leadlag.csv", index=False)
pd.concat(allscores).to_csv(W + "all_scores.csv")
