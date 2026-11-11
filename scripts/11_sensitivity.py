"""Reviewer-driven sensitivity: immune-only adjustment, BH-FDR across strata, Cdkn1a single-gene trend."""
import sys; sys.path.insert(0, "/home/claude/g12/scripts")
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
W = "/home/claude/g12/work/"
S = pd.read_csv(W + "all_scores.csv", index_col=0)
imm = ["Macrophage", "T_NK", "B", "Plasma", "Neutrophil"]; allc = imm + ["Endothelial", "Fibroblast"]
def resid(g, y, cols):
    C = g[cols].loc[:, g[cols].notna().all() & (g[cols].std() > 0)]
    X = np.c_[np.ones(len(g)), C.values]; b = np.linalg.lstsq(X, y, rcond=None)[0]; return y - X[:, 1:] @ b[1:]
rows = []
for (d, t, s), g in S.groupby(["dataset", "tissue", "Sex"]):
    y = g.Senescence.values; r = dict(dataset=d, tissue=t, sex=s, n=len(g))
    for lab, v in [("raw", y), ("adj_all", resid(g, y, allc)), ("adj_immune", resid(g, y, imm))]:
        r[f"rho_{lab}"], r[f"p_{lab}"] = stats.spearmanr(g.age, v)
    r["rho_met"], r["p_met"] = stats.spearmanr(g.age, g.Metabolic)
    rows.append(r)
R = pd.DataFrame(rows)
for lab in ["raw", "adj_all", "adj_immune", "met"]:
    for grp in [R.dataset.str.startswith("atlas"), R.dataset == "TMS_mouse"]:
        R.loc[grp, f"q_{lab}"] = multipletests(R.loc[grp, f"p_{lab}"], method="fdr_bh")[1]
# Cdkn1a single gene (log2 CPM), raw and immune-adjusted
def load(sp, tis):
    if tis == "gastrocnemius":
        X = pd.read_parquet(W + f"{sp}_gastroc_logcpm.parquet"); M = pd.read_csv(W + f"{sp}_gastroc_meta.csv", index_col=0)
    else:
        X = pd.read_parquet(W + f"{sp}_logcpm.parquet"); M = pd.read_csv(W + f"{sp}_meta.csv", index_col=0); M.index = M.index.astype(str); M = M[M.tissue == tis]
    return X.groupby(level=0).mean()[M.index], M
c = []
for sp, ds in [("rat", "atlas_rat"), ("mouse", "atlas_mouse")]:
    for tis in ["gWAT", "liver", "left ventricle", "kidney", "gastrocnemius"]:
        X, M = load(sp, tis)
        for sex, mm in M.groupby("Sex"):
            g = S[(S.dataset == ds) & (S.tissue == tis) & (S.Sex == sex)].copy(); g.index = g.index.astype(str)
            v = X.loc["Cdkn1a", mm.index].values; g = g.loc[mm.index] if set(mm.index) <= set(g.index) else None
            if g is None: continue
            rr, pr = stats.spearmanr(mm.age_m, v); ra, pa = stats.spearmanr(mm.age_m, resid(g, v, imm))
            c.append(dict(dataset=ds, tissue=tis, sex=sex, mean_log2cpm=v.mean(), rho_Cdkn1a=rr, p_Cdkn1a=pr, rho_Cdkn1a_immadj=ra, p_Cdkn1a_immadj=pa))
C = pd.DataFrame(c); R.to_csv(W + "sensitivity_adjustment.csv", index=False); C.to_csv(W + "cdkn1a_trend.csv", index=False)
pd.set_option("display.width", 250); print(R.round(3).to_string(index=False)); print(C.round(3).to_string(index=False))
a = R[R.dataset.str.startswith("atlas")]
print("atlas: raw q<.05 & rho>0:", ((a.q_raw < .05) & (a.rho_raw > 0)).sum(), " adj_all q<.05:", ((a.q_adj_all < .05) & (a.rho_adj_all > 0)).sum(),
      " adj_immune q<.05:", ((a.q_adj_immune < .05) & (a.rho_adj_immune > 0)).sum(), " median rho imm-adj", a.rho_adj_immune.median().round(2), a.rho_adj_immune.quantile([.25,.75]).round(2).tolist())
t = R[R.dataset == "TMS_mouse"]; print("TMS adj_immune q<.05:", (t.q_adj_immune < .05).sum(), "median", t.rho_adj_immune.median().round(2))
