"""G12 step 19: controls requested by anticipated reviewer objections.
(A) Specificity of the composition adjustment: the same adjustment is applied to random gene sets matched to the
    senescence core genes on their individual age correlation, and to the metabolic score.
(B) Female rats: senescence-macrophage coupling and the adult age contrast in female and male Fischer 344 rats (rat body map).
(C) Processing order: library numbers run in age blocks in the atlas, so within each age group the association
    between library number and the scores is tested (a technical drift would show up within age groups).
(E) Within-age coupling: share of the within-age variance of the senescence score (and of matched random sets)
    explained by the composition scores.
(D) Equivalence (TOST) of the pooled lead-lag and a pooled estimate that averages strata sharing animals first."""
import sys, gzip; sys.path.insert(0, "/home/claude/g12/scripts")
import numpy as np, pandas as pd
from scipy import stats
from lib import *
S = gene_sets(); rng = np.random.default_rng(19)
EXCL = set(S["SenMayo"]) | {g for v in CELL.values() for g in v}
NSET = 200

def rank_rho(Z, age):
    """Spearman rho of every row of Z with age (vectorised)."""
    R = Z.rank(axis=1).values; a = stats.rankdata(age)
    R = R - R.mean(1, keepdims=True); a = a - a.mean()
    return pd.Series((R @ a) / (np.sqrt((R ** 2).sum(1)) * np.sqrt((a ** 2).sum()) + 1e-12), index=Z.index)

def datasets():
    for sp in ["rat", "mouse"]:
        X = pd.read_parquet(W + f"{sp}_logcpm.parquet").groupby(level=0).mean()
        M = pd.read_csv(W + f"{sp}_meta.csv", index_col=0); M.index = M.index.astype(str)
        M = M[M.tissue.isin(["gWAT", "liver", "left ventricle", "kidney"])]
        yield f"atlas_{sp}", X, M[["tissue", "Sex", "age_m"]]
        G = pd.read_parquet(W + f"{sp}_gastroc_logcpm.parquet").groupby(level=0).mean(); GM = pd.read_csv(W + f"{sp}_gastroc_meta.csv", index_col=0)
        GM.index = GM.index.astype(str); yield f"atlas_{sp}", G, GM[["tissue", "Sex", "age_m"]]
    T = pd.read_parquet(W + "tms_logcpm.parquet"); TM = pd.read_csv(W + "tms_meta.csv", index_col=0)
    TM = TM[(TM.age_m >= 3) & (TM.tissue != "Pancreas")]
    yield "TMS_mouse", T, TM[["tissue", "Sex", "age_m"]]

# ---------------- (A) specificity ----------------
qc = pd.read_csv(W + "qc_excluded.csv"); bad = set(qc["sample"].astype(str))
rowsA = []
for name, X, M in datasets():
    M = M[~M.index.astype(str).isin(bad)]
    for (tis, sex), mm in M.groupby(["tissue", "Sex"]):
        cols = [c for c in mm.index if c in X.columns]
        z = zscore_tissue(X[cols]); z = z[z.notna().all(axis=1)]; age = mm.loc[cols, "age_m"].values.astype(float)
        C, _ = program_scores(z, CELL)
        rg = rank_rho(z, age)
        sen = [g for g in z.index.intersection(S["Senescence"])]
        pool = rg[[g for g in rg.index if g not in EXCL]]
        edges = np.quantile(rg.values, np.linspace(0, 1, 21)); edges[0] -= 1; edges[-1] += 1
        pbin = pd.cut(pool, edges, labels=False); sbin = pd.cut(rg[sen], edges, labels=False)
        bypool = {b: pool.index[pbin == b].values for b in np.unique(pbin)}
        def adj_rho(genes):
            s = z.loc[genes].mean(0); return trend(age, s)[0], trend(age, residualise(s, C))[0]
        r0, r1 = adj_rho(sen)
        null = []
        for _ in range(NSET):
            g = [rng.choice(bypool[b]) for b in sbin.values if b in bypool and len(bypool[b])]
            null.append(adj_rho(list(dict.fromkeys(g))))
        null = np.array(null)
        met = z.index.intersection(S["Metabolic"]); m0, m1 = adj_rho(list(met))
        rowsA.append(dict(dataset=name, tissue=tis, sex=sex, n=len(cols), rho_sen_raw=r0, rho_sen_adj=r1,
                          null_raw_median=np.median(null[:, 0]), null_adj_median=np.median(null[:, 1]),
                          null_adj_q05=np.quantile(null[:, 1], .05), null_adj_q95=np.quantile(null[:, 1], .95),
                          retained_sen=r1 / r0 if abs(r0) > 1e-9 else np.nan,
                          retained_null_median=np.median(null[:, 1] / null[:, 0]),
                          p_lower=(null[:, 1] <= r1).mean() if r0 > 0 else ((null[:, 1] >= r1).mean()),
                          rho_met_raw=m0, rho_met_adj=m1))
        print("A", name, tis, sex, round(r0, 2), round(r1, 2), round(np.median(null[:, 0]), 2), round(np.median(null[:, 1]), 2), flush=True)
A = pd.DataFrame(rowsA); A.to_csv(W + "control_specificity.csv", index=False)

# ---------------- (B) female rats, rat body map ----------------
B = pd.read_parquet(W + "bodymap_log.parquet"); BM = pd.read_csv(W + "bodymap_meta.csv", index_col=0)
rowsB = []
for (tis, sex), mm in BM.groupby(["tissue", "Sex"]):
    z = zscore_tissue(B[mm.index]); z = z[z.notna().all(axis=1)]
    P, _ = program_scores(z, {"Senescence": S["Senescence"]}); C, _ = program_scores(z, CELL)
    sen = P["Senescence"]; adj = residualise(sen, C); age = mm.age_m.values
    r_sm = stats.pearsonr(sen, C["Macrophage"])[0]
    old, young = age > 20, (age > 4) & (age < 6)
    def g_(y):
        a, b = y[old], y[young]; sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        return (a.mean() - b.mean()) / sp
    rowsB.append(dict(tissue=tis, sex=sex, n=len(mm), r_sen_macrophage=r_sm, d_raw_104_vs_21wk=g_(sen.values), d_adj_104_vs_21wk=g_(adj.values),
                      d_macrophage_104_vs_21wk=g_(C["Macrophage"].values)))
Bt = pd.DataFrame(rowsB); Bt.to_csv(W + "control_female_rat.csv", index=False); print(Bt.round(2))

# ---------------- (C) processing order ----------------
def libs(f):
    rows = {}
    for l in gzip.open(f, "rt"):
        if l.startswith("!Sample_description") or l.startswith("!Sample_title"):
            rows[l.split("\t")[0]] = [x.strip().strip('"') for x in l.rstrip("\n").split("\t")[1:]]
    t = pd.DataFrame(rows); t["lib"] = t["!Sample_description"].str.extract(r"(\d+)")[0]; return set(t.lib)
sc = pd.read_csv(W + "all_scores.csv", index_col=0); sc.index = sc.index.astype(str)
rowsC = []
for (ds, tis, sex), g in sc[sc.dataset.str.startswith("atlas") & (sc.tissue != "gastrocnemius")].groupby(["dataset", "tissue", "Sex"]):
    lib = pd.Series(g.index.astype(int), index=g.index)
    rho_lib_age = stats.spearmanr(lib, g.age)[0]
    for k in ["Senescence", "Macrophage", "Metabolic"]:
        zs, ws = [], []
        for a, gg in g.groupby("age"):
            if len(gg) >= 5:
                r = stats.spearmanr(lib[gg.index], gg[k])[0]
                if np.isfinite(r): zs.append(np.arctanh(np.clip(r, -.999, .999))); ws.append(len(gg) - 3)
        zbar = np.average(zs, weights=ws); se = 1 / np.sqrt(np.sum(ws)); p = 2 * stats.norm.sf(abs(zbar / se))
        rowsC.append(dict(dataset=ds, tissue=tis, sex=sex, score=k, rho_library_age=rho_lib_age, within_age_rho=np.tanh(zbar), p=p))
Ct = pd.DataFrame(rowsC)
from statsmodels.stats.multitest import multipletests
Ct["q"] = multipletests(Ct.p, method="fdr_bh")[1]; Ct.to_csv(W + "control_processing_order.csv", index=False); print(Ct.round(3))

# ---------------- (D) equivalence and clustered pooling ----------------
L = pd.read_csv(W + "leadlag.csv"); L = L[(L.p_Senescence < .05) & (L.p_Metabolic < .05)].copy()
L["se"] = (L.lagHi_Senescence - L.lagLo_Senescence) / (2 * 1.96)
def dl(y, se):
    w = 1 / se ** 2; m = np.sum(w * y) / w.sum(); Q = np.sum(w * (y - m) ** 2); k = len(y)
    tau2 = max(0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())); ws = 1 / (se ** 2 + tau2)
    mu = np.sum(ws * y) / ws.sum(); s = np.sqrt(1 / ws.sum()); return mu, s, tau2, k
def tost(mu, s, m): return max(stats.norm.sf((mu + m) / s), stats.norm.sf((m - mu) / s))
mu, s, tau2, k = dl(L.lag_Senescence.values, L.se.values)
# animals: atlas strata of the same species and sex (non-muscle tissues share animals); muscle came from a separate series
L["cluster"] = L.dataset + "_" + L.sex + np.where(L.tissue == "gastrocnemius", "_muscle", "")
G = L.groupby("cluster").agg(lag=("lag_Senescence", "mean"), se=("se", "mean")).reset_index()
mu2, s2, tau22, k2 = dl(G.lag.values, G.se.values)
D_ = pd.DataFrame([dict(model="all strata", k=k, pooled=mu, lo95=mu - 1.96 * s, hi95=mu + 1.96 * s, lo90=mu - 1.645 * s, hi90=mu + 1.645 * s, tau2=tau2,
                        tost_p_3mo=tost(mu, s, 3), tost_p_2mo=tost(mu, s, 2)),
                   dict(model="strata sharing animals averaged first", k=k2, pooled=mu2, lo95=mu2 - 1.96 * s2, hi95=mu2 + 1.96 * s2, lo90=mu2 - 1.645 * s2,
                        hi90=mu2 + 1.645 * s2, tau2=tau22, tost_p_3mo=tost(mu2, s2, 3), tost_p_2mo=tost(mu2, s2, 2))])
D_.to_csv(W + "control_equivalence.csv", index=False); print(G); print(D_.round(4).T)

# ---------------- (E) within-age coupling ----------------
rng = np.random.default_rng(7)
def wc(y, age):  # centre within age groups
    y = pd.Series(np.asarray(y, float)); a = pd.Series(age); return (y - y.groupby(a).transform("mean")).values
rows=[]
for name, X, M in datasets():
    M = M[~M.index.astype(str).isin(bad)]
    for (tis, sex), mm in M.groupby(["tissue", "Sex"]):
        cols=[c for c in mm.index if c in X.columns]; z=zscore_tissue(X[cols]); z=z[z.notna().all(axis=1)]; age=mm.loc[cols,"age_m"].values.astype(float)
        C,_=program_scores(z,CELL); mac=wc(C["Macrophage"].values,age)
        rg=rank_rho(z,age); sen=list(z.index.intersection(S["Senescence"])); pool=rg[[g for g in rg.index if g not in EXCL]]
        edges=np.quantile(rg.values,np.linspace(0,1,21)); edges[0]-=1; edges[-1]+=1
        pbin=pd.cut(pool,edges,labels=False); sbin=pd.cut(rg[sen],edges,labels=False); bypool={b:pool.index[pbin==b].values for b in np.unique(pbin)}
        r_sen=stats.pearsonr(wc(z.loc[sen].mean(0).values,age),mac)[0]
        # composition R2 within age (all 7 scores)
        Cw=np.column_stack([wc(C[k].values,age) for k in C.columns if C[k].notna().all()])
        def r2(y):
            yw=wc(y,age); b=np.linalg.lstsq(np.c_[np.ones(len(yw)),Cw],yw,rcond=None)[0]; e=yw-np.c_[np.ones(len(yw)),Cw]@b; return 1-e.var()/yw.var()
        R2s=r2(z.loc[sen].mean(0).values)
        nr,nR=[],[]
        for _ in range(100):
            g=list(dict.fromkeys([rng.choice(bypool[b]) for b in sbin.values if b in bypool]))
            s=z.loc[g].mean(0).values; nr.append(stats.pearsonr(wc(s,age),mac)[0]); nR.append(r2(s))
        met=z.loc[z.index.intersection(S["Metabolic"])].mean(0).values
        rows.append(dict(dataset=name,tissue=tis,sex=sex,within_r_sen_mac=r_sen,null_within_r_median=np.median(nr),null_within_r_q95=np.quantile(nr,.95),
                         p_within_r=(np.array(nr)>=r_sen).mean(),R2_sen=R2s,R2_null_median=np.median(nR),R2_null_q95=np.quantile(nR,.95),p_R2=(np.array(nR)>=R2s).mean(),
                         within_r_met_mac=stats.pearsonr(wc(met,age),mac)[0], R2_met=r2(met)))
        print(name,tis,sex,round(r_sen,2),round(np.median(nr),2),round(R2s,2),round(np.median(nR),2),flush=True)
pd.DataFrame(rows).to_csv(W+"control_within_age.csv",index=False)
