"""Shared helpers for G12: gene sets, program scores, cell-composition scores, T50 lead-lag."""
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from scipy import stats
D = "/home/claude/g12/data/"; W = "/home/claude/g12/work/"

def _h2m():
    hm = pd.read_pickle(W + "human2mouse.pkl")
    hg = pd.read_csv(D + "Homo_sapiens.gene_info.gz", sep="\t", usecols=["GeneID", "Symbol"]).rename(columns={"GeneID": "h", "Symbol": "hs"})
    mg = pd.read_csv(D + "Mus_musculus.gene_info.gz", sep="\t", usecols=["GeneID", "Symbol"]).rename(columns={"GeneID": "mouse_id", "Symbol": "ms"})
    return dict(hm.merge(hg, on="h").merge(mg, on="mouse_id")[["hs", "ms"]].values)

def gene_sets():
    H2M = _h2m(); hall = {}
    for l in open(D + "genesets/MSigDB_Hallmark_2020.txt"):
        p = [x for x in l.rstrip("\n").split("\t") if x]; hall[p[0]] = [H2M[g] for g in p[1:] if g in H2M]
    sen = open("/home/claude/g12/genesets_extra.txt").read().split("\t")[1].strip().split(",")
    leuk = {"Ccl3", "Ccl4", "Ccl5", "Ccl1", "Il2", "Il10", "Il13", "Csf2rb", "Lcp1", "Selplg", "Gmfg", "Cxcr2", "Il1b", "Tnf",
            "Mmp12", "Ccl8", "Ccl7", "Ccl24", "Cxcl10", "Pecam1"}
    S = {"SenMayo": sen, "Senescence": [g for g in sen if g not in leuk],
         "OXPHOS": hall["Oxidative Phosphorylation"], "FAO": hall["Fatty Acid Metabolism"],
         "Adipogenesis": hall["Adipogenesis"]}
    # "Metabolic" = union of OXPHOS and fatty-acid metabolism (core mitochondrial/lipid programme)
    S["Metabolic"] = sorted(set(S["OXPHOS"]) | set(S["FAO"]))
    return S

# marker panels (mouse symbols) for composition proxies; markers deliberately excluded from SenMayo overlap
CELL = {
    "Macrophage": ["Adgre1", "Cd68", "Lyz2", "C1qa", "C1qb", "C1qc", "Mrc1", "Cd163", "Csf1r", "Ms4a7"],
    "T_NK": ["Cd3e", "Cd3g", "Cd3d", "Cd2", "Cd8a", "Cd4", "Nkg7", "Gzma", "Klrd1"],
    "B": ["Cd79a", "Cd79b", "Ms4a1", "Cd19", "Pax5", "Cd22"],
    "Plasma": ["Jchain", "Mzb1", "Igkc", "Ighm", "Iglc2", "Ighg2c", "Igha"],
    "Neutrophil": ["S100a8", "S100a9", "Retnlg", "Ly6g", "Mmp8", "Ngp"],
    "Endothelial": ["Cdh5", "Kdr", "Emcn", "Tie1", "Flt1", "Cldn5"],
    "Fibroblast": ["Pdgfra", "Dcn", "Lum", "Col1a1", "Col1a2", "Col3a1", "Pdgfrb"],
}

def zscore_tissue(x, min_mean=1.0):
    x = x[x.mean(axis=1) > min_mean]
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1) + 1e-9, axis=0)

def program_scores(z, sets):
    out = {}; n = {}
    for k, v in sets.items():
        g = z.index.intersection(v); n[k] = len(g)
        out[k] = z.loc[g].mean(axis=0) if len(g) >= 3 else pd.Series(np.nan, index=z.columns)
    return pd.DataFrame(out), n

def residualise(y, C):
    """remove variance explained by composition covariates C (no age term); returns residual + mean"""
    C = C.loc[:, C.notna().all() & (C.std() > 0)]
    X = np.c_[np.ones(len(y)), C.values]
    b = np.linalg.lstsq(X, y.values, rcond=None)[0]
    return pd.Series(y.values - X[:, 1:] @ b[1:], index=y.index)

def t50(age, y):
    ag = np.sort(np.unique(age)); mu = np.array([y[age == a].mean() for a in ag])
    f = IsotonicRegression(increasing=bool(mu[-1] >= mu[0])).fit(ag, mu)
    fine = np.linspace(ag[0], ag[-1], 841); v = f.predict(fine); tot = v[-1] - v[0]
    if abs(tot) < 1e-9: return np.nan
    return fine[np.argmax((v - v[0]) / tot >= 0.5)]

def lead_lag(age, a, b, nboot=1000, seed=0):
    """T50(a), T50(b), their difference (a-b) with stratified-bootstrap 95% CI and two-sided bootstrap p"""
    rng = np.random.default_rng(seed); ages = np.unique(age)
    ta, tb = t50(age, a), t50(age, b); d = []
    for _ in range(nboot):
        idx = np.concatenate([rng.choice(np.where(age == x)[0], (age == x).sum()) for x in ages])
        d.append(t50(age[idx], a[idx]) - t50(age[idx], b[idx]))
    d = np.array(d); d = d[~np.isnan(d)]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean()) if len(d) else np.nan
    return ta, tb, ta - tb, np.percentile(d, 2.5), np.percentile(d, 97.5), min(p, 1.0)

def trend(age, y):
    r, p = stats.spearmanr(age, y); return r, p
