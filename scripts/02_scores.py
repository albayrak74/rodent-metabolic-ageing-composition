"""G12 step 2 (go/no-go): program scores per sample, age trends, immune adjustment, hinge (break-age) fits."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
W = "/home/claude/g12/work/"; D = "/home/claude/g12/data/"
rng = np.random.default_rng(1)

# gene sets: Hallmark (human symbols) -> mouse symbols via 1:1 orthologs
hm = pd.read_pickle(W + "human2mouse.pkl")
hi = pd.read_csv(D + "../data/Mus_musculus.gene_info.gz", sep="\t", usecols=["GeneID", "Symbol"])
import gzip
hsym = {}
for l in gzip.open(D + "Homo_sapiens.gene_info.gz", "rt") if False else []:
    pass
hall = {}
for l in open(D + "genesets/MSigDB_Hallmark_2020.txt"):
    p = [x for x in l.rstrip("\n").split("\t") if x]
    hall[p[0]] = p[1:]

def human_to_mouse(genes):
    # simple case conversion is unreliable; use NCBI human symbols via orthologs file built below
    return [H2M[g] for g in genes if g in H2M]

hg = pd.read_csv(D + "Homo_sapiens.gene_info.gz", sep="\t", usecols=["GeneID", "Symbol"]).rename(columns={"GeneID": "h", "Symbol": "hs"})
mg = hi.rename(columns={"GeneID": "mouse_id", "Symbol": "ms"})
H2M = dict(hm.merge(hg, on="h").merge(mg, on="mouse_id")[["hs", "ms"]].values)

senmayo = open("/home/claude/g12/genesets_extra.txt").read().split("\t")[1].strip().split(",")
SETS = {
    "SenMayo": senmayo,
    "Arrest": ["Cdkn1a", "Cdkn2a", "Cdkn2b"],
    "Immune": ["Ptprc", "Cd68", "Adgre1", "Lyz2", "C1qa", "C1qb", "C1qc", "Cd74", "Cd3e", "Cd79a", "Ms4a1", "Cd19", "Cd52", "Coro1a", "Laptm5"],
    "OXPHOS": human_to_mouse(hall["Oxidative Phosphorylation"]),
    "FAmetab": human_to_mouse(hall["Fatty Acid Metabolism"]),
    "Adipogenesis": human_to_mouse(hall["Adipogenesis"]),
    "Inflammatory": human_to_mouse(hall["Inflammatory Response"]),
}
# SenMayo without leukocyte-restricted genes (to reduce pure infiltration signal)
leuk = {"Ccl3", "Ccl4", "Ccl5", "Ccl1", "Il2", "Il10", "Il13", "Csf2rb", "Lcp1", "Selplg", "Gmfg", "Cxcr2", "Il1b", "Tnf", "Mmp12", "Ccl8", "Ccl7", "Ccl24", "Cxcl10"}
SETS["SenMayo_noLeuk"] = [g for g in senmayo if g not in leuk]

def hinge_fit(age, y, grid):
    """piecewise-linear with one break; returns best break, SSE, and SSE of linear"""
    X0 = np.c_[np.ones_like(age), age]
    sse_lin = np.sum((y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]) ** 2)
    best = (None, np.inf)
    for b in grid:
        X = np.c_[np.ones_like(age), age, np.maximum(age - b, 0)]
        r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        s = np.sum(r ** 2)
        if s < best[1]: best = (b, s)
    return best[0], best[1], sse_lin

out = []
for sp in ["rat", "mouse"]:
    X = pd.read_parquet(W + f"{sp}_logcpm.parquet")
    X = X.groupby(level=0).mean()
    M = pd.read_csv(W + f"{sp}_meta.csv", index_col=0, dtype={"sample": str})
    M.index = M.index.astype(str)
    for tis in sorted(M.tissue.unique()):
        for sex in sorted(M.Sex.unique()):
            mm = M[(M.tissue == tis) & (M.Sex == sex)]
            x = X[mm.index]
            x = x[x.mean(1) > 1]  # expressed in this tissue
            z = x.sub(x.mean(1), axis=0).div(x.std(1) + 1e-9, axis=0)
            sc = pd.DataFrame({k: z.loc[z.index.intersection(v)].mean(0) for k, v in SETS.items()})
            ngenes = {k: len(z.index.intersection(v)) for k, v in SETS.items()}
            sc["age"] = mm.age_m.values
            ages = np.sort(sc.age.unique()); grid = np.arange(ages[1], ages[-2] + 0.01, 0.5)
            for k in SETS:
                if ngenes[k] < 2: continue
                rho, p = stats.spearmanr(sc.age, sc[k])
                # immune-adjusted age effect
                d = sc.rename(columns={k: "y"})
                adj = smf.ols("y ~ age + Immune", d).fit() if k != "Immune" else None
                b, sse_h, sse_l = hinge_fit(sc.age.values, sc[k].values, grid)
                n = len(sc); F = ((sse_l - sse_h) / 2) / (sse_h / (n - 4))  # 2 extra params (break + slope change)
                # bootstrap break age
                bs = []
                for _ in range(200):
                    i = rng.integers(0, n, n)
                    bs.append(hinge_fit(sc.age.values[i], sc[k].values[i], grid)[0])
                out.append(dict(species=sp, tissue=tis, sex=sex, score=k, n=n, genes=ngenes[k], rho=rho, p=p,
                                age_beta_adjImm=None if adj is None else adj.params["age"],
                                p_adjImm=None if adj is None else adj.pvalues["age"],
                                break_age=b, break_lo=np.percentile(bs, 2.5), break_hi=np.percentile(bs, 97.5),
                                F_hinge=F, p_hinge=stats.f.sf(F, 2, n - 4)))
            sc.to_csv(W + f"scores_{sp}_{tis.replace(' ', '_')}_{sex}.csv")
R = pd.DataFrame(out); R.to_csv(W + "gonogo_scores.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(R.round(3).to_string(index=False))
