"""G12 step 17: robustness to the choice of senescence signature. KEGG 'Cellular senescence' (mouse), Reactome 'Cellular
Senescence' and Reactome 'Senescence-Associated Secretory Phenotype (SASP)' (human, mapped to mouse 1:1 orthologues) are scored
like the SenMayo core score; age trends are reported raw, after marker-score adjustment and after deconvolution adjustment."""
import sys; sys.path.insert(0, "/home/claude/g12/scripts")
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
from lib import *
from lib import _h2m
H2M = _h2m()
def read(fn, name, human):
    for l in open(D + "genesets/" + fn):
        p = [x for x in l.rstrip("\n").split("\t") if x]
        if p[0].startswith(name):
            g = [x.split(",")[0] for x in p[1:]]
            if human: return sorted({H2M[x] for x in g if x in H2M})
            ms = pd.read_csv(D + "Mus_musculus.gene_info.gz", sep="\t", usecols=["Symbol"]).Symbol
            up = {x.upper(): x for x in ms}
            return sorted({up[x.upper()] for x in g if x.upper() in up})
ALT = {"KEGG_senescence": read("KEGG_2019_Mouse.txt", "Cellular senescence", False),
       "Reactome_senescence": read("Reactome_2022.txt", "Cellular Senescence R-HSA-2559583", True),
       "Reactome_SASP": read("Reactome_2022.txt", "Senescence-Associated Secretory Phenotype", True)}
print({k: len(v) for k, v in ALT.items()})
sen = gene_sets()["SenMayo"]; print("overlap with SenMayo", {k: len(set(v) & set(sen)) for k, v in ALT.items()})
sc = pd.read_csv(W + "all_scores.csv", index_col=0); sc.index = sc.index.astype(str)
fr = pd.read_csv(W + "deconvolution_fractions.csv", index_col=0); fr.index = fr.index.astype(str)
IMMSTROMA = ["Macrophage", "Monocyte", "Neutrophil", "T", "B", "NK", "Endothelial", "Fibroblast"]
data = {}
for s_ in ["rat", "mouse"]:
    data[f"atlas_{s_}"] = pd.concat([pd.read_parquet(W + f"{s_}_logcpm.parquet").groupby(level=0).mean(),
                                    pd.read_parquet(W + f"{s_}_gastroc_logcpm.parquet").groupby(level=0).mean()], axis=1)
data["TMS_mouse"] = pd.read_parquet(W + "tms_logcpm.parquet")
rows = []
for (ds, tis, sex), s in sc.groupby(["dataset", "tissue", "Sex"]):
    X = data[ds]; X = X.loc[:, ~X.columns.duplicated()][s.index]; X = X[X.notna().all(axis=1)]
    z = zscore_tissue(X); P, n = program_scores(z, ALT); C = s[list(CELL)]
    F = fr[(fr.dataset == ds) & (fr.tissue == tis) & (fr.Sex == sex)].loc[s.index, IMMSTROMA].astype(float)
    age = s.age.values.astype(float)
    for k in ALT:
        if P[k].isna().all(): continue
        r0, p0 = trend(age, P[k]); r1, p1 = trend(age, residualise(P[k], C)); r2, p2 = trend(age, residualise(P[k], F))
        rows.append(dict(dataset=ds, tissue=tis, sex=sex, signature=k, n_genes=n[k], rho_raw=r0, p_raw=p0,
                         rho_marker_adj=r1, p_marker_adj=p1, rho_deconv_adj=r2, p_deconv_adj=p2, r_with_senmayo_core=np.corrcoef(P[k], s.Senescence)[0, 1]))
O = pd.DataFrame(rows); O["block"] = np.where(O.dataset.str.startswith("atlas"), "atlas", "TMS")
for k in ["raw", "marker_adj", "deconv_adj"]:
    O[f"q_{k}"] = O.groupby(["block", "signature"])[f"p_{k}"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
O.to_csv(W + "alt_signatures.csv", index=False)
print(O.groupby(["block", "signature"]).agg(raw=("rho_raw", "median"), marker=("rho_marker_adj", "median"), deconv=("rho_deconv_adj", "median"),
      sig_raw=("q_raw", lambda q: (q < .05).sum()), sig_marker=("q_marker_adj", lambda q: (q < .05).sum()), sig_deconv=("q_deconv_adj", lambda q: (q < .05).sum()),
      r_core=("r_with_senmayo_core", "median"), n=("tissue", "size")).round(2).to_string())
