"""G12 step 16: single-cell checks in Tabula Muris Senis FACS (Smart-seq2; 3, 18 and 24 months; figshare file 23939711).
(A) Cell source of the senescence signature at 3 months: per gene, mean expression in immune vs non-immune cell types
    (cell types weighted equally, so FACS sorting proportions do not matter). Metabolic genes serve as contrast.
(B) Age effect within cell types: mouse x cell-type pseudobulks of the senescence core score; age trend within immune and
    within non-immune cell types, compared with the immune vs non-immune difference."""
import sys; sys.path.insert(0, "/home/claude/g12/scripts")
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, statsmodels.formula.api as smf
from scipy import stats
from lib import *
S = gene_sets()
IMM = {"myeloid cell", "B cell", "T cell", "NK cell", "monocyte", "macrophage", "Kupffer cell", "myeloid leukocyte",
       "mature NK T cell", "neutrophil", "CD4-positive, alpha-beta T cell", "CD8-positive, alpha-beta T cell"}
MYE = {"myeloid cell", "monocyte", "macrophage", "Kupffer cell", "myeloid leukocyte", "neutrophil"}
STR = {"mesenchymal stem cell of adipose", "mesenchymal stem cell", "fibroblast of cardiac tissue", "mesangial cell",
       "kidney interstitial fibroblast", "valve cell", "smooth muscle cell"}
ENDO = {"endothelial cell", "endothelial cell of coronary artery", "endothelial cell of hepatic sinusoid", "fenestrated cell", "endocardial cell"}
def broad(c):
    return "myeloid" if c in MYE else "lymphoid" if c in IMM else "stromal" if c in STR else "endothelial" if c in ENDO else "parenchymal/other"
TIS = ["GAT", "SCAT", "BAT", "Heart", "Kidney", "Liver", "Limb_Muscle"]
A = ad.read_h5ad(D + "tms_sc/facs_raw.h5ad", backed="r")
genes = pd.Index(A.var_names)
sets = {k: [g for g in S[k] if g in genes] for k in ["SenMayo", "Senescence", "Metabolic"]}
srcrows = []; pb = []
for t in TIS:
    idx = np.where(A.obs.tissue.values == t)[0]
    X = sp.csr_matrix(A.X[idx]); o = A.obs.iloc[idx].copy()
    X = sp.diags(1e4 / np.asarray(X.sum(axis=1)).ravel()) @ X; L = X.copy(); L.data = np.log1p(L.data)
    o["immune"] = o.cell_ontology_class.astype(str).isin(IMM); o["ct"] = o.cell_ontology_class.astype(str)
    o["age_m"] = o.age.astype(str).str.replace("m", "").astype(float)
    # (A) cell source at 3 months, cell types weighted equally
    y = (o.age_m == 3).values
    cts = [c for c in o.ct[y].unique() if (o.ct[y] == c).sum() >= 20]
    M = pd.DataFrame({c: np.asarray(X[np.where(y & (o.ct == c).values)[0]].mean(axis=0)).ravel() for c in cts}, index=genes)
    imm_c = [c for c in cts if c in IMM]; non_c = [c for c in cts if c not in IMM]
    for k, g in sets.items():
        m = M.loc[g]; ri = m[imm_c].mean(axis=1); rn = m[non_c].mean(axis=1); ok = (ri + rn) > 0.5
        lr = np.log2((ri[ok] + 0.1) / (rn[ok] + 0.1))
        B = pd.DataFrame({b: m[[c for c in cts if broad(c) == b]].mean(axis=1) for b in ["myeloid", "lymphoid", "stromal", "endothelial", "parenchymal/other"]
                          if any(broad(c) == b for c in cts)})[ok]
        top = B.idxmax(axis=1).value_counts(normalize=True)
        r = dict(tissue=t, set=k, n_genes=int(ok.sum()), frac_higher_in_immune=(lr > 0).mean(), median_log2_imm_vs_non=lr.median(),
                 share_from_immune=(ri[ok].sum() / (ri[ok].sum() + rn[ok].sum())), n_immune_types=len(imm_c), n_nonimmune_types=len(non_c))
        for b in ["myeloid", "lymphoid", "stromal", "endothelial", "parenchymal/other"]:
            r[f"top_{b}"] = top.get(b, 0.0) if b in B.columns else np.nan
            r[f"share_{b}"] = (B[b].sum() / B.sum().sum()) if b in B.columns else np.nan
        srcrows.append(r)
    # (B) per-cell scores -> mouse x cell type pseudobulks
    for k in ["Senescence", "Metabolic"]:
        gi = genes.get_indexer(sets[k]); o[k] = np.asarray(L[:, gi].mean(axis=1)).ravel()
    for (mid, c), g in o.groupby(["mouse.id", "ct"], observed=True):
        if len(g) < 10: continue
        pb.append(dict(tissue=t, mouse=mid, ct=c, broad=broad(c), immune=bool(g.immune.iloc[0]), age=g.age_m.iloc[0], sex=g.sex.iloc[0], n=len(g),
                       Senescence=g.Senescence.mean(), Metabolic=g.Metabolic.mean()))
    print(t, "done", flush=True)
SRC = pd.DataFrame(srcrows); SRC.to_csv(W + "sc_signature_source.csv", index=False)
PB = pd.DataFrame(pb); PB.to_csv(W + "sc_pseudobulk.csv", index=False)
rows = []
for t, g in PB.groupby("tissue"):
    g = g.copy(); g["z"] = (g.Senescence - g.Senescence.mean()) / g.Senescence.std()
    f = smf.ols("z ~ C(immune) + age + C(ct)", data=g).fit() if g.ct.nunique() > 1 else None
    d = smf.ols("z ~ C(immune) + age", data=g).fit()
    r = dict(tissue=t, n_pseudobulks=len(g), immune_minus_nonimmune_sd=d.params["C(immune)[T.True]"],
             p_immune=d.pvalues["C(immune)[T.True]"], age_slope_within_celltype_sd_per_21mo=f.params["age"] * 21, p_age_within=f.pvalues["age"])
    for lab, sub in [("immune", g[g.immune]), ("nonimmune", g[~g.immune])]:
        if sub.age.nunique() > 1 and len(sub) > 5:
            ff = smf.ols("z ~ age + C(ct)", data=sub).fit() if sub.ct.nunique() > 1 else smf.ols("z ~ age", data=sub).fit()
            r[f"age_slope_{lab}_sd_per_21mo"] = ff.params["age"] * 21; r[f"p_age_{lab}"] = ff.pvalues["age"]
    for b, sub in g.groupby("broad"):
        if sub.age.nunique() > 1 and len(sub) > 5:
            ff = smf.ols("z ~ age + C(ct)", data=sub).fit() if sub.ct.nunique() > 1 else smf.ols("z ~ age", data=sub).fit()
            r[f"mean_z_{b}"] = sub.z.mean(); r[f"slope_{b}_per_21mo"] = ff.params["age"] * 21; r[f"p_{b}"] = ff.pvalues["age"]
    rows.append(r)
AG = pd.DataFrame(rows); AG.to_csv(W + "sc_age_within_celltype.csv", index=False)
pd.set_option("display.width", 250); print(SRC.round(2).to_string()); print(AG.round(3).to_string())
