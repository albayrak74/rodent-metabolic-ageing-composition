"""G12 step 12: reference-based deconvolution with a single-cell reference (Tabula Muris droplet, GSE132042 3-month channels)
and re-test of the composition-adjusted senescence trend. Signature genes overlapping SenMayo are excluded to avoid circularity."""
import sys, glob, tarfile, io; sys.path.insert(0, "/home/claude/g12/scripts")
import numpy as np, pandas as pd, scipy.io, scipy.sparse as sp
from scipy.optimize import nnls
from statsmodels.stats.multitest import multipletests
from lib import *
T10 = D + "tm10x/"
CLASS = {"macrophage": "Macrophage", "monocyte": "Monocyte", "classical monocyte": "Monocyte", "non-classical monocyte": "Monocyte",
         "granulocyte": "Neutrophil", "T cell": "T", "B cell": "B", "natural killer cell": "NK",
         "endothelial cell": "Endothelial", "kidney capillary endothelial cell": "Endothelial", "lung endothelial cell": "Endothelial",
         "endothelial cell of hepatic sinusoid": "Endothelial", "endocardial cell": "Endothelial",
         "fibroblast": "Fibroblast", "mesenchymal stem cell": "Fibroblast", "stromal cell": "Fibroblast", "mesangial cell": "Fibroblast",
         "hepatocyte": "Hepatocyte", "kidney proximal straight tubule epithelial cell": "Kidney_epithelium",
         "kidney loop of Henle ascending limb epithelial cell": "Kidney_epithelium", "kidney collecting duct epithelial cell": "Kidney_epithelium",
         "cardiac muscle cell": "Cardiomyocyte", "skeletal muscle satellite cell": "Muscle_satellite"}
IMMSTROMA = ["Macrophage", "Monocyte", "Neutrophil", "T", "B", "NK", "Endothelial", "Fibroblast"]

def build_reference():
    ann = pd.read_csv(T10 + "annotations_droplet.csv", low_memory=False)
    ann["cls"] = ann.cell_ontology_class.map(CLASS); ann = ann.dropna(subset=["cls"]).set_index("cell")
    sums = {}; ncell = {}
    for f in sorted(glob.glob(T10 + "GSM*.tar.gz")):
        print("ref", f.split("/")[-1], flush=True)
        tf = tarfile.open(f); mem = {m.name.split("/")[-1]: m for m in tf.getmembers() if m.isfile()}
        ch = [m.name.split("/")[0] for m in tf.getmembers()][0].split("-", 1)[1]
        genes = pd.read_csv(tf.extractfile(mem["genes.tsv"]), sep="\t", header=None)[1].values
        bc = pd.read_csv(tf.extractfile(mem["barcodes.tsv"]), header=None)[0].str.replace("-1", "", regex=False).values
        X = sp.csc_matrix(scipy.io.mmread(io.BytesIO(tf.extractfile(mem["matrix.mtx"]).read())))
        ids = np.array([f"{ch}_{b}" for b in bc]); AI = set(ann.index); keep = np.array([i in AI for i in ids])
        lab = ann.loc[ids[keep], "cls"].values; Xk = X[:, keep]
        Xk = Xk.multiply(1e4 / np.asarray(Xk.sum(axis=0)).ravel()).tocsc()  # per-cell normalisation to 10k
        for c in np.unique(lab):
            s = np.asarray(Xk[:, lab == c].sum(axis=1)).ravel()
            sums[c] = sums.get(c, 0) + pd.Series(s, index=genes).groupby(level=0).sum()
            ncell[c] = ncell.get(c, 0) + (lab == c).sum()
    R = pd.DataFrame({c: sums[c] / ncell[c] for c in sums}) * 100  # mean per-cell profile, scaled to CPM
    return R, pd.Series(ncell)

def signature(R, sen_genes, top=50):
    L = np.log2(R + 1); sig = set()
    for c in R.columns:
        other = L.drop(columns=c).max(axis=1); fc = (L[c] - other)[(R[c] > 10)]
        fc = fc.drop([g for g in fc.index if g in sen_genes], errors="ignore")
        sig |= set(fc.sort_values(ascending=False).index[:top][fc.sort_values(ascending=False)[:top] > 1])
    return sorted(sig)

def deconvolve(logcpm, R, sig):
    ok = logcpm.notna().all(axis=1); g = [x for x in sig if x in logcpm.index and ok[x]]; A = R.loc[g].values; B = (2 ** logcpm.loc[g] - 1).clip(lower=0)
    w = 1 / (A.mean(axis=1) + 1)  # down-weight very highly expressed genes
    F = np.array([nnls(A * w[:, None], B[c].values * w)[0] for c in B.columns])
    F = F / F.sum(axis=1, keepdims=True)
    return pd.DataFrame(F, index=logcpm.columns, columns=R.columns), len(g)

if __name__ == "__main__":
    S = gene_sets(); R, nc = build_reference()
    sig = signature(R, set(S["SenMayo"])); print("reference cells", nc.to_dict(), "signature genes", len(sig))
    R.loc[sig].to_csv(W + "deconv_reference_signature.csv"); nc.to_csv(W + "deconv_reference_ncells.csv")
    sc = pd.read_csv(W + "all_scores.csv", index_col=0); sc.index = sc.index.astype(str)
    data = {}
    for sp_ in ["rat", "mouse"]:
        data[f"atlas_{sp_}"] = pd.concat([pd.read_parquet(W + f"{sp_}_logcpm.parquet").groupby(level=0).mean(),
                                         pd.read_parquet(W + f"{sp_}_gastroc_logcpm.parquet").groupby(level=0).mean()], axis=1)
    data["TMS_mouse"] = pd.read_parquet(W + "tms_logcpm.parquet")
    rows = []; fr_all = []
    for (ds, tis, sex), s in sc.groupby(["dataset", "tissue", "Sex"]):
        X = data[ds]; X = X.loc[:, ~X.columns.duplicated()][s.index]
        Fr, ng = deconvolve(X, R, sig)
        Fr[["dataset", "tissue", "Sex", "age"]] = s[["dataset", "tissue", "Sex", "age"]].values; fr_all.append(Fr)
        C = Fr[IMMSTROMA].astype(float)
        adj = residualise(s["Senescence"], C)
        age = s.age.values.astype(float)
        r0, p0 = trend(age, s["Senescence"]); r1, p1 = trend(age, s["Senescence_adj"]); r2, p2 = trend(age, adj)
        rm, pm = trend(age, Fr["Macrophage"].astype(float)); val = np.corrcoef(Fr["Macrophage"].astype(float), s["Macrophage"])[0, 1]
        ta, tb, d, lo, hi, p = lead_lag(age, adj.values, s["Metabolic"].values, nboot=1000)
        rows.append(dict(dataset=ds, tissue=tis, sex=sex, n=len(s), n_sig_genes=ng,
                         rho_raw=r0, p_raw=p0, rho_marker_adj=r1, p_marker_adj=p1, rho_deconv_adj=r2, p_deconv_adj=p2,
                         rho_macrophage_fraction=rm, p_macrophage_fraction=pm, r_macrophage_fraction_vs_marker=val,
                         T50_Senescence_deconv_adj=ta, T50_Metabolic=tb, lag_deconv_adj=d, lagLo=lo, lagHi=hi, lagP=p))
        print(ds, tis, sex, round(r0, 2), round(r1, 2), round(r2, 2), "val", round(val, 2), flush=True)
    O = pd.DataFrame(rows)
    for k in ["raw", "marker_adj", "deconv_adj"]:
        O[f"q_{k}"] = np.nan
        for ds in O.dataset.str.startswith("atlas").map({True: "atlas", False: "TMS"}).unique():
            m = O.dataset.str.startswith("atlas") == (ds == "atlas")
            O.loc[m, f"q_{k}"] = multipletests(O.loc[m, f"p_{k}"], method="fdr_bh")[1]
    O.to_csv(W + "deconvolution_adjustment.csv", index=False); pd.concat(fr_all).to_csv(W + "deconvolution_fractions.csv")
