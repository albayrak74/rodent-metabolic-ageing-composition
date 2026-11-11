"""G12 step 1: load raw counts, map to a shared mouse-symbol gene space, keep metabolic tissues.
Outputs per-species parquet of log2-CPM (TMM-free; library-size normalised) plus sample metadata."""
import gzip, sys, numpy as np, pandas as pd
D = "/home/claude/g12/data/"
OUT = "/home/claude/g12/work/"
import os; os.makedirs(OUT, exist_ok=True)

def meta(g):
    rows = {}; titles = None
    for l in gzip.open(D + g + "_series_matrix.txt.gz", "rt"):
        if l.startswith("!Sample_title"):
            titles = [x.strip('"\n') for x in l.split("\t")[1:]]
        if l.startswith("!Sample_characteristics_ch1"):
            v = [x.strip('"\n') for x in l.split("\t")[1:]]
            rows.setdefault(v[0].split(":")[0], [x.split(": ", 1)[-1] for x in v])
    m = pd.DataFrame(rows)
    m["sample"] = [t.rsplit("_", 1)[1] for t in titles]
    m["age_m"] = m["age"].str.split().str[0].astype(float)
    return m.set_index("sample")

def gene_info(f):
    gi = pd.read_csv(D + f, sep="\t", usecols=["GeneID", "Symbol", "dbXrefs", "type_of_gene"])
    return gi

ri = gene_info("Rattus_norvegicus.gene_info.gz"); mi = gene_info("Mus_musculus.gene_info.gz")
orth = pd.read_csv(D + "orth_hmr.tsv", sep="\t")
# NCBI gene_orthologs is human-anchored: join human->mouse and human->rat, plus direct mouse<->rat rows
hm = orth[(orth["#tax_id"] == 9606) & (orth.Other_tax_id == 10090)][["GeneID", "Other_GeneID"]].rename(columns={"GeneID": "h", "Other_GeneID": "mouse_id"})
hr = orth[(orth["#tax_id"] == 9606) & (orth.Other_tax_id == 10116)][["GeneID", "Other_GeneID"]].rename(columns={"GeneID": "h", "Other_GeneID": "rat_id"})
hm = hm[~hm.h.duplicated(keep=False) & ~hm.mouse_id.duplicated(keep=False)]
hr = hr[~hr.h.duplicated(keep=False) & ~hr.rat_id.duplicated(keep=False)]
d1 = orth[(orth["#tax_id"] == 10090) & (orth.Other_tax_id == 10116)][["GeneID", "Other_GeneID"]]
d1.columns = ["mouse_id", "rat_id"]
d2 = orth[(orth["#tax_id"] == 10116) & (orth.Other_tax_id == 10090)][["Other_GeneID", "GeneID"]]
d2.columns = ["mouse_id", "rat_id"]
rm = pd.concat([hm.merge(hr, on="h")[["mouse_id", "rat_id"]], d1, d2]).drop_duplicates()
hm.to_pickle(OUT + "human2mouse.pkl")
# keep strict one-to-one
rm = rm[~rm.mouse_id.duplicated(keep=False) & ~rm.rat_id.duplicated(keep=False)]
# rat ensembl -> rat GeneID
# rat count matrix rows are mostly rat gene symbols (plus some Ensembl IDs for unnamed genes)
e1 = ri[["Symbol", "GeneID"]].rename(columns={"Symbol": "ens", "GeneID": "rat_id"})
e2 = ri.assign(ens=ri.dbXrefs.str.extract(r"Ensembl:(ENSRNOG\d+)")[0]).dropna(subset=["ens"])[["ens", "GeneID"]].rename(columns={"GeneID": "rat_id"})
ens = pd.concat([e1, e2]); ens = ens[~ens.ens.duplicated(keep=False)]
ratrows = set(l.strip() for l in open(D + "rat_ids.txt"))
ens = ens[ens.ens.isin(ratrows)]
msym = mi[["GeneID", "Symbol"]].rename(columns={"GeneID": "mouse_id", "Symbol": "msym"})
rat2m = ens.merge(rm, on="rat_id").merge(msym, on="mouse_id")
rat2m = rat2m[~rat2m.msym.duplicated(keep=False) & ~rat2m.ens.duplicated(keep=False)]
print("rat Ensembl -> mouse symbol 1:1 pairs:", len(rat2m))
pd.to_pickle(rat2m, OUT + "rat2mouse.pkl")

TIS = ["gWAT", "liver", "left ventricle", "kidney", "right ventricle", "spleen"]
for sp, g, f in [("rat", "GSE314970", "GSE314970_rat_raw_counts_wo_muscle.txt.gz"),
                 ("mouse", "GSE314817", "GSE314817_mouse_raw_counts_wo_muscle.txt.gz")]:
    m = meta(g)
    keep = m.index[m.tissue.isin(TIS)]
    hdr = pd.read_csv(D + f, sep="\t", nrows=0).columns
    cols = ["GeneName"] + [c for c in hdr[1:] if c in set(keep)]
    X = pd.read_csv(D + f, sep="\t", usecols=cols, index_col=0)
    print(sp, "raw", X.shape, "samples in meta not in counts:", len(set(keep) - set(X.columns)))
    lib = X.sum(0)
    if sp == "rat":
        X = X.loc[X.index.intersection(rat2m.ens)]
        X.index = rat2m.set_index("ens").loc[X.index, "msym"].values
    else:
        X = X.groupby(level=0).sum()
    cpm = np.log2(X.div(lib, axis=1) * 1e6 + 1)
    cpm.astype("float32").to_parquet(OUT + f"{sp}_logcpm.parquet")
    m.loc[X.columns].to_csv(OUT + f"{sp}_meta.csv")
    print(sp, "logCPM", cpm.shape, m.loc[X.columns].groupby(["tissue"]).size().to_dict())
