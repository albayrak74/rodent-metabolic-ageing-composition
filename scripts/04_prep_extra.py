"""G12 step 4: add gastrocnemius (atlas muscle series) and build external validation sets
(Tabula Muris Senis bulk GSE132040; rat BodyMap GSE53960) on mouse-symbol gene space, log2-CPM."""
import gzip, glob, re, numpy as np, pandas as pd
D = "/home/claude/g12/data/"; W = "/home/claude/g12/work/"
rat2m = pd.read_pickle(W + "rat2mouse.pkl")  # columns ens (rat row id/symbol), msym
r2m = dict(zip(rat2m.ens, rat2m.msym))

def logcpm(X):
    return np.log2(X.div(X.sum(axis=0), axis=1) * 1e6 + 1)

def series_meta(g):
    rows = {}; acc = None; titles = None
    for l in gzip.open(D + g + "_series_matrix.txt.gz", "rt"):
        if l.startswith("!Sample_geo_accession"): acc = [x.strip('"\n') for x in l.split("\t")[1:]]
        if l.startswith("!Sample_title"): titles = [x.strip('"\n') for x in l.split("\t")[1:]]
        if l.startswith("!Sample_characteristics_ch1"):
            v = [x.strip('"\n') for x in l.split("\t")[1:]]
            rows.setdefault(v[0].split(":")[0], [x.split(": ", 1)[-1] for x in v])
    m = pd.DataFrame(rows, index=acc); m["title"] = titles
    return m

# ---- atlas gastrocnemius (rat GSE226118, mouse GSE226117) ----
for sp, g in [("rat", "GSE226118"), ("mouse", "GSE226117")]:
    m = series_meta(g)
    m = m[m.tissue == "gastrocnemius"]
    cols = {}
    for acc in m.index:
        f = glob.glob(D + f"muscle/{acc}_*_raw_counts.txt.gz")[0]
        s = pd.read_csv(f, sep="\t", index_col=0).iloc[:, 0]
        cols[acc] = s
    X = pd.DataFrame(cols).fillna(0)
    lib = X.sum(axis=0)
    if sp == "rat":
        X = X.loc[X.index.intersection(list(r2m))]; X.index = [r2m[i] for i in X.index]
    X = X.groupby(level=0).sum()
    L = np.log2(X.div(lib, axis=1) * 1e6 + 1)
    age = m.time.str.extract(r"(\d+)")[0].astype(float)
    M = pd.DataFrame({"tissue": "gastrocnemius", "Sex": m.Sex, "age_m": age}, index=m.index)
    L.astype("float32").to_parquet(W + f"{sp}_gastroc_logcpm.parquet"); M.to_csv(W + f"{sp}_gastroc_meta.csv")
    print(sp, "gastrocnemius", L.shape, M.groupby(["Sex", "age_m"]).size().to_dict())

# ---- Tabula Muris Senis bulk ----
tm = pd.read_csv(D + "GSE132040_MACA_Bulk_metadata.csv.gz")
tm = tm[tm["source name"].str.match(r"^(GAT|SCAT|BAT|Liver|Heart|Kidney|Limb_Muscle|Limb|Pancreas)_")]
tm["tissue"] = tm["source name"].str.rsplit("_", n=1).str[0]
tm = tm.set_index("Sample name")
X = pd.read_csv(D + "GSE132040_190214_A00111_0269_AHH3J3DSXX_190214_A00111_0270_BHHMFWDSXX.csv.gz", index_col=0)
X.columns = [c.replace(".gencode.vM19", "") for c in X.columns]
X = X[[c for c in X.columns if c in tm.index]]
X = X[~X.index.str.startswith("__")]
L = logcpm(X)
M = pd.DataFrame({"tissue": tm.loc[X.columns, "tissue"], "Sex": tm.loc[X.columns, "characteristics: sex"].map({"m": "male", "f": "female"}),
                  "age_m": tm.loc[X.columns, "characteristics: age"].astype(float)})
L.astype("float32").to_parquet(W + "tms_logcpm.parquet"); M.to_csv(W + "tms_meta.csv")
print("TMS", L.shape, M.groupby("tissue").size().to_dict())

# ---- rat BodyMap (F344; AceView FPKM-like values) ----
rows = {}
for f in sorted(glob.glob(D + "bodymap/*.txt.gz")):
    nm = re.search(r"SEQC_(\w+?)_([FM])_(\d+)_(\d)", f)
    if nm.group(1) not in ("Lvr", "Hrt", "Kdn", "Msc"): continue
    s = pd.read_csv(f, sep="\t", index_col=0).iloc[:, 0]
    rows[f.split("/")[-1].split("_")[0]] = (s, nm.group(1), nm.group(2), int(nm.group(3)))
X = pd.DataFrame({k: v[0] for k, v in rows.items()}).fillna(0)
X = X.loc[X.index.intersection(list(r2m))]; X.index = [r2m[i] for i in X.index]; X = X.groupby(level=0).sum()
L = np.log2(X + 1)
M = pd.DataFrame({k: dict(tissue={"Lvr": "liver", "Hrt": "heart", "Kdn": "kidney", "Msc": "muscle"}[v[1]],
                          Sex={"F": "female", "M": "male"}[v[2]], age_m=v[3] / 4.345) for k, v in rows.items()}).T
L.astype("float32").to_parquet(W + "bodymap_log.parquet"); M.to_csv(W + "bodymap_meta.csv")
print("BodyMap", L.shape, M.groupby(["tissue"]).size().to_dict())
