"""G12 step 7: final elastic-net clocks per tissue/species; cross-species transfer (1:1 orthologs);
external validation (TMS bulk mouse, BodyMap F344 rat); conserved predictor genes; SHAP on XGBoost."""
import sys; sys.path.insert(0, "/home/claude/g12/scripts")
import numpy as np, pandas as pd, warnings, json, pickle
from lib import W, gene_sets, CELL
from sklearn.linear_model import ElasticNetCV
from xgboost import XGBRegressor
from scipy import stats
import shap
warnings.filterwarnings("ignore"); SEED = 7; NG = 2000

def load_atlas(sp, tis):
    if tis == "gastrocnemius":
        X = pd.read_parquet(W + f"{sp}_gastroc_logcpm.parquet"); M = pd.read_csv(W + f"{sp}_gastroc_meta.csv", index_col=0)
    else:
        X = pd.read_parquet(W + f"{sp}_logcpm.parquet"); M = pd.read_csv(W + f"{sp}_meta.csv", index_col=0)
        M.index = M.index.astype(str); M = M[M.tissue == tis]
    return X.groupby(level=0).mean()[M.index], M

def zgenes(X):  # per-dataset gene standardisation (removes platform/pipeline offsets)
    return X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1) + 1e-9, axis=0)

TMS = pd.read_parquet(W + "tms_logcpm.parquet"); TMSm = pd.read_csv(W + "tms_meta.csv", index_col=0)
BM = pd.read_parquet(W + "bodymap_log.parquet"); BMm = pd.read_csv(W + "bodymap_meta.csv", index_col=0)
EXT = {"gWAT": [("TMS_mouse", "GAT")], "liver": [("TMS_mouse", "Liver"), ("BodyMap_rat", "liver")],
       "left ventricle": [("TMS_mouse", "Heart"), ("BodyMap_rat", "heart")], "kidney": [("TMS_mouse", "Kidney"), ("BodyMap_rat", "kidney")],
       "gastrocnemius": [("TMS_mouse", "Limb_Muscle"), ("BodyMap_rat", "muscle")]}

def ext(name, tis):
    if name == "TMS_mouse":
        m = TMSm[(TMSm.tissue == tis) & (TMSm.age_m >= 3)]; return TMS[m.index].groupby(level=0).mean(), m
    m = BMm[BMm.tissue == tis]; return BM[m.index], m

def fit_en(X, y, genes):
    Z = zgenes(X.loc[genes]).T.values
    return ElasticNetCV(l1_ratio=[.1, .5, .9], alphas=30, cv=5, max_iter=10000, random_state=SEED).fit(Z, y)

def evaluate(model, genes, Xt, yt):
    Xt = Xt.reindex(genes); cover = Xt.notna().all(axis=1).mean()
    Z = zgenes(Xt.fillna(Xt.mean(axis=1).fillna(0))).fillna(0).T.values
    p = model.predict(Z); r, pr = stats.pearsonr(p, yt); rho, prho = stats.spearmanr(p, yt)
    return dict(coverage=cover, r=r, p_r=pr, rho=rho, p_rho=prho, MAE=np.mean(np.abs(p - yt))), p

S = gene_sets(); senset = set(S["SenMayo"]); metset = set(S["Metabolic"]); cellset = set(g for v in CELL.values() for g in v)
rows = []; coefs = []; shaprows = []; extpred = []
for tis in ["gWAT", "liver", "left ventricle", "kidney", "gastrocnemius"]:
    data = {sp: load_atlas(sp, tis) for sp in ["rat", "mouse"]}
    common = data["rat"][0].index.intersection(data["mouse"][0].index)
    fitted = {}
    for sp, (X, M) in data.items():
        y = M.age_m.values.astype(float)
        e = X.loc[X.mean(axis=1) > 1]
        g_own = e.var(axis=1).sort_values(ascending=False).index[:NG]
        g_orth = e.loc[e.index.intersection(common)].var(axis=1).sort_values(ascending=False).index[:NG]
        mod = fit_en(X, y, g_own); mod_o = fit_en(X, y, g_orth)
        fitted[sp] = (mod_o, g_orth, X, y)
        pickle.dump(dict(model=mod, genes=list(g_own), species=sp, tissue=tis), open(W + f"clock_{sp}_{tis.replace(' ', '_')}.pkl", "wb"))
        c = pd.Series(mod.coef_, index=g_own); c = c[c != 0]
        coefs.append(pd.DataFrame(dict(species=sp, tissue=tis, gene=c.index, coef=c.values)))
        # SHAP on XGBoost (own gene space)
        Z = zgenes(X.loc[g_own]).T
        xg = XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.3, random_state=SEED, n_jobs=2).fit(Z.values, y)
        sv = shap.TreeExplainer(xg).shap_values(Z.values)
        imp = pd.Series(np.abs(sv).mean(0), index=g_own).sort_values(ascending=False)
        dirn = pd.Series([stats.spearmanr(Z[g], y)[0] for g in imp.index[:50]], index=imp.index[:50])
        shaprows.append(pd.DataFrame(dict(species=sp, tissue=tis, gene=imp.index[:50], mean_abs_shap=imp.values[:50], rho_age=dirn.values)))
        # external validation
        for name, et in EXT[tis]:
            if (name == "TMS_mouse") != (sp == "mouse"): continue
            Xt, mt = ext(name, et); res, p = evaluate(mod, list(g_own), Xt, mt.age_m.values.astype(float))
            rows.append(dict(kind="external", train=f"{sp}_atlas", test=name, tissue=tis, n=len(mt), **res))
            extpred.append(pd.DataFrame(dict(test=name, tissue=tis, age=mt.age_m.values, sex=mt.Sex.values, pred=p)))
    # cross-species transfer in ortholog space, plus external sets of the other species
    for a, b in [("mouse", "rat"), ("rat", "mouse")]:
        mod_o, g, _, _ = fitted[a]; Xb, yb = fitted[b][2], fitted[b][3]
        res, p = evaluate(mod_o, list(g), Xb, yb)
        rows.append(dict(kind="cross-species", train=f"{a}_atlas", test=f"{b}_atlas", tissue=tis, n=len(yb), **res))
        for name, et in EXT[tis]:
            if (name == "TMS_mouse") == (b == "mouse"):
                Xt, mt = ext(name, et); res, _ = evaluate(mod_o, list(g), Xt, mt.age_m.values.astype(float))
                rows.append(dict(kind="cross-species external", train=f"{a}_atlas", test=name, tissue=tis, n=len(mt), **res))
    print(tis, "done", flush=True)

R = pd.DataFrame(rows); R.to_csv(W + "transfer_validation.csv", index=False)
C = pd.concat(coefs); C.to_csv(W + "clock_coefficients.csv", index=False)
SH = pd.concat(shaprows); SH.to_csv(W + "shap_top50.csv", index=False)
pd.concat(extpred).to_csv(W + "external_predictions.csv", index=False)
# conserved predictors: non-zero in both species' clocks for the tissue, same sign
cons = []
for tis, g in C.groupby("tissue"):
    r = g[g.species == "rat"].set_index("gene").coef; m = g[g.species == "mouse"].set_index("gene").coef
    both = r.index.intersection(m.index)
    for gene in both:
        if np.sign(r[gene]) == np.sign(m[gene]):
            cons.append(dict(tissue=tis, gene=gene, coef_rat=r[gene], coef_mouse=m[gene], senmayo=gene in senset, metabolic=gene in metset, cellmarker=gene in cellset))
CO = pd.DataFrame(cons); CO.to_csv(W + "conserved_predictors.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
print(R.round(3).to_string(index=False)); print(CO.groupby("tissue").size()); print(C.groupby(["species", "tissue"]).size())
