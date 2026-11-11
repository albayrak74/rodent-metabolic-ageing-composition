"""G12 step 8: (a) BodyMap adult contrast (21 vs 104 wk); (b) do tissue clocks survive composition adjustment?
(c) is CV age-acceleration correlated with composition/senescence scores?"""
import sys; sys.path.insert(0, "/home/claude/g12/scripts")
import numpy as np, pandas as pd, warnings
from lib import W, CELL, gene_sets, zscore_tissue, program_scores, residualise
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats
warnings.filterwarnings("ignore"); SEED = 7

# (a) BodyMap adults
E = pd.read_csv(W + "external_predictions.csv")
b = E[(E.test == "BodyMap_rat") & (E.age > 4)]
outa = []
for tis, g in b.groupby("tissue"):
    old = g.age > 20
    outa.append(dict(tissue=tis, n_21wk=(~old).sum(), n_104wk=old.sum(), mean_pred_21wk=g.pred[~old].mean(), mean_pred_104wk=g.pred[old].mean(),
                     diff=g.pred[old].mean() - g.pred[~old].mean(), AUC=roc_auc_score(old, g.pred),
                     p_MW=stats.mannwhitneyu(g.pred[old], g.pred[~old]).pvalue))
A = pd.DataFrame(outa); A.to_csv(W + "bodymap_adult_contrast.csv", index=False); print(A.round(3).to_string(index=False))

# (b)+(c)
def load(sp, tis):
    if tis == "gastrocnemius":
        X = pd.read_parquet(W + f"{sp}_gastroc_logcpm.parquet"); M = pd.read_csv(W + f"{sp}_gastroc_meta.csv", index_col=0)
    else:
        X = pd.read_parquet(W + f"{sp}_logcpm.parquet"); M = pd.read_csv(W + f"{sp}_meta.csv", index_col=0)
        M.index = M.index.astype(str); M = M[M.tissue == tis]
    return X.groupby(level=0).mean()[M.index], M

S = gene_sets(); P = pd.read_csv(W + "age_models_pred.csv"); P["sample"] = P["sample"].astype(str)
rows = []
for sp in ["rat", "mouse"]:
    for tis in ["gWAT", "liver", "left ventricle", "kidney", "gastrocnemius"]:
        X, M = load(sp, tis); y = M.age_m.values.astype(float)
        z = zscore_tissue(X)
        C, _ = program_scores(z, CELL); C = C.loc[:, C.notna().all()]
        Pr, _ = program_scores(z, S)
        top = z.loc[X.loc[z.index].var(axis=1).sort_values(ascending=False).index[:2000]]
        # residualise every gene on composition scores (unsupervised; no age information used)
        Cm = np.c_[np.ones(len(y)), C.values]
        B = np.linalg.lstsq(Cm, top.T.values, rcond=None)[0]
        R_ = top.T.values - Cm[:, 1:] @ B[1:]
        res = {}
        for lab, F in [("raw", top.T.values), ("composition_adjusted", R_), ("composition_only", C.values)]:
            p = np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(F, y.astype(int)):
                mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
                m = ElasticNetCV(l1_ratio=[.1, .5, .9], alphas=30, cv=4, max_iter=10000, random_state=SEED).fit((F[tr] - mu) / sd, y[tr])
                p[te] = m.predict((F[te] - mu) / sd)
            res[lab] = (np.mean(np.abs(p - y)), stats.pearsonr(p, y)[0])
        # (c) age acceleration from the main CV elastic net vs composition / senescence
        pp = P[(P.species == sp) & (P.tissue == tis) & (P.model == "ElasticNet")].set_index("sample").loc[M.index]
        accel = pp.pred - pp.age
        accel = accel - np.polyval(np.polyfit(pp.age, accel, 1), pp.age)  # remove regression-to-mean
        imm = C[[c for c in ["Macrophage", "T_NK", "B", "Plasma", "Neutrophil"] if c in C]].mean(axis=1)
        # partial correlations controlling for age
        def pc(a, b_):
            ra = a - np.polyval(np.polyfit(y, a, 1), y); rb = b_ - np.polyval(np.polyfit(y, b_, 1), y); return stats.spearmanr(ra, rb)
        r_imm = pc(accel.values, imm.values); r_sen = pc(accel.values, Pr["Senescence"].values); r_met = pc(accel.values, Pr["Metabolic"].values)
        rows.append(dict(species=sp, tissue=tis, n=len(y), MAE_raw=res["raw"][0], r_raw=res["raw"][1], MAE_adj=res["composition_adjusted"][0], r_adj=res["composition_adjusted"][1],
                         MAE_comp_only=res["composition_only"][0], r_comp_only=res["composition_only"][1], MAE_null=np.mean(np.abs(y - y.mean())),
                         accel_vs_immune_rho=r_imm[0], accel_vs_immune_p=r_imm[1], accel_vs_sen_rho=r_sen[0], accel_vs_sen_p=r_sen[1],
                         accel_vs_met_rho=r_met[0], accel_vs_met_p=r_met[1]))
        print(sp, tis, flush=True)
R = pd.DataFrame(rows); R.to_csv(W + "clock_composition.csv", index=False)
pd.set_option("display.width", 250); print(R.round(3).to_string(index=False))
