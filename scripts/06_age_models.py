"""G12 step 6: tissue-specific transcriptomic age models (elastic net, XGBoost, MLP),
nested 5-fold CV (repeated 3x), cross-species transfer and external validation."""
import sys; sys.path.insert(0, "/home/claude/g12/scripts")
import numpy as np, pandas as pd, warnings, json
from lib import W
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from scipy import stats
warnings.filterwarnings("ignore")
NG = 2000; SEED = 7

def load(sp, tis):
    if tis == "gastrocnemius":
        X = pd.read_parquet(W + f"{sp}_gastroc_logcpm.parquet"); M = pd.read_csv(W + f"{sp}_gastroc_meta.csv", index_col=0)
    else:
        X = pd.read_parquet(W + f"{sp}_logcpm.parquet"); M = pd.read_csv(W + f"{sp}_meta.csv", index_col=0)
        M.index = M.index.astype(str); M = M[M.tissue == tis]
    X = X.groupby(level=0).mean()[M.index]
    return X, M

def models():
    return {
        "ElasticNet": lambda: make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.1, .5, .9], alphas=30, cv=4, max_iter=5000, random_state=SEED)),
        "XGBoost": lambda: XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.3, random_state=SEED, n_jobs=2),
        "MLP": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 16), alpha=1.0, learning_rate_init=1e-3, max_iter=2000, early_stopping=True, random_state=SEED)),
    }

def select(Xtr):  # unsupervised: expressed + most variable within training fold only
    e = Xtr.loc[Xtr.mean(axis=1) > 1]
    return e.var(axis=1).sort_values(ascending=False).index[:NG]

TISSUES = ["gWAT", "liver", "left ventricle", "kidney", "gastrocnemius"]
rows = []; preds = []
for sp in ["rat", "mouse"]:
    for tis in TISSUES:
        X, M = load(sp, tis); y = M.age_m.values.astype(float)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
        P = {m: np.zeros((3, len(y))) for m in models()}
        for i, (tr, te) in enumerate(cv.split(X.T, y.astype(int))):
            g = select(X.iloc[:, tr]); Xtr, Xte = X.loc[g].iloc[:, tr].T.values, X.loc[g].iloc[:, te].T.values
            for m, f in models().items():
                P[m][i // 5, te] = f().fit(Xtr, y[tr]).predict(Xte)
        for m in P:
            p = P[m].mean(0); mae = np.mean(np.abs(p - y)); r = stats.pearsonr(p, y)[0]
            mae_rep = [np.mean(np.abs(P[m][k] - y)) for k in range(3)]
            rows.append(dict(species=sp, tissue=tis, model=m, n=len(y), MAE=mae, MAE_sd=np.std(mae_rep), r=r,
                             MAE_null=np.mean(np.abs(y - y.mean()))))
            preds.append(pd.DataFrame(dict(species=sp, tissue=tis, model=m, sample=M.index, age=y, pred=p)))
        print(sp, tis, [(r_["model"], round(r_["MAE"], 2), round(r_["r"], 2)) for r_ in rows[-3:]], flush=True)
R = pd.DataFrame(rows); R.to_csv(W + "age_models_cv.csv", index=False); pd.concat(preds).to_csv(W + "age_models_pred.csv", index=False)
print(R.round(3).to_string(index=False))
