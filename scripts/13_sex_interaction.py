"""G12 step 13: formal test of sex differences in age trends (mouse atlas and Tabula Muris Senis).
For each tissue with both sexes: score ~ age + sex + age:sex (OLS, scores standardised within tissue and sex as in the main analysis);
the age:sex term tests whether the age slope differs between females and males. Also reports the sex-specific Spearman trends."""
import sys; sys.path.insert(0, "/home/claude/g12/scripts")
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import stats
W = "/home/claude/g12/work/"
sc = pd.read_csv(W + "all_scores.csv", index_col=0); sc.index = sc.index.astype(str)
dc = pd.read_csv(W + "deconvolution_fractions.csv", index_col=0); dc.index = dc.index.astype(str)
rows = []
for (ds, tis), s in sc[sc.dataset != "atlas_rat"].groupby(["dataset", "tissue"]):
    if s.Sex.nunique() < 2: continue
    for k in ["Senescence", "Senescence_adj", "Metabolic", "Macrophage"]:
        d = s[[k, "age", "Sex"]].dropna().rename(columns={k: "y"})
        if d.Sex.value_counts().min() < 8: continue
        f = smf.ols("y ~ age * C(Sex)", data=d).fit()
        term = [t for t in f.params.index if t.startswith("age:")][0]
        r = dict(dataset=ds, tissue=tis, score=k, n_female=(d.Sex == "female").sum(), n_male=(d.Sex == "male").sum(),
                 slope_female=f.params["age"], slope_male=f.params["age"] + f.params[term], interaction_beta=f.params[term],
                 interaction_p=f.pvalues[term])
        for sx in ["female", "male"]:
            q = d[d.Sex == sx]; r[f"rho_{sx}"], r[f"p_{sx}"] = stats.spearmanr(q.age, q.y)
        rows.append(r)
O = pd.DataFrame(rows)
O["interaction_q"] = O.groupby("score").interaction_p.transform(lambda p: multipletests(p, method="fdr_bh")[1])
O["same_direction"] = np.sign(O.rho_female) == np.sign(O.rho_male)
O.to_csv(W + "sex_interaction.csv", index=False)
print(O.groupby("score").agg(n=("tissue", "size"), sig_q=("interaction_q", lambda q: (q < .05).sum()),
                             same_dir=("same_direction", "sum")).to_string())
print(O[O.interaction_q < .05][["dataset", "tissue", "score", "rho_female", "rho_male", "interaction_q"]].to_string())
