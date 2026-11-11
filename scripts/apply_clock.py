"""Apply a fitted tissue clock to new bulk RNA-seq data.
usage: python apply_clock.py models/clock_rat_liver.json counts.tsv [--rat-symbols]
counts.tsv: genes x samples raw counts (first column = gene symbol). Rat symbols are used as-is when they
match the mouse orthologue symbol; otherwise map rat genes to mouse symbols with 1:1 orthologues first.
Predictions are relative within the scored dataset (genes are z-scored within it); use >= ~10 samples
spanning a range of ages, and report cell composition alongside predicted age."""
import sys, json, numpy as np, pandas as pd
m = json.load(open(sys.argv[1])); X = pd.read_csv(sys.argv[2], sep="\t", index_col=0)
L = np.log2(X.div(X.sum(axis=0), axis=1) * 1e6 + 1).groupby(level=0).mean()
L = L.reindex(m["genes"]); cov = L.notna().all(axis=1).mean()
Z = L.sub(L.mean(axis=1), axis=0).div(L.std(axis=1) + 1e-9, axis=0).fillna(0)
pred = m["intercept"] + np.asarray(m["coef"]) @ Z.values
print(f"# gene coverage: {cov:.2f}")
print(pd.Series(pred, index=X.columns, name="predicted_age_months").round(2).to_csv(sep="\t"))
