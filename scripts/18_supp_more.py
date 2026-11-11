"""G12 step 18: supplementary tables eTable 3E (single-cell source and within-cell-type age trends), 3F (alternative
senescence signatures) and 4B (power simulation and pooled lead-lag)."""
import pandas as pd, numpy as np
W = "/home/claude/g12/work/"; O = "/home/claude/g12/supp/"
f2 = lambda x: "" if pd.isna(x) else f"{x:.2f}"; pv = lambda x: "" if pd.isna(x) else ("<.001" if x < .001 else f"{x:.3f}")
S = pd.read_csv(W + "sc_signature_source.csv"); S = S[S.set.isin(["Senescence", "Metabolic"])]
A = pd.read_csv(W + "sc_age_within_celltype.csv")
e = S[["tissue", "set", "n_genes"]].copy()
for b in ["myeloid", "lymphoid", "stromal", "endothelial", "parenchymal/other"]: e[f"top_{b.split('/')[0]}"] = S[f"top_{b}"].map(f2)
e["share_immune"] = S.share_from_immune.map(f2)
e = e.merge(A[["tissue", "n_pseudobulks", "immune_minus_nonimmune_sd", "p_immune", "age_slope_within_celltype_sd_per_21mo", "p_age_within"]], on="tissue")
e.loc[e.set == "Metabolic", ["n_pseudobulks", "immune_minus_nonimmune_sd", "p_immune", "age_slope_within_celltype_sd_per_21mo", "p_age_within"]] = np.nan
for c in ["immune_minus_nonimmune_sd", "age_slope_within_celltype_sd_per_21mo"]: e[c] = e[c].map(f2)
for c in ["p_immune", "p_age_within"]: e[c] = e[c].map(pv)
e["n_pseudobulks"] = e.n_pseudobulks.map(lambda x: "" if pd.isna(x) else str(int(x)))
e["set"] = e.set.str.replace("Senescence", "Senescence core")
e.rename(columns={"age_slope_within_celltype_sd_per_21mo": "slope_within_celltype_SD_per_21mo", "immune_minus_nonimmune_sd": "immune_minus_nonimmune_SD"}).to_csv(O + "eTable3f.csv", index=False)
L = pd.read_csv(W + "alt_signatures.csv")
t = pd.DataFrame({"dataset": L.dataset.str.replace("atlas_", "atlas ").str.replace("TMS_mouse", "TMS"), "tissue": L.tissue, "sex": L.sex.str[0].str.upper(),
                  "signature": L.signature.str.replace("_", " "), "n_genes": L.n_genes, "rho_raw": L.rho_raw.map(f2), "q_raw": L.q_raw.map(pv),
                  "rho_marker_adj": L.rho_marker_adj.map(f2), "q_marker_adj": L.q_marker_adj.map(pv), "rho_deconv_adj": L.rho_deconv_adj.map(f2),
                  "q_deconv_adj": L.q_deconv_adj.map(pv), "r_with_core": L.r_with_senmayo_core.map(f2)})
t.to_csv(O + "eTable3e.csv", index=False)
P = pd.read_csv(W + "power_simulation.csv"); pl = pd.read_csv(W + "pooled_lag.csv").iloc[0]
w = P.pivot_table(index=["dataset", "tissue", "sex", "n"], columns="shift", values="power").reset_index()
w.columns = ["dataset", "tissue", "sex", "n"] + [f"power_true_lead_{-int(c)}mo" if c != 0 else "false_positive_rate" for c in w.columns[4:]]
w = w[["dataset", "tissue", "sex", "n", "false_positive_rate", "power_true_lead_3mo", "power_true_lead_6mo", "power_true_lead_9mo"]]
w["dataset"] = w.dataset.str.replace("atlas_", "atlas ").str.replace("TMS_mouse", "TMS"); w["sex"] = w.sex.str[0].str.upper()
for c in w.columns[4:]: w[c] = w[c].map(f2)
w.loc[len(w)] = ["pooled (random effects, k = 9)", "", "", "", f"lag {pl.pooled_lag:.1f} months", f"95% CI {pl.lo:.1f} to {pl.hi:.1f}", f"tau2 = {pl.tau2:.2f}", ""]
w.to_csv(O + "eTable4b.csv", index=False); print(open(O + "eTable4b.csv").read()); print(open(O + "eTable3f.csv").read())
