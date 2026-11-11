"""Figure 4 (age models) and Table 1 (conserved predictors)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
W = "/home/claude/g12/work/"; F = "/home/claude/g12/figures/"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600})
TIS = ["gWAT", "liver", "left ventricle", "kidney", "gastrocnemius"]; TL = ["Gonadal\nWAT", "Liver", "Heart\n(LV)", "Kidney", "Skeletal\nmuscle"]
C = pd.read_csv(W + "age_models_cv.csv"); T = pd.read_csv(W + "transfer_validation.csv"); K = pd.read_csv(W + "clock_composition.csv")
fig = plt.figure(figsize=(7.4, 6.2)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.62, wspace=0.42)
# A: CV MAE
cols = {"ElasticNet": "#2166ac", "XGBoost": "#67a9cf", "MLP": "#d1e5f0"}
for i, sp in enumerate(["rat", "mouse"]):
    ax = fig.add_subplot(gs[0, i]); x = np.arange(len(TIS))
    for j, m in enumerate(cols):
        g = C[(C.species == sp) & (C.model == m)].set_index("tissue").loc[TIS]
        ax.bar(x + (j - 1) * 0.26, g.MAE, 0.26, yerr=g.MAE_sd, color=cols[m], edgecolor="k", lw=0.4, label=m, capsize=1.2, error_kw={"lw": 0.6})
    null = C[(C.species == sp)].groupby("tissue").MAE_null.first().loc[TIS]
    ax.scatter(x, null, marker="_", s=180, color="#b2182b", lw=1.4, label="Predict mean age", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(TL); ax.set_ylabel("Cross-validated MAE (months)"); ax.set_ylim(0, 11)
    ax.set_title(f"{'A' if i == 0 else 'B'}  {'Rat (Sprague Dawley, male)' if sp == 'rat' else 'Mouse (C57BL/6J, both sexes)'}", loc="left", fontsize=8)
    if i == 1: ax.legend(frameon=False, fontsize=6.3, ncol=2, loc="upper left")
# C: transfer / external correlations
ax = fig.add_subplot(gs[1, 0])
rows = [("rat_atlas", "BodyMap_rat", "external", "Rat → rat body map (F344)"), ("mouse_atlas", "TMS_mouse", "external", "Mouse → Tabula Muris Senis"),
        ("mouse_atlas", "rat_atlas", "cross-species", "Mouse → rat (orthologues)"), ("rat_atlas", "mouse_atlas", "cross-species", "Rat → mouse (orthologues)")]
M = np.full((len(rows), len(TIS)), np.nan)
for a, (tr, te, kind, _) in enumerate(rows):
    for b, t in enumerate(TIS):
        g = T[(T.train == tr) & (T.test == te) & (T.kind == kind) & (T.tissue == t)]
        if len(g): M[a, b] = g.r.values[0]
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
for a in range(M.shape[0]):
    for b in range(M.shape[1]):
        if not np.isnan(M[a, b]): ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(M[a, b]) > .6 else "k")
ax.set_xticks(range(len(TIS))); ax.set_xticklabels(TL); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[3] for r in rows])
ax.set_title("C  Transfer: Pearson r, predicted vs true age", loc="left", fontsize=8); ax.spines[:].set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.ax.tick_params(labelsize=5.5)
# D: composition dependence
ax = fig.add_subplot(gs[1, 1]); x = np.arange(len(TIS))
for k, (sp, mk) in enumerate([("rat", "o"), ("mouse", "s")]):
    g = K[K.species == sp].set_index("tissue").loc[TIS]; off = (k - 0.5) * 0.25
    for lab, col, c in [("r_raw", "#2166ac", "All genes"), ("r_comp_only", "#999999", "Composition scores only"), ("r_adj", "#b2182b", "Composition-adjusted genes")]:
        ax.scatter(x + off, g[lab], marker=mk, color=col, s=16, label=f"{c} ({sp})" if True else None, zorder=3)
    for xi, a_, b_ in zip(x + off, g.r_raw, g.r_adj): ax.plot([xi, xi], [a_, b_], color="#bbbbbb", lw=0.6, zorder=1)
ax.axhline(0, color="grey", lw=0.5); ax.set_xticks(x); ax.set_xticklabels(TL); ax.set_ylim(-0.2, 1.05); ax.set_ylabel("Cross-validated Pearson r")
ax.set_title("D  Clock accuracy after removing composition", loc="left", fontsize=8)
h, l = ax.get_legend_handles_labels(); ax.legend(h, l, frameon=False, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
fig.savefig(F + "Figure4.tif", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight"); fig.savefig(F + "Figure4.png", dpi=200, bbox_inches="tight"); plt.close()

# Table 1: conserved predictors
CO = pd.read_csv(W + "conserved_predictors.csv"); SH = pd.read_csv(W + "shap_top50.csv")
rows = []
for t, tl in zip(TIS, ["Gonadal WAT", "Liver", "Heart (left ventricle)", "Kidney", "Skeletal muscle (gastrocnemius)"]):
    g = CO[CO.tissue == t].copy(); g["w"] = g[["coef_rat", "coef_mouse"]].abs().min(axis=1); g = g.sort_values("w", ascending=False)
    up = [x for x, c in zip(g.gene, g.coef_rat) if c > 0]; dn = [x for x, c in zip(g.gene, g.coef_rat) if c < 0]
    a = set(SH[(SH.tissue == t) & (SH.species == "rat")].gene); b = set(SH[(SH.tissue == t) & (SH.species == "mouse")].gene)
    rows.append({"Tissue": tl, "Conserved predictors, n": len(g), "Positive coefficient (both species)": ", ".join(up) or "–", "Negative coefficient (both species)": ", ".join(dn) or "–",
                 "Top-50 SHAP genes shared by both species": ", ".join(sorted(a & b)) or "none"})
pd.DataFrame(rows).to_csv(W + "Table1.csv", index=False); print(pd.DataFrame(rows).to_string(index=False))
