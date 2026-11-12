"""Figure 4 (age models) and Table 1 (conserved predictors)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
W = "/home/claude/g12/work/"; F = "/home/claude/g12/figures/"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600})
TIS = ["gWAT", "liver", "left ventricle", "kidney", "gastrocnemius"]; TL = ["Gonadal\nWAT", "Liver", "Heart\n(LV)", "Kidney", "Skeletal\nmuscle"]
C = pd.read_csv(W + "age_models_cv.csv"); T = pd.read_csv(W + "transfer_validation.csv"); K = pd.read_csv(W + "clock_composition.csv")
TL = ["gWAT", "Liver", "LV", "Kidney", "Muscle"]
fig = plt.figure(figsize=(5.6, 8.6)); gs = fig.add_gridspec(3, 2, top=0.86, bottom=0.13, height_ratios=[1, 0.9, 1], hspace=0.75, wspace=0.35)
# A, B: CV MAE
cols = {"ElasticNet": "#2166ac", "XGBoost": "#67a9cf", "MLP": "#d1e5f0"}
for i, sp in enumerate(["rat", "mouse"]):
    ax = fig.add_subplot(gs[0, i]); x = np.arange(len(TIS))
    for j, m in enumerate(cols):
        g = C[(C.species == sp) & (C.model == m)].set_index("tissue").loc[TIS]
        ax.bar(x + (j - 1) * 0.26, g.MAE, 0.26, yerr=g.MAE_sd, color=cols[m], edgecolor="k", lw=0.4, label=m, capsize=1.2, error_kw={"lw": 0.6})
    null = C[(C.species == sp)].groupby("tissue").MAE_null.first().loc[TIS]
    ax.scatter(x, null, marker="_", s=150, color="#b2182b", lw=1.4, label="Predict mean age", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(TL, rotation=30, ha="right", rotation_mode="anchor"); ax.set_ylim(0, 9.5)
    if i == 0: ax.set_ylabel("MAE (months)")
    ax.set_title(f"{'A' if i == 0 else 'B'}  {'Rat' if sp == 'rat' else 'Mouse'}", loc="left", fontsize=12)
    if i == 1: ax.legend(frameon=False, fontsize=11, ncol=1, loc="upper left", handlelength=1.2) if False else None
h0, l0 = ax.get_legend_handles_labels(); fig.legend(h0, l0, fontsize=12, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.88))
# C: transfer / external correlations
ax = fig.add_subplot(gs[1, :])
rows = [("rat_atlas", "BodyMap_rat", "external", "Rat → F344"), ("mouse_atlas", "TMS_mouse", "external", "Mouse → TMS"),
        ("mouse_atlas", "rat_atlas", "cross-species", "Mouse → rat"), ("rat_atlas", "mouse_atlas", "cross-species", "Rat → mouse")]
M = np.full((len(rows), len(TIS)), np.nan)
for a, (tr, te, kind, _) in enumerate(rows):
    for b, t in enumerate(TIS):
        g = T[(T.train == tr) & (T.test == te) & (T.kind == kind) & (T.tissue == t)]
        if len(g): M[a, b] = g.r.values[0]
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
for a in range(M.shape[0]):
    for b in range(M.shape[1]):
        if not np.isnan(M[a, b]): ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=12, color="white" if abs(M[a, b]) > .6 else "k")
ax.set_xticks(range(len(TIS))); ax.set_xticklabels(TL); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[3] for r in rows])
ax.set_title("C  Transfer (Pearson r)", loc="left", fontsize=12); ax.spines[:].set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.ax.tick_params(labelsize=12)
# D: composition dependence
ax = fig.add_subplot(gs[2, :]); x = np.arange(len(TIS))
from matplotlib.lines import Line2D
for k, (sp, mk) in enumerate([("rat", "o"), ("mouse", "s")]):
    g = K[K.species == sp].set_index("tissue").loc[TIS]; off = (k - 0.5) * 0.25
    for lab, col in [("r_raw", "#2166ac"), ("r_comp_only", "#999999"), ("r_adj", "#b2182b")]:
        ax.scatter(x + off, g[lab], marker=mk, color=col, s=18, zorder=3)
    for xi, a_, b_ in zip(x + off, g.r_raw, g.r_adj): ax.plot([xi, xi], [a_, b_], color="#bbbbbb", lw=0.6, zorder=1)
ax.axhline(0, color="gray", lw=0.5); ax.set_xticks(x); ax.set_xticklabels(TL); ax.set_ylim(-0.2, 1.05); ax.set_ylabel("Pearson r"); ax.set_xlim(-0.5, 4.5)
ax.set_title("D  Accuracy and composition", loc="left", fontsize=12)
hs = [Line2D([], [], ls="", marker="o", color=c) for c in ["#2166ac", "#999999", "#b2182b"]] + [Line2D([], [], ls="", marker=m, color="k") for m in ["o", "s"]]
ax.legend(hs, ["All genes", "Composition scores only", "Composition-adjusted genes", "Rat", "Mouse"], frameon=False, fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
fig.savefig(F + "Figure4.tif", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight"); fig.savefig(F + "Figure4.png", dpi=200, bbox_inches="tight"); plt.close()

# Table 1: conserved predictors
CO = pd.read_csv(W + "conserved_predictors.csv"); SH = pd.read_csv(W + "shap_top50.csv")
rows = []
for t, tl in zip(TIS, ["Gonadal WAT", "Liver", "Heart (left ventricle)", "Kidney", "Skeletal muscle (gastrocnemius)"]):
    g = CO[CO.tissue == t].copy(); g["w"] = g[["coef_rat", "coef_mouse"]].abs().min(axis=1); g = g.sort_values("w", ascending=False)
    up = [x for x, c in zip(g.gene, g.coef_rat) if c > 0]; dn = [x for x, c in zip(g.gene, g.coef_rat) if c < 0]
    a = set(SH[(SH.tissue == t) & (SH.species == "rat")].gene); b = set(SH[(SH.tissue == t) & (SH.species == "mouse")].gene)
    rows.append({"Tissue": tl, "Genes, n": len(g), "Positive coefficient (both species)": ", ".join(up) or "–", "Negative coefficient (both species)": ", ".join(dn) or "–",
                 "Top-50 SHAP genes shared by both species": ", ".join(sorted(a & b)) or "none"})
pd.DataFrame(rows).to_csv(W + "Table1.csv", index=False); print(pd.DataFrame(rows).to_string(index=False))
