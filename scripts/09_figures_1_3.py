"""G12 figures 1-3 (trend heatmap, trajectories, lead-lag forest)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
W = "/home/claude/g12/work/"; F = "/home/claude/g12/figures/"
import os; os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600})
L = pd.read_csv(W + "leadlag.csv")
TN = {"gWAT": "gWAT", "GAT": "gWAT", "SCAT": "scWAT", "BAT": "BAT", "liver": "Liver", "Liver": "Liver",
      "left ventricle": "LV", "Heart": "Heart", "kidney": "Kidney", "Kidney": "Kidney", "gastrocnemius": "Muscle",
      "Limb_Muscle": "Muscle", "Pancreas": "Pancreas"}
DN = {"atlas_rat": "Rat", "atlas_mouse": "Mouse", "TMS_mouse": "TMS"}
L["label"] = [f"{DN[d]} · {TN[t]} · {'M' if s == 'male' else 'F'}" for d, t, s in zip(L.dataset, L.tissue, L.sex)]
L = L.sort_values(["dataset", "tissue", "sex"], key=lambda c: c.map({"atlas_rat": 0, "atlas_mouse": 1, "TMS_mouse": 2}) if c.name == "dataset" else c)

# ---- Figure 1: heatmap of age correlations ----
cols = [("rho_Senescence", "Senescence"), ("rho_Senescence_adj", "Senescence,\nadjusted"),
        ("rho_Macrophage", "Macrophage"), ("rho_Metabolic", "Metabolic")]
M = L[[c for c, _ in cols]].values; P = L[[c.replace("rho", "p") for c, _ in cols]].values
fig, ax = plt.subplots(figsize=(6.0, 9.2))
im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i, j]:.2f}" + ("*" if P[i, j] < 0.05 else ""), ha="center", va="center", fontsize=12,
                    color="white" if abs(M[i, j]) > 0.55 else "black")
ax.set_xticks(range(len(cols))); ax.set_xticklabels([n for _, n in cols], fontsize=12, rotation=35, ha="left", rotation_mode="anchor")
ax.set_yticks(range(len(L))); ax.set_yticklabels(L.label, fontsize=12); ax.xaxis.tick_top()
for y in np.where(L.dataset.values[1:] != L.dataset.values[:-1])[0]: ax.axhline(y + 0.5, color="k", lw=0.8)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Spearman ρ with age", fontsize=12)
pass
fig.tight_layout(); fig.savefig(F + "Figure1.tif", pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight"); fig.savefig(F + "Figure1.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- Figure 2: trajectories ----
S = pd.read_csv(W + "all_scores.csv")
panels = [("atlas_mouse", "gWAT", "male"), ("atlas_mouse", "liver", "male"), ("atlas_mouse", "kidney", "female"),
          ("atlas_rat", "kidney", "male"), ("atlas_rat", "left ventricle", "male"), ("atlas_rat", "liver", "male")]
series = [("Senescence", "Senescence", "#b2182b", "-"), ("Senescence_adj", "Senescence, adjusted", "#b2182b", "--"),
          ("Macrophage", "Macrophage", "#4d4d4d", ":"), ("Metabolic", "Metabolic", "#2166ac", "-")]
fig, axs = plt.subplots(2, 3, figsize=(6.4, 6.4), sharex=True)
for ax, (d, t, s) in zip(axs.flat, panels):
    g = S[(S.dataset == d) & (S.tissue == t) & (S.Sex == s)]
    for k, lab, c, ls in series:
        m = g.groupby("age")[k].agg(["mean", "sem"])
        ax.errorbar(m.index, m["mean"], yerr=m["sem"], color=c, ls=ls, lw=1.2, marker="o", ms=2.5, capsize=1.5, label=lab)
    ax.axhline(0, color="grey", lw=0.4)
    ax.set_title(f"{DN[d]} {TN[t]}\n{'male' if s == 'male' else 'female'}", fontsize=12)
    ax.set_xticks([6, 12, 18, 24]);
for ax in axs[1]: ax.set_xlabel("Age (months)")
for ax in axs[:, 0]: ax.set_ylabel("Score (mean z ± SEM)")
h_, l_ = axs[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, fontsize=12, frameon=False, loc="lower center", ncol=2)
fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(F + "Figure2.tif", pil_kwargs={"compression": "tiff_lzw"}); fig.savefig(F + "Figure2.png", dpi=200); plt.close()

# ---- Figure 3: lead-lag forest (only strata where BOTH programmes change, P<.05) ----
for sen in ["Senescence", "Senescence_adj"]:
    L[f"both_{sen}"] = (L[f"p_{sen}"] < 0.05) & (L["p_Metabolic"] < 0.05)
fig, axs = plt.subplots(2, 1, figsize=(6.0, 8.0), sharey=False)
for ax, sen, ttl in zip(axs, ["Senescence", "Senescence_adj"], ["A  Senescence core score", "B  Composition-adjusted score"]):
    g = L[L[f"both_{sen}"]].reset_index(drop=True)
    y = np.arange(len(g))[::-1]
    ax.errorbar(g[f"lag_{sen}"], y, xerr=[g[f"lag_{sen}"] - g[f"lagLo_{sen}"], g[f"lagHi_{sen}"] - g[f"lag_{sen}"]], fmt="o", color="k", ms=3, capsize=2, lw=0.9)
    ax.axvline(0, color="grey", lw=0.6, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(g.label, fontsize=12); ax.set_xlim(-22, 22)
    ax.set_xlabel("T50 senescence − T50 metabolic (months)\n← senescence earlier   metabolic earlier →", fontsize=12)
    ax.set_title(f"{ttl}  (n = {len(g)} strata)", fontsize=12, loc="left")
fig.tight_layout(); fig.savefig(F + "Figure3.tif", pil_kwargs={"compression": "tiff_lzw"}); fig.savefig(F + "Figure3.png", dpi=200, bbox_inches="tight"); plt.close()
L.to_csv(W + "leadlag_labelled.csv", index=False)
print(L[["label", "both_Senescence", "lag_Senescence", "lagLo_Senescence", "lagHi_Senescence", "lagP_Senescence", "both_Senescence_adj", "lag_Senescence_adj", "lagLo_Senescence_adj", "lagHi_Senescence_adj", "lagP_Senescence_adj"]][L.both_Senescence | L.both_Senescence_adj].round(2).to_string(index=False))
