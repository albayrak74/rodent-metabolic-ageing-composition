"""G12 step 14: supplementary tables for the deconvolution sensitivity analysis (eTable 3C) and sex interaction tests (eTable 3D)."""
import pandas as pd
W = "/home/claude/g12/work/"; O = "/home/claude/g12/supp/"
d = pd.read_csv(W + "deconvolution_adjustment.csv")
f = lambda x: f"{x:.2f}"; q = lambda x: "<.001" if x < .001 else f"{x:.3f}"
t = pd.DataFrame({"dataset": d.dataset.str.replace("atlas_", "atlas ").str.replace("TMS_mouse", "TMS"), "tissue": d.tissue, "sex": d.sex.str[0].str.upper(),
                  "n": d.n, "rho_raw": d.rho_raw.map(f), "rho_marker_adj": d.rho_marker_adj.map(f), "rho_deconv_adj": d.rho_deconv_adj.map(f),
                  "q_deconv_adj": d.q_deconv_adj.map(q), "r_mac_frac_vs_marker": d.r_macrophage_fraction_vs_marker.map(f),
                  "rho_mac_frac_age": d.rho_macrophage_fraction.map(f),
                  "lag_deconv_adj (95% CI)": [f"{a:.1f} ({b:.1f} to {c:.1f})" for a, b, c in zip(d.lag_deconv_adj, d.lagLo, d.lagHi)]})
t.to_csv(O + "eTable3c.csv", index=False)
s = pd.read_csv(W + "sex_interaction.csv")
u = pd.DataFrame({"dataset": s.dataset.str.replace("atlas_mouse", "atlas mouse").str.replace("TMS_mouse", "TMS"), "tissue": s.tissue,
                  "score": s.score.str.replace("Senescence_adj", "Senescence (marker-adjusted)"), "n_F": s.n_female, "n_M": s.n_male,
                  "rho_F": s.rho_female.map(f), "rho_M": s.rho_male.map(f), "interaction_beta": s.interaction_beta.map(lambda x: f"{x:.3f}"),
                  "interaction_P": s.interaction_p.map(q), "interaction_q": s.interaction_q.map(q)})
u.to_csv(O + "eTable3d.csv", index=False); print(len(t), len(u))
