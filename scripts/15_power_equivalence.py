"""G12 step 15: how large a senescence lead could the T50 design have detected?
(a) Power simulation in each stratum used for the raw lead-lag analysis: real ages, group sizes, score amplitudes and residual
    noise; logistic trajectories with the metabolic midpoint at its observed T50 and the senescence midpoint shifted by L months.
    Detection = bootstrap 95% CI of T50(sen) - T50(met) entirely below zero.
(b) Random-effects pooling (DerSimonian-Laird) of the observed lags, giving the largest pooled lead compatible with the data."""
import sys; sys.path.insert(0, "/home/claude/g12/scripts")
import numpy as np, pandas as pd
from lib import *
rng = np.random.default_rng(1)
L = pd.read_csv(W + "leadlag.csv"); sc = pd.read_csv(W + "all_scores.csv", index_col=0)
E = L[(L.p_Senescence < .05) & (L.p_Metabolic < .05)].copy()
SHIFTS = [0, -3, -6, -9]; NSIM = int(sys.argv[1]) if len(sys.argv) > 1 else 100; NBOOT = 200

def params(age, y):
    ag = np.unique(age); mu = np.array([y[age == a].mean() for a in ag])
    resid = y - np.array([mu[list(ag).index(a)] for a in age])
    return mu[-1] - mu[0], resid.std(ddof=1)

rows = []
for _, r in E.iterrows():
    s = sc[(sc.dataset == r.dataset) & (sc.tissue == r.tissue) & (sc.Sex == r.sex)]
    age = s.age.values.astype(float); lo_a, hi_a = age.min(), age.max()
    As, ss = params(age, s.Senescence.values); Am, sm = params(age, s.Metabolic.values)
    mid = float(np.clip(r.T50_Metabolic, lo_a + 3, hi_a - 3)); width = 2.5
    for sh in SHIFTS:
        det = 0; est = []
        for k in range(NSIM):
            ms = np.clip(mid + sh, lo_a, hi_a)
            ysen = As / (1 + np.exp(-(age - ms) / width)) + rng.normal(0, ss, len(age))
            ymet = Am / (1 + np.exp(-(age - mid) / width)) + rng.normal(0, sm, len(age))
            _, _, d, lo, hi, p = lead_lag(age, ysen, ymet, nboot=NBOOT, seed=k)
            det += hi < 0; est.append(d)
        rows.append(dict(dataset=r.dataset, tissue=r.tissue, sex=r.sex, n=len(age), shift=sh, power=det / NSIM,
                         median_estimate=np.nanmedian(est), amp_sen_over_sd=As / ss, amp_met_over_sd=Am / sm))
        print(r.dataset, r.tissue, r.sex, sh, det / NSIM, flush=True)
P = pd.DataFrame(rows); P.to_csv(W + "power_simulation.csv", index=False)
# (b) random-effects pooling of observed raw lags
y = E.lag_Senescence.values; se = (E.lagHi_Senescence - E.lagLo_Senescence).values / 3.92; w = 1 / se ** 2
fe = (w * y).sum() / w.sum(); Q = (w * (y - fe) ** 2).sum(); k = len(y)
tau2 = max(0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())); wr = 1 / (se ** 2 + tau2)
pooled = (wr * y).sum() / wr.sum(); pse = np.sqrt(1 / wr.sum())
out = dict(k=k, pooled_lag=pooled, lo=pooled - 1.96 * pse, hi=pooled + 1.96 * pse, tau2=tau2, Q=Q)
pd.DataFrame([out]).to_csv(W + "pooled_lag.csv", index=False); print(out)
