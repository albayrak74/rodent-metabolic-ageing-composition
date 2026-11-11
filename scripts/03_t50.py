"""Lead-lag check: age at which each program reaches 50% of its 6->27 mo change (T50), isotonic fit on group means, bootstrap within age groups."""
import numpy as np, pandas as pd, glob
from sklearn.isotonic import IsotonicRegression
rng=np.random.default_rng(2)
def t50(age,y):
    ag=np.sort(np.unique(age)); mu=np.array([y[age==a].mean() for a in ag])
    inc = mu[-1]>=mu[0]
    f=IsotonicRegression(increasing=inc).fit(ag,mu); fine=np.linspace(ag[0],ag[-1],421); v=f.predict(fine)
    tot=v[-1]-v[0]
    if abs(tot)<1e-9: return np.nan
    frac=(v-v[0])/tot
    return fine[np.argmax(frac>=0.5)]
rows=[]
for f in sorted(glob.glob("work/scores_*.csv")):
    sp,rest=f.split("scores_")[1].split("_",1); tis,sex=rest[:-4].rsplit("_",1)
    if tis not in ["gWAT","liver","left_ventricle","kidney"]: continue
    s=pd.read_csv(f,index_col=0); age=s.age.values
    for k in ["SenMayo_noLeuk","Immune","OXPHOS","FAmetab"]:
        if k not in s: continue
        y=s[k].values; est=t50(age,y); bs=[]
        for _ in range(500):
            idx=np.concatenate([rng.choice(np.where(age==a)[0],(age==a).sum()) for a in np.unique(age)])
            bs.append(t50(age[idx],y[idx]))
        rows.append(dict(sp=sp,tis=tis,sex=sex,score=k,T50=est,lo=np.nanpercentile(bs,2.5),hi=np.nanpercentile(bs,97.5)))
R=pd.DataFrame(rows); pd.set_option("display.width",200); pd.set_option("display.max_rows",200)
print(R.round(1).to_string(index=False)); R.to_csv("work/t50.csv",index=False)
