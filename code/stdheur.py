import numpy as np, sys, json
from motif_bench import load, build, triangles, inject, heuristics
from sklearn.metrics import roc_auc_score
name=sys.argv[1]; rng=np.random.default_rng(0); E=load(name); _,_,adj0=build(E); tri=triangles(adj0)
for ratio in [0.01,0.05,0.10]:
    inj=inject(E,tri,ratio,rng); E2=sorted(E+[e for _,es in inj for e in es],key=lambda e:e[2]); mult,birth,adj=build(E2)
    motifs=tri+[m for m,_ in inj]; y=np.r_[np.zeros(len(tri)),np.ones(len(inj))]; H=heuristics(motifs,mult,birth,adj)
    print(json.dumps(dict(data=name,ratio=ratio,**{k:round(roc_auc_score(y,v),3) for k,v in H.items()})),flush=True)
