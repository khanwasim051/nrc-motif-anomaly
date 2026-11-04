import numpy as np, json
import coupling as C, coupling2 as C2
from motif_bench import load, build, triangles
from camouflage import edge_events
from coupling import SCALES
E=load("email_eu"); mult,birth,adj=build(E); tri=triangles(adj)
ts=np.array([e[2] for e in E]); tmin,tmax=ts.min(),ts.max()
pa=[t for t in tri if min(mult[(t[0],t[1])],mult[(t[0],t[2])],mult[(t[1],t[2])])>=3]
ev=edge_events(E); out={}
for v in ["anchored","joint_bursts"]:
  A=[];N=[]
  for s in range(3):
    rng=np.random.default_rng(100+s); pool=[pa[i] for i in rng.choice(len(pa),3000,replace=False)]
    ev2,y=C.inject(pool,ev,v,0.25,rng,tmin=tmin,tmax=tmax); nidx=C.node_index(ev2)
    Z=np.stack([C2.chan_coincidence(pool,ev2,nidx,scales=[d]) for d in SCALES])
    A.append(Z[:,y==1].mean(1)); N.append(Z[:,y==0].mean(1))
  out[v]=dict(anom=np.mean(A,0).tolist(),norm=np.mean(N,0).tolist())
json.dump(out,open("zscale.json","w")); print(out)
