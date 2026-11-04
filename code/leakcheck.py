import numpy as np, sys, os, json
BASE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,BASE)
from sklearn.metrics import roc_auc_score as auc, average_precision_score as ap
import coupling as C, coupling2 as C2
from motif_bench import load, build, triangles, heuristics
from camouflage import edge_events
from collections import defaultdict
def activity_novelty(pool, ev2, nidx, W=86400.0):
    out=[]
    for a,b,c in pool:
        nodes=[a,b,c]; vals=[]
        for (u,v) in [(a,b),(a,c),(b,c)]:
            P=np.sort(np.concatenate([nidx[x][0][~np.isin(nidx[x][1],nodes)] for x in (u,v)]))
            t=ev2[(u,v)][1:]; cnt=np.searchsorted(P,t+W,side='right')-np.searchsorted(P,t-W); vals.extend(-np.log1p(cnt))
        out.append(np.mean(vals) if vals else 0.0)
    return np.array(out)
name=sys.argv[1]; seeds=int(sys.argv[2]) if len(sys.argv)>2 else 3
E=load(name); mult,birth,adj=build(E); tri=triangles(adj)
ts=np.array([e[2] for e in E]); tmin,tmax=ts.min(),ts.max()
pa=[t for t in tri if min(mult[(t[0],t[1])],mult[(t[0],t[2])],mult[(t[1],t[2])])>=3]
ev=edge_events(E)
for v in ['anchored','joint_bursts','periodic']:
  for rho in [0.1,0.25]:
    R=defaultdict(list)
    for s in range(seeds):
      rng=np.random.default_rng(100+s)
      pool=[pa[i] for i in rng.choice(len(pa),min(3000,len(pa)),replace=False)]
      ev2,y=C.inject(pool,ev,v,rho,rng,tmin=tmin,tmax=tmax); nidx=C.node_index(ev2)
      bb={p:t[0] for p,t in ev2.items()}; mm=defaultdict(int,{p:len(t) for p,t in ev2.items()})
      R['structural max'].append(max(auc(y,x) for x in heuristics(pool,mm,bb,adj).values()))
      R['activity novelty'].append(auc(y,activity_novelty(pool,ev2,nidx)))
      n1,n2=C.naive_rate(pool,ev2,600),C.naive_rate(pool,ev2,3600)
      R['best naive'].append(max(auc(y,n1),auc(y,n2))); R['best naive AP'].append(max(ap(y,n1),ap(y,n2)))
      A=C2.chan_coincidence(pool,ev2,nidx); R['NRC'].append(auc(y,A)); R['NRC AP'].append(ap(y,A))
      if len(sys.argv)>3:
        open(sys.argv[3],'a').write(json.dumps(dict(data=name,variant=v,rho=rho,seed=s,**{k:float(x[-1]) for k,x in R.items()}))+'\n')
    print(json.dumps(dict(data=name,variant=v,rho=rho,**{k:round(float(np.mean(x)),4) for k,x in R.items()})),flush=True)
