import numpy as np, sys, json
import coupling as C
from coupling2 import pools_for
from motif_bench import load, build, triangles
from camouflage import edge_events
name=sys.argv[1]
E=load(name); mult,birth,adj=build(E); tri=triangles(adj)
ts=np.array([e[2] for e in E]); tmin,tmax=ts.min(),ts.max()
pa=[t for t in tri if min(mult[(t[0],t[1])],mult[(t[0],t[2])],mult[(t[1],t[2])])>=3]
ev=edge_events(E); nidx=C.node_index(ev)
emp=0;small=0;tot=0;nojoint=0;nolat=0;nanom=0;outside=0;nt=0
for s in range(3):
    rng=np.random.default_rng(100+s); pool=[pa[i] for i in rng.choice(len(pa),min(3000,len(pa)),replace=False)]
    for m in pool:
        T,P=pools_for(m,ev,nidx)
        for Pj in P:
            tot+=1; emp+=len(Pj)==0; small+=len(Pj)<50
    idx=np.random.default_rng(7+s).choice(len(pool),max(1,int(.05*len(pool))),replace=False)
    for k in idx:
        m=pool[k]; act=C._active_times(nidx,m); nanom+=1
        nojoint+=len(act)==0
        r=np.random.default_rng(k)
        for _ in range(5):
            per=86400*r.uniform(.5,2); ph=r.uniform(0,per)
            lat=tmin+ph+per*np.arange(int((tmax-tmin)/per)+1)
            if len(act):
                near=np.abs(act[np.clip(np.searchsorted(act,lat),0,len(act)-1)]-lat)<=3600
                nolat+= (not near.any())/5
print(json.dumps(dict(data=name,pool_edges=tot,empty_pool=emp/tot,pool_lt_k=small/tot,anom=nanom,joint_no_activity=nojoint/nanom,periodic_fallback=nolat/nanom)))
