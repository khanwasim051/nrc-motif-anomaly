import numpy as np, sys, json
from motif_bench import load, build, triangles
from camouflage import edge_events
import coupling as C
name=sys.argv[1]; E=load(name); mult,birth,adj=build(E); tri=triangles(adj)
pa=[t for t in tri if min(mult[(t[0],t[1])],mult[(t[0],t[2])],mult[(t[1],t[2])])>=3]
rng=np.random.default_rng(0); pool=[pa[i] for i in rng.choice(len(pa),min(2000,len(pa)),replace=False)]
ev=edge_events(E); nidx=C.node_index(ev)
ne=[];ps=[];dens=[]
span=(E[-1][2]-E[0][2])/86400
for a,b,c in pool:
    for u,v in [(a,b),(a,c),(b,c)]:
        ne.append(len(ev[(u,v)]))
        P=np.concatenate([nidx[x][0][~np.isin(nidx[x][1],[a,b,c])] for x in (u,v)]); ps.append(len(P))
        # events of the endpoints per active day
        dens.append(len(P)/max(1,len(np.unique((P//86400)))))
print(json.dumps(dict(data=name,span_days=round(span),events_per_edge=float(np.median(ne)),pool=float(np.median(ps)),pool_per_active_day=round(float(np.median(dens)),1))))
