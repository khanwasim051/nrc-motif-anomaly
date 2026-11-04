"""Vary the injected coordination window delta independently of NRC's scales."""
import numpy as np, os, sys, json
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
from sklearn.metrics import roc_auc_score as auc, average_precision_score as ap
import coupling as C, coupling2 as C2
from motif_bench import load, build, triangles
from camouflage import edge_events
from coupling import SCALES
name, out = sys.argv[1], sys.argv[2]
E = load(name); mult, birth, adj = build(E); tri = triangles(adj)
ts = np.array([e[2] for e in E]); tmin, tmax = ts.min(), ts.max()
pa = [t for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3]
ev = edge_events(E)
for dl in [60.0, 600.0, 3600.0, 14400.0]:
    for v in ["anchored", "joint_bursts"]:
        for rho in [0.1, 0.25]:
            for s in range(3):
                rng = np.random.default_rng(100 + s)
                pool = [pa[i] for i in rng.choice(len(pa), min(3000, len(pa)), replace=False)]
                ev2, y = C.inject(pool, ev, v, rho, rng, delta=dl, tmin=tmin, tmax=tmax); nidx = C.node_index(ev2)
                Z = {d: C2.chan_coincidence(pool, ev2, nidx, scales=[d]) for d in SCALES}
                nv = {d: C.naive_rate(pool, ev2, d) for d in sorted(set(SCALES) | {600.0})}
                S = np.sum([Z[d] for d in SCALES], 0)
                rec = dict(data=name, delta=dl, variant=v, rho=rho, seed=s,
                           nrc=auc(y, S), nrc_ap=ap(y, S),
                           naive_10m_1h=max(auc(y, nv[600.0]), auc(y, nv[3600.0])),
                           naive_10m_1h_ap=max(ap(y, nv[600.0]), ap(y, nv[3600.0])),
                           naive_best5=max(auc(y, nv[d]) for d in nv), naive_best5_ap=max(ap(y, nv[d]) for d in nv),
                           single={str(int(d)): auc(y, Z[d]) for d in SCALES})
                rec["single_best"] = max(rec["single"].values())
                open(out, "a").write(json.dumps(rec) + "\n"); print(json.dumps(rec)[:200], flush=True)
