"""Per-seed values for the sensitivity study (same instances as ablation.py 'sens' mode)."""
import numpy as np, os, sys, json
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
from sklearn.metrics import roc_auc_score as auc, average_precision_score as ap
import coupling as C, coupling2 as C2
from motif_bench import load, build, triangles
from camouflage import edge_events
name, out = sys.argv[1], sys.argv[2]
E = load(name); mult, birth, adj = build(E); tri = triangles(adj)
ts = np.array([e[2] for e in E]); tmin, tmax = ts.min(), ts.max()
pa = [t for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3]
ev = edge_events(E)
for v in ["anchored", "joint_bursts"]:
    for rho, frac in [(0.05, 0.05), (0.1, 0.05), (0.25, 0.05), (0.5, 0.05)]:
        for s in range(3):
            rng = np.random.default_rng(100 + s)
            pool = [pa[i] for i in rng.choice(len(pa), min(3000, len(pa)), replace=False)]
            ev2, y = C.inject(pool, ev, v, rho, rng, frac=frac, tmin=tmin, tmax=tmax); nidx = C.node_index(ev2)
            n1, n2 = C.naive_rate(pool, ev2, 600), C.naive_rate(pool, ev2, 3600); A = C2.chan_coincidence(pool, ev2, nidx)
            rec = dict(data=name, variant=v, rho=rho, seed=s, naive_auroc=max(auc(y, n1), auc(y, n2)), naive_ap=max(ap(y, n1), ap(y, n2)), nrc_auroc=auc(y, A), nrc_ap=ap(y, A))
            open(out, "a").write(json.dumps(rec) + "\n")
