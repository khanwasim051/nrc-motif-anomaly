"""NRC ablations and sensitivity on identical instances.
mode 'abl': null type (global / fixed 1-day window / adaptive kNN), scale aggregation (single scales, max, sum),
            k and lambda sweeps. Cells: anchored & joint_bursts at rho in {0.10, 0.25}.
mode 'sens': rho in {0.05, 0.10, 0.25, 0.50} at 5% anomalies, and anomaly fraction in {1%, 5%, 10%} at rho=0.25."""
import numpy as np, os, sys, json
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
from collections import defaultdict
from sklearn.metrics import roc_auc_score as auc, average_precision_score as ap
import coupling as C, coupling2 as C2
from motif_bench import load, build, triangles
from camouflage import edge_events
from coupling import SCALES

name, mode = sys.argv[1], sys.argv[2]; out = sys.argv[3]; seeds = 3
E = load(name); mult, birth, adj = build(E); tri = triangles(adj)
ts = np.array([e[2] for e in E]); tmin, tmax = ts.min(), ts.max()
pa = [t for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3]
ev = edge_events(E)

def instance(v, rho, frac, s):
    rng = np.random.default_rng(100 + s)
    pool = [pa[i] for i in rng.choice(len(pa), min(3000, len(pa)), replace=False)]
    ev2, y = C.inject(pool, ev, v, rho, rng, frac=frac, tmin=tmin, tmax=tmax)
    return pool, ev2, y, C.node_index(ev2)

def emit(rec):
    open(out, "a").write(json.dumps(rec) + "\n"); print(json.dumps(rec), flush=True)

if mode == "abl":
    for v in ["anchored", "joint_bursts"]:
        for rho in [0.1, 0.25]:
            R = defaultdict(list)
            for s in range(seeds):
                pool, ev2, y, nidx = instance(v, rho, 0.05, s)
                n1, n2 = C.naive_rate(pool, ev2, 600), C.naive_rate(pool, ev2, 3600)
                R["naive (best of 10m/1h)"].append(max(auc(y, n1), auc(y, n2)))
                R["global null, 10 min"].append(auc(y, C.nrc_score(pool, ev2, nidx, scales=[600.0])))
                R["global null, sum of scales"].append(auc(y, C.nrc_sum(pool, ev2, nidx)))
                R["fixed 1-day window, sum"].append(auc(y, C.nrc_local(pool, ev2, nidx)))
                Z = {d: C2.chan_coincidence(pool, ev2, nidx, scales=[d]) for d in SCALES}
                for d in SCALES: R[f"adaptive, single scale {int(d)}s"].append(auc(y, Z[d]))
                R["adaptive, max over scales"].append(auc(y, np.max(np.stack([Z[d] for d in SCALES]), 0)))
                R["adaptive, sum over scales (NRC)"].append(auc(y, np.sum(np.stack([Z[d] for d in SCALES]), 0)))
                for k in [10, 25, 100]:
                    R[f"NRC k={k}"].append(auc(y, C2.chan_coincidence(pool, ev2, nidx, k=k)))
                for lam in [0.0, 20.0]:
                    R[f"NRC lambda={lam:g}"].append(auc(y, C2.chan_coincidence(pool, ev2, nidx, prior=lam)))
            emit(dict(data=name, variant=v, rho=rho, **{k: round(float(np.mean(x)), 4) for k, x in R.items()}))
else:
    for v in ["anchored", "joint_bursts"]:
        for rho, frac in [(0.05, 0.05), (0.1, 0.05), (0.25, 0.05), (0.5, 0.05), (0.25, 0.01), (0.25, 0.10)]:
            R = defaultdict(list)
            for s in range(seeds):
                pool, ev2, y, nidx = instance(v, rho, frac, s)
                n1, n2 = C.naive_rate(pool, ev2, 600), C.naive_rate(pool, ev2, 3600)
                a1, a2 = auc(y, n1), auc(y, n2); A = C2.chan_coincidence(pool, ev2, nidx)
                R["naive AUROC"].append(max(a1, a2)); R["naive AP"].append(max(ap(y, n1), ap(y, n2)))
                R["NRC AUROC"].append(auc(y, A)); R["NRC AP"].append(ap(y, A))
            emit(dict(data=name, variant=v, rho=rho, frac=frac, **{k: round(float(np.mean(x)), 4) for k, x in R.items()}))
