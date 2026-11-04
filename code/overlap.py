"""Overlap / label-contamination audit and edge-disjoint control (reviewer request).
For every setting: (i) distribution of re-timed events per anomalous motif and share of unchanged anomalies;
(ii) share of normal motifs sharing an edge with an anomalous motif, sharing an edge that actually had events
re-timed (contaminated), or sharing a node (activity pool touched); (iii) AUROC/AP of naive, NRC, burstiness
and IForest on (a) the standard candidate sample, (b) the same with unchanged anomalies and contaminated
normals removed, (c) an edge-disjoint candidate sample."""
import numpy as np, os, sys, json
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
from collections import defaultdict
from sklearn.metrics import roc_auc_score as auc, average_precision_score as ap
from sklearn.ensemble import IsolationForest
import coupling as C, coupling2 as C2
from motif_bench import load, build, triangles
from camouflage import edge_events
from baselines_dyn import burstiness, motif_features

MOVED = defaultdict(int)
C.RECORD = []                                   # filled by coupling._retime with the new times actually placed
def _sync_moved():
    for p, new in C.RECORD: MOVED[p] += len(new)
    C.RECORD.clear()

name = sys.argv[1]; out = sys.argv[2]
E = load(name); mult, birth, adj = build(E); tri = triangles(adj)
ts = np.array([e[2] for e in E]); tmin, tmax = ts.min(), ts.max()
pa = [t for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3]
ev = edge_events(E)
E3 = lambda m: [(m[0], m[1]), (m[0], m[2]), (m[1], m[2])]

def scores(pool, ev2, y, s):
    nidx = C.node_index(ev2)
    n1, n2 = C.naive_rate(pool, ev2, 600), C.naive_rate(pool, ev2, 3600)
    F = motif_features(pool, ev2, nidx)
    return {"naive": (n1, n2),                                    # both scales; oracle chosen per metric and label set below
            "NRC": C2.chan_coincidence(pool, ev2, nidx), "burstiness": burstiness(pool, ev2),
            "IForest": -IsolationForest(n_estimators=200, random_state=s).fit(F).score_samples(F)}

def audit(pool, y):
    anom = [pool[i] for i in np.where(y == 1)[0]]
    mv = np.array([sum(MOVED[e] for e in E3(m)) for m in anom])
    aedges = {e for m in anom for e in E3(m)}; moved_e = {e for e in aedges if MOVED[e] > 0}
    anodes = {x for m in anom for x in m}
    share_e = np.zeros(len(pool), bool); contam = np.zeros(len(pool), bool); share_n = np.zeros(len(pool), bool)
    for i, m in enumerate(pool):
        if y[i]: continue
        es = E3(m); share_e[i] = any(e in aedges for e in es); contam[i] = any(e in moved_e for e in es)
        share_n[i] = any(x in anodes for x in m)
    nn = (y == 0).sum()
    return mv, dict(n_cand=len(pool), n_anom=int(y.sum()), moved_median=float(np.median(mv)), moved_q10=float(np.quantile(mv, .1)),
                    moved_q90=float(np.quantile(mv, .9)), unchanged=float((mv == 0).mean()),
                    norm_share_edge=float(share_e.sum() / nn), norm_contam=float(contam.sum() / nn), norm_share_node=float(share_n.sum() / nn)), contam

def edge_disjoint(rng, cap=3000):
    used = set(); sel = []
    for i in rng.permutation(len(pa)):
        es = E3(pa[i])
        if any(e in used for e in es): continue
        used.update(es); sel.append(pa[i])
        if len(sel) >= cap: break
    return sel

for v in ["anchored", "joint_bursts", "periodic"]:
    for rho in [0.1, 0.25]:
        for s in range(3):
            for mode in ["standard", "edge_disjoint"]:
                MOVED.clear(); C.RECORD.clear(); rng = np.random.default_rng(100 + s)
                pool = [pa[i] for i in rng.choice(len(pa), min(3000, len(pa)), replace=False)] if mode == "standard" else edge_disjoint(rng)
                ev2, y = C.inject(pool, ev, v, rho, rng, tmin=tmin, tmax=tmax); _sync_moved()
                mv, A, contam = audit(pool, y)
                sc = scores(pool, ev2, y, s)
                # naive rate: oracle scale (10 min or 1 h) chosen separately for each metric and label set, as in every other table
                M = lambda f, yy, x, k: max(f(yy, z[k]) for z in x) if isinstance(x, tuple) else f(yy, x[k])
                al = slice(None)
                rec = dict(data=name, variant=v, rho=rho, seed=s, mode=mode, **A,
                           **{k + " AUROC": M(auc, y, x, al) for k, x in sc.items()}, **{k + " AP": M(ap, y, x, al) for k, x in sc.items()})
                if mode == "standard":            # cleaned labels: drop unchanged anomalies and contaminated normals
                    keep = np.ones(len(pool), bool); ai = np.where(y == 1)[0]; keep[ai[mv == 0]] = False; keep &= ~contam
                    rec.update({k + " AUROC clean": M(auc, y[keep], x, keep) for k, x in sc.items()})
                    rec.update({k + " AP clean": M(ap, y[keep], x, keep) for k, x in sc.items()})
                open(out, "a").write(json.dumps(rec) + "\n"); print(json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in rec.items()}), flush=True)
