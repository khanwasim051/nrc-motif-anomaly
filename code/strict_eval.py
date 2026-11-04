"""Camouflaged protocol with the endpoint-wise activity constraint (both endpoints of every re-timed edge active on
other edges within w of the new time). Same seeds, candidate sets and detectors as the main results."""
import numpy as np, sys, json
from collections import defaultdict
from sklearn.metrics import roc_auc_score as auc, average_precision_score as ap
from sklearn.ensemble import IsolationForest
import coupling as C, coupling2 as C2
from motif_bench import load, build, triangles, heuristics
from camouflage import edge_events
from baselines_dyn import motif_features, burstiness
from leakcheck_fn import activity_novelty
name, out = sys.argv[1], sys.argv[2]; W = float(sys.argv[3]) if len(sys.argv) > 3 else 3600.0
E = load(name); mult, birth, adj = build(E); tri = triangles(adj)
ts = np.array([e[2] for e in E]); tmin, tmax = ts.min(), ts.max()
pa = [t for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3]; ev = edge_events(E)
for v in ["anchored", "joint_bursts", "periodic"]:
    for rho in [0.1, 0.25]:
        for s in range(3):
            for k in C.STRICT: C.STRICT[k] = 0
            rng = np.random.default_rng(100 + s); pool = [pa[i] for i in rng.choice(len(pa), min(3000, len(pa)), replace=False)]
            ev2, y = C.inject(pool, ev, v, rho, rng, tmin=tmin, tmax=tmax, endpoint_w=W); nidx = C.node_index(ev2)
            bb = {p: t[0] for p, t in ev2.items()}; mm = defaultdict(int, {p: len(t) for p, t in ev2.items()})
            n1, n2 = C.naive_rate(pool, ev2, 600), C.naive_rate(pool, ev2, 3600); A = C2.chan_coincidence(pool, ev2, nidx)
            F = motif_features(pool, ev2, nidx); IF = -IsolationForest(n_estimators=200, random_state=s).fit(F).score_samples(F)
            an = auc(y, activity_novelty(pool, ev2, nidx))
            rec = dict(data=name, variant=v, rho=rho, seed=s, w=W, struct=max(auc(y, x) for x in heuristics(pool, mm, bb, adj).values()),
                       actnov=an, actnov_dev=abs(an - 0.5), naive=max(auc(y, n1), auc(y, n2)), naive_ap=max(ap(y, n1), ap(y, n2)),
                       nrc=auc(y, A), nrc_ap=ap(y, A), iforest=auc(y, IF), burst=auc(y, burstiness(pool, ev2)),
                       proposed=C.STRICT["proposed"], accepted=C.STRICT["accepted"], infeasible=C.STRICT["infeasible"])
            open(out, "a").write(json.dumps(rec) + "\n"); print(json.dumps(rec)[:160], flush=True)
