"""Empirical null calibration of NRC's per-scale Z(delta) and of the summed score.
Activity-pool randomisation: for each edge-disjoint candidate triangle, each edge keeps its birth event and its
|T_e|-1 later events are redrawn (without replacement where possible) from the part of the edge's activity pool A_j
after its birth, i.e. from the times after b_e at which its endpoints are active on edges outside the motif, so the
birth stays the first event. A global approximation of NRC's null (draws are not local in time).
Also reports Z on the unrandomised organic motifs."""
import numpy as np, os, sys, json
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
import coupling as C, coupling2 as C2
from coupling2 import pools_for
from motif_bench import load, build, triangles
from camouflage import edge_events
from coupling import SCALES
name = sys.argv[1]; out = sys.argv[2]
E = load(name); mult, birth, adj = build(E); tri = triangles(adj)
pa = [t for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3]
ev = edge_events(E); nidx = C.node_index(ev)
E3 = lambda m: [(m[0], m[1]), (m[0], m[2]), (m[1], m[2])]
TH = [1.645, 2.326, 3.090]
for s in range(3):
    rng = np.random.default_rng(500 + s); used = set(); pool = []
    for i in rng.permutation(len(pa)):
        es = E3(pa[i])
        if any(e in used for e in es): continue
        used.update(es); pool.append(pa[i])
        if len(pool) >= 3000: break
    ev2 = dict(ev)
    for m in pool:
        T, P = pools_for(m, ev, nidx)
        for e, Tj, Pj in zip(E3(m), T, P):
            n = len(Tj) - 1
            Pa = Pj[Pj > Tj[0]]                      # only pool events after the birth, so b_e stays the first event
            if len(Pa) == 0 or n == 0: continue
            new = rng.choice(Pa, n, replace=len(Pa) < n)
            ev2[e] = np.sort(np.r_[Tj[0], new])
    for kind, evx in [("randomised", ev2), ("organic", ev)]:
        Z = np.stack([C2.chan_coincidence(pool, evx, nidx, scales=[d]) for d in SCALES])   # 5 x n
        S = Z.sum(0)
        rec = dict(data=name, seed=s, kind=kind, n=len(pool),
                   z_mean=Z.mean(1).tolist(), z_sd=Z.std(1).tolist(),
                   z_fpr={str(t): (Z > t).mean(1).tolist() for t in TH},
                   S_mean=float(S.mean()), S_sd=float(S.std()), S_sd_indep=float(np.sqrt(5)),
                   S_fpr_nominal={str(t): float((S / np.sqrt(5) > t).mean()) for t in TH},
                   z_corr_mean=float(np.mean(np.corrcoef(Z)[np.triu_indices(5, 1)])))
        open(out, "a").write(json.dumps(rec) + "\n"); print(json.dumps(rec)[:400], flush=True)
