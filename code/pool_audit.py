"""Audit of the Monte Carlo resampling pools: for every candidate edge, n = |T_e| - 1 later events and the post-birth pool
A_j+ = {s in A_j : s > b_e}. Counts edges with an empty A_j+ (edge kept unchanged), with 0 < |A_j+| < n (draws with
replacement, the fallback), and with |A_j+| < k = 50 (local window smaller than k). Replays exactly the instances of
calib.py (edge-disjoint, organic stream) and rndnull.py (main candidate sets, injected stream).
usage: python pool_audit.py GRAPH OUT.jsonl"""
import numpy as np, sys, json
import coupling as C
from coupling2 import pools_for
from motif_bench import load, build, triangles
from camouflage import edge_events
name, out = sys.argv[1], sys.argv[2]
E = load(name); mult, birth, adj = build(E); tri = triangles(adj)
ts = np.array([e[2] for e in E]); tmin, tmax = ts.min(), ts.max()
pa = [t for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3]
ev = edge_events(E); E3 = lambda m: [(m[0], m[1]), (m[0], m[2]), (m[1], m[2])]
def audit(pool, evx, nidx):
    c = dict(edges=0, motifs=len(pool), empty=0, fallback=0, small_local=0, motifs_fallback=0, motifs_empty=0, deficit=[], n=[], pool=[])
    for m in pool:
        T, P = pools_for(m, evx, nidx); fb = em = False
        for Tj, Pj in zip(T, P):
            n = len(Tj) - 1; Pa = Pj[Pj > Tj[0]]; c["edges"] += 1
            if len(Pa) == 0: c["empty"] += 1; em = True
            elif len(Pa) < n: c["fallback"] += 1; fb = True; c["deficit"].append(n - len(Pa)); c["n"].append(n); c["pool"].append(len(Pa))
            if 0 < len(Pa) < 50: c["small_local"] += 1
        c["motifs_fallback"] += fb; c["motifs_empty"] += em
    return c
nidx0 = C.node_index(ev)
for s in range(3):                                   # calib.py instances
    rng = np.random.default_rng(500 + s); used = set(); pool = []
    for i in rng.permutation(len(pa)):
        es = E3(pa[i])
        if any(e in used for e in es): continue
        used.update(es); pool.append(pa[i])
        if len(pool) >= 3000: break
    open(out, "a").write(json.dumps(dict(data=name, use="calibration", seed=s, **audit(pool, ev, nidx0))) + "\n")
for v in ["anchored", "joint_bursts", "periodic"]:   # rndnull.py instances
    for rho in [0.1, 0.25]:
        for s in range(3):
            rng = np.random.default_rng(100 + s)
            pool = [pa[i] for i in rng.choice(len(pa), min(3000, len(pa)), replace=False)]
            ev2, y = C.inject(pool, ev, v, rho, rng, tmin=tmin, tmax=tmax)
            open(out, "a").write(json.dumps(dict(data=name, use="monte_carlo_null", variant=v, rho=rho, seed=s, **audit(pool, ev2, C.node_index(ev2)))) + "\n")
print(name, "done")
