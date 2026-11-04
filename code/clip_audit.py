"""Audit of the boundary handling of Algorithm 1 with feasible-time sampling: per run, the number of selected later
events, the share placed, the share left unchanged because no feasible proposal (after the birth and within the
observation window) was found in 100 draws, the mean number of redraws, and a check that every new time lies in
[t_min, t_max]. Same instances as the main results. Usage: python clip_audit.py G out.jsonl"""
import numpy as np, os, sys, json
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
import coupling as C
from motif_bench import load, build, triangles
from camouflage import edge_events
name, out = sys.argv[1], sys.argv[2]
E = load(name); mult, birth, adj = build(E); tri = triangles(adj)
ts = np.array([e[2] for e in E]); tmin, tmax = ts.min(), ts.max()
pa = [t for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3]
ev = edge_events(E)
for v in ["anchored", "joint_bursts", "periodic"]:
    for rho in [0.1, 0.25]:
        for s in range(3):
            rng = np.random.default_rng(100 + s)
            pool = [pa[i] for i in rng.choice(len(pa), min(3000, len(pa)), replace=False)]
            for k in C.GEN: C.GEN[k] = 0
            ev2, y = C.inject(pool, ev, v, rho, rng, tmin=tmin, tmax=tmax)
            inside = all(ev2[e].min() >= tmin and ev2[e].max() <= tmax for e in ev2)
            births = all(ev2[e][0] == ev[e][0] and np.sum(ev2[e] <= ev[e][0]) <= np.sum(ev[e] <= ev[e][0]) for e in ev)  # birth kept, nothing placed at or before it (organic ties at the birth second may move later)
            open(out, "a").write(json.dumps(dict(data=name, variant=v, rho=rho, seed=s, inside_window=inside, births_first=births, **C.GEN)) + "\n")
