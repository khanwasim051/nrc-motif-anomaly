"""Per-endpoint activity audit of re-timed events: for every re-timed event t' on edge (u, v) of an anomalous motif,
d(t', A_u^out) and d(t', A_v^out), where A_x^out are x's events on edges outside the motif (original stream).
Reference: the same distances for organic later events on edges of normal motifs."""
import numpy as np, sys, json
import coupling as C
from motif_bench import load, build, triangles
from camouflage import edge_events
NEW = []
C.RECORD = NEW                                  # coupling._retime appends (edge, new times placed)
name, out = sys.argv[1], sys.argv[2]
W = [600.0, 3600.0, 86400.0]
E = load(name); mult, birth, adj = build(E); tri = triangles(adj)
ts = np.array([e[2] for e in E]); tmin, tmax = ts.min(), ts.max()
pa = [t for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3]
ev = edge_events(E); nidx = C.node_index(ev)
def dist(x, A):
    if len(A) == 0: return np.full(len(x), np.inf)
    i = np.searchsorted(A, x); lo = np.abs(x - A[np.clip(i - 1, 0, len(A) - 1)]); hi = np.abs(A[np.clip(i, 0, len(A) - 1)] - x); return np.minimum(lo, hi)
def ext(x, motif): return nidx[x][0][~np.isin(nidx[x][1], list(motif))]
for v in ["anchored", "joint_bursts", "periodic"]:
    for rho in [0.1, 0.25]:
        for s in range(3):
            NEW.clear(); rng = np.random.default_rng(100 + s)
            pool = [pa[i] for i in rng.choice(len(pa), min(3000, len(pa)), replace=False)]
            ev2, y = C.inject(pool, ev, v, rho, rng, tmin=tmin, tmax=tmax)
            anom = [pool[i] for i in np.where(y == 1)[0]]; e2m = {}
            for m in anom:
                for e in [(m[0], m[1]), (m[0], m[2]), (m[1], m[2])]: e2m.setdefault(e, m)
            du, dv = [], []
            for (u, w_), tt in NEW:
                m = e2m[(u, w_)]; du.append(dist(tt, ext(u, m))); dv.append(dist(tt, ext(w_, m)))
            du, dv = np.concatenate(du), np.concatenate(dv)
            # organic reference: later events of edges of 300 normal motifs
            nm = [pool[i] for i in np.where(y == 0)[0][:300]]; ou, ov = [], []
            for m in nm:
                for (u, w_) in [(m[0], m[1]), (m[0], m[2]), (m[1], m[2])]:
                    tt = ev[(u, w_)][1:]; ou.append(dist(tt, ext(u, m))); ov.append(dist(tt, ext(w_, m)))
            ou, ov = np.concatenate(ou), np.concatenate(ov)
            rec = dict(data=name, variant=v, rho=rho, seed=s, n_retimed=int(len(du)))
            for w in W:
                rec[f"retimed_both_{int(w)}"] = float(np.mean((du <= w) & (dv <= w))); rec[f"retimed_one_{int(w)}"] = float(np.mean((du <= w) | (dv <= w)))
                rec[f"organic_both_{int(w)}"] = float(np.mean((ou <= w) & (ov <= w))); rec[f"organic_one_{int(w)}"] = float(np.mean((ou <= w) | (ov <= w)))
            open(out, "a").write(json.dumps(rec) + "\n")
