"""Camouflaged coordination anomalies on UCI.
Anomalous motifs are REAL organic triangles: structure, degrees, common neighbours, per-edge event counts
and per-edge birth times are left exactly as observed. Only a fraction rho of each edge's later events is
re-timed to sit within +/- delta of events on the motif's other edges (coordinated bursts).
We then check trivial heuristics, a naive synchrony statistic, and a surrogate-calibrated synchrony test."""
import numpy as np, os, sys
BASE = os.path.dirname(os.path.abspath(__file__))
from collections import defaultdict
from sklearn.metrics import roc_auc_score
sys.path.insert(0, BASE)
from motif_bench import load, build, triangles, heuristics

def edge_events(E):
    ev = defaultdict(list)
    for u, v, t in E: ev[(u, v)].append(t)
    return {k: np.sort(np.array(v)) for k, v in ev.items()}

def coincidences(a, b, delta):
    """# events in a with an event of b within delta."""
    if len(a) == 0 or len(b) == 0: return 0
    i = np.searchsorted(b, a)
    lo = np.abs(a - b[np.clip(i - 1, 0, len(b) - 1)]); hi = np.abs(b[np.clip(i, 0, len(b) - 1)] - a)
    return int((np.minimum(lo, hi) <= delta).sum())

def motif_sync(trains, delta):
    s = 0
    for i in range(3):
        for j in range(3):
            if i != j: s += coincidences(trains[i], trains[j], delta)
    return s

def make_anomalies(tri_pool, ev, frac, rho, delta, rng):
    """Re-time a fraction rho of non-first events of edges 2,3 to near events of edge 1 (keeps births/counts)."""
    idx = rng.choice(len(tri_pool), int(frac * len(tri_pool)), replace=False)
    new_ev = dict(ev)
    for k in idx:
        a, b, c = tri_pool[k]; pairs = [(a, b), (a, c), (b, c)]
        anchor = ev[pairs[0]]
        for p in pairs[1:]:
            t = ev[p].copy()
            later = np.arange(1, len(t)); m = rng.random(len(later)) < rho
            pick = later[m]
            if len(pick):
                tgt = rng.choice(anchor, len(pick)) + rng.uniform(-delta, delta, len(pick))
                t[pick] = np.maximum(tgt, t[0] + 1)          # never move before the edge's birth
            new_ev[p] = np.sort(t)
    y = np.zeros(len(tri_pool)); y[idx] = 1
    return new_ev, y

def node_streams(ev):
    ns = defaultdict(list)
    for (u, v), t in ev.items(): ns[u].extend(t); ns[v].extend(t)
    return {k: np.sort(np.array(v)) for k, v in ns.items()}

def calibrated_sync(tri_pool, ev, delta, R, rng, tmin, tmax):
    """Excess coincidences over a surrogate null: each edge train is circularly shifted by an independent
    random offset (preserves each edge's own burstiness/rhythm, breaks cross-edge alignment)."""
    span = tmax - tmin; out = []
    for (a, b, c) in tri_pool:
        tr = [ev[(a, b)], ev[(a, c)], ev[(b, c)]]
        obs = motif_sync(tr, delta)
        null = []
        for _ in range(R):
            sh = [tmin + np.sort((x - tmin + rng.uniform(0, span)) % span) for x in tr]
            null.append(motif_sync(sh, delta))
        null = np.array(null)
        out.append((obs - null.mean()) / (null.std() + 1.0))
    return np.array(out)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    E = load("uci"); mult, birth, adj = build(E); tri = triangles(adj)
    pool = [t for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3]
    ev = edge_events(E); ts = np.array([e[2] for e in E]); tmin, tmax = ts.min(), ts.max()
    print(f"UCI organic triangles with >=3 events per edge: {len(pool)}")
    for delta in [600.0, 3600.0]:
        for rho in [0.1, 0.25, 0.5]:
            ev2, y = make_anomalies(pool, ev, 0.05, rho, delta, rng)
            # birth times / multiplicities / structure are unchanged by construction -> recompute anyway
            b2 = {p: t[0] for p, t in ev2.items()}; m2 = {p: len(t) for p, t in ev2.items()}
            H = heuristics(pool, defaultdict(int, m2), b2, adj)
            naive = np.array([motif_sync([ev2[(a, b)], ev2[(a, c)], ev2[(b, c)]], delta) for a, b, c in pool])
            naive_n = naive / np.array([m2[(a, b)] + m2[(a, c)] + m2[(b, c)] for a, b, c in pool])
            cal = calibrated_sync(pool, ev2, delta, 20, rng, tmin, tmax)
            print(f"delta={int(delta):5d}s rho={rho:.2f} | " + "  ".join(f"{k}: {roc_auc_score(y, v):.3f}" for k, v in H.items()) +
                  f" | naive sync count {roc_auc_score(y, naive):.3f}  naive sync rate {roc_auc_score(y, naive_n):.3f}  surrogate-calibrated {roc_auc_score(y, cal):.3f}", flush=True)
