"""Self-contained checks of the camouflaged generator and of NRC on a synthetic event stream (no data download).
Run from the package root:  python tests/test_protocol.py
Checks: (1) Proposition 1 - structure, every edge multiplicity and every birth time are unchanged by injection, for all
three coordination types, with and without the endpoint constraint; (2) only events after the birth are moved, and every new time lies inside the observation window (feasible-time sampling);
(3) all structural heuristic scores are identical before and after injection; (4) NRC runs and ranks coordinated motifs
above chance (AUROC > 0.55 at rho = 0.5) on this small synthetic stream; (5) the Monte Carlo randomisations draw only
after each edge's birth, so births stay the first events."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from sklearn.metrics import roc_auc_score
import coupling as C, coupling2 as C2
from motif_bench import build, triangles, heuristics
from camouflage import edge_events

def synthetic_stream(n_nodes=60, n_days=30, seed=1):
    """Nodes with daily activity sessions interacting on a random graph with many triangles."""
    r = np.random.default_rng(seed); E = []
    groups = [list(range(i, i + 6)) for i in range(0, n_nodes, 6)]
    pairs = {(min(a, b), max(a, b)) for g in groups for a in g for b in g if a != b}
    pairs |= {tuple(sorted(r.choice(n_nodes, 2, replace=False))) for _ in range(120)}
    for u, v in pairs:
        for _ in range(r.integers(4, 12)):
            day = r.integers(0, n_days); E.append((int(u), int(v), float(day * 86400 + r.uniform(8, 20) * 3600)))
    return sorted(E, key=lambda e: e[2])

E = synthetic_stream(); mult, birth, adj = build(E); tri = triangles(adj)
ev = edge_events(E); ts = np.array([e[2] for e in E]); tmin, tmax = ts.min(), ts.max()
pool = [t for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3]
assert len(pool) >= 100, len(pool)
fails = 0
for variant in ["anchored", "joint_bursts", "periodic"]:
    for w in [None, 3600.0]:
        rng = np.random.default_rng(100)
        ev2, y = C.inject(pool, ev, variant, 0.5, rng, frac=0.1, tmin=tmin, tmax=tmax, endpoint_w=w)
        same_edges = set(ev2) == set(ev)
        same_mult = all(len(ev2[e]) == len(ev[e]) for e in ev)
        same_birth = all(np.min(ev2[e]) == np.min(ev[e]) for e in ev)
        later_only = all(np.sum(ev2[e] <= np.min(ev[e])) == 1 for e in ev)
        in_window = all(ev2[e].max() <= max(tmax, ev[e].max()) for e in ev)
        moved = sum(int(np.sum(~np.isin(ev2[e], ev[e]))) for e in ev)
        nidx = C.node_index(ev2)
        nrc = C2.chan_coincidence(pool, ev2, nidx); naive = C.naive_rate(pool, ev2, 600)
        mm = {p: len(t) for p, t in ev2.items()}; bb = {p: t[0] for p, t in ev2.items()}
        m0 = {p: len(t) for p, t in ev.items()}; b0 = {p: t[0] for p, t in ev.items()}
        H0, H1 = heuristics(pool, m0, b0, adj), heuristics(pool, mm, bb, adj)
        struct_same = all(np.array_equal(H0[k], H1[k]) for k in H0)
        a_nrc, a_nv = roc_auc_score(y, nrc), roc_auc_score(y, naive)
        ok = same_edges and same_mult and same_birth and later_only and in_window and moved > 0 and struct_same and a_nrc > 0.55
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {variant:13s} endpoint_w={w}: edges/multiplicities/births unchanged="
              f"{same_edges and same_mult and same_birth}, births kept first={later_only}, inside window={in_window}, events moved={moved}, "
              f"structural scores identical={struct_same}, AUROC NRC={a_nrc:.3f}, naive={a_nv:.3f}")
# (5) the Monte Carlo randomisations (Sections 7.9, 7.12) keep every birth as the first event
import rndnull
from coupling2 import pools_for
nidx0 = C.node_index(ev); rng = np.random.default_rng(0); viol = 0; pre = 0
for m in pool:
    T, P = pools_for(m, ev, nidx0)
    for Tj, Pj in zip(T, P):
        pre += int((Pj < Tj[0]).any())
        x = rndnull.local_draw(Tj, Pj, rng); viol += int(x.min() < Tj[0] or len(x) != len(Tj))
        Pa = Pj[Pj > Tj[0]]
        if len(Pa) and len(Tj) > 1:
            g = np.sort(np.r_[Tj[0], rng.choice(Pa, len(Tj) - 1, replace=len(Pa) < len(Tj) - 1)]); viol += int(g.min() < Tj[0])
ok = viol == 0; fails += not ok
print(f"{'PASS' if ok else 'FAIL'} randomisation keeps births first: {pre} of {3 * len(pool)} edges have pool events before their birth, violations = {viol}")
print("ALL CHECKS PASSED" if not fails else f"{fails} CHECK(S) FAILED"); sys.exit(1 if fails else 0)
