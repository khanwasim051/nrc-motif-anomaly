"""Reproduce the standard motif-anomaly injection protocol (clique injection, as introduced for motif-level detection by MADG)
and test how far trivial, training-free heuristics get on it."""
import numpy as np, os, itertools, sys
BASE = os.path.dirname(os.path.abspath(__file__))
from collections import defaultdict
from sklearn.metrics import roc_auc_score

RAW = os.path.join(BASE, "data", "raw")

def load(name):
    E = []
    if name == "uci":
        for l in open(f"{RAW}/uci"):
            if l.startswith("%"): continue
            u, v, _, t = l.split(); E.append((int(u), int(v), float(t)))
    elif name == "radoslaw":                       # KONECT radoslaw_email: u v weight t (manufacturing company e-mail)
        for l in open(os.path.join(BASE, "data", "other", "radoslaw.txt")):
            if l.startswith("%"): continue
            u, v, _, t = l.split(); E.append((int(u), int(v), float(t)))
    elif name == "thiers13":                       # SocioPatterns high school 2013: t i j, 20 s resolution
        for l in open(os.path.join(BASE, "data", "other", "thiers13.txt")):
            t, u, v = l.split(); E.append((int(u), int(v), float(t)))
    elif name in ("email_eu", "mathoverflow", "collegemsg"):
        f = {"email_eu": "email-Eu-core-temporal.txt", "mathoverflow": "sx-mathoverflow.txt", "collegemsg": "CollegeMsg.txt"}[name]
        for l in open(os.path.join(BASE, "data", "snap", f)):
            u, v, t = l.split(); E.append((int(u), int(v), float(t)))
    else:
        f = {"btc_alpha": "soc-sign-bitcoinalpha.csv", "btc_otc": "soc-sign-bitcoinotc.csv"}[name]
        for l in open(f"{RAW}/{f}"):
            u, v, _, t = l.strip().split(","); E.append((int(u), int(v), float(t)))
    E = [(min(u, v), max(u, v), t) for u, v, t in E if u != v]
    E.sort(key=lambda e: e[2])
    return E

def build(E):
    mult = defaultdict(int); birth = {}; adj = defaultdict(set)
    for u, v, t in E:
        mult[(u, v)] += 1
        birth.setdefault((u, v), t)
        adj[u].add(v); adj[v].add(u)
    return mult, birth, adj

def triangles(adj):
    tri = []
    for u in adj:
        for v in adj[u]:
            if v <= u: continue
            for w in adj[u] & adj[v]:
                if w > v: tri.append((u, v, w))
    return tri

def inject(E, tri, ratio, rng):
    """Standard protocol: random node triplets that never co-occur in any triangle, joined into a clique
    at a random time stamp."""
    nodes = np.array(sorted({x for e in E for x in e[:2]}))
    in_tri = defaultdict(set)
    for t in tri:
        for a, b in itertools.combinations(t, 2): in_tri[a].add(b); in_tri[b].add(a)
    times = np.array([e[2] for e in E])
    k = max(1, int(ratio * len(tri))); out = []; seen = set()
    while len(out) < k:
        a, b, c = sorted(rng.choice(nodes, 3, replace=False))
        if (a, b, c) in seen or b in in_tri[a] or c in in_tri[a] or c in in_tri[b]: continue
        seen.add((a, b, c)); ts = float(rng.choice(times))
        out.append(((a, b, c), [(a, b, ts), (a, c, ts), (b, c, ts)]))
    return out

def heuristics(motifs, mult, birth, adj):
    """Training-free scores; higher = more anomalous."""
    S = defaultdict(list)
    for (a, b, c) in motifs:
        pairs = [(a, b), (a, c), (b, c)]
        m = [mult[p] for p in pairs]
        bt = [birth[p] for p in pairs]
        cn = [len(adj[x] & adj[y]) - 1 for x, y in pairs]       # common neighbours outside the motif
        S["low edge multiplicity"].append(-np.log1p(m).sum())
        S["synchronised edge birth"].append(-np.std(bt))
        S["few common neighbours"].append(-np.sum(cn))
        S["low degree"].append(-sum(len(adj[x]) for x in (a, b, c)))
    return {k: np.array(v) for k, v in S.items()}

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for name in ["uci", "btc_alpha", "btc_otc"]:
        E = load(name)
        _, _, adj0 = build(E); tri = triangles(adj0)
        for ratio in [0.01, 0.05, 0.10]:
            inj = inject(E, tri, ratio, rng)
            E2 = sorted(E + [e for _, es in inj for e in es], key=lambda e: e[2])
            mult, birth, adj = build(E2)
            motifs = tri + [m for m, _ in inj]
            y = np.r_[np.zeros(len(tri)), np.ones(len(inj))]
            H = heuristics(motifs, mult, birth, adj)
            print(f"{name:9s} {int(ratio*100):2d}%  nodes {len(adj0):5d} triangles {len(tri):6d} injected {len(inj):5d} | " +
                  "  ".join(f"{k}: {roc_auc_score(y, v):.3f}" for k, v in H.items()), flush=True)
