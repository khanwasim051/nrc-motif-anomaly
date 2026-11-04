"""Fingerprint of one camouflaged instance (same construction as slade_ckpt.py), used to check that the laptop
builds exactly the same anomalies as the server before training."""
import numpy as np, hashlib
import coupling as C
from motif_bench import load, build, triangles
from camouflage import edge_events
_cache = {}
def fingerprint(name, variant, rho, seed):
    if name not in _cache:
        E = load(name); mult, birth, adj = build(E); tri = triangles(adj)
        ts = np.array([e[2] for e in E])
        pa = [t for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3]
        _cache[name] = (pa, edge_events(E), ts.min(), ts.max())
    pa, ev, tmin, tmax = _cache[name]
    rng = np.random.default_rng(100 + seed)
    pool = [pa[i] for i in rng.choice(len(pa), min(3000, len(pa)), replace=False)]
    for k in C.GEN: C.GEN[k] = 0
    ev2, y = C.inject(pool, ev, variant, rho, rng, tmin=tmin, tmax=tmax)
    h = hashlib.sha256()
    for k in np.where(y == 1)[0]:
        m = pool[k]
        for e in [(m[0], m[1]), (m[0], m[2]), (m[1], m[2])]: h.update(np.round(ev2[e], 3).tobytes())
    return dict(anomalies=int(y.sum()), placed=C.GEN["placed"], infeasible=C.GEN["infeasible"], hash=h.hexdigest()[:16])
