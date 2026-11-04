"""Activity-pool randomisation baseline detector (Monte Carlo null instead of NRC's closed form).
For each motif: B randomisations redraw each edge's later events from the part of its activity pool A_j after its birth (birth stays the first event);
per scale, z = (C_obs - mean_B) / sqrt(var_B + 1) of the total cross-edge coincidence count; score = sum over scales."""
import numpy as np, os, sys, json, time
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
from sklearn.metrics import roc_auc_score as auc, average_precision_score as ap
import coupling as C, coupling2 as C2
from coupling2 import pools_for
from motif_bench import load, build, triangles
from camouflage import edge_events
from coupling import SCALES
SC = np.array(SCALES); B = int(os.environ.get("RND_B", "30"))
PAIRS = [(i, j) for i in range(3) for j in range(3) if i != j]
def counts(T):
    c = np.zeros(len(SC))
    for i, j in PAIRS:
        ti, tj = T[i], T[j]
        k = np.searchsorted(ti, tj); lo = np.abs(tj - ti[np.clip(k - 1, 0, len(ti) - 1)]); hi = np.abs(ti[np.clip(k, 0, len(ti) - 1)] - tj)
        d = np.minimum(lo, hi); c += (d[:, None] <= SC[None, :]).sum(0)
    return c
K_LOCAL = 50
def local_draw(Tj, Pj, rng, k=K_LOCAL):
    """Replace each later event t of T_j by a draw from the same local neighbourhood NRC uses: the window of
    k_j = min(k, |A_j+|) consecutive events of the post-birth pool A_j+ centred on t's insertion position."""
    t = Tj[1:]; Pa = Pj[Pj > Tj[0]]                 # only pool events after the birth, so b_e stays the first event
    if len(Pa) == 0 or len(t) == 0: return np.sort(Tj)
    kk = min(k, len(Pa)); pos = np.searchsorted(Pa, t)
    lo = np.clip(pos - kk // 2, 0, len(Pa) - kk); idx = lo + rng.integers(0, kk, len(t))
    return np.sort(np.r_[Tj[0], Pa[idx]])

def rnd_score(pool, ev2, nidx, rng, local=False):
    out = np.zeros(len(pool))
    for m, mot in enumerate(pool):
        T, P = pools_for(mot, ev2, nidx); obs = counts(T); sims = np.zeros((B, len(SC)))
        for b in range(B):
            Tb = []
            for Tj, Pj in zip(T, P):
                n = len(Tj) - 1
                if not len(Pj): Tb.append(np.sort(Tj)); continue
                if local: Tb.append(local_draw(Tj, Pj, rng)); continue
                Pa = Pj[Pj > Tj[0]]                 # only pool events after the birth
                if not len(Pa) or n == 0: Tb.append(np.sort(Tj)); continue
                new = rng.choice(Pa, n, replace=len(Pa) < n)
                Tb.append(np.sort(np.r_[Tj[0], new]))
            sims[b] = counts(Tb)
        out[m] = np.sum((obs - sims.mean(0)) / np.sqrt(sims.var(0) + 1.0))
    return out
if __name__ == "__main__":
    name, out = sys.argv[1], sys.argv[2]; LOCAL = len(sys.argv) > 3 and sys.argv[3] == "local"
    E = load(name); mult, birth, adj = build(E); tri = triangles(adj)
    ts = np.array([e[2] for e in E]); tmin, tmax = ts.min(), ts.max()
    pa = [t for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3]
    ev = edge_events(E)
    for v in ["anchored", "joint_bursts", "periodic"]:
        for rho in [0.1, 0.25]:
            for s in range(3):
                rng = np.random.default_rng(100 + s)
                pool = [pa[i] for i in rng.choice(len(pa), min(3000, len(pa)), replace=False)]
                ev2, y = C.inject(pool, ev, v, rho, rng, tmin=tmin, tmax=tmax); nidx = C.node_index(ev2)
                t0 = time.time(); R = rnd_score(pool, ev2, nidx, np.random.default_rng(900 + s), local=LOCAL); sec = time.time() - t0
                rec = dict(data=name, variant=v, rho=rho, seed=s, null="local" if LOCAL else "global", rnd=auc(y, R), rnd_ap=ap(y, R), sec=sec)
                open(out, "a").write(json.dumps(rec) + "\n"); print(json.dumps(rec), flush=True)
