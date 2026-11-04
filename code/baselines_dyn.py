"""All detectors on identical camouflaged instances (same seeds/pools/injection as leakcheck.py).
New baselines: MIDAS and MIDAS-R (Bhatia et al., AAAI 2020; exact counts instead of count-min sketches,
which is the zero-collision limit of the original), a burstiness heuristic, and IsolationForest on
motif-level temporal features. Also recomputes naive synchrony and NRC so every row shares instances."""
import numpy as np, os, sys, json, time
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
from collections import defaultdict
from sklearn.metrics import roc_auc_score as auc, average_precision_score as ap
from sklearn.ensemble import IsolationForest
import coupling as C, coupling2 as C2
from motif_bench import load, build, triangles
from camouflage import edge_events
from coupling import _near, SCALES

TICK = 3600.0  # MIDAS time tick (1 hour)

def midas_scores(ev2, alpha=0.5):
    """Per-event MIDAS and MIDAS-R scores. Returns dict edge -> (midas array, midasr array) aligned with ev2[edge]."""
    rows = sorted(((t, u, v) for (u, v), ts in ev2.items() for t in ts))
    t0 = rows[0][0]
    cur_e = defaultdict(float); tot_e = defaultdict(float)          # MIDAS (edge)
    rc = {}; rt = defaultdict(float)                                # MIDAS-R decayed counts, lazy decay
    last_tick = 1
    out_m = defaultdict(list); out_r = defaultdict(list)
    def chi(a, s, t):
        return 0.0 if (s == 0 or t <= 1) else (a - s / t) ** 2 * t * t / (s * (t - 1))
    def bump(k, tick):
        v, lt = rc.get(k, (0.0, tick)); v = v * alpha ** (tick - lt) + 1; rc[k] = (v, tick); rt[k] += 1; return v
    for t, u, v in rows:
        tick = int((t - t0) // TICK) + 1
        if tick > last_tick:
            cur_e.clear(); last_tick = tick
        e = (u, v)
        cur_e[e] += 1; tot_e[e] += 1
        a_e = bump(("e", e), tick); a_u = bump(("n", u), tick); a_v = bump(("n", v), tick)
        out_m[e].append(chi(cur_e[e], tot_e[e], tick))
        out_r[e].append(max(chi(a_e, rt[("e", e)], tick), chi(a_u, rt[("n", u)], tick), chi(a_v, rt[("n", v)], tick)))
    return out_m, out_r

def motif_mean(pool, per_edge):
    """Mean per-event score over each motif's later events (births excluded, as they are untouched)."""
    s = []
    for a, b, c in pool:
        vals = []
        for e in [(a, b), (a, c), (b, c)]: vals.extend(per_edge[e][1:])
        s.append(np.mean(np.log1p(vals)) if vals else 0.0)
    return np.array(s)

def burstiness(pool, ev2):
    """Goh-Barabasi burstiness B = (sd - mean)/(sd + mean) of the motif's merged inter-event times."""
    s = []
    for a, b, c in pool:
        t = np.sort(np.concatenate([ev2[e][1:] for e in [(a, b), (a, c), (b, c)]]))
        d = np.diff(t)
        s.append((d.std() - d.mean()) / (d.std() + d.mean() + 1e-9) if len(d) > 1 else 0.0)
    return np.array(s)

def motif_features(pool, ev2, nidx):
    F = []
    for a, b, c in pool:
        E3 = [(a, b), (a, c), (b, c)]; T = [ev2[e] for e in E3]
        cnt = sorted(len(x) for x in T)
        iet = [np.diff(x) for x in T]
        med = sorted(np.median(d) if len(d) else 0 for d in iet)
        cv = sorted((d.std() / (d.mean() + 1e-9)) if len(d) > 1 else 0 for d in iet)
        allt = np.sort(np.concatenate([x[1:] for x in T])); n = sum(len(x) for x in T)
        sync = [sum(_near(T[j], T[i], d).sum() for i in range(3) for j in range(3) if i != j) / n for d in SCALES]
        nodes = [a, b, c]; act = []
        for (u, v), tt in zip(E3, T):
            P = np.sort(np.concatenate([nidx[x][0][~np.isin(nidx[x][1], nodes)] for x in (u, v)]))
            act.extend(np.searchsorted(P, tt[1:] + 86400, side="right") - np.searchsorted(P, tt[1:] - 86400))
        span = (allt[-1] - allt[0]) if len(allt) > 1 else 0
        F.append(list(np.log1p(cnt)) + list(np.log1p(med)) + cv + sync + [np.log1p(np.mean(act)) if act else 0, np.log1p(span)])
    return np.array(F, float)

if __name__ == "__main__":
    name = sys.argv[1]; seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(BASE, "baselines_dyn.jsonl")
    E = load(name); mult, birth, adj = build(E); tri = triangles(adj)
    ts = np.array([e[2] for e in E]); tmin, tmax = ts.min(), ts.max()
    pa = [t for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3]
    ev = edge_events(E)
    VS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["anchored", "joint_bursts", "periodic"]
    RH = [float(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0.1, 0.25]
    for v in VS:
        for rho in RH:
            R = defaultdict(list); T_ = defaultdict(list)
            for s in range(seeds):
                rng = np.random.default_rng(100 + s)
                pool = [pa[i] for i in rng.choice(len(pa), min(3000, len(pa)), replace=False)]
                ev2, y = C.inject(pool, ev, v, rho, rng, tmin=tmin, tmax=tmax); nidx = C.node_index(ev2)
                sc = {}
                t0 = time.time(); n1, n2 = C.naive_rate(pool, ev2, 600), C.naive_rate(pool, ev2, 3600); T_["naive"].append(time.time() - t0)
                sc["naive 10m"] = n1; sc["naive 1h"] = n2
                t0 = time.time(); sc["NRC"] = C2.chan_coincidence(pool, ev2, nidx); T_["NRC"].append(time.time() - t0)
                t0 = time.time(); m, r = midas_scores(ev2); sc["MIDAS"] = motif_mean(pool, m); sc["MIDAS-R"] = motif_mean(pool, r); T_["MIDAS"].append(time.time() - t0)
                sc["burstiness"] = burstiness(pool, ev2)
                t0 = time.time(); F = motif_features(pool, ev2, nidx)
                sc["IForest(features)"] = -IsolationForest(n_estimators=200, random_state=s).fit(F).score_samples(F); T_["IForest"].append(time.time() - t0)
                for k, x in sc.items():
                    R[k + " AUROC"].append(auc(y, x)); R[k + " AP"].append(ap(y, x))
                if len(sys.argv) > 6:
                    open(sys.argv[6], "a").write(json.dumps(dict(data=name, variant=v, rho=rho, seed=s, **{k: float(x[-1]) for k, x in R.items()})) + "\n")
            rec = dict(data=name, variant=v, rho=rho, **{k: round(float(np.mean(x)), 4) for k, x in R.items()},
                       **{"sec " + k: round(float(np.mean(x)), 1) for k, x in T_.items()})
            open(out, "a").write(json.dumps(rec) + "\n"); print(json.dumps(rec), flush=True)
