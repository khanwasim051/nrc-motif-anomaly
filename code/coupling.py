"""Node-rhythm conditional coupling (NRC) score for coordinated motif anomalies — training-free prototype.

For a triangle motif with edge event trains T1,T2,T3, every ordered edge pair (i,j) shares one node.
Organic coincidences between T_j and T_i largely come from the endpoints' own activity rhythms
(sessions, daily cycles). Null for edge j: its n_j events are placed like a random draw from the activity
pool P_j = events of j's endpoints on edges OUTSIDE the motif. Then for scale delta
    f_j(delta) = fraction of P_j within delta of T_i,  E[C_ij] = n_j f,  Var = n_j f (1 - f)
and z_ij(delta) = (C_ij - n_j f) / sqrt(n_j f (1 - f) + 1). Motif score = max over scales of the
Stouffer combination over the 6 ordered pairs. Closed form, no resampling, no training."""
import numpy as np, os, sys
BASE = os.path.dirname(os.path.abspath(__file__))
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, BASE)
from motif_bench import load, build, triangles
from camouflage import edge_events, coincidences

SCALES = [60.0, 300.0, 900.0, 3600.0, 4 * 3600.0]

def node_index(ev):
    """node -> (sorted times, partner ids) over all its events."""
    t = defaultdict(list); p = defaultdict(list)
    for (u, v), ts in ev.items():
        t[u].extend(ts); p[u].extend([v] * len(ts)); t[v].extend(ts); p[v].extend([u] * len(ts))
    out = {}
    for k in t:
        tt = np.array(t[k]); pp = np.array(p[k]); o = np.argsort(tt); out[k] = (tt[o], pp[o])
    return out

def near_frac(pool, ref, delta):
    if len(pool) == 0 or len(ref) == 0: return 0.0
    i = np.searchsorted(ref, pool)
    lo = np.abs(pool - ref[np.clip(i - 1, 0, len(ref) - 1)]); hi = np.abs(ref[np.clip(i, 0, len(ref) - 1)] - pool)
    return float((np.minimum(lo, hi) <= delta).mean())

def nrc_score(motifs, ev, nidx, scales=SCALES, max_pool=20000, rng=None):
    rng = rng or np.random.default_rng(0)
    out = np.zeros(len(motifs))
    for k, (a, b, c) in enumerate(motifs):
        nodes = {a, b, c}; E3 = [(a, b), (a, c), (b, c)]; T = [ev[e] for e in E3]
        pools = []
        for (u, v) in E3:
            ps = []
            for x in (u, v):
                tt, pp = nidx[x]; ps.append(tt[~np.isin(pp, list(nodes))])
            P = np.concatenate(ps)
            if len(P) > max_pool: P = rng.choice(P, max_pool, replace=False)
            pools.append(P)
        best = -np.inf
        for d in scales:
            zs = []
            for i in range(3):
                for j in range(3):
                    if i == j: continue
                    n = len(T[j]); C = coincidences(T[j], T[i], d); f = near_frac(pools[j], T[i], d)
                    zs.append((C - n * f) / np.sqrt(n * f * (1 - f) + 1.0))
            best = max(best, np.sum(zs) / np.sqrt(len(zs)))
        out[k] = best
    return out

def _near(x, ref, delta):
    if len(ref) == 0: return np.zeros(len(x), bool)
    i = np.searchsorted(ref, x)
    lo = np.abs(x - ref[np.clip(i - 1, 0, len(ref) - 1)]); hi = np.abs(ref[np.clip(i, 0, len(ref) - 1)] - x)
    return np.minimum(lo, hi) <= delta

def nrc_local(motifs, ev, nidx, scales=SCALES, W=86400.0, prior=1.0):
    """Time-local version: for each event t of edge j, the null probability that it falls within delta of T_i
    is the fraction of the endpoints' out-of-motif events in [t-W, t+W] that fall within delta of T_i
    (Beta-smoothed). Score = sum over scales of the Stouffer z over all events and ordered pairs."""
    out = np.zeros(len(motifs))
    for k, (a, b, c) in enumerate(motifs):
        nodes = list({a, b, c}); E3 = [(a, b), (a, c), (b, c)]; T = [ev[e] for e in E3]
        pools = []
        for (u, v) in E3:
            P = np.sort(np.concatenate([nidx[x][0][~np.isin(nidx[x][1], nodes)] for x in (u, v)])); pools.append(P)
        tot = 0.0
        for d in scales:
            obs = 0.0; exp = 0.0; var = 0.0
            for i in range(3):
                for j in range(3):
                    if i == j: continue
                    tj = T[j]; P = pools[j]
                    hit = _near(tj, T[i], d)
                    if len(P):
                        pn = _near(P, T[i], d).astype(float); cs = np.r_[0, np.cumsum(pn)]
                        lo = np.searchsorted(P, tj - W); hi = np.searchsorted(P, tj + W, side="right")
                        base = (_near(P, T[i], d).mean() if len(P) else 0.0)
                        p = (cs[hi] - cs[lo] + prior * base) / (hi - lo + prior)
                    else:
                        p = np.full(len(tj), 0.0)
                    obs += hit.sum(); exp += p.sum(); var += (p * (1 - p)).sum()
            tot += (obs - exp) / np.sqrt(var + 1.0)
        out[k] = tot
    return out

def nrc_sum(motifs, ev, nidx, scales=SCALES):
    """Global-null version with scales summed (not maxed)."""
    return np.sum(np.stack([nrc_score(motifs, ev, nidx, scales=[d]) for d in scales]), 0)

# ------------------------------------------------------------- anomaly variants (all keep structure,
# per-edge event counts and per-edge birth times; only later events are re-timed)

GEN = dict(selected=0, placed=0, infeasible=0, redraws=0)
RECORD = None   # if set to a list, _retime appends (edge, new times placed) for audits

def _feasible(c, b, tmax):
    """A new time must lie strictly after the edge's birth b and inside the observation window (<= t_max)."""
    return (c > b) & (c <= tmax)

def _retime(ev, pairs, new_times_fn, rho, rng, tmax=np.inf, tries=100):
    """Feasible-time sampling: every selected later event receives a proposal from new_times_fn; proposals at or before
    the birth or after t_max are redrawn (up to `tries` proposals per event), so the accepted time follows the coordination
    distribution conditioned on the feasible range. An event without a feasible proposal keeps its original time."""
    for p in pairs:
        t = ev[p].copy(); later = np.arange(1, len(t)); pick = later[rng.random(len(later)) < rho]
        if not len(pick): continue
        GEN["selected"] += len(pick)
        new = new_times_fn(len(pick)); ok = _feasible(new, t[0], tmax); k = 1
        while not ok.all() and k < tries:
            bad = np.flatnonzero(~ok); GEN["redraws"] += len(bad)
            new[bad] = new_times_fn(len(bad)); ok[bad] = _feasible(new[bad], t[0], tmax); k += 1
        GEN["placed"] += int(ok.sum()); GEN["infeasible"] += int((~ok).sum())
        t[pick[ok]] = new[ok]; ev[p] = np.sort(t)
        if RECORD is not None: RECORD.append((p, new[ok].copy()))

STRICT = dict(proposed=0, accepted=0, infeasible=0)

def _retime_strict(ev, pairs, new_times_fn, rho, rng, motif, nidx, w, tries=100, tmax=np.inf):
    """Endpoint-wise activity constraint: a new time t' on edge (u, v) is accepted only if both u and v have an
    event on an edge outside the motif within w of t'. Up to `tries` proposals per event; if none is accepted the
    event keeps its original time (counted as infeasible)."""
    ext = {x: np.sort(nidx[x][0][~np.isin(nidx[x][1], list(motif))]) for x in motif}
    def near(A, t):
        if len(A) == 0: return False
        i = np.searchsorted(A, t); return min(abs(t - A[max(i - 1, 0)]), abs(A[min(i, len(A) - 1)] - t)) <= w
    for p in pairs:
        t = ev[p].copy(); later = np.arange(1, len(t)); pick = later[rng.random(len(later)) < rho]
        if not len(pick): continue
        for k in pick:
            STRICT["proposed"] += 1; ok = False
            for _ in range(tries):
                c = float(new_times_fn(1)[0])
                if not _feasible(c, t[0], tmax): continue           # before the birth or after t_max: redraw
                if near(ext[p[0]], c) and near(ext[p[1]], c): ok = True; break
            if ok: t[k] = c; STRICT["accepted"] += 1
            else: STRICT["infeasible"] += 1
        ev[p] = np.sort(t)

def _active_times(nidx, motif):
    """Times of the motif nodes' activity on edges outside the motif (sorted)."""
    ts = [nidx[x][0][~np.isin(nidx[x][1], list(motif))] for x in motif if x in nidx]
    return np.sort(np.concatenate(ts)) if ts else np.zeros(0)

def inject(pool, ev, variant, rho, rng, frac=0.05, delta=600.0, tmin=0.0, tmax=1.0, endpoint_w=None):
    idx = rng.choice(len(pool), max(1, int(frac * len(pool))), replace=False)
    ev_nodes = node_index(ev) if (variant in ("joint_bursts", "periodic", "periodic_strict") or endpoint_w) else None
    ev2 = dict(ev)
    for k in idx:
        a, b, c = pool[k]; E3 = [(a, b), (a, c), (b, c)]
        def RT(pairs, fn, rho_, rng_, motif=(a, b, c)):
            if endpoint_w: _retime_strict(ev2, pairs, fn, rho_, rng_, motif, ev_nodes, endpoint_w, tmax=tmax)
            else: _retime(ev2, pairs, fn, rho_, rng_, tmax=tmax)
        if variant == "anchored":                      # V1: edges 2,3 re-timed near edge-1 events
            anc = ev[E3[0]]
            RT(E3[1:], lambda m: rng.choice(anc, m) + rng.uniform(-delta, delta, m), rho, rng)
        elif variant == "joint_bursts":                # V2: all three edges pulled into new shared bursts
            # leak-free placement: burst centres are drawn from times when the motif's own nodes are active
            act = _active_times(ev_nodes, (a, b, c))
            centres = rng.choice(act, 5) if len(act) else rng.uniform(tmin, tmax, 5)
            RT(E3, lambda m: rng.choice(centres, m) + rng.uniform(-delta, delta, m), rho, rng)
        elif variant == "periodic_strict":             # V3 without fallback: redraw (P, phase) until the lattice meets the activity constraint
            act = _active_times(ev_nodes, (a, b, c)); lat = None
            for _ in range(200):
                per = 86400.0 * rng.uniform(0.5, 2.0); ph = rng.uniform(0, per)
                L = tmin + ph + per * np.arange(int((tmax - tmin) / per) + 1)
                near = np.abs(act[np.clip(np.searchsorted(act, L), 0, len(act) - 1)] - L) <= 3600 if len(act) else np.zeros(len(L), bool)
                if near.any(): lat = L[near]; break
            if lat is None: lat = act[rng.choice(len(act), 1)] if len(act) else np.array([rng.uniform(tmin, tmax)])
            def f(m, lat=lat):
                return rng.choice(lat, m) + rng.normal(0, delta / 3, m)
            RT(E3, f, rho, rng)
        elif variant == "periodic":                    # V3: bot-like common schedule (period ~1 day, phase shared)
            per = 86400.0 * rng.uniform(0.5, 2.0); ph = rng.uniform(0, per)
            act = _active_times(ev_nodes, (a, b, c))
            lat = tmin + ph + per * np.arange(int((tmax - tmin) / per) + 1)
            if len(act):                               # leak-free: keep lattice points inside active periods
                near = np.abs(act[np.clip(np.searchsorted(act, lat), 0, len(act) - 1)] - lat) <= 3600
                lat = lat[near] if near.any() else lat
            def f(m, lat=lat):
                return rng.choice(lat, m) + rng.normal(0, delta / 3, m)
            RT(E3, f, rho, rng)
    y = np.zeros(len(pool)); y[idx] = 1
    return ev2, y

def naive_rate(motifs, ev, delta=600.0):
    s = []
    for a, b, c in motifs:
        T = [ev[(a, b)], ev[(a, c)], ev[(b, c)]]
        C = sum(coincidences(T[j], T[i], delta) for i in range(3) for j in range(3) if i != j)
        s.append(C / sum(len(x) for x in T))
    return np.array(s)

def naive_rate_multiscale(motifs, ev):
    return np.max(np.stack([naive_rate(motifs, ev, d) for d in SCALES]) /
                  (np.stack([naive_rate(motifs, ev, d) for d in SCALES]).std(1, keepdims=True) + 1e-9), 0)

if __name__ == "__main__":
    name = sys.argv[1]; seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    E = load(name); mult, birth, adj = build(E); tri = triangles(adj)
    ts = np.array([e[2] for e in E]); tmin, tmax = ts.min(), ts.max()
    pool_all = [t for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3]
    ev = edge_events(E)
    print(f"{name}: eligible triangles {len(pool_all)}", flush=True)
    for variant in ["anchored", "joint_bursts", "periodic"]:
        for rho in [0.05, 0.10, 0.25]:
            R = defaultdict(list)
            for s in range(seeds):
                rng = np.random.default_rng(100 + s)
                pool = pool_all if len(pool_all) <= 3000 else [pool_all[i] for i in rng.choice(len(pool_all), 3000, replace=False)]
                ev2, y = inject(pool, ev, variant, rho, rng, tmin=tmin, tmax=tmax)
                nidx = node_index(ev2)
                sc = {"naive rate (10 min)": naive_rate(pool, ev2, 600.0),
                      "naive rate (1 h)": naive_rate(pool, ev2, 3600.0),
                      "naive multiscale": naive_rate_multiscale(pool, ev2),
                      "NRC (ours)": nrc_score(pool, ev2, nidx, rng=rng)}
                for k, v in sc.items():
                    R[k].append((roc_auc_score(y, v), average_precision_score(y, v)))
            print(f"  {variant:12s} rho={rho:.2f} | " + "  ".join(f"{k}: {np.mean([a for a, _ in v]):.3f}/{np.mean([b for _, b in v]):.3f}" for k, v in R.items()), flush=True)
