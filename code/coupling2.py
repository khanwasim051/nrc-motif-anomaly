"""NRC v2: adaptive node-rhythm null + periodicity channel.
Channel A (coincidence): for each event t of edge j, null prob. of being within delta of T_i is estimated
from the k nearest-in-time out-of-motif events of j's endpoints (adaptive window, Beta-smoothed to the
endpoint-global rate). Stouffer z over events, ordered pairs; summed over scales.
Channel B (periodicity): cross-edge phase alignment. For period P, R_m(P) = |mean exp(2*pi*i*(t_a - t_b)/P)|
over cross-edge event pairs of the motif; R_p(P) = same with each motif edge's events replaced by the same
number of its endpoints' out-of-motif events (node-rhythm null, e.g. daily cycles). Score = max over the
period grid of sqrt(n_pairs) * (R_m - R_p).
Channels are combined after robust standardisation across motifs: S = z(A) + z(B)."""
import numpy as np, os, sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from coupling import _near, SCALES

def pools_for(motif, ev, nidx):
    a, b, c = motif; nodes = [a, b, c]; E3 = [(a, b), (a, c), (b, c)]
    T = [ev[e] for e in E3]
    P = [np.sort(np.concatenate([nidx[x][0][~np.isin(nidx[x][1], nodes)] for x in e])) for e in E3]
    return T, P

def chan_coincidence(motifs, ev, nidx, k=50, prior=5.0, scales=SCALES):
    out = np.zeros(len(motifs))
    for m, mot in enumerate(motifs):
        T, P = pools_for(mot, ev, nidx); tot = 0.0
        for d in scales:
            obs = exp = var = 0.0
            for i in range(3):
                for j in range(3):
                    if i == j: continue
                    tj, Pj = T[j], P[j]
                    hit = _near(tj, T[i], d)
                    if len(Pj) == 0:
                        p = np.zeros(len(tj))
                    else:
                        pn = _near(Pj, T[i], d).astype(float); cs = np.r_[0, np.cumsum(pn)]; g = pn.mean()
                        pos = np.searchsorted(Pj, tj); kk = min(k, len(Pj))
                        lo = np.clip(pos - kk // 2, 0, len(Pj) - kk); hi = lo + kk      # k nearest-in-time (centred)
                        p = (cs[hi] - cs[lo] + prior * g) / (kk + prior)
                    obs += hit.sum(); exp += p.sum(); var += (p * (1 - p)).sum()
            tot += (obs - exp) / np.sqrt(var + 1.0)
        out[m] = tot
    return out

PERIODS = np.exp(np.linspace(np.log(0.4 * 86400), np.log(2.5 * 86400), 48))

def _phase_R(trains, periods, cap=150, rng=None):
    tr = [t[:cap] if len(t) <= cap else rng.choice(t, cap, replace=False) for t in trains]
    diffs = []
    for i in range(3):
        for j in range(i + 1, 3):
            if len(tr[i]) and len(tr[j]): diffs.append((tr[i][:, None] - tr[j][None, :]).ravel())
    if not diffs: return np.zeros(len(periods)), 0
    d = np.abs(np.concatenate(diffs))
    ph = 2 * np.pi * d[None, :] / periods[:, None]
    w = (d[None, :] > 0.33 * periods[:, None]).astype(float)         # ignore short lags (bursts), keep only
    nw = w.sum(1)                                                     # pairs at least ~1/3 period apart
    R = np.abs((w * np.exp(1j * ph)).sum(1)) / np.maximum(nw, 1)
    return R * np.sqrt(np.maximum(nw, 1)), int(np.median(nw))

def chan_periodic(motifs, ev, nidx, n_null=3, rng=None):
    rng = rng or np.random.default_rng(0); out = np.zeros(len(motifs))
    for m, mot in enumerate(motifs):
        T, P = pools_for(mot, ev, nidx)
        Tl = [t[1:] for t in T]                                      # births excluded (kept organic by design)
        Rm, n = _phase_R(Tl, PERIODS, rng=rng)
        if n == 0: continue
        Rn = np.zeros(len(PERIODS))
        for _ in range(n_null):
            surr = [rng.choice(Pj, len(t), replace=len(Pj) < len(t)) if len(Pj) else t for t, Pj in zip(Tl, P)]
            Rn += _phase_R(surr, PERIODS, rng=rng)[0] / n_null
        out[m] = np.max(Rm - Rn)
    return out

def rz(v):
    med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med)) + 1e-9
    return (v - med) / mad

def nrc2(motifs, ev, nidx, rng=None):
    A = chan_coincidence(motifs, ev, nidx); B = chan_periodic(motifs, ev, nidx, rng=rng)
    return {"A coincidence": A, "B periodic": B, "NRC2 = z(A)+z(B)": rz(A) + rz(B)}
