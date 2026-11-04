"""Recompute the naive-rate entries of an existing ov_<graph>.jsonl with the per-metric oracle used everywhere else in the
paper (best of the 10-min and 1-h scales chosen separately for AUROC and for AP, and separately on the cleaned labels).
Replays exactly the instances of overlap.py (same RNG use) without the expensive detectors, checks that the stored
AUROC-selected values are reproduced bit-for-bit, and writes a corrected file.
usage: python naive_refresh.py GRAPH OV_IN OV_OUT"""
import numpy as np, os, sys, json
sys.argv_saved = list(sys.argv); name, fin, fout = sys.argv[1:4]
from sklearn.metrics import roc_auc_score as auc, average_precision_score as ap
import coupling as C
from motif_bench import load, build, triangles
from camouflage import edge_events
from collections import defaultdict
MOVED = defaultdict(int); C.RECORD = []
def _sync():
    for p, new in C.RECORD: MOVED[p] += len(new)
    C.RECORD.clear()
E = load(name); mult, birth, adj = build(E); tri = triangles(adj)
ts = np.array([e[2] for e in E]); tmin, tmax = ts.min(), ts.max()
pa = [t for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3]
ev = edge_events(E); E3 = lambda m: [(m[0], m[1]), (m[0], m[2]), (m[1], m[2])]
def edge_disjoint(rng, cap=3000):
    used = set(); sel = []
    for i in rng.permutation(len(pa)):
        es = E3(pa[i])
        if any(e in used for e in es): continue
        used.update(es); sel.append(pa[i])
        if len(sel) >= cap: break
    return sel
def contam_mask(pool, y):
    anom = [pool[i] for i in np.where(y == 1)[0]]
    mv = np.array([sum(MOVED[e] for e in E3(m)) for m in anom])
    moved_e = {e for m in anom for e in E3(m) if MOVED[e] > 0}
    contam = np.array([(not y[i]) and any(e in moved_e for e in E3(m)) for i, m in enumerate(pool)])
    return mv, contam
rows = [json.loads(l) for l in open(fin)]; idx = {(r["variant"], r["rho"], r["seed"], r["mode"]): r for r in rows}
maxdiff = 0.0
for v in ["anchored", "joint_bursts", "periodic"]:
    for rho in [0.1, 0.25]:
        for s in range(3):
            for mode in ["standard", "edge_disjoint"]:
                MOVED.clear(); C.RECORD.clear(); rng = np.random.default_rng(100 + s)
                pool = [pa[i] for i in rng.choice(len(pa), min(3000, len(pa)), replace=False)] if mode == "standard" else edge_disjoint(rng)
                ev2, y = C.inject(pool, ev, v, rho, rng, tmin=tmin, tmax=tmax); _sync()
                n1, n2 = C.naive_rate(pool, ev2, 600), C.naive_rate(pool, ev2, 3600)
                r = idx[(v, rho, s, mode)]
                old = n1 if auc(y, n1) >= auc(y, n2) else n2                        # the old AUROC-selected rule
                maxdiff = max(maxdiff, abs(auc(y, old) - r["naive AUROC"]), abs(ap(y, old) - r["naive AP"]))
                r["naive AP (AUROC-selected scale)"] = r["naive AP"]
                r["naive AUROC"] = max(auc(y, n1), auc(y, n2)); r["naive AP"] = max(ap(y, n1), ap(y, n2))
                if mode == "standard":
                    mv, contam = contam_mask(pool, y)
                    keep = np.ones(len(pool), bool); ai = np.where(y == 1)[0]; keep[ai[mv == 0]] = False; keep &= ~contam
                    maxdiff = max(maxdiff, abs(auc(y[keep], old[keep]) - r["naive AUROC clean"]), abs(ap(y[keep], old[keep]) - r["naive AP clean"]))
                    r["naive AUROC clean (standard-selected scale)"] = r["naive AUROC clean"]; r["naive AP clean (AUROC-selected scale)"] = r["naive AP clean"]
                    r["naive AUROC clean"] = max(auc(y[keep], n1[keep]), auc(y[keep], n2[keep]))
                    r["naive AP clean"] = max(ap(y[keep], n1[keep]), ap(y[keep], n2[keep]))
with open(fout, "w") as fh:
    for r in rows: fh.write(json.dumps(r) + "\n")
print(name, "max abs difference to stored AUROC-selected values:", maxdiff)
