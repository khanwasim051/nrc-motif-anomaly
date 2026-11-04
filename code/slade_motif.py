"""Run SLADE on the (standard or camouflaged) motif benchmark and evaluate at motif level.
Motif score = mean SLADE event score over the motif's later events (births excluded, as they are untouched)."""
import numpy as np, os, sys, os, subprocess, json, time
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from sklearn.metrics import roc_auc_score as auc, average_precision_score as ap
import coupling as C
from motif_bench import load, build, triangles
from camouflage import edge_events
SL = os.environ.get("SLADE_DIR", os.path.join(BASE, "SLADE"))
name, variant, rho, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
epochs = int(os.environ.get("SLADE_EPOCHS", "10"))
E = load(name); mult, birth, adj = build(E); tri = triangles(adj)
ts = np.array([e[2] for e in E]); tmin, tmax = ts.min(), ts.max()
pa = [t for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3]
ev = edge_events(E); rng = np.random.default_rng(100 + seed)
pool = [pa[i] for i in rng.choice(len(pa), min(3000, len(pa)), replace=False)]
ev2, y = C.inject(pool, ev, variant, rho, rng, tmin=tmin, tmax=tmax)
# event stream (same seed/pool as eval3.py, so results are directly comparable)
rows = [(u, v, t) for (u, v), tt in ev2.items() for t in tt]
rows.sort(key=lambda r: r[2])
nodes = sorted({x for r in rows for x in r[:2]}); nid = {n: i + 1 for i, n in enumerate(nodes)}
anom_edges = {e for k in np.where(y == 1)[0] for e in [(pool[k][0], pool[k][1]), (pool[k][0], pool[k][2]), (pool[k][1], pool[k][2])]}
tag = f"{name}_{variant}_{rho}_{seed}"
with open(f"{SL}/data/ml_{tag}.csv", "w") as f:
    f.write(",u,i,ts,label,idx\n")
    t0 = rows[0][2]
    for k, (u, v, t) in enumerate(rows):
        f.write(f"{k},{nid[u]},{nid[v]},{t - t0},{1.0 if (u, v) in anom_edges else 0.0},{k + 1}\n")
out = os.path.join(BASE, "scores", f"slade_scores_{tag}.npy"); os.makedirs(os.path.dirname(out), exist_ok=True)
st = time.time()
r = subprocess.run([sys.executable, "SLADE_main.py", "-d", tag, "--n_epoch", str(epochs)], cwd=SL,
                   env=dict(os.environ, SLADE_DUMP=out), capture_output=True, text=True)
if r.returncode != 0: print(r.stderr[-2000:]); sys.exit(1)
s = np.load(out)
# per-edge mean score over later events
from collections import defaultdict
acc = defaultdict(list); first = set()
for k, (u, v, t) in enumerate(rows):
    if (u, v) in first: acc[(u, v)].append(s[k])
    else: first.add((u, v))
ms = np.array([np.mean(sum([acc[e] for e in [(a, b), (a, c), (b, c)]], [])) for a, b, c in pool])
print(json.dumps(dict(data=name, variant=variant, rho=rho, seed=seed, slade_auroc=float(auc(y, ms)), slade_ap=float(ap(y, ms)), minutes=round((time.time() - st) / 60, 1))), flush=True)
