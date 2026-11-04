"""SLADE with the authors' default configuration (10 epochs, lr 3e-6, batch 100) and score checkpoints.
Motif-level aggregation rules are compared on the same event scores."""
import numpy as np, os, sys, subprocess, json, time
BASE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BASE)
from sklearn.metrics import roc_auc_score as auc, average_precision_score as ap
from collections import defaultdict
import coupling as C
from motif_bench import load, build, triangles
from camouflage import edge_events
SL = os.path.join(BASE, "SLADE")
name, variant, rho, seed, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
EPOCHS = int(os.environ.get("SLADE_EPOCHS", "10")); CK = os.environ.get("SLADE_CKPTS", "1,2,3,5,10")
E = load(name); mult, birth, adj = build(E); tri = triangles(adj)
ts = np.array([e[2] for e in E]); tmin, tmax = ts.min(), ts.max()
pa = [t for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3]
ev = edge_events(E); rng = np.random.default_rng(100 + seed)
pool = [pa[i] for i in rng.choice(len(pa), min(3000, len(pa)), replace=False)]
ev2, y = C.inject(pool, ev, variant, rho, rng, tmin=tmin, tmax=tmax)
rows = sorted(((u, v, t) for (u, v), tt in ev2.items() for t in tt), key=lambda r: r[2])
nodes = sorted({x for r in rows for x in r[:2]}); nid = {n: i + 1 for i, n in enumerate(nodes)}
anom_edges = {e for k in np.where(y == 1)[0] for e in [(pool[k][0], pool[k][1]), (pool[k][0], pool[k][2]), (pool[k][1], pool[k][2])]}
tag = f"ck_{name}_{variant}_{rho}_{seed}"
with open(f"{SL}/data/ml_{tag}.csv", "w") as f:
    f.write(",u,i,ts,label,idx\n"); t0 = rows[0][2]
    for k, (u, v, t) in enumerate(rows): f.write(f"{k},{nid[u]},{nid[v]},{t - t0},{1.0 if (u, v) in anom_edges else 0.0},{k + 1}\n")
dump = os.path.join(BASE, "scores_ck", f"{tag}.npy"); os.makedirs(os.path.dirname(dump), exist_ok=True)
for f in [dump[:-4] + "_loss.json"]:
    if os.path.exists(f): os.remove(f)
st = time.time()
r = subprocess.run([sys.executable, "SLADE_main.py", "-d", tag, "--n_epoch", str(EPOCHS)], cwd=SL,
                   env=dict(os.environ, SLADE_DUMP=dump, SLADE_CKPTS=CK), capture_output=True, text=True)
if r.returncode != 0: print(r.stderr[-2000:]); sys.exit(1)
loss = json.load(open(dump[:-4] + "_loss.json"))
E3 = lambda m: [(m[0], m[1]), (m[0], m[2]), (m[1], m[2])]
for ep in [int(x) for x in CK.split(",")]:
    s = np.load(dump[:-4] + f"_ep{ep}.npy")
    acc = defaultdict(list); seen = set()
    for k, (u, v, t) in enumerate(rows):
        if (u, v) in seen: acc[(u, v)].append(s[k])
        else: seen.add((u, v))
    agg = defaultdict(list)
    for m in pool:
        ed = [np.array(acc[e]) for e in E3(m)]; a = np.concatenate(ed)
        agg["mean"].append(a.mean()); agg["median"].append(np.median(a)); agg["max"].append(a.max())
        agg["top-3 mean"].append(np.sort(a)[-3:].mean()); agg["min over edges of edge mean"].append(min(x.mean() for x in ed))
    rec = dict(data=name, variant=variant, rho=rho, seed=seed, epoch=ep, loss=loss[ep - 1]["loss"],
               **{f"{k} AUROC": float(auc(y, v)) for k, v in agg.items()}, **{f"{k} AP": float(ap(y, v)) for k, v in agg.items()})
    open(out, "a").write(json.dumps(rec) + "\n")
print(json.dumps(dict(tag=tag, minutes=round((time.time() - st) / 60, 1), losses=[round(x["loss"], 4) for x in loss])), flush=True)
