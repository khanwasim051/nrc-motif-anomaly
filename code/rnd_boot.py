"""Paired two-level cluster bootstrap (settings, then seeds) for the Monte Carlo null comparison."""
import json, numpy as np
GS = ["uci", "email_eu", "mathoverflow", "radoslaw", "thiers13"]
cells = [(v, r) for v in ["anchored", "joint_bursts", "periodic"] for r in [0.1, 0.25]]
L = lambda f: [json.loads(l) for l in open(f)]
rng = np.random.default_rng(0); B = 20000
def boot2(D):
    nc, ns = D.shape; ci = rng.integers(0, nc, (B, nc)); si = rng.integers(0, ns, (B, nc, ns))
    return np.quantile(D[ci[:, :, None], si].mean((1, 2)), [.025, .975]).tolist()
res = {}
for g in GS:
    ov = {(o["variant"], o["rho"], o["seed"]): o["NRC AUROC"] for o in L(f"ov_{g}.jsonl") if o["mode"] == "standard"}
    gl = {(o["variant"], o["rho"], o["seed"]): o["rnd"] for o in L(f"rnd_{g}.jsonl")}
    lo = {(o["variant"], o["rho"], o["seed"]): o["rnd"] for o in L(f"rndloc_{g}.jsonl")}
    A = lambda a, b: np.array([[a[(v, r, s)] - b[(v, r, s)] for s in range(3)] for v, r in cells])
    res[g] = {}
    for name, D in [("local-global", A(lo, gl)), ("NRC-local", A(ov, lo)), ("NRC-global", A(ov, gl))]:
        res[g][name] = dict(mean=float(D.mean()), ci=boot2(D))
        print(g, name, "%+.3f [%+.3f, %+.3f]" % (D.mean(), *res[g][name]["ci"]))
json.dump(res, open("rnd_boot.json", "w"), indent=1)
