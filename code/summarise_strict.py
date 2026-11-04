"""Table 17 and Tables S8-S9: endpoint-wise activity constraint (strict_<graph>.jsonl from strict_eval.py) and the
per-endpoint activity audit (epa_<graph>.jsonl from endpoint_audit.py). Bootstrap as in summarise_heldout.py, but with one generator default_rng(0) shared by all graphs in the order of GS."""
import json, numpy as np
GS = ["uci", "email_eu", "mathoverflow", "radoslaw", "thiers13"]
CELLS = [(v, r) for v in ["anchored", "joint_bursts", "periodic"] for r in [0.1, 0.25]]
B = 20000
def boot2(rng, D):
    nc, ns = D.shape; ci = rng.integers(0, nc, (B, nc)); si = rng.integers(0, ns, (B, nc, ns))
    return np.quantile(D[ci[:, :, None], si].mean((1, 2)), [.025, .975]).tolist()
L = lambda f: [json.loads(l) for l in open(f)]
out = {"strict": {}, "settings": [], "audit": []}
rng = np.random.default_rng(0)   # one generator for all graphs, in the order of GS
for g in GS:
    R = L(f"strict_{g}.jsonl"); get = lambda v, r, s: [x for x in R if x["variant"] == v and x["rho"] == r and x["seed"] == s][0]
    D = np.array([[get(v, r, s)["nrc"] - get(v, r, s)["naive"] for s in range(3)] for v, r in CELLS])
    cell = []
    for v, r in CELLS:
        x = [get(v, r, s) for s in range(3)]; m = lambda f: float(np.mean([y[f] for y in x]))
        st = dict(data=g, variant=v, rho=r, struct=m("struct"), actnov=m("actnov"), naive=m("naive"), nrc=m("nrc"), naive_ap=m("naive_ap"), nrc_ap=m("nrc_ap"),
                  iforest=m("iforest"), infeas=float(np.mean([y["infeasible"] / y["proposed"] for y in x])))
        out["settings"].append(st); cell.append(st)
    out["strict"][g] = dict(mean=float(D.mean()), ci=boot2(rng, D), mean_np=float(D[:4].mean()), ci_np=boot2(rng, D[:4]), mean_p=float(D[4:].mean()), ci_p=boot2(rng, D[4:]),
        infeas=float(sum(x["infeasible"] for x in R) / sum(x["proposed"] for x in R)), infeas_rng=[min(c["infeas"] for c in cell), max(c["infeas"] for c in cell)],
        actnov=[min(c["actnov"] for c in cell), max(c["actnov"] for c in cell)], naive=float(np.mean([c["naive"] for c in cell])), nrc=float(np.mean([c["nrc"] for c in cell])),
        iforest=float(np.mean([c["iforest"] for c in cell])), nrc_if=float(np.mean([c["nrc"] - c["iforest"] for c in cell])))
    E = L(f"epa_{g}.jsonl")
    for v in ["anchored", "joint_bursts", "periodic"]:
        x = [e for e in E if e["variant"] == v]
        out["audit"].append(dict(data=g, variant=v, **{k: float(np.mean([e[k] for e in x])) for k in
            ["retimed_both_3600", "retimed_one_3600", "retimed_both_86400", "organic_both_3600", "organic_one_3600", "organic_both_86400"]}))
json.dump(out, open("strict_summary.json", "w"), indent=1)
