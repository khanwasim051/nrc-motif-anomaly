"""Table 16 and Table S7: held-out validation summaries (reads ov_<graph>.jsonl from overlap.py and the main-table outputs).
Bootstrap: two-level cluster bootstrap (settings, then seeds), 20,000 replicates, one generator default_rng(0) per graph,
drawn in the order (all settings, anchored + joint, periodic) for each comparison in turn."""
import json, numpy as np
GS = ["radoslaw", "thiers13"]
CELLS = [(v, r) for v in ["anchored", "joint_bursts", "periodic"] for r in [0.1, 0.25]]
COMPS = [("standard", "NRC AUROC", "naive AUROC"), ("standard", "NRC AUROC clean", "naive AUROC clean"),
         ("edge_disjoint", "NRC AUROC", "naive AUROC"), ("standard", "NRC AP", "naive AP"), ("standard", "NRC AUROC", "IForest AUROC")]
B = 20000
def boot2(rng, D):
    nc, ns = D.shape; ci = rng.integers(0, nc, (B, nc)); si = rng.integers(0, ns, (B, nc, ns))
    return np.quantile(D[ci[:, :, None], si].mean((1, 2)), [.025, .975]).tolist()
L = lambda f: [json.loads(l) for l in open(f)]
main = {"radoslaw": L("heldout_radoslaw_main.jsonl"), "thiers13": L("heldout_thiers13_main.jsonl")}
out, settings = {}, []
for g in GS:
    R = L(f"ov_{g}.jsonl"); get = lambda m, v, r, s: [x for x in R if x["mode"] == m and x["variant"] == v and x["rho"] == r and x["seed"] == s][0]
    rng = np.random.default_rng(0); out[g] = {}
    for m, a, b in COMPS:
        D = np.array([[get(m, v, r, s)[a] - get(m, v, r, s)[b] for s in range(3)] for v, r in CELLS])
        out[g][f"{m}|{a}-{b}"] = dict(mean=float(D.mean()), ci=boot2(rng, D), mean_np=float(D[:4].mean()), ci_np=boot2(rng, D[:4]),
                                      mean_p=float(D[4:].mean()), ci_p=boot2(rng, D[4:]), pos=int((D > 0).sum()),
                                      sd=float(np.sqrt((D.std(axis=1, ddof=1) ** 2).mean())))
    for v, r in CELLS:
        mm = lambda m, f: float(np.mean([get(m, v, r, s)[f] for s in range(3)]))
        mn = [x for x in main[g] if x["variant"] == v and x["rho"] == r]
        settings.append(dict(data=g, variant=v, rho=r, struct=float(np.mean([x["structural max"] for x in mn])), actnov=float(np.mean([x["activity novelty"] for x in mn])),
                             naive=mm("standard", "naive AUROC"), nrc=mm("standard", "NRC AUROC"), naive_ap=mm("standard", "naive AP"), nrc_ap=mm("standard", "NRC AP"),
                             naive_c=mm("standard", "naive AUROC clean"), nrc_c=mm("standard", "NRC AUROC clean"),
                             naive_e=mm("edge_disjoint", "naive AUROC"), nrc_e=mm("edge_disjoint", "NRC AUROC"), iforest=mm("standard", "IForest AUROC")))
json.dump(out, open("heldout.json", "w"), indent=1); json.dump(settings, open("heldout_settings.json", "w"), indent=1)
