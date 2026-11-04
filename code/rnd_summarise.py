"""Summarise global and local Monte Carlo null detectors against NRC and the naive rate (Table 14, Table S6)."""
import json, sys, numpy as np
GS = ["uci", "email_eu", "mathoverflow", "radoslaw", "thiers13"]
CELLS = [(v, r) for v in ["anchored", "joint_bursts", "periodic"] for r in [0.1, 0.25]]
L = lambda f: [json.loads(l) for l in open(f)]
out = {}
for g in GS:
    ov = {(o["variant"], o["rho"], o["seed"]): o for o in L(f"ov_{g}.jsonl") if o["mode"] == "standard"}
    gl = {(o["variant"], o["rho"], o["seed"]): o for o in L(f"rnd_{g}.jsonl")}
    lo = {(o["variant"], o["rho"], o["seed"]): o for o in L(f"rndloc_{g}.jsonl")}
    keys = sorted(ov); assert set(keys) == set(gl) == set(lo) and len(keys) == 18, g
    out[f"{g}|wins_global"] = int(sum(ov[k]["NRC AUROC"] > gl[k]["rnd"] for k in keys))
    out[f"{g}|wins_local"] = int(sum(ov[k]["NRC AUROC"] > lo[k]["rnd"] for k in keys))
    out[f"{g}|local_gt_global"] = int(sum(lo[k]["rnd"] > gl[k]["rnd"] for k in keys))
    out[f"{g}|n"] = len(keys)
    for v, r in CELLS:
        ks = [k for k in keys if k[0] == v and k[1] == r]
        m = lambda d, f: float(np.mean([d[k][f] for k in ks]))
        out[f"{g}|{v}|{r}"] = dict(naive=m(ov, "naive AUROC"), nrc=m(ov, "NRC AUROC"), rnd=m(gl, "rnd"), rnd_ap=m(gl, "rnd_ap"),
                                   sec=m(gl, "sec"), loc=m(lo, "rnd"), loc_ap=m(lo, "rnd_ap"), loc_sec=m(lo, "sec"))
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "rnd_summary.json", "w"), indent=1)
for g in GS:
    xs = [out[f"{g}|{v}|{r}"] for v, r in CELLS]; a = lambda f: np.mean([x[f] for x in xs])
    print(g, "naive %.3f glob %.3f loc %.3f nrc %.3f | NRC>glob %d NRC>loc %d loc>glob %d /18 | sec %.0f %.0f" % (
        a("naive"), a("rnd"), a("loc"), a("nrc"), out[g+"|wins_global"], out[g+"|wins_local"], out[g+"|local_gt_global"], a("sec"), a("loc_sec")))
    for (v, r), x in zip(CELLS, xs): print("   ", v, r, "naive %.3f glob %.3f loc %.3f nrc %.3f" % (x["naive"], x["rnd"], x["loc"], x["nrc"]))
