"""Summaries for the held-out sensitivity analyses (Section 7.16, Tables 18-19)."""
import json, numpy as np
L = lambda f: [json.loads(l) for l in open(f)]
out = {"strength": {}, "prevalence": {}, "delta": {}, "hyper": {}}
for g in ["thiers13", "radoslaw"]:
    R = L(f"sens_ho_{g}.jsonl")
    for v in ["anchored", "joint_bursts", "periodic"]:
        for rho in [0.05, 0.1, 0.25, 0.5]:
            x = [r for r in R if r["variant"] == v and r["rho"] == rho]; assert len(x) == 3, (g, v, rho, len(x))
            d = np.array([r["nrc_auroc"] - r["naive_auroc"] for r in x])
            out["strength"][f"{g}|{v}|{rho}"] = dict(naive=np.mean([r["naive_auroc"] for r in x]), nrc=np.mean([r["nrc_auroc"] for r in x]),
                naive_ap=np.mean([r["naive_ap"] for r in x]), nrc_ap=np.mean([r["nrc_ap"] for r in x]), diff=d.mean(), diff_sd=d.std(ddof=1), wins=int((d > 0).sum()))
S = [r for r in L("sensitivity.jsonl") if r["data"] == "thiers13"]
for r in S:
    out["prevalence"][f"{r['variant']}|{r['rho']}|{r['frac']}"] = {k: r[k] for k in ["naive AUROC", "naive AP", "NRC AUROC", "NRC AP"]}
D = [r for r in L("ds_thiers13.jsonl")]
near = lambda d: str(int(min([60, 300, 900, 3600, 14400], key=lambda s: abs(np.log(s / d)))))
for dl in [60.0, 600.0, 3600.0, 14400.0]:
    x = [r for r in D if r["delta"] == dl]
    out["delta"][str(int(dl))] = dict(naive=np.mean([r["naive_best5"] for r in x]), nrc=np.mean([r["nrc"] for r in x]), matched=np.mean([r["single"][near(dl)] for r in x]),
        d_nrc=np.mean([r["nrc"] - r["naive_best5"] for r in x]), d_matched=np.mean([r["single"][near(dl)] - r["naive_best5"] for r in x]), n=len(x))
A = [r for r in L("ablation.jsonl") if r["data"] == "thiers13"]
for k in A[0]:
    if k in ("data", "variant", "rho"): continue
    out["hyper"][k] = float(np.mean([r[k] for r in A]))
json.dump(out, open("ho_sens.json", "w"), indent=1)
for k, v in out["delta"].items(): print("delta", k, {a: round(b, 3) for a, b in v.items()})
for k, v in out["hyper"].items(): print("hyper", k, round(v, 3))
for k, v in out["strength"].items(): print(k, {a: round(b, 3) for a, b in v.items()})
