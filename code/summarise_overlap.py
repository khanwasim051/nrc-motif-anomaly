"""Table 10 and Tables S2-S3: means over seeds of the label-noise audit and the AUROC/AP under main,
cleaned-label and edge-disjoint evaluation (reads ov_<graph>.jsonl written by overlap.py)."""
import json, sys, numpy as np
GS = ["uci", "email_eu", "mathoverflow", "thiers13"]
STD = ["n_cand", "n_anom", "moved_median", "unchanged", "norm_share_edge", "norm_contam", "norm_share_node",
       "naive AUROC", "NRC AUROC", "burstiness AUROC", "IForest AUROC", "naive AP", "NRC AP",
       "naive AUROC clean", "NRC AUROC clean", "IForest AUROC clean", "naive AP clean", "NRC AP clean"]
ED = STD[:13]
out = {}
for g in GS:
    R = [json.loads(l) for l in open(f"ov_{g}.jsonl")]
    for v in ["anchored", "joint_bursts", "periodic"]:
        for rho in [0.1, 0.25]:
            for mode, F in [("standard", STD), ("edge_disjoint", ED)]:
                x = [r for r in R if r["variant"] == v and r["rho"] == rho and r["mode"] == mode]
                out[f"{g}|{v}|{rho}|{mode}"] = {f: float(np.mean([r[f] for r in x])) for f in F}
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "overlap_summary.json", "w"), indent=1)
