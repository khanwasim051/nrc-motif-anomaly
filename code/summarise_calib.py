"""Table 11: null calibration of NRC, means over seeds of the per-seed statistics written by calib.py."""
import json, sys, numpy as np
GS = ["uci", "email_eu", "mathoverflow", "thiers13"]
out = {}
for g in GS:
    R = [json.loads(l) for l in open(f"cal_{g}.jsonl")]
    for kind in ["randomised", "organic"]:
        x = [r for r in R if r["kind"] == kind]
        m = lambda f: np.mean([r[f] for r in x], axis=0)
        out[f"{g}|{kind}"] = dict(n=float(m("n")), z_mean=m("z_mean").tolist(), z_sd=m("z_sd").tolist(),
            fpr={q: np.mean([r["z_fpr"][q] for r in x], axis=0).tolist() for q in ["1.645", "2.326", "3.09"]},
            S_mean=float(m("S_mean")), S_sd=float(m("S_sd")), corr=float(m("z_corr_mean")),
            S_fpr={q: float(np.mean([r["S_fpr_nominal"][q] for r in x])) for q in ["1.645", "2.326", "3.09"]})
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "calib_summary.json", "w"), indent=1)
