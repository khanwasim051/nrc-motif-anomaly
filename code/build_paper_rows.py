"""Assemble the main camouflaged-protocol table (Table 4) from per-run outputs.
Inputs (in the working directory): perrun_main.jsonl (leakcheck.py), slade_1epoch_perrun.jsonl (slade_motif.py).
Output (working directory): paper_rows.json with one row per (graph, type, rho):
[graph, type, rho, structural max, activity novelty, best naive AUROC, NRC AUROC, best naive AP, NRC AP, SLADE AUROC]."""
import json, numpy as np
from collections import defaultdict
M = defaultdict(list); S = defaultdict(list)
for l in open("perrun_main.jsonl"):
    r = json.loads(l); M[(r["data"], r["variant"], r["rho"])].append(r)
for l in open("slade_1epoch_perrun.jsonl"):
    r = json.loads(l); S[(r["data"], r["variant"], r["rho"])].append(r["slade_auroc"])
rows = []
for d in ["uci", "email_eu", "mathoverflow"]:
    for v in ["anchored", "joint_bursts", "periodic"]:
        for rho in [0.1, 0.25]:
            rs = M[(d, v, rho)]; m = lambda k: round(float(np.mean([x[k] for x in rs])), 4)
            rows.append([d, v, rho, m("structural max"), m("activity novelty"), m("best naive"), m("NRC"), m("best naive AP"), m("NRC AP"),
                         round(float(np.mean(S[(d, v, rho)])), 3) if S[(d, v, rho)] else None])
json.dump(rows, open("paper_rows.json", "w")); print(len(rows), "rows")
