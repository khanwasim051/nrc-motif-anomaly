"""Assemble the v2 SLADE outputs (slade_parts/<graph>_<type>_<rho>_<seed>.jsonl from slade_v2.sh) into the raw files
used by the summaries: slade_ck_uci.jsonl and slade_ck_other.jsonl (ten-epoch checkpoint runs), slade_1epoch_perrun.jsonl
(epoch-1 checkpoint of every run, mean aggregation = the one-epoch main result) and slade_runtime.jsonl (wall-clock
minutes per run from slade_progress.log, minute resolution, two runs in parallel on two cores; ten-epoch Email-Eu/MathOverflow times from the laptop GPU log)."""
import json, glob, os, re, sys, datetime
# usage: python assemble_slade.py OUTDIR [PARTS_DIR] [GPU_JSONL] [CPU_PROGRESS_LOG] [GPU_LOG]
A = sys.argv + [None] * 6
OUT = A[1]; SRC = A[2] or "slade_parts"; GPU = A[3] or "results/slade_ckpt_v2.jsonl"
CPULOG = A[4] or "slade_progress.log"; GPULOG = A[5] or "results/run_gpu3_log.txt"
rows = {}
for f in sorted(glob.glob(f"{SRC}/*.jsonl")):
    tag = os.path.basename(f)[:-6]
    if not os.path.exists(f"{SRC}/{tag}.done"): sys.exit(f"incomplete: {tag}")
    rows[tag] = [json.loads(l) for l in open(f)]
assert len(rows) == 54, len(rows)
ck_uci, one = [], []
# Email-Eu / MathOverflow ten-epoch checkpoints come from the laptop GPU (run_gpu3.py, instances checked by fingerprint)
ck_oth = [json.loads(l) for l in open(GPU) if l.strip()]
assert len(ck_oth) == 60, len(ck_oth)
for tag, R in rows.items():
    if len(R) > 1: assert R[0]["data"] == "uci", tag; ck_uci.extend(R)
    e1 = [r for r in R if r["epoch"] == 1][0]
    one.append(dict(data=e1["data"], variant=e1["variant"], rho=e1["rho"], seed=e1["seed"], slade_auroc=e1["mean AUROC"], slade_ap=e1["mean AP"], epochs=max(r["epoch"] for r in R), source=e1.get("source", "cpu")))
order = {"uci": 0, "email_eu": 1, "mathoverflow": 2}; vo = {"anchored": 0, "joint_bursts": 1, "periodic": 2}
key = lambda r: (order[r["data"]], vo[r["variant"]], r["rho"], r["seed"], r.get("epoch", 0))
for name, R in [("slade_ck_uci", ck_uci), ("slade_ck_other", ck_oth), ("slade_1epoch_perrun", one)]:
    with open(f"{OUT}/{name}.jsonl", "w") as fh:
        for r in sorted(R, key=key): fh.write(json.dumps(r) + "\n")
# run times from the progress log (last START and DONE of each tag)
st, dn = {}, {}
for l in open(CPULOG):
    m = re.match(r"(\d\d):(\d\d) (START|DONE ) (\S+)", l)
    if not m: continue
    t = int(m[1]) * 60 + int(m[2]); (st if m[3] == "START" else dn)[m[4]] = t
with open(f"{OUT}/slade_runtime.jsonl", "w") as fh:
    for tag in sorted(rows):
        if tag in st and tag in dn:
            fh.write(json.dumps(dict(tag=tag, epochs=max(r["epoch"] for r in rows[tag]), minutes=(dn[tag] - st[tag]) % 1440)) + "\n")
    for l in open(GPULOG):
        m = re.search(r"\[(\d+)/12\] running (\S+) (\S+) (\S+) seed (\d)", l)
        if m: cur = f"{m[2]}_{m[3]}_{m[4]}_{m[5]}"
        m = re.search(r"\[\d+/12\] finished in ([\d.]+) min", l)
        if m: fh.write(json.dumps(dict(tag=cur, epochs=10, minutes=float(m[1]), device="gpu_rtx3050")) + "\n")
print(len(ck_uci), len(ck_oth), len(one))
