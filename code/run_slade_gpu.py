"""SLADE, 10 epochs with checkpoints at epochs 1, 2, 3, 5, 10, on the corrected (v2) camouflaged anomalies.
12 jobs: Email-Eu and MathOverflow x anchored and joint bursts x rho = 0.25 x seeds 0, 1, 2.
Results go to results/slade_ckpt_v2.jsonl (raw file slade_ck_other.jsonl in the package).
Safe to stop (Ctrl+C) and restart: finished jobs are skipped."""
import subprocess, sys, os, json, time
os.makedirs("results", exist_ok=True)
OUT = os.path.join("results", "slade_ckpt_v2.jsonl"); LOG = os.path.join("results", "run_gpu3_log.txt")
EXP = json.load(open("expected_fingerprints.json"))
JOBS = [(d, v, 0.25, s) for s in [0, 1, 2] for d in ["email_eu", "mathoverflow"] for v in ["joint_bursts", "anchored"]]
def log(msg):
    print(msg, flush=True); open(LOG, "a").write(time.strftime("%Y-%m-%d %H:%M ") + msg + "\n")
def done(d, v, r, s):
    if not os.path.exists(OUT): return False
    return any((x["data"], x["variant"], x["rho"], x["seed"], x["epoch"]) == (d, v, r, s, 10) for x in map(json.loads, open(OUT)))
# 1. check that this laptop builds exactly the same anomalies as the server
log("Checking the anomaly generator (a few minutes) ...")
from fingerprint import fingerprint
bad = []
for d, v, r, s in JOBS:
    got, exp = fingerprint(d, v, r, s), EXP[f"{d}|{v}|{r}|{s}"]
    log(f"   {d} {v} seed {s}: {'OK' if got == exp else 'MISMATCH ' + json.dumps(got)}")
    if got != exp: bad.append((d, v, s))
if bad:
    log("STOP: the generator does not reproduce the server's instances. Please send results\\run_gpu3_log.txt."); sys.exit(1)
log("Generator check passed.")
# 2. train
for i, (d, v, r, s) in enumerate(JOBS, 1):
    if done(d, v, r, s): log(f"[{i}/12] skip {d} {v} {r} seed {s} (already done)"); continue
    log(f"[{i}/12] running {d} {v} {r} seed {s} ...")
    t0 = time.time()
    rc = subprocess.run([sys.executable, "slade_ckpt.py", d, v, str(r), str(s), OUT],
                        env=dict(os.environ, SLADE_EPOCHS="10", SLADE_CKPTS="1,2,3,5,10")).returncode
    log(f"[{i}/12] {'finished' if rc == 0 else 'FAILED'} in {(time.time() - t0) / 60:.1f} min")
log("ALL DONE - please send results\\slade_ckpt_v2.jsonl and results\\run_gpu3_log.txt")
