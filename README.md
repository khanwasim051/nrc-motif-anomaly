# NRC: camouflaged motif-anomaly benchmark and Node-Rhythm Conditional coupling

Reproducibility package (https://github.com/khanwasim051/nrc-motif-anomaly) for the paper

> **Training-Free Detection of Coordinated Motif Anomalies in Dynamic Graphs via Node-Rhythm Conditional Coupling**
> (submitted to the *International Journal of Data Science and Analytics*).

It contains the NRC implementation, the camouflaged anomaly generator (anchored, joint-burst and periodic
coordination), the standard clique-injection protocol, all baselines except the third-party SLADE code
(supplied as a patch), the evaluation and statistics scripts, the random seeds, the software versions,
and the raw per-run outputs behind every table and figure.

## Contents

```
code/                 all scripts (run them from this folder)
  motif_bench.py      data loading/preprocessing, triangle enumeration, standard clique injection, structural heuristics
  camouflage.py       per-edge event trains
  coupling.py         camouflaged generator (Algorithm 1), naive synchrony rate, global/fixed-window null variants
  coupling2.py        NRC (Algorithm 2): chan_coincidence(), activity pools
  baselines_dyn.py    MIDAS, MIDAS-R (exact counts), burstiness, isolation forest over 16 motif features
  leakcheck.py        main camouflaged table: structural heuristics, activity novelty, naive rate, NRC
  ...                 one script per experiment, see the table below
results/raw/          per-run outputs (JSON lines), one line per graph x setting x seed unless noted
results/summary/      aggregated files read by the manuscript builder (paper_rows.json, stats.json, ...)
figures/              Figures 1-4 (PDF vector and 300-dpi PNG)
slade/                patch for the official SLADE code and the three modified files
data/                 download instructions and SHA-256 checksums (datasets are not redistributed)
tests/                self-contained checks of the generator and NRC on a synthetic stream
CONFIG.md             exact settings of the protocol, NRC, every baseline and the statistics
reproduce_summaries.sh  rebuilds results/summary from results/raw and verifies every value (verify_summaries.py)
```

## Quick verification (about five minutes, no data download)

```bash
pip install -r requirements.txt
python tests/test_protocol.py      # generator invariants (Proposition 1) and NRC on a synthetic stream
bash reproduce_summaries.sh        # rebuilds all 12 files in results/summary from results/raw and compares them
```

`tests/test_protocol.py` injects all three coordination types, with and without the endpoint constraint, into a
synthetic stream and checks that edges, edge multiplicities, birth times and all structural heuristic scores are
unchanged, that only later events move, and that NRC ranks coordinated motifs above chance.
`reproduce_summaries.sh` recomputes every aggregated number behind the tables, including all bootstrap intervals,
from the per-run outputs and reports `ALL SUMMARIES REPRODUCED` when every value matches the shipped summaries.
Exact settings of every method and baseline are listed in `CONFIG.md`.

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # NRC, generator, baselines, statistics, figures
ln -s ../data code/data                  # scripts read code/data/raw and code/data/snap
```

Download the datasets as described in `data/README.md` and check the SHA-256 sums.
All experiments except SLADE run on a CPU; the reported results were produced on a 2-core Linux virtual machine
with 8 GB of memory (Python 3.13.18).

## Seeds

| Purpose | Seed |
|---|---|
| Candidate sampling and anomaly injection, seed s in {0, 1, 2} | `numpy.random.default_rng(100 + s)` |
| Isolation forest | `random_state = s` |
| Activity-preserving randomisation (null calibration) | `default_rng(500 + s)` |
| Monte Carlo null detectors (global and local) | `default_rng(900 + s)` |
| Cluster bootstrap (20,000 replicates) | `default_rng(0)` |
| Standard clique injection (Table 3) | `default_rng(0)` per graph |
| SLADE | the official default `--seed 0` |

## Reproducing the paper

Run every command from `code/`. Experiment scripts append JSON lines to the file named on the command line
(or print them); our own outputs for every command are in `results/raw/`. Summary scripts write their JSON file to the
working directory; the copies used for the manuscript are in `results/summary/`.
To rebuild tables and figures from our outputs without re-running experiments, copy them into `code/`:
`cp ../results/raw/*.jsonl ../results/raw/*.json .`

| Paper item | Command (per graph `G` in `uci email_eu mathoverflow`) | Raw output |
|---|---|---|
| Table 2 (statistics) | `python dataset_stats.py` | `dataset_stats.jsonl` |
| Table 3 (standard protocol) | `python stdheur.py G` (also `btc_alpha`, `btc_otc`) | `std_heuristics.jsonl` |
| Table 4 (main camouflaged results) | `python leakcheck.py G 3 perrun_main.jsonl` and SLADE (below); then `python build_paper_rows.py` | `perrun_main.jsonl`, `slade_1epoch_perrun.jsonl` |
| Table 5, Table S1 (baselines) | `python baselines_dyn.py G 3 baselines_means.jsonl anchored,joint_bursts,periodic 0.1,0.25 perrun_baselines.jsonl` | `perrun_baselines.jsonl`, `baselines_means.jsonl` |
| Table 6 (activity statistics) | `python actstats.py G` | `actstats.jsonl` |
| Table 7 (ablation) | `python ablation.py G abl ablation.jsonl` | `ablation.jsonl` (means over 3 seeds) |
| Table 8 (prevalence) | `python ablation.py G sens sensitivity.jsonl` | `sensitivity.jsonl` (means over 3 seeds) |
| Figures 2-3 (strength, ±1 SD) | `python sens_seeds.py G sens_seeds.jsonl`; `python make_figs.py` | `sens_seeds.jsonl` |
| Table 9 (run time) | `python baselines_dyn.py G 1 timing.jsonl anchored 0.1` (machine otherwise idle) | `timing_all.jsonl` |
| Table 10, Tables S2-S3 (label noise, edge-disjoint) | `python overlap.py G ov_G.jsonl`; then `python summarise_overlap.py` | `ov_*.jsonl`; `results/summary/overlap_summary.json` |
| Table 11 (null calibration) | `python calib.py G cal_G.jsonl`; then `python summarise_calib.py` | `cal_*.jsonl`; `results/summary/calib_summary.json` |
| Table 12, Table S4 (paired differences, bootstrap) | `python stats2.py` (reads `ov_*.jsonl`) | `results/summary/stats2.json` |
| Win/tie/loss counts, Table 6 | `python stats.py` (reads `paper_rows.json`, `baselines_means.jsonl`, `actstats.jsonl`) | `results/summary/stats.json` |
| Table 13, Table S5 (SLADE 10 epochs) | `python slade_ckpt.py G TYPE RHO SEED slade_ck.jsonl` | `slade_ck_uci.jsonl` (CPU), `slade_ck_other.jsonl` (Email-Eu, MathOverflow; GPU); assembled with `python assemble_slade.py OUTDIR` |
| Table 14, Table S6 (global and local Monte Carlo nulls; `G` also `radoslaw`, `thiers13`) | `python rndnull.py G rnd_G.jsonl`; `python rndnull.py G rndloc_G.jsonl local`; then `python rnd_summarise.py rnd_summary.json` and `python rnd_boot.py` (both read `ov_*.jsonl`) | `rnd_*.jsonl`, `rndloc_*.jsonl`; `results/summary/rnd_summary.json`, `rnd_boot.json` |
| Table 15, Figure 4 (coordination window) | `python deltasweep.py G ds_G.jsonl`; `python make_dsfig.py` | `ds_*.jsonl` |
| Figure 1 (panel c) | `python zscale.py`; `python make_method_fig.py` | `zscale.json` |
| Table 16, Table S7 (held-out validation, NRC frozen) | `python leakcheck.py radoslaw 3 heldout_radoslaw_main.jsonl`; `python overlap.py radoslaw ov_radoslaw.jsonl`; same with `thiers13`; then `python summarise_heldout.py` | `heldout_*`, `ov_radoslaw.jsonl`, `ov_thiers13.jsonl`; freeze record in `results/freeze_record.txt` |
| Table S8 (per-endpoint activity audit, Sec. 4.3) | `python endpoint_audit.py G epa_G.jsonl` (also `radoslaw`, `thiers13`) | `epa_*.jsonl` |
| Table 17, Table S9 (endpoint-wise activity constraint, w = 1 h) | `python strict_eval.py G strict_G.jsonl 3600` (also `radoslaw`, `thiers13`); then `python summarise_strict.py` (also reads `epa_*.jsonl`) | `strict_*.jsonl`; `results/summary/strict_summary.json` |
| Tables 18-19, Table S10 (held-out sensitivity, Sec. 7.16) | `python sens_seeds_ho.py thiers13 sens_ho_thiers13.jsonl` (also `radoslaw`); `python ablation.py thiers13 abl ablation.jsonl`; `python ablation.py thiers13 sens sensitivity.jsonl`; `python deltasweep.py thiers13 ds_thiers13.jsonl`; `python calib.py thiers13 cal_thiers13.jsonl`; then `python ho_sens_summary.py` | `sens_ho_*.jsonl`, `ds_thiers13.jsonl`, `cal_thiers13.jsonl`, Thiers13 rows of `ablation.jsonl` and `sensitivity.jsonl`; `results/summary/ho_sens.json` |
| Generator fallbacks, pool sizes (Sec. 4.2, 5.2) | `python gen_audit.py G` | `gen_audit.jsonl` |
| Naive-rate entries of `ov_*.jsonl` with the per-metric oracle scale (replays the instances without the other detectors; checks the stored values first) | `python naive_refresh.py G ov_G.jsonl ov_G_new.jsonl` | `ov_*.jsonl` |
| Monte Carlo resampling pools: empty post-birth pools, draws with replacement, local windows below k (Sec. 6.3, 7.9) | `python pool_audit.py G pool_audit.jsonl` | `pool_audit.jsonl` |
| Boundary audit: new times inside the window, infeasible share (Sec. 4.2) | `python clip_audit.py G` | `clip_audit.jsonl` |
| SLADE run times (Sec. 7.7) | `python assemble_slade.py OUTDIR slade_parts results/slade_ckpt_v2.jsonl slade_progress.log results/run_gpu3_log.txt` | `slade_runtime.jsonl`; logs in `results/logs/` |

Approximate CPU time with one core per graph: main table 5-15 min, baselines 10-40 min, overlap 20-60 min,
calibration 10-30 min, randomisation null 10-20 min, coordination-window sweep 20-60 min, ablation 30-90 min.

### SLADE

SLADE is third-party code and is not included. To reproduce the SLADE results:

```bash
git clone https://github.com/jhsk777/SLADE.git code/SLADE
cd code/SLADE && git checkout 30f8c8f && git apply --ignore-whitespace ../../slade/slade_motif_patch.diff && cd ..
pip install -r ../requirements-slade.txt     # separate environment recommended
SLADE_EPOCHS=1 python slade_motif.py G TYPE RHO SEED        # one-epoch runs (Table 4)
SLADE_EPOCHS=10 SLADE_CKPTS=1,2,3,5,10 python slade_ckpt.py G TYPE RHO SEED slade_ck.jsonl   # Table 13, S5
python run_slade_gpu.py     # Email-Eu/MathOverflow ten-epoch jobs (GPU); first checks instance fingerprints (fingerprint.py)
```

The patch only (i) exports per-event anomaly scores after replaying the full stream from a fresh memory,
(ii) saves score checkpoints and per-epoch training loss, (iii) replaces `torch_scatter` by
`torch.Tensor.scatter_reduce`, and (iv) guards the validation AUROC against single-class batches.
If `git apply` fails, copy the files in `slade/modified_files/` over the clone.

## Raw output format

Each line of a `.jsonl` file is one run, identified by `data`, `variant` (`anchored`, `joint_bursts`, `periodic`),
`rho`, and `seed` (and `delta`, `mode` or `epoch` where relevant), followed by the metrics of every detector on that
run (AUROC and AP). Files marked "means over 3 seeds" contain one line per setting.

## Licence and citation

Code: MIT licence (see `LICENSE`). Please cite the paper (see `CITATION.cff`).
The SLADE patch modifies code from https://github.com/jhsk777/SLADE and is subject to that repository's terms.
