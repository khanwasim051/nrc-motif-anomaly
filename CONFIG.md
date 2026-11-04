# Configuration of every method and experiment

All values below are the ones used for the reported results; the scripts in `code/` contain no other tunable settings.

## Camouflaged protocol (Algorithm 1; `code/coupling.py`, function `inject`)

| Setting | Value |
|---|---|
| Eligible triangles | triangles whose three edges each have ≥ 3 events |
| Candidates per run | min(3,000, eligible) — 2,220 on UCI, 3,000 on all other graphs |
| Anomaly fraction | 5% (prevalence analysis: 1%, 5%, 10% at ρ = 0.25) |
| Re-timing probability ρ | {0.10, 0.25} main; {0.05, 0.10, 0.25, 0.50} strength analysis |
| Coordination window δ | 600 s (window analysis: 60, 600, 3,600, 14,400 s) |
| Anchored | events of e2, e3 moved to U(−δ, δ) around randomly chosen events of e1 |
| Joint bursts | 5 burst centres drawn from the motif nodes' out-of-motif event times; events moved to U(−δ, δ) around them |
| Periodic | period U(0.5, 2) days, phase U(0, period); lattice points within 1 h of motif-node activity (full lattice if none); jitter N(0, (δ/3)²) |
| Feasible-time sampling | a proposal must satisfy birth < t′ ≤ t_max; otherwise redrawn (≤ 100 proposals per event); an event without a feasible proposal keeps its time and is counted as infeasible; birth events are never moved. Counters `coupling.GEN` (selected, placed, infeasible, redraws); audit `code/clip_audit.py` → `clip_audit.jsonl` |
| Endpoint constraint (Sec. 7.15) | `endpoint_w = 3600`: both endpoints active on other edges within 1 h; ≤ 100 proposals, else unchanged |
| Edge-disjoint control | random scan of eligible triangles, accept if no edge used before |
| Seeds | candidate sampling and injection `default_rng(100 + s)`, s ∈ {0, 1, 2} |

## NRC (Algorithm 2; `code/coupling2.py`, function `chan_coincidence`)

k = 50 (rank window), λ = 5 (shrinkage), scales Δ = {60, 300, 900, 3,600, 14,400} s, score = sum over scales.
Frozen since the end of development; checksums in `results/freeze_record.txt`.

## Baselines

| Detector | Configuration | Script |
|---|---|---|
| Structural heuristics | low edge multiplicity (−Σ log(1 + |T_e|)), synchronised edge birth (−SD of birth times), few common neighbours outside the motif, low degree sum; best of four reported | `motif_bench.py` (`heuristics`) |
| Activity novelty | Eq. (2), W = 1 day, all later events of every candidate | `leakcheck.py` |
| Naive synchrony rate | δ ∈ {600, 3,600} s, better of the two per setting (oracle); six scales in Sec. 7.13 | `coupling.py` (`naive_rate`) |
| Burstiness | Goh–Barabási B of the merged later inter-event times | `baselines_dyn.py` |
| MIDAS / MIDAS-R | exact counts, 1-hour ticks, MIDAS-R decay α = 0.5; motif score = mean log(1 + score) over later events | `baselines_dyn.py` |
| Isolation forest | 16 motif features (Sec. 6.3), `n_estimators=200`, `random_state=s`, other scikit-learn defaults | `baselines_dyn.py`, `overlap.py` |
| SLADE | official code at commit 30f8c8f plus `slade/slade_motif_patch.diff`; batch 100, learning rate 3·10⁻⁶ with decay 0.8 per epoch, 20 neighbours, memory 256, message 128, `--seed 0`; 1 epoch (main), 10 epochs with checkpoints 1, 2, 3, 5, 10 (Sec. 7.11) | `slade_motif.py`, `slade_ckpt.py` |
| Monte Carlo nulls | 30 randomisations per motif (`RND_B`), global: uniform from the whole pool; local: uniform from the rank window of k = 50 (`python rndnull.py G out local`); `default_rng(900 + s)` | `rndnull.py` |
| TADDY | analysed formally (Proposition 2), not run | — |

## Statistics

Two-level cluster bootstrap over settings, then seeds; 20,000 replicates; `default_rng(0)`. The order in which
intervals are drawn is fixed in each summary script (`stats2.py`, `summarise_heldout.py`, `summarise_strict.py`,
`rnd_boot.py`). Null calibration: activity-pool randomisation with `default_rng(500 + s)`.

## Software

Python 3.13.18, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2 (`requirements.txt`).
SLADE: see `requirements-slade.txt`.
