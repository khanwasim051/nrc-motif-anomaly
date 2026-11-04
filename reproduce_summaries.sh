#!/usr/bin/env bash
# Rebuild every aggregated result file (results/summary/*.json) from the raw per-run outputs in results/raw,
# then compare them with the shipped summaries. Needs only requirements.txt; takes a few minutes on one CPU core.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
cp "$ROOT"/results/raw/*.jsonl "$ROOT"/results/raw/*.json "$WORK"/
cd "$WORK"
C="$ROOT/code"
python "$C/build_paper_rows.py"            # paper_rows.json   (Table 4)
python "$C/stats.py"            > /dev/null # stats.json        (win/tie/loss counts)
python "$C/stats2.py"           > /dev/null # stats2.json       (Tables 12, S4: paired differences, bootstrap)
python "$C/summarise_overlap.py"            # overlap_summary.json  (Tables 10, S2, S3)
python "$C/summarise_calib.py"              # calib_summary.json    (Table 11)
python "$C/summarise_heldout.py"            # heldout.json, heldout_settings.json (Tables 16, S7)
python "$C/summarise_strict.py"             # strict_summary.json   (Tables 17, S8, S9)
python "$C/rnd_summarise.py" rnd_summary.json > /dev/null   # Tables 14, S6
python "$C/rnd_boot.py"         > /dev/null # rnd_boot.json
python "$C/ho_sens_summary.py"  > /dev/null # ho_sens.json      (Tables 18, 19, S10)
mkdir -p "$WORK/fig_out"; sed 's#\.\./figures/#fig_out/#g' "$C/make_dsfig.py" > make_dsfig_local.py
python make_dsfig_local.py      > /dev/null # dsweep_summary.json (Table 15, Fig. 4)
python "$ROOT/verify_summaries.py" "$WORK" "$ROOT/results/summary"
