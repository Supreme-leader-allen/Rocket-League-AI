#!/usr/bin/env bash
# Random-action baseline. No training. This is the ruler every other experiment
# is measured against, so it has to exist before 01 means anything.
# 01_ground.sh runs this automatically if its output is missing.
set -euo pipefail
source "$(dirname "$0")/params.sh"
cd "$REPO_ROOT"
banner "00  baseline (random actions)"

# A few hundred episodes is plenty for per-episode means/SEMs; the old
# ROUND*2 (6000 episodes at ROUND=3000, single process) took 5+ hours.
EPISODES=$(( ROUND * 2 ))
[ "$EPISODES" -lt 10 ] && EPISODES=10
[ "$EPISODES" -gt 400 ] && EPISODES=400

python Run_Baseline.py --episodes "$EPISODES" --workers "$N_PROC" \
    --metrics-dir "$OUT/metrics" 2>&1 | tee "$OUT/logs/00_baseline.log"

echo "00 done -> $OUT/metrics/baseline_random.*.csv"
