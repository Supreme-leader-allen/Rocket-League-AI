#!/usr/bin/env bash
# Random-action baseline. No training. This is the ruler every other experiment
# is measured against, so it has to exist before 01 means anything.
set -euo pipefail
source "$(dirname "$0")/params.sh"
cd "$REPO_ROOT"
banner "00  baseline (random actions)"

EPISODES=$(( ROUND * 2 ))
[ "$EPISODES" -lt 10 ] && EPISODES=10

python Run_Baseline.py --episodes "$EPISODES" 2>&1 | tee "$OUT/logs/00_baseline.log"

# Run_Baseline.py writes relative to CWD; collect the shards into $OUT.
mv metrics/baseline_random*.csv "$OUT/metrics/" 2>/dev/null || true
echo "00 done -> $OUT/metrics/baseline_random.*.csv"
