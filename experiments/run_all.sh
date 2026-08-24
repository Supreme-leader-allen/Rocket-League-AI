#!/usr/bin/env bash
# THE ONLY COMMAND YOU NEED.
#
#   local smoke test :  N_PROC=8  ROUND=5    bash experiments/run_all.sh
#   on the server    :  N_PROC=12 ROUND=3000 bash experiments/run_all.sh
#
# Identical script both times. ROUND is the only thing that changes.
#   1 round = 100,000 timesteps.  ROUND=5 is ~1 minute, ROUND=3000 is ~6 hours.
#
# Optional:
#   CORE_ONLY=1   run only 00-02 (the three required experiments)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/params.sh"

CORE_ONLY="${CORE_ONLY:-0}"
EXPERIMENTS=(00_baseline.sh 01_ground.sh 02_aerial.sh)
[ "$CORE_ONLY" = "1" ] || EXPERIMENTS+=(03_full_info.sh 04_no_anneal.sh 05_aux_encoder.sh)

echo "output -> $OUT"
echo "queue  -> ${EXPERIMENTS[*]}"

FAILED=()
for exp in "${EXPERIMENTS[@]}"; do
    # A blocked or failing ablation must not take the rest of the queue down.
    if bash "$HERE/$exp"; then :; else FAILED+=("$exp"); echo "!! $exp failed, continuing"; fi
done

echo
echo "============================================================"
echo "  done.  results: $OUT"
[ ${#FAILED[@]} -gt 0 ] && echo "  FAILED: ${FAILED[*]}"
echo "============================================================"
echo "pull results back:   runpodctl send $OUT/metrics"
echo "AND STOP THE POD."
