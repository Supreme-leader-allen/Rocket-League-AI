#!/usr/bin/env bash
# Model B, aerial. Warm-starts from 01's weights.
set -euo pipefail
source "$(dirname "$0")/params.sh"
cd "$REPO_ROOT"
banner "02  Model B (aerial, warm-started from 01)"

CKPT=$(latest_checkpoint_path "$OUT/checkpoints/01_ground") || {
    echo "FAIL: no checkpoint from 01. Run 01_ground.sh first." >&2; exit 1; }
DONE_TS=$(latest_checkpoint_ts "$OUT/checkpoints/01_ground")

# Learner restores cumulative_timesteps from the checkpoint, so this limit must
# be ABSOLUTE and above where 01 stopped. Passing $LIMIT directly would make
# this run exit instantly having trained nothing (docs/ISSUES.md P1).
B_LIMIT=$(( DONE_TS + LIMIT ))
echo "warm start : $CKPT"
echo "01 reached : ${DONE_TS} steps"
echo "this run   : trains to ${B_LIMIT} total (= ${LIMIT} new steps)"

AERIAL_CHECKPOINT_TO_LOAD="$CKPT" \
AERIAL_TIMESTEP_LIMIT="$B_LIMIT" \
AERIAL_N_PROC="$N_PROC" \
AERIAL_SAVE_EVERY_TS="$SAVE_EVERY" \
AERIAL_CHECKPOINT_DIR="$OUT/checkpoints/02_aerial" \
    python Train_Aerial.py 2>&1 | tee "$OUT/logs/02_aerial.log"

mv metrics/aerial_*.csv "$OUT/metrics/" 2>/dev/null || true
echo "02 done"
