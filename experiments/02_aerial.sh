#!/usr/bin/env bash
# Model B, aerial. Warm-starts from 01's weights.
set -euo pipefail
source "$(dirname "$0")/params.sh"
cd "$REPO_ROOT"
banner "02  Model B (aerial, warm-started from 01)"

CKPT=$(latest_checkpoint_path "$OUT/checkpoints/01_ground") || {
    echo "FAIL: no checkpoint from 01. Run 01_ground.sh first." >&2; exit 1; }

# Train_Aerial.py now derives its own absolute timestep_limit internally
# (LOADED_TIMESTEPS, read off $CKPT's own digit-named folder, plus
# AERIAL_ADDITIONAL_TIMESTEPS below) -- fixed docs/ISSUES.md P1 at the
# Python layer, so this script no longer has to compute an absolute
# B_LIMIT itself. Passing an absolute AERIAL_TIMESTEP_LIMIT (the
# previous interface) is silently ignored by the current Train_Aerial.py
# and would have made this run train with the wrong budget.
echo "warm start : $CKPT"
echo "this run   : trains ${LIMIT} new steps beyond wherever $CKPT left off"

# RUN_LABEL/CSV_PATH/FITNESS_CSV_PATH used to be bare constants in
# Train_Aerial.py with no override, so this script's real output was
# unavoidably labeled "model_b_aerial" -- not "02_aerial" like every
# other script's own filename -- and had to be written to a
# repo-relative metrics/ dir and mv'd into $OUT afterward instead of
# landing directly in $OUT/metrics/ the way PBT_CSV_PATH already lets
# 01_ground.sh do it. Confirmed by running this real pipeline (not
# caught by Plot_Results.py's earlier synthetic-CSV testing). Now
# exposed the same way, so this writes straight into $OUT/metrics/ with
# a consistent label, no mv step needed.
AERIAL_CHECKPOINT_TO_LOAD="$CKPT" \
AERIAL_ADDITIONAL_TIMESTEPS="$LIMIT" \
AERIAL_N_PROC="$N_PROC" \
AERIAL_SAVE_EVERY_TS="$SAVE_EVERY" \
AERIAL_CHECKPOINT_DIR="$OUT/checkpoints/02_aerial" \
AERIAL_RUN_LABEL="02_aerial" \
AERIAL_CSV_PATH="$OUT/metrics/02_aerial.csv" \
AERIAL_FITNESS_CSV_PATH="$OUT/metrics/02_aerial_fitness.csv" \
    python Train_Aerial.py 2>&1 | tee "$OUT/logs/02_aerial.log"

echo "02 done"
