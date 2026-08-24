#!/usr/bin/env bash
# Model A, ground 4v4. The main result.
set -euo pipefail
source "$(dirname "$0")/params.sh"
cd "$REPO_ROOT"
banner "01  Model A (ground 4v4)"

PBT_N_PROC="$N_PROC" \
PBT_TIMESTEP_LIMIT="$LIMIT" \
PBT_SAVE_EVERY_TS="$SAVE_EVERY" \
PBT_RUN_LABEL="01_ground" \
PBT_CHECKPOINT_DIR="$OUT/checkpoints/01_ground" \
PBT_CSV_PATH="$OUT/metrics/01_ground.csv" \
PBT_FITNESS_CSV_PATH="$OUT/metrics/01_ground_fitness.csv" \
    python Train_Ground.py 2>&1 | tee "$OUT/logs/01_ground.log"

echo "01 done -> $(latest_checkpoint_path "$OUT/checkpoints/01_ground" || echo 'NO CHECKPOINT SAVED')"
