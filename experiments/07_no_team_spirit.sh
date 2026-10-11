#!/usr/bin/env bash
# Ablation: no team spirit. Identical to 01 except every car keeps its own
# shaping reward throughout. Compare its coordination metrics (overcommit,
# spacing, goal-side rate) against 01's to measure what team spirit does.
set -euo pipefail
source "$(dirname "$0")/params.sh"
cd "$REPO_ROOT"
banner "07  ablation: no team spirit"

TEAM_SPIRIT_START=0 TEAM_SPIRIT_FINAL=0 \
PBT_N_PROC="$N_PROC" PBT_TIMESTEP_LIMIT="$LIMIT" PBT_SAVE_EVERY_TS="$SAVE_EVERY" \
PBT_RUN_LABEL="07_no_team_spirit" \
PBT_CHECKPOINT_DIR="$OUT/checkpoints/07_no_team_spirit" \
PBT_CSV_PATH="$OUT/metrics/07_no_team_spirit.csv" \
PBT_FITNESS_CSV_PATH="$OUT/metrics/07_no_team_spirit_fitness.csv" \
    python Train_Ground.py 2>&1 | tee "$OUT/logs/07_no_team_spirit.log"
echo "07 done"
