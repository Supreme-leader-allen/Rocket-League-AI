#!/usr/bin/env bash
# Ablation: full information. Same as 01 but the view cone is opened to 180
# degrees, so nothing is ever occluded. Supports "imperfect information is
# necessary."
set -euo pipefail
source "$(dirname "$0")/params.sh"
cd "$REPO_ROOT"
banner "03  ablation: full information"

if ! grep -q 'FOV_HALF_ANGLE_DEG = float(os.environ' Observation.py; then
    cat >&2 <<'MSG'
BLOCKED: Observation.py hardcodes FOV_HALF_ANGLE_DEG as a literal, so this
ablation cannot run without editing it. Expose it first:

    FOV_HALF_ANGLE_DEG = float(os.environ.get("FOV_HALF_ANGLE_DEG", 55.0))

Also see docs/ISSUES.md P0: the view cone does not currently work at all, so this
ablation is meaningless until that is fixed.
MSG
    exit 1
fi

FOV_HALF_ANGLE_DEG=180 \
PBT_N_PROC="$N_PROC" PBT_TIMESTEP_LIMIT="$LIMIT" PBT_SAVE_EVERY_TS="$SAVE_EVERY" \
PBT_RUN_LABEL="03_full_info" \
PBT_CHECKPOINT_DIR="$OUT/checkpoints/03_full_info" \
PBT_CSV_PATH="$OUT/metrics/03_full_info.csv" \
PBT_FITNESS_CSV_PATH="$OUT/metrics/03_full_info_fitness.csv" \
    python Train_Ground.py 2>&1 | tee "$OUT/logs/03_full_info.log"
echo "03 done"
