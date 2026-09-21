#!/usr/bin/env bash
# Ablation: shaping never decays. Supports the claim that annealing dense
# shaping to zero is what produces coordination (Liu et al.).
set -euo pipefail
source "$(dirname "$0")/params.sh"
cd "$REPO_ROOT"
banner "04  ablation: no reward annealing"

if ! grep -q 'ANNEAL_TIMESTEPS = float(os.environ' Train_Ground.py; then
    cat >&2 <<'MSG'
BLOCKED: Train_Ground.py hardcodes ANNEAL_TIMESTEPS as a literal. Expose it first:

    ANNEAL_TIMESTEPS = float(os.environ.get("ANNEAL_TIMESTEPS", 200_000_000))

Note 0 is NOT the value you want -- AnnealedCombinedReward treats <=0 as
"already fully annealed". Use a huge value so the anneal never progresses.
MSG
    exit 1
fi

# Annealing is now driven by (estimated) cumulative timesteps, not
# wall-clock seconds (docs/ISSUES.md P3) -- set it far above $LIMIT so
# this run's progress never reaches it and shaping stays constant.
ANNEAL_TIMESTEPS=1000000000 \
PBT_N_PROC="$N_PROC" PBT_TIMESTEP_LIMIT="$LIMIT" PBT_SAVE_EVERY_TS="$SAVE_EVERY" \
PBT_RUN_LABEL="04_no_anneal" \
PBT_CHECKPOINT_DIR="$OUT/checkpoints/04_no_anneal" \
PBT_CSV_PATH="$OUT/metrics/04_no_anneal.csv" \
PBT_FITNESS_CSV_PATH="$OUT/metrics/04_no_anneal_fitness.csv" \
    python Train_Ground.py 2>&1 | tee "$OUT/logs/04_no_anneal.log"
echo "04 done"
