#!/usr/bin/env bash
# Ablation: Lucy-SKG auxiliary encoder. Three stages -- log data, train the
# encoder, retrain Model A with the encoded features concatenated on. The only
# ablation that already has env-var switches.
set -euo pipefail
source "$(dirname "$0")/params.sh"
cd "$REPO_ROOT"
banner "05  ablation: auxiliary encoder"

echo "--- 5a: logging observation data ---"
AUX_LOGGING=1 AUX_DATA_DIR="$OUT/aux_data" \
PBT_N_PROC="$N_PROC" PBT_TIMESTEP_LIMIT="$LIMIT" PBT_SAVE_EVERY_TS="$SAVE_EVERY" \
PBT_RUN_LABEL="05a_logging" \
PBT_CHECKPOINT_DIR="$OUT/checkpoints/05a_logging" \
PBT_CSV_PATH="$OUT/metrics/05a_logging.csv" \
PBT_FITNESS_CSV_PATH="$OUT/metrics/05a_logging_fitness.csv" \
    python Train_Ground.py 2>&1 | tee "$OUT/logs/05a_logging.log"

echo "--- 5b: training the encoder ---"
python Train_Auxiliary_Encoder.py --data-dir "$OUT/aux_data" \
    --sr-checkpoint "$OUT/auxiliary_encoder.pt" 2>&1 | tee "$OUT/logs/05b_encoder.log"

echo "--- 5c: retraining with the encoder ---"
AUX_ENCODER_CHECKPOINT="$OUT/auxiliary_encoder.pt" \
PBT_N_PROC="$N_PROC" PBT_TIMESTEP_LIMIT="$LIMIT" PBT_SAVE_EVERY_TS="$SAVE_EVERY" \
PBT_RUN_LABEL="05_aux" \
PBT_CHECKPOINT_DIR="$OUT/checkpoints/05_aux" \
PBT_CSV_PATH="$OUT/metrics/05_aux.csv" \
PBT_FITNESS_CSV_PATH="$OUT/metrics/05_aux_fitness.csv" \
    python Train_Ground.py 2>&1 | tee "$OUT/logs/05_aux.log"
echo "05 done"
