#!/usr/bin/env bash
# Population-based training (Liu et al. 2019 / FTW-style evolution) over
# Train_Ground.py. The evolution pillar -- previously only reachable via a
# bare `python Pbt.py`, not wired into the experiment matrix at all
# (docs/ISSUES.md). Opt-in only: see run_all.sh's INCLUDE_PBT flag, since a
# full PBT pass costs meaningfully more compute than any single ablation
# here (population x generations x timesteps-per-generation, not just
# timesteps-per-run) -- you should consciously choose to pay that, not have
# it silently added to every run_all.sh invocation.
#
# ROUND means the same thing here as everywhere else: 1 round = 100,000
# timesteps, and ROUND is the total EACH MEMBER trains for. It's split evenly
# across generations. So ROUND=3000 is 300M per member, comparable to one
# 01_ground.sh run, and 1.2B timesteps across a population of 4. It used to
# be per generation, which made ROUND=3000 silently mean 12 billion timesteps.
#
# POPULATION_SIZE / N_GENERATIONS stay small (2 / 2) at or below the local
# smoke-test ROUND (5). Override PBT_POPULATION_SIZE / PBT_N_GENERATIONS
# explicitly to change them.
set -euo pipefail
source "$(dirname "$0")/params.sh"
cd "$REPO_ROOT"
banner "06  PBT (population-based training / evolution)"

if ! ls "$OUT"/metrics/baseline_random.*.csv >/dev/null 2>&1; then
    echo "no baseline in $OUT/metrics -- running 00_baseline.sh first"
    bash "$(dirname "$0")/00_baseline.sh"
fi

if [ "$ROUND" -le 5 ]; then
    PBT_POPULATION_SIZE="${PBT_POPULATION_SIZE:-2}"
    PBT_N_GENERATIONS="${PBT_N_GENERATIONS:-2}"
else
    PBT_POPULATION_SIZE="${PBT_POPULATION_SIZE:-4}"
    PBT_N_GENERATIONS="${PBT_N_GENERATIONS:-10}"
fi

GENERATION_TIMESTEPS=$(( LIMIT / PBT_N_GENERATIONS ))
echo "population=$PBT_POPULATION_SIZE  generations=$PBT_N_GENERATIONS  generation_timesteps=$GENERATION_TIMESTEPS  n_proc_per_member=$N_PROC"

PBT_OUT_DIR="$OUT" \
PBT_POPULATION_SIZE="$PBT_POPULATION_SIZE" \
PBT_N_GENERATIONS="$PBT_N_GENERATIONS" \
PBT_GENERATION_TIMESTEPS="$GENERATION_TIMESTEPS" \
PBT_N_PROC_PER_MEMBER="$N_PROC" \
    python Pbt.py 2>&1 | tee "$OUT/logs/06_pbt.log"

echo "06 done -> $OUT/checkpoints/pbt/pbt_member_*, $OUT/metrics/pbt_member_*.csv, $OUT/metrics/pbt/generations.jsonl"
