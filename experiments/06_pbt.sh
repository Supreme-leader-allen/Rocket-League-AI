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
# ROUND maps onto PBT's own, orthogonal scale like this:
#   - GENERATION_TIMESTEPS = ROUND * 100,000, exactly like every other
#     script's LIMIT -- "1 round" still means the same thing, just applied
#     per member per generation instead of per whole run.
#   - POPULATION_SIZE / N_GENERATIONS stay small (2 / 2) at or below the
#     documented local-smoke-test ROUND (5), so a bare `ROUND=5` invocation
#     actually finishes in a comparable ballpark to the other scripts
#     instead of taking population x generations times longer; above that
#     they step up to Pbt.py's own real-run defaults (4 members, 10
#     generations). Override PBT_POPULATION_SIZE / PBT_N_GENERATIONS
#     directly if you want something else at any ROUND.
#   - N_PROC_PER_MEMBER gets the full measured $N_PROC, not a fraction of
#     it: Pbt.py trains population members sequentially, one at a time
#     (confirmed by reading main()), never in parallel, so there's no
#     population-size split to account for the way there would be if
#     members trained concurrently.
set -euo pipefail
source "$(dirname "$0")/params.sh"
cd "$REPO_ROOT"
banner "06  PBT (population-based training / evolution)"

if [ "$ROUND" -le 5 ]; then
    PBT_POPULATION_SIZE="${PBT_POPULATION_SIZE:-2}"
    PBT_N_GENERATIONS="${PBT_N_GENERATIONS:-2}"
else
    PBT_POPULATION_SIZE="${PBT_POPULATION_SIZE:-4}"
    PBT_N_GENERATIONS="${PBT_N_GENERATIONS:-10}"
fi

echo "population=$PBT_POPULATION_SIZE  generations=$PBT_N_GENERATIONS  generation_timesteps=$LIMIT  n_proc_per_member=$N_PROC"

PBT_OUT_DIR="$OUT" \
PBT_POPULATION_SIZE="$PBT_POPULATION_SIZE" \
PBT_N_GENERATIONS="$PBT_N_GENERATIONS" \
PBT_GENERATION_TIMESTEPS="$LIMIT" \
PBT_N_PROC_PER_MEMBER="$N_PROC" \
    python Pbt.py 2>&1 | tee "$OUT/logs/06_pbt.log"

echo "06 done -> $OUT/checkpoints/pbt/pbt_member_*, $OUT/metrics/pbt_member_*.csv, $OUT/metrics/pbt/generations.jsonl"
