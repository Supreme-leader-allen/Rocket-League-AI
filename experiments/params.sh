#!/usr/bin/env bash
# All experiment parameters, in three layers. Sourced by every script here.
# You only ever edit LAYER 3.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# =============================================================================
# LAYER 1 — FROZEN. Not variables. Listed so you know not to touch them.
# =============================================================================
#   policy/critic layers  [2048, 2048, 1024, 1024]   change -> Model B can't load Model A
#   zero_padding          4                          change -> observation size breaks
#   team size             4v4                        change -> different research question
#   action space          full LookupTableAction     change -> "no abstraction" claim dies
#   GoalReward            10.0, team-shared          this IS the credit-assignment mechanism

# =============================================================================
# LAYER 2 — MEASURED, not chosen. Run Benchmark.py to get it.
# =============================================================================
# Parallel environment processes. NOT your core count, and NOT Train_Ground.py's
# default of 32 (that came from a tutorial).
N_PROC="${N_PROC:?run 'python Benchmark.py' first, then: export N_PROC=<saturation point>}"

# =============================================================================
# LAYER 3 — YOURS. The only layer you edit.
# =============================================================================
# 1 round = 100,000 timesteps = one PPO iteration.
#
#   local smoke test :  ROUND=5      (~1 minute,  does the pipeline work?)
#   real run         :  ROUND=3000   (~6 hours,   300M timesteps)
#
# Nothing else changes between local and server. Same scripts, bigger ROUND.
ROUND="${ROUND:-5}"

# Where results go. /workspace is RunPod's persistent volume; anything written
# outside it is lost when the pod stops.
if [ -d /workspace ]; then
    OUT="${OUT:-/workspace/output}"
else
    OUT="${OUT:-$REPO_ROOT/output}"
fi

# =============================================================================
# Derived. Don't edit.
# =============================================================================
TS_PER_ROUND=100000
LIMIT=$(( ROUND * TS_PER_ROUND ))

# Learner has NO save-on-exit: it only saves every save_every_ts. If that
# interval is >= the limit, the run finishes having saved nothing at all.
SAVE_EVERY=$(( LIMIT / 8 ))
[ "$SAVE_EVERY" -lt 1 ] && SAVE_EVERY=1

mkdir -p "$OUT/checkpoints" "$OUT/metrics" "$OUT/logs"

# Highest checkpoint number under a save folder == that run's cumulative
# timestep count. Needed because Learner restores cumulative_timesteps on load
# and loops `while cumulative < timestep_limit`, so a warm-started run needs an
# ABSOLUTE limit above where the previous one stopped, or it exits immediately
# having trained nothing (docs/ISSUES.md P1).
latest_checkpoint_ts() {
    local base="$1" run
    run=$(ls -d "${base}"-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    [ -n "$run" ] || return 1
    ls "$run" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1
}

latest_checkpoint_path() {
    local base="$1" run ts
    run=$(ls -d "${base}"-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    [ -n "$run" ] || return 1
    ts=$(ls "$run" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
    [ -n "$ts" ] || return 1
    echo "$run/$ts"
}

banner() {
    echo
    echo "============================================================"
    echo "  $1"
    echo "  ROUND=$ROUND  ($LIMIT timesteps)   n_proc=$N_PROC"
    echo "============================================================"
}
