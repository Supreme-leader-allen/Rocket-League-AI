#!/usr/bin/env bash
# THE single command: ROUND=3000 ./exp.sh
#
# Does everything needed to go from a freshly-deployed Pod to a running
# experiment in one call: builds a persistent venv (skipped automatically
# once it exists, so re-running after a pause/resume costs nothing extra),
# verifies the environment, benchmarks this machine, and launches training.
#
# Env vars:
#   ROUND=<n>        forwarded to experiments/*.sh via params.sh (default 5)
#   CONDITION=<name> which experiments/<name>.sh to run (default 01_ground;
#                    use CONDITION=run_all for the full six-condition matrix)
#   N_PROC=<n>       skip auto-benchmarking and use this value directly
#   BENCH_ARGS=...   extra flags passed to Benchmark.py, e.g. BENCH_ARGS=--quick
#   VENV_DIR=<path>  where the persistent venv lives (default ./venv --
#                    put this repo on your network volume and it survives
#                    Pod stop/restart automatically, per docs/COMPUTE.md)
#
# What this does NOT replace: docs/ISSUES.md and docs/COMPUTE.md are still
# worth reading once. This script automates the mechanical setup steps, not
# the judgment calls those two files document.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

VENV_DIR="${VENV_DIR:-$HERE/venv}"
CONDITION="${CONDITION:-01_ground}"

echo "============================================================"
echo "  exp.sh: repo=$HERE  venv=$VENV_DIR  condition=$CONDITION"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. One-time environment setup. Skipped automatically on every run after
#    the first, which is the whole point of putting VENV_DIR on the network
#    volume: a Pod stop wipes the container disk (pip installs vanish) but
#    NOT the network volume, so a resumed Pod hits the "reusing" branch below
#    and skips straight past the slow part instead of reinstalling torch/
#    rocketsim from scratch every time.
# ---------------------------------------------------------------------------
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[1/4] No venv at $VENV_DIR -- building it now (one-time cost)."
    python3 -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --no-cache-dir -r requirements.txt
else
    echo "[1/4] Reusing existing venv at $VENV_DIR (skipping install)."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# ---------------------------------------------------------------------------
# 2. Same sanity checks experiments/setup_pod.sh runs -- GPU visible to the
#    driver, numpy<2, CUDA visible to torch, RocketSim loads. Fails loudly
#    and stops here rather than burning GPU-hours on a misconfigured run.
# ---------------------------------------------------------------------------
echo "[2/4] Verifying environment"
nvidia-smi > /dev/null || { echo "FAIL: no GPU visible. Wrong pod type." >&2; exit 1; }
python - <<'PY'
import numpy, torch
assert numpy.__version__.startswith("1."), f"FAIL: rlgym-rocket-league needs numpy<2, got {numpy.__version__}"
assert torch.cuda.is_available(), "FAIL: CUDA not visible to torch (training would silently fall back to CPU)"
from rlgym.rocket_league.sim import RocketSimEngine
RocketSimEngine()
print(f"  OK -- torch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")
PY

# ---------------------------------------------------------------------------
# 3. Benchmark, unless N_PROC was already supplied. This is params.sh's
#    Layer 2 ("MEASURED, not chosen"). Auto-applying it here trades away the
#    manual look-before-you-commit checkpoint params.sh normally wants --
#    the full Benchmark.py output is still printed below so you can review
#    the bottleneck verdict after the fact, but nothing pauses for you to
#    read it before training starts. If you want that pause back, run
#    `python Benchmark.py` yourself first and pass N_PROC= explicitly.
# ---------------------------------------------------------------------------
if [ -z "${N_PROC:-}" ]; then
    echo "[3/4] N_PROC not set -- running Benchmark.py ${BENCH_ARGS:-}"
    BENCH_OUT="$(python Benchmark.py ${BENCH_ARGS:-})"
    echo "$BENCH_OUT"
    N_PROC="$(echo "$BENCH_OUT" | grep -oE 'Set PBT_N_PROC=[0-9]+' | grep -oE '[0-9]+' | tail -1)"
    if [ -z "$N_PROC" ]; then
        echo "FAIL: could not parse N_PROC out of Benchmark.py's output above -- set N_PROC= manually." >&2
        exit 1
    fi
    echo "  -> auto-selected N_PROC=$N_PROC"
else
    echo "[3/4] Using supplied N_PROC=$N_PROC (skipping benchmark)"
fi
export N_PROC

# ---------------------------------------------------------------------------
# 4. Launch. ROUND is read straight from the inherited environment by
#    params.sh inside experiments/$CONDITION.sh -- nothing to forward here.
# ---------------------------------------------------------------------------
echo "[4/4] Launching experiments/$CONDITION.sh  (N_PROC=$N_PROC ROUND=${ROUND:-5})"
bash "experiments/$CONDITION.sh"

echo
echo "============================================================"
echo "  exp.sh done. Pull results, then STOP THE POD:"
echo "    runpodctl send \${OUT:-output}/metrics"
echo "============================================================"
