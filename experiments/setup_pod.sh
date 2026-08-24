#!/usr/bin/env bash
# Set up a fresh RunPod pod. Run once per pod:
#
#   bash experiments/setup_pod.sh
#
# Assumes a RunPod *PyTorch* template (CUDA + torch preinstalled). The bare
# Ubuntu template also works but adds a ~2.5 GB torch download.
#
# Every step ends in a check. If one fails, stop and fix it before training --
# the failures caught here (CUDA invisible, numpy 2.x) do not crash training,
# they just make it silently slow or silently wrong.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== 1/4  GPU visible to the driver? ==="
nvidia-smi || { echo "FAIL: no GPU. Wrong pod type." >&2; exit 1; }

echo
echo "=== 2/4  Installing dependencies ==="
# rlgym-rocket-league requires numpy<2. Many images ship numpy 2.x, so pip will
# downgrade it here -- that is expected, not an error.
pip install --no-cache-dir -r requirements.txt

echo
echo "=== 3/4  Verifying the environment ==="
python - <<'PY'
import numpy
print("numpy", numpy.__version__)
assert numpy.__version__.startswith("1."), \
    "FAIL: rlgym-rocket-league needs numpy<2.  pip install 'numpy<2'"

import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), (
    "FAIL: CUDA not visible. Training will NOT crash -- it will silently run "
    "the PPO update on CPU, roughly 20x slower."
)
print("gpu:", torch.cuda.get_device_name(0))

import rlgym_ppo                                     # noqa: F401
from rlgym.rocket_league.sim import RocketSimEngine
RocketSimEngine()   # loads the collision meshes bundled in the package
print("rlgym + rlgym_ppo + RocketSim OK")
PY

echo
echo "=== 4/4  Measuring this machine ==="
echo "Benchmark.py takes 2-4 minutes. Do NOT skip it: it prints the N_PROC value"
echo "and the throughput estimate you check the real run against."
python Benchmark.py

cat <<'MSG'

Setup complete. Next:

  N_PROC=<saturation point printed above> ROUND=3000 bash experiments/run_all.sh

In the first minute, compare the trainer's "Steps per Second" against the
estimate above. A large shortfall means a misconfiguration -- catching it now
instead of six hours in is the entire point of measuring.
MSG
