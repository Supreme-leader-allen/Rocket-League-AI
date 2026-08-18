"""
Measures this machine's actual training throughput and prints a concrete
configuration recommendation. Run this BEFORE choosing hardware, sizing a
cloud instance, or estimating how long a run will take -- see COMPUTE.md
for why estimating from intuition goes wrong on this project specifically.

    python Benchmark.py              # full sweep, ~2-4 minutes
    python Benchmark.py --quick      # skip the multi-process sweep
    python Benchmark.py --procs 1,8,16,32   # test specific process counts

What it measures:

1. Single-process environment throughput, using this project's real env
   (PartialInfoObs + the full reward stack + HumanlikeAction), not a
   stripped-down stand-in. Also measures DefaultObs for comparison, which
   is what tells you how much the partial-information layer is costing.
2. Multi-process scaling, to find where collection saturates. This is the
   number PBT_N_PROC should be set to. It is NOT simply your core count --
   scaling flattens before that.
3. PPO update cost per iteration, on whichever device torch will actually
   use, with the exact layer sizes and batch/epoch/buffer settings read
   out of Train_Ground.py (so this stays honest if you change them).
4. Peak device memory, which is what you should size a GPU against.

It then combines all four into an end-to-end estimate and says which side
is the bottleneck, because the answer differs by machine and determines
which optimization is worth doing first.

Caveats, stated up front so nobody over-trusts the output:
- Collection is measured with workers running flat out. Under rlgym_ppo
  the workers block on the central inference process, so real throughput
  lands below this. Treat the estimate as an optimistic ceiling.
- Inter-process communication is not modeled at all. Above roughly
  30-40k SPS it becomes the real limit and these numbers get too rosy.
- The update figure assumes a full experience buffer. Early iterations,
  before the buffer fills, are cheaper.
"""

import argparse
import os
import re
import sys
import time
import multiprocessing as mp

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(PROJECT_DIR, "Train_Ground.py")

N_AGENTS = 8  # 4v4, fixed by FixedTeamSizeMutator in build_rlgym_v2_env


# --------------------------------------------------------------------------
# Read the real config out of Train_Ground.py rather than duplicating it.
# Duplicated constants drift; a benchmark that measures last month's config
# is worse than no benchmark. If a pattern stops matching we fail loudly
# instead of silently falling back to a default.
# --------------------------------------------------------------------------

def read_train_config():
    with open(TRAIN_SCRIPT) as f:
        src = f.read()

    def find(pattern, cast=int):
        m = re.search(pattern, src)
        if m is None:
            raise RuntimeError(
                f"Could not find {pattern!r} in Train_Ground.py. The training "
                f"script changed shape -- update Benchmark.py's read_train_config()."
            )
        return cast(m.group(1).replace("_", ""))

    def find_list(name):
        m = re.search(rf"{name}\s*=\s*\[([\d,\s]+)\]", src)
        if m is None:
            raise RuntimeError(f"Could not find {name} in Train_Ground.py.")
        return [int(x) for x in m.group(1).replace(" ", "").rstrip(",").split(",")]

    return dict(
        exp_buffer_size=find(r"exp_buffer_size=([\d_]+)"),
        ppo_batch_size=find(r"ppo_batch_size=([\d_]+)"),
        ppo_minibatch_size=find(r"ppo_minibatch_size=([\d_]+)"),
        ppo_epochs=find(r"ppo_epochs=([\d_]+)"),
        ts_per_iteration=find(r"ts_per_iteration=([\d_]+)"),
        timestep_limit=find(r'PBT_TIMESTEP_LIMIT",\s*([\d_]+)'),
        policy_layers=find_list("POLICY_LAYER_SIZES"),
        critic_layers=find_list("CRITIC_LAYER_SIZES"),
    )


# --------------------------------------------------------------------------
# 1 + 2. Environment throughput
# --------------------------------------------------------------------------

def build_env(obs_mode="project", metrics_dir=None):
    """
    Mirrors build_rlgym_v2_env in Train_Ground.py. Deliberately does NOT
    import it: that function wraps the env in rlgym_ppo's RLGymV2GymWrapper
    and writes metrics CSVs into the real run's paths. We want the bare env
    and throwaway CSV paths. Keep the two in sync if the env config changes.
    """
    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league.done_conditions import (
        GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition)
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import (
        MutatorSequence, FixedTeamSizeMutator, KickoffMutator)
    from rlgym.rocket_league.obs_builders import DefaultObs

    sys.path.insert(0, PROJECT_DIR)
    from Rewards import (SpeedTowardBallReward, InAirReward,
                         VelocityBallToGoalReward, AnnealedCombinedReward)
    from Observation import PartialInfoObs
    from Actions import HumanlikeAction
    from Metrics import CoordinationMetrics, FitnessTracker

    metrics_dir = metrics_dir or os.path.join(PROJECT_DIR, ".benchmark_tmp")
    os.makedirs(metrics_dir, exist_ok=True)

    shaping = AnnealedCombinedReward(
        weighted_rewards=[(SpeedTowardBallReward(), 0.01, 0.0),
                          (VelocityBallToGoalReward(), 0.1, 0.02),
                          (InAirReward(), 0.002, 0.0)],
        anneal_seconds=2 * 60 * 60)

    entries = [(shaping, 1.0)]
    if obs_mode == "project":
        entries.append((FitnessTracker(GoalReward(),
                                       csv_path=os.path.join(metrics_dir, "fit.csv"),
                                       run_label="benchmark"), 10.0))
        entries.append((CoordinationMetrics(csv_path=os.path.join(metrics_dir, "coord.csv"),
                                            run_label="benchmark"), 0.0))
        obs_builder = PartialInfoObs(zero_padding=4)
    else:
        entries.append((GoalReward(), 10.0))
        obs_builder = DefaultObs(zero_padding=4)

    return RLGym(
        state_mutator=MutatorSequence(FixedTeamSizeMutator(blue_size=4, orange_size=4),
                                      KickoffMutator()),
        obs_builder=obs_builder,
        action_parser=HumanlikeAction(repeats=8, reaction_delay_ticks=3),
        reward_fn=CombinedReward(*entries),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(NoTouchTimeoutCondition(timeout_seconds=30),
                                     TimeoutCondition(timeout_seconds=300)),
        transition_engine=RocketSimEngine(),
        renderer=None)


def step_env(env, n_steps):
    """Returns agent-timesteps per second. Random actions: the policy is not
    what we're measuring here, and a real policy would add its cost twice."""
    import numpy as np
    obs = env.reset()
    agents = list(obs.keys())
    t0 = time.perf_counter()
    for _ in range(n_steps):
        actions = {a: np.array([np.random.randint(0, 90)]) for a in agents}
        obs, _, term, trunc = env.step(actions)
        if any(term.values()) or any(trunc.values()):
            obs = env.reset()
    dt = time.perf_counter() - t0
    return n_steps * len(agents) / dt


def _scale_worker(n_steps, idx, q):
    # Runs in a spawned child. Benchmark.py is re-imported as __mp_main__ with
    # the __main__ guard false, so build_env/step_env are already in scope.
    env = build_env("project", metrics_dir=os.path.join(PROJECT_DIR, ".benchmark_tmp", str(idx)))
    q.put(step_env(env, n_steps))


def measure_scaling(proc_counts, n_steps=200):
    results = []
    for nproc in proc_counts:
        q = mp.Queue()
        ps = [mp.Process(target=_scale_worker, args=(n_steps, i, q)) for i in range(nproc)]
        for p in ps:
            p.start()
        rates = [q.get() for _ in ps]
        for p in ps:
            p.join()
        total = sum(rates)
        results.append((nproc, total))
        print(f"    {nproc:3d} processes -> {total:9,.0f} SPS aggregate "
              f"({total / nproc:6,.0f} per process)")
        # Stop early once adding processes buys less than 5%: everything past
        # the knee is wasted memory and context-switching.
        if len(results) >= 2 and total < results[-2][1] * 1.05:
            print(f"    (saturated -- stopping sweep)")
            break
    return results


# --------------------------------------------------------------------------
# 3 + 4. PPO update cost and peak device memory
# --------------------------------------------------------------------------

def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # rlgym_ppo's Learner resolves device="auto" to cuda-or-cpu and never
        # selects mps. We benchmark it anyway so you can see what the hardware
        # could do, but the real run will be on CPU unless you patch Learner.
        return "mps", "Apple GPU (NOTE: rlgym_ppo will NOT use this -- see COMPUTE.md)"
    return "cpu", "CPU only"


def measure_update(cfg, device, obs_size=212, n_actions=90):
    import torch
    import torch.nn as nn

    def mlp(sizes, out):
        mods, prev = [], obs_size
        for s in sizes:
            mods += [nn.Linear(prev, s), nn.ReLU()]
            prev = s
        return nn.Sequential(*mods, nn.Linear(prev, out))

    pol = mlp(cfg["policy_layers"], n_actions).to(device)
    val = mlp(cfg["critic_layers"], 1).to(device)
    popt = torch.optim.Adam(pol.parameters(), lr=1e-4)
    vopt = torch.optim.Adam(val.parameters(), lr=1e-4)

    mb = cfg["ppo_minibatch_size"]
    # PPOLearner.learn() calls exp.get_all_batches_shuffled(batch_size), which
    # walks the WHOLE buffer in chunks -- not one batch. Missing this is the
    # single easiest way to underestimate update cost by 3x on this config.
    n_batches = cfg["exp_buffer_size"] // cfg["ppo_batch_size"]
    n_slices = cfg["ppo_batch_size"] // mb
    total_slices = cfg["ppo_epochs"] * n_batches * n_slices

    obs = torch.randn(mb, obs_size, device=device)
    tgt = torch.zeros(mb, 1, dtype=torch.long, device=device)

    def one_slice():
        lp = pol(obs).log_softmax(-1)
        (-lp.gather(1, tgt).mean()).backward()
        ((val(obs).squeeze(-1)) ** 2).mean().backward()

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    for _ in range(2):
        one_slice()
    popt.zero_grad(); vopt.zero_grad()
    sync()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(cfg["ppo_epochs"]):
        for _ in range(n_batches):
            for _ in range(n_slices):
                one_slice()
            popt.step(); vopt.step()
            popt.zero_grad(); vopt.zero_grad()
    sync()
    update_s = time.perf_counter() - t0

    if device == "cuda":
        peak_gib = torch.cuda.max_memory_allocated() / 2 ** 30
    elif device == "mps":
        peak_gib = torch.mps.driver_allocated_memory() / 2 ** 30
    else:
        peak_gib = float("nan")

    n_params = (sum(p.numel() for p in pol.parameters())
                + sum(p.numel() for p in val.parameters()))
    return update_s, total_slices, peak_gib, n_params


def measure_inference(cfg, device, n_proc, obs_size=212, n_actions=90):
    import torch
    import torch.nn as nn

    def mlp(sizes, out):
        mods, prev = [], obs_size
        for s in sizes:
            mods += [nn.Linear(prev, s), nn.ReLU()]
            prev = s
        return nn.Sequential(*mods, nn.Linear(prev, out))

    pol = mlp(cfg["policy_layers"], n_actions).to(device)
    val = mlp(cfg["critic_layers"], 1).to(device)
    # Learner batches inference across min_inference_size envs; each env
    # contributes all 8 of its agents.
    batch = max(1, int(round(n_proc * 0.9))) * N_AGENTS
    x = torch.randn(batch, obs_size, device=device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    with torch.no_grad():
        for _ in range(10):
            pol(x); val(x)
        sync()
        t0 = time.perf_counter()
        for _ in range(200):
            pol(x); val(x)
        sync()
        per_call = (time.perf_counter() - t0) / 200
    return per_call, batch


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the multi-process sweep (assumes cpu_count-2 processes)")
    ap.add_argument("--procs", type=str, default=None,
                    help="comma-separated process counts to test, e.g. 1,8,16,32")
    ap.add_argument("--steps", type=int, default=250, help="env steps per sweep point")
    args = ap.parse_args()

    cfg = read_train_config()
    ncpu = os.cpu_count() or 4

    print("=" * 72)
    print("Rocket-League-AI throughput benchmark")
    print("=" * 72)
    print(f"  CPUs visible           : {ncpu}")
    print(f"  policy / critic layers : {cfg['policy_layers']} / {cfg['critic_layers']}")
    print(f"  exp_buffer_size        : {cfg['exp_buffer_size']:,}")
    print(f"  ppo_batch_size         : {cfg['ppo_batch_size']:,}")
    print(f"  ppo_minibatch_size     : {cfg['ppo_minibatch_size']:,}")
    print(f"  ppo_epochs             : {cfg['ppo_epochs']}")
    print(f"  ts_per_iteration       : {cfg['ts_per_iteration']:,}")
    print(f"  timestep_limit         : {cfg['timestep_limit']:,}")

    # ---- 1. single process ----
    print("\n[1/4] Single-process environment throughput")
    env = build_env("project")
    step_env(env, 20)
    sps_project = step_env(env, args.steps)
    print(f"    project env (PartialInfoObs + full reward stack) : {sps_project:9,.0f} SPS")
    env = build_env("default")
    step_env(env, 20)
    sps_default = step_env(env, args.steps)
    print(f"    DefaultObs, bare rewards (reference)             : {sps_default:9,.0f} SPS")
    obs_overhead = sps_default / sps_project
    print(f"    -> the partial-information layer costs {obs_overhead:.1f}x")

    # ---- 2. scaling ----
    if args.quick:
        best_procs, best_sps = max(1, ncpu - 2), sps_project * max(1, ncpu - 2)
        print(f"\n[2/4] Multi-process scaling  (--quick: assuming {best_procs} procs, extrapolated)")
    else:
        if args.procs:
            counts = [int(x) for x in args.procs.split(",")]
        else:
            counts, n = [], 1
            while n <= ncpu * 2:
                counts.append(n)
                n = n * 2 if n < 4 else n + max(4, ncpu // 4)
        print(f"\n[2/4] Multi-process scaling  (sweeping {counts})")
        results = measure_scaling(counts, n_steps=args.steps)
        best_procs, best_sps = max(results, key=lambda r: r[1])

    print(f"    -> collection saturates at {best_procs} processes, {best_sps:,.0f} SPS")

    # ---- 3 + 4. update ----
    device, device_name = pick_device()
    print(f"\n[3/4] PPO update cost   (device: {device} -- {device_name})")
    update_s, n_slices, peak_gib, n_params = measure_update(cfg, device)
    print(f"    {n_slices} forward+backward slices of {cfg['ppo_minibatch_size']:,}"
          f"  ({n_params / 1e6:.2f}M params total)")
    print(f"    {update_s:.2f}s per {cfg['ts_per_iteration']:,}-step iteration")
    print(f"    -> update alone caps throughput at {cfg['ts_per_iteration'] / update_s:,.0f} SPS")
    if peak_gib == peak_gib:  # not nan
        print(f"    peak device memory: {peak_gib:.2f} GiB")

    print(f"\n[4/4] Inference cost")
    infer_s, infer_batch = measure_inference(cfg, device, best_procs)
    calls = cfg["ts_per_iteration"] / infer_batch
    infer_total = infer_s * calls
    print(f"    {infer_s * 1000:.2f} ms per call (batch {infer_batch})"
          f" -> {infer_total:.2f}s per {cfg['ts_per_iteration']:,} steps")

    # ---- synthesis ----
    t_collect = cfg["ts_per_iteration"] / best_sps
    total = t_collect + infer_total + update_s
    sps = cfg["ts_per_iteration"] / total
    hours = cfg["timestep_limit"] / sps / 3600

    print("\n" + "=" * 72)
    print("RECOMMENDATION")
    print("=" * 72)
    print(f"  Set PBT_N_PROC={best_procs}")
    print(f"\n  Per {cfg['ts_per_iteration']:,}-step iteration:")
    print(f"    collection {t_collect:6.2f}s | inference {infer_total:5.2f}s | update {update_s:6.2f}s")
    print(f"  Estimated end-to-end: {sps:,.0f} SPS")
    print(f"  timestep_limit ({cfg['timestep_limit']:,}) would take ~{hours:.1f} hours"
          f"  ({hours / 24:.1f} days)")

    ratio = t_collect / update_s if update_s else float("inf")
    print(f"\n  Bottleneck: ", end="")
    if ratio > 1.5:
        print("COLLECTION (CPU side)")
        print(f"    Collection is {ratio:.1f}x the update. Worth doing, in order:")
        print(f"    1. Fix copy.deepcopy in Observation.py:_masked_state_for -- it is")
        print(f"       ~80% of step time and the partial-info layer costs {obs_overhead:.1f}x here.")
        print(f"    2. Add CPU cores. A faster GPU will buy you almost nothing.")
    elif ratio < 0.67:
        print("PPO UPDATE (GPU/compute side)")
        print(f"    The update is {1 / ratio:.1f}x collection. Worth doing, in order:")
        print(f"    1. Lower exp_buffer_size (currently {cfg['exp_buffer_size']:,}). Dropping it")
        print(f"       to {cfg['ppo_batch_size']:,} cuts update cost {cfg['exp_buffer_size'] // cfg['ppo_batch_size']}x."
              f" This changes sample reuse, so it is a")
        print(f"       training tradeoff, not a free win -- see COMPUTE.md.")
        print(f"    2. A faster GPU. Do NOT buy for VRAM: peak use is {peak_gib:.1f} GiB.")
        print(f"    Optimizing the env / deepcopy first would be wasted work here.")
    else:
        print("BALANCED")
        print(f"    Collection and update are within 1.5x of each other. Neither")
        print(f"    single-sided optimization will help much -- you need both, or")
        print(f"    accept the current rate.")

    print(f"\n  Reality check: these numbers ignore rlgym_ppo's inter-process")
    print(f"  communication, which becomes the real limit somewhere around")
    print(f"  30-40k SPS. Compare against the Steps per Second that the actual")
    print(f"  training run prints in its first minute, and trust that one.")
    print("=" * 72)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
