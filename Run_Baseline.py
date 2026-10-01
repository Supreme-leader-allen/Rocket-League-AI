"""
Runs N episodes with random actions -- no learned policy, no training --
and logs the same CoordinationMetrics used during training, tagged
run_label="baseline_random". This is the literal "baseline model" from
your research plan: a policy with no learned coordination at all, for
comparison against the trained agent via analyze.py's t-test.

Deliberately needs no trained checkpoint and no FrozenPolicy machinery
(unlike the self-play opponent pool in self_play.py) -- random actions
require nothing but the action space itself, so this only depends on
things already confirmed working in train_ground.py.

Run with:
    python Run_Baseline.py --episodes 400 --workers 8

--workers runs episodes in parallel processes (each writes its own CSV
shard, same per-process sharding as training). This is why the baseline
was never run before: single-process, the old experiments/00_baseline.sh
asked for ROUND*2 = 6000 episodes at ROUND=3000, roughly 5+ hours on one
core. A few hundred episodes is plenty for a per-episode mean/SEM
comparison, and 00_baseline.sh now caps it.

Verified against rlgym_ppo.util.rlgym_v2_gym_wrapper.RLGymV2GymWrapper's actual
source (it does NOT follow the agent-ID-dict-keyed Gymnasium multi-agent
convention this originally assumed):
- reset() returns a plain np.ndarray of shape (n_agents, obs_dim) -- not a
  dict, not a (obs, info) tuple. There is no per-agent key at all; agent
  identity is only recoverable via the wrapper's internal agent_map (row
  index -> AgentID), which this script doesn't need since it just fires
  random actions at every row.
- action_space is a single shared gym.spaces.Discrete attribute (all
  agents share one action space in this project), not a per-agent
  callable -- env.action_space(agent) raised TypeError: 'Discrete' object
  is not callable, confirmed by running it.
- step(actions) takes actions as a plain (n_agents, 1) array indexed by
  row position (matching agent_map order from the last reset()), not an
  AgentID-keyed dict -- confirmed against the wrapper's step() source,
  which does `for i in range(len(actions)): ... action_dict[agent_map[i]]
  = actions[i]`. It returns (obs, rews, done, truncated, info) where rews
  is a plain list and done/truncated are already-aggregated bools (the
  wrapper ORs across all agents internally), not per-agent dicts.
"""

import argparse
import multiprocessing as mp
import os

import numpy as np


def _run_worker(args) -> int:
    n_episodes, metrics_dir = args
    from Train_Ground import build_rlgym_v2_env

    # Same reward/obs/action/state-mutator config as Train_Ground.py --
    # only the run_label/csv_path differ, via build_rlgym_v2_env's
    # parameters (see Train_Ground.py's docstring on why those are
    # parameterized rather than hardcoded).
    env = build_rlgym_v2_env(
        run_label="baseline_random",
        csv_path=os.path.join(metrics_dir, "baseline_random.csv"),
        fitness_csv_path=os.path.join(metrics_dir, "baseline_random_fitness.csv"),
    )
    n_actions = env.action_space.n

    for _ in range(n_episodes):
        obs = env.reset()  # np.ndarray, shape (n_agents, obs_dim)
        n_agents = obs.shape[0]
        terminated = truncated = False
        while not (terminated or truncated):
            actions = np.random.randint(0, n_actions, size=(n_agents, 1))
            obs, reward, terminated, truncated, info = env.step(actions)

    # CoordinationMetrics/FitnessTracker only write an episode when the
    # NEXT one starts (see Metrics.py), so one more reset flushes the last.
    env.reset()
    return n_episodes


def run(n_episodes: int, workers: int, metrics_dir: str) -> None:
    workers = max(1, min(workers, n_episodes))
    shares = [n_episodes // workers + (1 if i < n_episodes % workers else 0) for i in range(workers)]
    done = 0
    if workers == 1:
        done = _run_worker((shares[0], metrics_dir))
    else:
        with mp.Pool(workers) as pool:
            for n in pool.imap_unordered(_run_worker, [(n, metrics_dir) for n in shares]):
                done += n
                print(f"baseline: {done}/{n_episodes} episodes")

    print(f"Done. Metrics written to {metrics_dir}/baseline_random.<pid>.csv "
          f"(run_label=baseline_random) -- see Metrics.py for the per-process shard note.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--metrics-dir", default="metrics")
    args = parser.parse_args()
    run(args.episodes, args.workers, args.metrics_dir)
