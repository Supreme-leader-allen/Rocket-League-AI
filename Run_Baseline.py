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
    python run_baseline.py --episodes 200

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

import numpy as np

from Train_Ground import build_rlgym_v2_env


def run(n_episodes: int) -> None:
    # Same reward/obs/action/state-mutator config as train_ground.py --
    # only the run_label/csv_path differ, via build_rlgym_v2_env's
    # parameters (see train_ground.py's docstring on why those are
    # parameterized rather than hardcoded).
    env = build_rlgym_v2_env(
        run_label="baseline_random",
        csv_path="metrics/baseline_random.csv",
        fitness_csv_path="metrics/baseline_random_fitness.csv",
    )
    n_actions = env.action_space.n

    for ep in range(n_episodes):
        obs = env.reset()  # np.ndarray, shape (n_agents, obs_dim)
        n_agents = obs.shape[0]

        terminated = truncated = False
        while not (terminated or truncated):
            actions = np.random.randint(0, n_actions, size=(n_agents, 1))
            obs, reward, terminated, truncated, info = env.step(actions)

        if (ep + 1) % 20 == 0:
            print(f"baseline: {ep + 1}/{n_episodes} episodes")

    print(f"Done. Metrics written to metrics/baseline_random.<pid>.csv "
          f"(run_label=baseline_random) -- see metrics.py for the per-process shard note.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()
    run(args.episodes)