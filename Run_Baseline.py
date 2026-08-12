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

Verify before trusting the output:
- RLGymV2GymWrapper's exact reset()/step() signature. This assumes a
  gym-like reset() -> obs (or (obs, info)) and step(actions) ->
  (obs, reward, terminated, truncated, info), which is the standard
  Gymnasium convention rlgym_ppo's wrapper is built to match, but wasn't
  directly confirmed against source. If reset()/step() return something
  else in your installed version, adjust the loop below accordingly --
  the error message on first run will make it obvious what's mismatched.
"""

import argparse

import numpy as np

from Train_Ground import build_rlgym_v2_env


def run(n_episodes: int) -> None:
    # Same reward/obs/action/state-mutator config as train_ground.py --
    # only the run_label/csv_path differ, via build_rlgym_v2_env's
    # parameters (see train_ground.py's docstring on why those are
    # parameterized rather than hardcoded).
    env = build_rlgym_v2_env(run_label="baseline_random", csv_path="metrics/baseline_random.csv")

    for ep in range(n_episodes):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        agents = list(obs.keys())

        terminated = truncated = False
        while not (terminated or truncated):
            actions = {}
            for agent in agents:
                action_space = env.action_space(agent)
                # Discrete LookupTableAction space -- sample a random valid index.
                # If action_space exposes .sample() (Gymnasium Discrete/Box),
                # prefer that; fall back to randint over its size otherwise.
                if hasattr(action_space, "sample"):
                    actions[agent] = action_space.sample()
                else:
                    actions[agent] = np.random.randint(action_space)

            step_result = env.step(actions)
            obs, reward, terminated_d, truncated_d, info = step_result
            terminated = all(terminated_d.values()) if isinstance(terminated_d, dict) else terminated_d
            truncated = all(truncated_d.values()) if isinstance(truncated_d, dict) else truncated_d

        if (ep + 1) % 20 == 0:
            print(f"baseline: {ep + 1}/{n_episodes} episodes")

    print(f"Done. Metrics written to metrics/baseline_random.<pid>.csv "
          f"(run_label=baseline_random) -- see metrics.py for the per-process shard note.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()
    run(args.episodes)