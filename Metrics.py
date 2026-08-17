"""
CoordinationMetrics: per-episode data collection for your research
comparison (trained agents vs. baseline, t-test). It hooks in as a
zero-weight entry in CombinedReward, so it's called every step alongside
training with no separate wiring needed and has zero effect on the
actual reward signal:

    reward_fn = CombinedReward(
        (shaping, 1.0),
        (GoalReward(), 10.0),
        (CoordinationMetrics(csv_path="metrics/ground_training.csv",
                              run_label="model_a_ground"), 0.0),
    )

Two metric families, matching your research plan's mechanical-skill vs.
coordination split, but with two metrics ADDED on the coordination side
beyond what you listed. Raw teammate distance alone can't tell "well
spread across the field" apart from "just nowhere near the ball" -- it's
ambiguous as a coordination signal by itself. overcommit_rate and
simultaneous_air_rate are added because they measure the specific
failure mode coordination is supposed to prevent: multiple teammates
converging on the same objective (the ball, or an aerial) instead of
splitting into first-man/second-man/third-man roles. Those two are
probably your strongest single coordination signals in this file --
lead with them if you're picking one or two metrics to highlight.

Mechanical skill (from your original list):
- avg_dist_to_ball: mean distance from each car to the ball, uu.
- avg_vel_toward_ball: mean per-agent velocity component toward the
  ball, uu/s (unnormalized version of SpeedTowardBallReward).
- air_time_fraction: fraction of steps each car spends airborne,
  averaged across the team. Proxy for aerial mechanical skill.

Coordination (teammate distance + boost from your original list, plus
two derived metrics):
- avg_teammate_pairwise_dist: mean distance between all teammate pairs,
  uu. Spacing / anti-clumping.
- boost_stddev_teammates: standard deviation of boost amount across
  teammates, averaged over the episode. A team executing rotations
  tends to have boost spread unevenly by role (the car about to
  challenge has less, the car rotating back has more); a team that
  isn't rotating tends to have boost more uniformly low.
- overcommit_rate: fraction of steps where 2+ teammates are within
  CHALLENGE_DIST_UU of the ball at once.
- simultaneous_air_rate: fraction of steps where 2+ teammates are
  airborne at once.

Output: one CSV row per completed episode. IMPORTANT: rlgym_ppo runs
n_proc separate environment processes in parallel, and build_rlgym_v2_env
(hence this class) gets constructed once per process -- so with
n_proc=32 you'd have 32 instances all wanting to write to the same
csv_path concurrently. Rather than rely on small-write append being
atomic on every OS (usually true on Linux/Mac, less guaranteed on
Windows), each instance writes to its own per-process shard,
`<csv_path stem>.<pid><ext>`, so there's never any concurrent-write risk
at all. analyze.py and run_baseline.py both read shards with a glob
pattern (`metrics/ground_training.*.csv`) rather than a single file --
don't point analyze.py at csv_path directly, use the glob form.

Caveat: the very last episode of a run is only flushed when the *next*
episode's reset() fires, so it's dropped if training stops mid-episode.
Negligible if you're logging thousands of episodes across 32 shards.
"""

import csv
import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState

CHALLENGE_DIST_UU = 750.0

CSV_COLUMNS = [
    "run_label", "episode_id", "episode_length_steps",
    "avg_dist_to_ball", "avg_vel_toward_ball", "air_time_fraction",
    "avg_teammate_pairwise_dist", "boost_stddev_teammates",
    "overcommit_rate", "simultaneous_air_rate",
]


class CoordinationMetrics(RewardFunction[AgentID, GameState, float]):
    def __init__(self, csv_path: str, run_label: str):
        super().__init__()
        # Per-process shard -- see module docstring on why this isn't just
        # csv_path directly.
        p = Path(csv_path)
        self.csv_path = str(p.with_name(f"{p.stem}.{os.getpid()}{p.suffix}"))
        self.run_label = run_label
        self._episode_id = 0
        self._buffers = None
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        directory = os.path.dirname(self.csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(CSV_COLUMNS)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        if self._buffers is not None and self._buffers["steps"] > 0:
            self._flush()
        self._episode_id += 1
        self._buffers = {
            "steps": 0,
            "dist_to_ball": [], "vel_toward_ball": [], "air": [],
            "teammate_dist": [], "boost_std": [],
            "overcommit": [], "simultaneous_air": [],
        }

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                     is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        self._record_step(agents, state)
        return {agent: 0.0 for agent in agents}

    def _record_step(self, agents: List[AgentID], state: GameState) -> None:
        teams: Dict[bool, list] = {True: [], False: []}
        for agent in agents:
            teams[state.cars[agent].is_orange].append(agent)

        ball_pos = state.ball.position
        dists_to_ball, vels_toward_ball, airborne_flags = [], [], []

        for agent in agents:
            car = state.cars[agent]
            physics = car.physics
            to_ball = ball_pos - physics.position
            dist = np.linalg.norm(to_ball)
            dists_to_ball.append(dist)
            if dist > 1e-6:
                vels_toward_ball.append(np.dot(physics.linear_velocity, to_ball / dist))
            airborne_flags.append(not car.on_ground)

        pairwise_dists, overcommit_flags, air_overlap_flags, boost_stds = [], [], [], []
        for team_agents in teams.values():
            if len(team_agents) < 2:
                continue
            positions = [state.cars[a].physics.position for a in team_agents]
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    pairwise_dists.append(np.linalg.norm(positions[i] - positions[j]))

            challengers = sum(
                1 for a in team_agents
                if np.linalg.norm(state.cars[a].physics.position - ball_pos) < CHALLENGE_DIST_UU
            )
            overcommit_flags.append(challengers >= 2)

            airborne_count = sum(1 for a in team_agents if not state.cars[a].on_ground)
            air_overlap_flags.append(airborne_count >= 2)

            boost_stds.append(np.std([state.cars[a].boost_amount for a in team_agents]))

        b = self._buffers
        b["steps"] += 1
        b["dist_to_ball"].append(np.mean(dists_to_ball))
        if vels_toward_ball:
            b["vel_toward_ball"].append(np.mean(vels_toward_ball))
        b["air"].append(np.mean(airborne_flags))
        if pairwise_dists:
            b["teammate_dist"].append(np.mean(pairwise_dists))
        if boost_stds:
            b["boost_std"].append(np.mean(boost_stds))
        if overcommit_flags:
            b["overcommit"].append(np.mean(overcommit_flags))
        if air_overlap_flags:
            b["simultaneous_air"].append(np.mean(air_overlap_flags))

    def _flush(self) -> None:
        b = self._buffers

        def avg(key):
            return float(np.mean(b[key])) if b[key] else ""

        row = [
            self.run_label, self._episode_id, b["steps"],
            avg("dist_to_ball"), avg("vel_toward_ball"), avg("air"),
            avg("teammate_dist"), avg("boost_std"),
            avg("overcommit"), avg("simultaneous_air"),
        ]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)


FITNESS_CSV_COLUMNS = ["run_label", "episode_id", "episode_length_steps", "episode_return"]


class FitnessTracker(RewardFunction[AgentID, GameState, float]):
    """
    Pass-through wrapper for population-based training's fitness signal.
    Wraps an inner reward function (use GoalReward -- the actual task
    objective, win/loss) and logs each episode's summed return per
    CombinedReward's weight while returning the inner reward UNCHANGED, so
    training behavior is identical to using the inner reward function
    directly:

        reward_fn = CombinedReward(
            (shaping, 1.0),
            (FitnessTracker(GoalReward(), csv_path="metrics/ground_fitness.csv",
                             run_label="model_a_ground"), 10.0),
            (coordination_metrics, 0.0),
        )

    DON'T use CoordinationMetrics' columns as PBT fitness -- see pbt.py's
    module docstring for why that would invalidate them as an independent
    coordination measurement. This class exists specifically to give PBT
    a fitness signal that's the real task objective instead.

    Same per-process CSV sharding as CoordinationMetrics, same last-episode
    caveat -- see that class's docstring for both.
    """

    def __init__(self, inner: RewardFunction, csv_path: str, run_label: str):
        super().__init__()
        self.inner = inner
        p = Path(csv_path)
        self.csv_path = str(p.with_name(f"{p.stem}.{os.getpid()}{p.suffix}"))
        self.run_label = run_label
        self._episode_id = 0
        self._episode_steps = 0
        self._episode_sum = 0.0
        self._has_data = False
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        directory = os.path.dirname(self.csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(FITNESS_CSV_COLUMNS)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        if self._has_data:
            self._flush()
        self._episode_id += 1
        self._episode_steps = 0
        self._episode_sum = 0.0
        self._has_data = False
        self.inner.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                     is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = self.inner.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        if rewards:
            # Team-shared reward (GoalReward) is identical across teammates,
            # so averaging rather than summing keeps this a per-agent return
            # rather than inflating it by team size.
            self._episode_sum += float(np.mean(list(rewards.values())))
            self._episode_steps += 1
            self._has_data = True
        return rewards

    def _flush(self) -> None:
        row = [self.run_label, self._episode_id, self._episode_steps, self._episode_sum]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)


class AuxiliaryDataLogger(RewardFunction[AgentID, GameState, float]):
    """
    Logs (observation, reward) pairs to compact binary .npz shards for
    train_auxiliary_encoder.py to train the Lucy-SKG-style SR/RP
    networks on (see auxiliary.py). Zero weight -- always returns 0.0,
    no effect on training, same pattern as CoordinationMetrics/
    FitnessTracker above.

    Reads the observation PartialInfoObs (obs.py) writes into
    shared_info["aux_obs"][agent] each step. Confirmed against
    rlgym.api.rlgym.RLGym.step's source: it calls
    `obs_builder.build_obs(...)` before `reward_fn.get_rewards(...)` on
    every step, so shared_info["aux_obs"] is always populated by the time
    this reads it. The .get(agent) fallback below is kept as harmless
    defensive coding, not because the ordering is actually in question.

    reward_proxy: an inner RewardFunction used ONLY to compute what gets
    logged as the "reward" target for Reward Prediction -- NOT used for
    actual training reward. Defaults to VelocityBallToGoalReward (a
    dense, always-computable signal, in the spirit of Lucy-SKG's own
    dense reward function) if none is given.
    """

    def __init__(self, data_dir: str, reward_proxy: RewardFunction = None, flush_every: int = 2000):
        super().__init__()
        if reward_proxy is None:
            from Rewards import VelocityBallToGoalReward
            reward_proxy = VelocityBallToGoalReward()
        self.reward_proxy = reward_proxy
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._obs_buffer = []
        self._reward_buffer = []
        self._shard_index = 0

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.reward_proxy.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                     is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        reward_values = self.reward_proxy.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        aux_obs = shared_info.get("aux_obs", {})

        for agent in agents:
            obs = aux_obs.get(agent)
            if obs is None:
                continue
            self._obs_buffer.append(np.asarray(obs, dtype=np.float32))
            self._reward_buffer.append(float(reward_values.get(agent, 0.0)))

        if len(self._obs_buffer) >= self.flush_every:
            self._flush()

        return {agent: 0.0 for agent in agents}

    def _flush(self) -> None:
        if not self._obs_buffer:
            return
        path = self.data_dir / f"shard_{os.getpid()}_{self._shard_index:06d}.npz"
        np.savez_compressed(
            path,
            obs=np.stack(self._obs_buffer, axis=0),
            reward=np.array(self._reward_buffer, dtype=np.float32),
        )
        self._shard_index += 1
        self._obs_buffer = []
        self._reward_buffer = []