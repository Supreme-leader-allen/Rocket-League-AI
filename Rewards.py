"""
Reward functions for the ground (Model A) and aerial (Model B) runs.

SpeedTowardBallReward, InAirReward, and VelocityBallToGoalReward are the
individual dense-shaping terms from the RLGym tutorial. AnnealedCombinedReward
wraps CombinedReward-style (reward, weight) pairs and linearly decays a
chosen subset of those weights over training, per Liu et al. (2019):
heavy shaping early (when the sparse team-goal signal is too rare to learn
from), decaying toward zero so the agent ends up optimizing mostly for the
shared sparse outcome (GoalReward), which is what actually produces
coordinated rather than individually-selfish play.

GoalReward (imported from rlgym.rocket_league.reward_functions in your
train_*.py files) is left un-annealed and un-wrapped -- it's already
team-shared by construction, which is the credit-assignment mechanism
itself, not something this file needs to implement.
"""

from typing import List, Dict, Any, Sequence, Tuple

import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values


class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for moving quickly toward the ball."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                     is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_physics = car.physics if car.is_orange else car.inverted_physics
            ball_physics = state.ball if car.is_orange else state.inverted_ball
            player_vel = car_physics.linear_velocity
            pos_diff = ball_physics.position - car_physics.position
            dist_to_ball = np.linalg.norm(pos_diff)
            dir_to_ball = pos_diff / dist_to_ball

            speed_toward_ball = np.dot(player_vel, dir_to_ball)
            rewards[agent] = max(speed_toward_ball / common_values.CAR_MAX_SPEED, 0.0)
        return rewards


class InAirReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for being in the air. Weight ~0 for Model A (ground),
    raised for Model B (aerial) -- see train_ground.py / train_aerial.py."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                     is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: float(not state.cars[agent].on_ground) for agent in agents}


class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """Rewards the agent for hitting the ball toward the opponent's goal."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                     is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            goal_y = -common_values.BACK_NET_Y if car.is_orange else common_values.BACK_NET_Y

            ball_vel = ball.linear_velocity
            pos_diff = np.array([0, goal_y, 0]) - ball.position
            dist = np.linalg.norm(pos_diff)
            dir_to_goal = pos_diff / dist

            vel_toward_goal = np.dot(ball_vel, dir_to_goal)
            rewards[agent] = max(vel_toward_goal / common_values.BALL_MAX_SPEED, 0)
        return rewards


class TouchBallReward(RewardFunction[AgentID, GameState, float]):
    """1.0 on every step the agent touched the ball (car.ball_touches > 0,
    RocketSim's per-step touch counter -- confirmed against
    rlgym.rocket_league.sim.rocketsim_engine's touch callback).

    The single most important early-training reward for RLGym bots
    (ZealanL's RLGym-PPO-Guide, "making a good bot"): a random policy almost
    never touches the ball, so nothing downstream of a touch -- shots,
    passes, goals -- can be learned until touching is. It is individual, not
    team-shared, which is why it is annealed down (see SHAPING_SCHEDULES)."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                     is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: float(state.cars[agent].ball_touches > 0) for agent in agents}


class FaceBallReward(RewardFunction[AgentID, GameState, float]):
    """Cosine between the car's forward vector and the direction to the
    ball, in [-1, 1]. Cheap orientation shaping that makes driving toward
    the ball learnable before SpeedTowardBallReward has anything to reward."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                     is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        ball_pos = state.ball.position
        for agent in agents:
            physics = state.cars[agent].physics
            pos_diff = ball_pos - physics.position
            dist = np.linalg.norm(pos_diff)
            rewards[agent] = float(np.dot(physics.forward, pos_diff / dist)) if dist > 1e-6 else 0.0
        return rewards


# (reward class, initial weight, final weight), annealed over ANNEAL_TIMESTEPS
# by AnnealedCombinedReward. GoalReward (weight 10, team-shared, never
# annealed) is added separately by the training scripts.
#
# Rebalanced 2026-09-30 (docs/ISSUES.md "P0 -- the policy never learned").
# The previous weights were the RLGym quickstart tutorial's (speed 0.01,
# ball-to-goal 0.1, air 0.002): measured on this env with random actions,
# their per-step sum averaged ~0.003 against +-10 per goal, so the only
# signal PPO saw was rare goals scored mostly by luck, and the policy stayed
# exactly uniform (entropy == ln 90) for two full runs. These follow the
# proportions in ZealanL's RLGym-PPO-Guide: touching the ball dominates
# early, approach/orientation terms make touching discoverable, and a
# little InAirReward keeps the bot from unlearning jump.
#
# Fading to a floor, not to zero (changed 2026-10-10, docs/ISSUES.md "P0 --
# shaping annealed to zero un-learned everything"). The first 300M-step run
# on these terms learned for real -- touches/min 2.5 (random) -> 46, air time
# 0.83 -> 0.35, entropy down to 0.93 of max -- then collapsed back to
# exactly the random baseline in step with the anneal: decline began once
# shaping was ~70% annealed, and by 200M (fully annealed) behavior and
# entropy were both back to random. With only the team-shared goal left, a
# 4v4 goal is too rare and too diluted across 8 cars to carry any per-agent
# signal, and the entropy bonus then pulls the policy back to uniform.
#
# So the individual terms (touch, speed, face, air) now anneal only to 50%
# of their starting weight -- the collapse began around 30%. Ball-to-goal
# velocity is NOT annealed: it is identical for every agent on a team, so
# it is a team-shared reward in exactly Liu et al.'s credit-assignment
# sense, just dense enough to learn from. Liu et al. annealed to zero, but
# in 2v2 with far more training; at this budget in 4v4 that measurably
# fails, and experiments/04_no_anneal.sh remains the ablation for whether
# annealing matters at all.
SHAPING_SCHEDULES = {
    "ground": [
        (TouchBallReward, 5.0, 2.5),
        (SpeedTowardBallReward, 0.5, 0.25),
        (FaceBallReward, 0.1, 0.05),
        (VelocityBallToGoalReward, 1.0, 1.0),
        (InAirReward, 0.03, 0.015),
    ],
    # Model B: warm-started from Model A, so touching is already learned;
    # aerial play is new and gets dense InAirReward guidance again.
    "aerial": [
        (TouchBallReward, 2.5, 2.5),
        (SpeedTowardBallReward, 0.25, 0.25),
        (FaceBallReward, 0.05, 0.05),
        (VelocityBallToGoalReward, 1.0, 1.0),
        (InAirReward, 0.2, 0.05),
    ],
}


def build_shaping(phase: str, anneal_timesteps: float, n_proc: int, initial_timesteps: float,
                  team_spirit: Tuple[float, float, float] = (0.0, 0.0, 1.0)):
    """AnnealedCombinedReward for `phase` ("ground" or "aerial").
    team_spirit = (start, final, ramp_timesteps) -- see AnnealedCombinedReward."""
    return AnnealedCombinedReward(
        weighted_rewards=[(cls(), w0, w1) for cls, w0, w1 in SHAPING_SCHEDULES[phase]],
        anneal_timesteps=anneal_timesteps,
        n_proc=n_proc,
        initial_timesteps=initial_timesteps,
        team_spirit_start=team_spirit[0],
        team_spirit_final=team_spirit[1],
        team_spirit_timesteps=team_spirit[2],
    )


class AnnealedCombinedReward(RewardFunction[AgentID, GameState, float]):
    """
    Combines (reward_fn, initial_weight, final_weight) triples and linearly
    anneals each weight from initial to final over `anneal_timesteps` of
    (estimated) cumulative environment timesteps.

    Previously annealed over wall-clock time instead, measured from this
    process's own start -- confirmed broken (docs/ISSUES.md P3): a fresh
    process starts a fresh clock, so the anneal silently restarted from
    zero on every checkpoint resume (never completing at all on any
    session-limited platform) and, on any run longer than the hardcoded
    anneal_seconds, left shaping stuck at zero for the remainder. Neither
    failure is possible with a timestep-based budget, since
    cumulative_timesteps is what checkpoint resume actually restores and
    what timestep_limit is actually budgeted in.

    Exact global cumulative_timesteps isn't directly available inside an
    environment subprocess without forking rlgym_ppo internals to push it
    in every step (same category of limitation as self_play.py's
    FrozenPolicy) -- rlgym_ppo runs n_proc independent environment
    processes and Learner's cumulative_timesteps lives only in the parent
    process. What's tracked instead is an estimate:

        estimated_cumulative = initial_timesteps + local_agent_steps * n_proc

    `local_agent_steps` accumulates len(agents) every get_rewards() call
    (confirmed against rlgym_ppo.batched_agents.batched_agent_manager.
    BatchedAgentManager._collect_response's source: it counts
    `n_collected = prev_n_agents` for a multi-agent env, i.e.
    cumulative_timesteps advances by the AGENT count each step, not by
    1 -- so a 4v4's 8 agents must be counted, not the env.step() call).
    Multiplying by `n_proc` estimates the other processes' contributions,
    assuming they advance at roughly the same rate, which is the same
    "close enough in practice" approximation the wall-clock version made
    explicitly, just no longer one that resets to zero on resume.
    `initial_timesteps` recovers the actual resume point from the loaded
    checkpoint (Train_Ground.py/Train_Aerial.py pass it in, read off the
    checkpoint path's own digit-named folder -- the same technique
    Pbt.py's and Train_Aerial.py's timestep_limit fixes use).

    Terms with initial_weight == final_weight (e.g. GoalReward, passed
    with no annealing) are just held constant.

    Team spirit (added 2026-10-11, docs/ISSUES.md "Team spirit"). After the
    weighted sum, each agent's shaping reward is blended with its team's
    average:

        r_i' = (1 - tau) * r_i + tau * mean_{j on i's team}(r_j)

    tau ramps linearly from team_spirit_start to team_spirit_final over
    team_spirit_timesteps (same estimated-timestep clock as the anneal, but
    its own budget, so experiments/04_no_anneal.sh can freeze the anneal
    without also freezing team spirit). This is OpenAI Five's mechanism.

    Why: run 3 learned to score in 97% of episodes, but by ball-chasing --
    overcommit rate 0.28 vs 0.005 random, teammate spacing 1500 vs 4400 uu,
    nobody goal-side 25% of the time. Touch and speed-to-ball pay the car
    that gets to the ball, so four cars racing for it is exactly what they
    reward. At tau = 1 a car earns the same whether it or a teammate makes
    the touch, so there is no reason to fight a teammate for the ball, but
    the reward stays dense -- unlike annealing the shaping away, which run 3
    showed collapses learning. Starting at tau = 0 keeps individual credit
    while the bot is still learning to touch the ball at all.

    Terms that are already identical across a team (ball-to-goal velocity)
    are unchanged by the blend. Goal reward is outside this class.
    """

    def __init__(self, weighted_rewards: Sequence[Tuple[RewardFunction, float, float]],
                 anneal_timesteps: float, n_proc: int = 1, initial_timesteps: float = 0.0,
                 team_spirit_start: float = 0.0, team_spirit_final: float = 0.0,
                 team_spirit_timesteps: float = 1.0):
        super().__init__()
        self._entries = list(weighted_rewards)
        self._ts_start = team_spirit_start
        self._ts_final = team_spirit_final
        self._ts_timesteps = team_spirit_timesteps
        self._anneal_timesteps = anneal_timesteps
        self._n_proc = max(1, n_proc)
        self._initial_timesteps = initial_timesteps
        self._local_agent_steps = 0

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        for reward_fn, _, _ in self._entries:
            reward_fn.reset(agents, initial_state, shared_info)

    def _estimated_cumulative(self) -> float:
        return self._initial_timesteps + self._local_agent_steps * self._n_proc

    def _current_progress(self) -> float:
        if self._anneal_timesteps <= 0:
            return 1.0
        return min(1.0, self._estimated_cumulative() / self._anneal_timesteps)

    def _current_team_spirit(self) -> float:
        ramp = 1.0 if self._ts_timesteps <= 0 else min(1.0, self._estimated_cumulative() / self._ts_timesteps)
        return self._ts_start + (self._ts_final - self._ts_start) * ramp

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                     is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        progress = self._current_progress()
        # read by Metrics.CoordinationMetrics, so each episode row records
        # how far the shaping anneal had got
        shared_info["shaping_progress"] = progress
        totals = {agent: 0.0 for agent in agents}

        for reward_fn, initial_weight, final_weight in self._entries:
            weight = initial_weight + (final_weight - initial_weight) * progress
            per_agent = reward_fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            for agent in agents:
                totals[agent] += weight * per_agent[agent]

        tau = self._current_team_spirit()
        shared_info["team_spirit"] = tau
        if tau > 0:
            team_sum: Dict[int, float] = {}
            team_n: Dict[int, int] = {}
            for agent in agents:
                team = state.cars[agent].team_num
                team_sum[team] = team_sum.get(team, 0.0) + totals[agent]
                team_n[team] = team_n.get(team, 0) + 1
            totals = {
                agent: (1 - tau) * totals[agent]
                + tau * team_sum[state.cars[agent].team_num] / team_n[state.cars[agent].team_num]
                for agent in agents
            }

        self._local_agent_steps += len(agents)
        return totals