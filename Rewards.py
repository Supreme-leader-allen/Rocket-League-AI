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
    """

    def __init__(self, weighted_rewards: Sequence[Tuple[RewardFunction, float, float]],
                 anneal_timesteps: float, n_proc: int = 1, initial_timesteps: float = 0.0):
        super().__init__()
        self._entries = list(weighted_rewards)
        self._anneal_timesteps = anneal_timesteps
        self._n_proc = max(1, n_proc)
        self._initial_timesteps = initial_timesteps
        self._local_agent_steps = 0

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        for reward_fn, _, _ in self._entries:
            reward_fn.reset(agents, initial_state, shared_info)

    def _current_progress(self) -> float:
        if self._anneal_timesteps <= 0:
            return 1.0
        estimated_cumulative = self._initial_timesteps + self._local_agent_steps * self._n_proc
        return min(1.0, estimated_cumulative / self._anneal_timesteps)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                     is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        progress = self._current_progress()
        totals = {agent: 0.0 for agent in agents}

        for reward_fn, initial_weight, final_weight in self._entries:
            weight = initial_weight + (final_weight - initial_weight) * progress
            per_agent = reward_fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            for agent in agents:
                totals[agent] += weight * per_agent[agent]

        self._local_agent_steps += len(agents)
        return totals