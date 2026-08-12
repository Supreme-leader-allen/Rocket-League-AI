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

import time
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
    anneals each weight from initial to final over `anneal_seconds` of
    wall-clock training time.

    Wall-clock rather than a global timestep counter is a deliberate choice:
    rlgym_ppo runs many environment processes in parallel (n_proc), each
    with its own copy of this reward function, so there's no cheap way for
    one process to know the *global* timestep count without extra
    plumbing (e.g. a multiprocessing.Value updated via a metrics_logger
    callback). Wall-clock time is process-local, needs no synchronization,
    and tracks training progress closely enough in practice since all
    workers run continuously and roughly in step with each other. If you
    later want exact timestep-based annealing, that shared-counter
    approach is the correct upgrade -- this is the simple version to start
    with.

    Terms with initial_weight == final_weight (e.g. GoalReward, passed
    with no annealing) are just held constant.
    """

    def __init__(self, weighted_rewards: Sequence[Tuple[RewardFunction, float, float]], anneal_seconds: float):
        super().__init__()
        self._entries = list(weighted_rewards)
        self._anneal_seconds = anneal_seconds
        self._start_time = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        if self._start_time is None:
            self._start_time = time.time()
        for reward_fn, _, _ in self._entries:
            reward_fn.reset(agents, initial_state, shared_info)

    def _current_progress(self) -> float:
        if self._anneal_seconds <= 0:
            return 1.0
        elapsed = time.time() - self._start_time
        return min(1.0, elapsed / self._anneal_seconds)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                     is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        progress = self._current_progress()
        totals = {agent: 0.0 for agent in agents}

        for reward_fn, initial_weight, final_weight in self._entries:
            weight = initial_weight + (final_weight - initial_weight) * progress
            per_agent = reward_fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            for agent in agents:
                totals[agent] += weight * per_agent[agent]

        return totals