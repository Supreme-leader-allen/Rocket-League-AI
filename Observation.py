"""
PartialInfoObs: wraps DefaultObs to remove perfect information.

Design choice worth calling out: instead of slicing into DefaultObs's
output vector (which would require knowing its exact internal per-car
block layout -- position/rotation/velocity/boost/etc ordering, which
isn't something to guess at and get subtly wrong), this mutates each
acting agent's *view* of the GameState before handing it to DefaultObs.
Cars outside the acting car's forward view cone have their position and
velocity replaced with a memorized last-seen value (or zeroed out if
unseen for too long). DefaultObs then builds the observation normally
from that doctored state, so its internal layout never needs to be known
or trusted to stay stable across rlgym versions.

Only position and linear_velocity are masked below -- those are the two
Car/PhysicsObject fields directly confirmed from the reward code you
already have (SpeedTowardBallReward uses car_physics.position and
car_physics.linear_velocity). Rotation and boost amount likely also leak
information about occluded cars and are good candidates to add once
you've confirmed their exact attribute names against your installed
rlgym.rocket_league.api -- check the Car/PhysicsObject class directly
(e.g. via `python -c "from rlgym.rocket_league.api import Car; help(Car)"`)
before extending the masking loop below.

Cost note: this does one shallow-ish copy + mutation of the relevant car
physics per acting agent per step (8 agents in a 4v4), which is more
work than DefaultObs alone. Fine for getting training running; worth
profiling before you scale up n_proc a lot.
"""

import copy
from typing import List, Dict, Any

import numpy as np
from rlgym.api import ObsBuilder, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league import common_values

FOV_HALF_ANGLE_DEG = 55.0     # ~110 degree horizontal cone -- rough stand-in for a player's screen + attention
NOISE_STD_UU = 60.0           # gaussian position noise (uu) applied to cars in view but far away
NOISE_START_DIST_UU = 3000.0  # noise ramps in past this distance; nothing added for close-range cars
MAX_STALENESS_STEPS = 45      # ~3s at 15 decisions/sec before a fully-occluded car is zeroed out entirely


class PartialInfoObs(ObsBuilder[AgentID, np.ndarray, GameState, Any]):
    def __init__(self, zero_padding: int = 3):
        super().__init__()
        self._base = DefaultObs(
            zero_padding=zero_padding,
            pos_coef=np.asarray([1 / common_values.SIDE_WALL_X,
                                  1 / common_values.BACK_NET_Y,
                                  1 / common_values.CEILING_Z]),
            ang_coef=1 / np.pi,
            lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
            ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
            boost_coef=1 / 100.0,
        )
        # memory[agent][other_agent] = (position, linear_velocity, staleness_steps)
        self._memory: Dict[AgentID, Dict[AgentID, Any]] = {}

    def get_obs_space(self, agent: AgentID):
        return self._base.get_obs_space(agent)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self._base.reset(agents, initial_state, shared_info)
        self._memory = {agent: {} for agent in agents}

    def build_obs(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        out = {}
        for agent in agents:
            masked_state = self._masked_state_for(agent, state)
            # DefaultObs.build_obs takes a list of agents; give it just this
            # one so the doctored state only affects this agent's own obs.
            out[agent] = self._base.build_obs([agent], masked_state, shared_info)[agent]
        return out

    def _masked_state_for(self, agent: AgentID, state: GameState) -> GameState:
        acting_car = state.cars[agent]
        acting_physics = acting_car.physics if acting_car.is_orange else acting_car.inverted_physics
        forward = acting_physics.forward  # readonly property on PhysicsObject, not a method -- confirmed via help(PhysicsObject)
        acting_pos = acting_physics.position

        masked_state = copy.deepcopy(state)
        memory = self._memory.setdefault(agent, {})

        for other_id, other_car in state.cars.items():
            if other_id == agent:
                continue

            other_physics = other_car.physics if other_car.is_orange else other_car.inverted_physics
            to_other = other_physics.position - acting_pos
            dist = np.linalg.norm(to_other)
            if dist < 1e-6:
                continue
            cos_angle = np.dot(forward, to_other / dist)
            in_view = cos_angle >= np.cos(np.radians(FOV_HALF_ANGLE_DEG))

            masked_other = masked_state.cars[other_id]
            masked_physics = masked_other.physics if masked_other.is_orange else masked_other.inverted_physics

            if in_view:
                noise_scale = max(0.0, dist - NOISE_START_DIST_UU) / NOISE_START_DIST_UU
                noisy_pos = other_physics.position + np.random.normal(0, NOISE_STD_UU * noise_scale, size=3)
                memory[other_id] = (noisy_pos, other_physics.linear_velocity, 0)
                masked_physics.position = noisy_pos
                masked_physics.linear_velocity = other_physics.linear_velocity
            elif other_id in memory:
                last_pos, last_vel, staleness = memory[other_id]
                staleness += 1
                if staleness > MAX_STALENESS_STEPS:
                    masked_physics.position = np.zeros(3)
                    masked_physics.linear_velocity = np.zeros(3)
                    del memory[other_id]
                else:
                    memory[other_id] = (last_pos, last_vel, staleness)
                    masked_physics.position = last_pos
                    masked_physics.linear_velocity = last_vel
            else:
                masked_physics.position = np.zeros(3)
                masked_physics.linear_velocity = np.zeros(3)

        return masked_state