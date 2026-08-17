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

Lucy-SKG-style auxiliary abstraction (see auxiliary.py): if the
AUX_ENCODER_CHECKPOINT environment variable is set, PartialInfoObs loads
a trained StateRepresentationNet encoder and concatenates its output
onto every observation. Unset (the default), behavior is identical to
before this feature existed -- nothing changes unless you've actually
trained an encoder via train_auxiliary_encoder.py. When enabled,
get_obs_space's reported size grows by ENCODED_DIM (16) accordingly,
which matters because rlgym_ppo sizes its policy network's input layer
from get_obs_space -- if this reported size and what build_obs actually
returns ever disagree, you'll get a shape-mismatch error immediately on
startup rather than something subtle later.

Also writes each agent's raw (pre-concatenation) observation into
shared_info["aux_obs"][agent] every step, which metrics.py's
AuxiliaryDataLogger reads to build training data for the encoder. The
raw, not the enriched, observation is logged deliberately -- training
the encoder on its own already-encoded output would be circular.
"""

import copy
import os
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

AUX_ENCODER_CHECKPOINT = os.environ.get("AUX_ENCODER_CHECKPOINT") or None


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
        # Lazily constructed on first build_obs call, once we know the raw
        # observation size (needed to build the encoder's input layer).
        self._aux_encoder = None
        self._aux_encoder_load_attempted = False

    def get_obs_space(self, agent: AgentID):
        space_type, size = self._base.get_obs_space(agent)
        if AUX_ENCODER_CHECKPOINT:
            from Auxiliary import ENCODED_DIM
            size = size + ENCODED_DIM
        return space_type, size

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self._base.reset(agents, initial_state, shared_info)
        self._memory = {agent: {} for agent in agents}

    def build_obs(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        out = {}
        aux_obs_log = shared_info.setdefault("aux_obs", {})

        for agent in agents:
            masked_state = self._masked_state_for(agent, state)
            # DefaultObs.build_obs takes a list of agents; give it just this
            # one so the doctored state only affects this agent's own obs.
            raw_obs = self._base.build_obs([agent], masked_state, shared_info)[agent]
            aux_obs_log[agent] = raw_obs

            encoder = self._get_aux_encoder(raw_obs.shape[0])
            if encoder is not None:
                encoded = encoder.encode(raw_obs)
                out[agent] = np.concatenate([raw_obs, encoded])
            else:
                out[agent] = raw_obs

        return out

    def _get_aux_encoder(self, obs_size: int):
        if not AUX_ENCODER_CHECKPOINT or self._aux_encoder_load_attempted:
            return self._aux_encoder
        self._aux_encoder_load_attempted = True
        from Auxiliary import AuxiliaryEncoder
        try:
            self._aux_encoder = AuxiliaryEncoder(AUX_ENCODER_CHECKPOINT, obs_size)
        except Exception as e:
            print(f"PartialInfoObs: failed to load AUX_ENCODER_CHECKPOINT="
                  f"{AUX_ENCODER_CHECKPOINT!r} ({e}) -- continuing without it. "
                  f"Check that obs_size ({obs_size}) matches what the checkpoint "
                  f"was trained with (train_auxiliary_encoder.py's --data-dir).")
            self._aux_encoder = None
        return self._aux_encoder

    def _masked_state_for(self, agent: AgentID, state: GameState) -> GameState:
        acting_car = state.cars[agent]
        acting_physics = acting_car.physics if acting_car.is_orange else acting_car.inverted_physics
        # PhysicsObject.forward is a @property (rotation_mtx[:, 0]), not a
        # method -- confirmed via source; calling it as forward() raised
        # TypeError: 'numpy.ndarray' object is not callable.
        forward = acting_physics.forward
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