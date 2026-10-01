"""
PartialInfoObs: wraps DefaultObs to remove perfect information.

Design choice worth calling out: instead of slicing into DefaultObs's
output vector (which would require knowing its exact internal per-car
block layout -- position/rotation/velocity/boost/etc ordering, which
isn't something to guess at and get subtly wrong), this mutates each
acting agent's *view* of the GameState before handing it to DefaultObs.
Cars outside the acting car's forward view cone are replaced with a
memorized last-seen snapshot (or blanked out if unseen for too long).
DefaultObs then builds the observation normally from that doctored
state, so its internal layout never needs to be known or trusted to stay
stable across rlgym versions.

What gets masked (rewritten 2026-09-30): the WHOLE car, not just position
and linear_velocity. The earlier version only overwrote those two fields,
so an occluded car still leaked its live rotation (forward/up), angular
velocity, boost amount, on_ground, is_boosting and supersonic flags --
exactly the information a human can't see behind them. Now each remembered
car is a frozen copy of the Car as it looked at its last sighting
(including that sighting's position noise), and a forgotten or never-seen
car is a blank Car: zero physics, zero rotation matrix (so forward/up are
zero -- an impossible orientation, never confused with a real one), zero
boost, airborne, not boosting.

Frame handling (fixes the class of bug behind docs/ISSUES.md P0 for good):
memory and masking work entirely in the TRUE (blue) frame -- car.physics,
never car.inverted_physics. Each masked copy gets its _inverted_physics
cache cleared, so when DefaultObs asks an orange agent's view for
inverted_physics it is recomputed from the masked true-frame physics. The
visibility test itself is frame-invariant (the inversion is a 180 degree
rotation about z, which preserves dot products), so there is no longer any
frame choice that can disagree with DefaultObs's.

Copy cost: GameState is shallow-copied, its cars dict is copied, and only
the cars being masked are replaced with new Car objects. Nothing is
mutated in place, so the live state is never touched. This replaces a
copy.deepcopy(state) per agent per step, which docs/COMPUTE.md measured as
most of the environment step time.

Extra features appended after DefaultObs's vector (EXTRA_OBS_DIM = 31):
- pending actions (PENDING_SLOTS x 8 = 24): the control rows of decisions
  already queued in HumanlikeAction's reaction delay but not executed yet.
  Without them the policy can't see what it has already committed to, which
  makes the problem non-Markov (see Actions.py's docstring).
- staleness per other car (3 ally slots + 4 enemy slots = 7), in
  DefaultObs's ally-then-enemy order: 0.0 = visible now, rising toward 1.0
  while remembered, 1.0 = unknown (forgotten, never seen, or a padding
  slot). Without this flag, a remembered car and a live one look identical
  to the network, and a forgotten car's zeroed position is the field
  centre, a perfectly legal place for a car to be.

Observation size is therefore 212 + 31 = 243 for 4v4 (DefaultObs with
zero_padding=4 is 212). Changing the observation invalidates any checkpoint
trained on the old 212-wide input; see docs/ISSUES.md for why that was
accepted this once.

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
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league import common_values

# ~110 degree horizontal cone -- rough stand-in for a player's screen +
# attention. Exposed as an env var (not just a literal) so an ablation
# run can open it to 180 (nothing ever occluded) to test whether the
# partial-information constraint itself matters -- see
# experiments/03_full_info.sh.
FOV_HALF_ANGLE_DEG = float(os.environ.get("FOV_HALF_ANGLE_DEG", 55.0))
NOISE_STD_UU = 60.0           # gaussian position noise (uu) applied to cars in view but far away
NOISE_START_DIST_UU = 3000.0  # noise ramps in past this distance; nothing added for close-range cars
MAX_STALENESS_STEPS = 45      # ~3s at 15 decisions/sec before a fully-occluded car is zeroed out entirely

PENDING_SLOTS = 3             # HumanlikeAction's default reaction_delay_ticks
CONTROL_DIM = 8
MAX_TEAM_SIZE = 4
STALENESS_SLOTS = (MAX_TEAM_SIZE - 1) + MAX_TEAM_SIZE
EXTRA_OBS_DIM = PENDING_SLOTS * CONTROL_DIM + STALENESS_SLOTS

AUX_ENCODER_CHECKPOINT = os.environ.get("AUX_ENCODER_CHECKPOINT") or None


def _blank_car(car: Car) -> Car:
    """A copy of `car` carrying no information beyond its team: zero
    physics and rotation, no boost, airborne, not boosting/supersonic."""
    blank = copy.copy(car)
    physics = PhysicsObject()
    physics.position = np.zeros(3, dtype=np.float32)
    physics.linear_velocity = np.zeros(3, dtype=np.float32)
    physics.angular_velocity = np.zeros(3, dtype=np.float32)
    physics.rotation_mtx = np.zeros((3, 3), dtype=np.float32)
    blank.physics = physics
    blank._inverted_physics = None
    blank.boost_amount = 0.0
    blank.demo_respawn_timer = 0.0
    blank.boost_active_time = 0.0
    blank.supersonic_time = 0.0
    blank.on_ground = False
    return blank


def _sighted_car(car: Car, position: np.ndarray) -> Car:
    """A frozen copy of `car` as seen right now, with `position` (noisy)
    in place of its exact position. Shares nothing mutable with `car`."""
    seen = copy.copy(car)
    physics = copy.copy(car.physics)
    physics.position = position
    seen.physics = physics
    seen._inverted_physics = None
    return seen


class PartialInfoObs(ObsBuilder[AgentID, np.ndarray, GameState, Any]):
    def __init__(self, zero_padding: int = MAX_TEAM_SIZE):
        # 4, not 3: DefaultObs's zero_padding is "max cars per team," and
        # this project's fixed 4v4 needs exactly that (confirmed by direct
        # execution -- 3 silently under-declares the observation size and
        # crashes policy-network construction with a shape mismatch; see
        # Train_Ground.py's comment and docs/ISSUES.md).
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
        self._cos_fov = np.cos(np.radians(FOV_HALF_ANGLE_DEG))
        # memory[agent][other_agent] = [last_seen_car_copy, staleness_steps]
        self._memory: Dict[AgentID, Dict[AgentID, list]] = {}
        # Lazily constructed on first build_obs call, once we know the raw
        # observation size (needed to build the encoder's input layer).
        self._aux_encoder = None
        self._aux_encoder_load_attempted = False

    def get_obs_space(self, agent: AgentID):
        space_type, size = self._base.get_obs_space(agent)
        size = size + EXTRA_OBS_DIM
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
        pending_controls = shared_info.get("pending_controls", {})

        for agent in agents:
            masked_state, staleness = self._masked_state_for(agent, state)
            # DefaultObs.build_obs takes a list of agents; give it just this
            # one so the doctored state only affects this agent's own obs.
            base_obs = self._base.build_obs([agent], masked_state, shared_info)[agent]
            raw_obs = np.concatenate([
                base_obs,
                self._pending_features(pending_controls.get(agent)),
                self._staleness_features(agent, state, staleness),
            ]).astype(np.float32)
            aux_obs_log[agent] = raw_obs

            encoder = self._get_aux_encoder(raw_obs.shape[0])
            if encoder is not None:
                encoded = encoder.encode(raw_obs)
                out[agent] = np.concatenate([raw_obs, encoded])
            else:
                out[agent] = raw_obs

        return out

    @staticmethod
    def _pending_features(pending) -> np.ndarray:
        feats = np.zeros((PENDING_SLOTS, CONTROL_DIM), dtype=np.float32)
        if pending is not None and len(pending):
            pending = np.asarray(pending, dtype=np.float32)[-PENDING_SLOTS:]
            feats[:len(pending)] = pending
        return feats.reshape(-1)

    @staticmethod
    def _staleness_features(agent: AgentID, state: GameState, staleness: Dict[AgentID, float]) -> np.ndarray:
        # Same iteration order DefaultObs._build_obs uses: state.cars dict
        # order, split into allies then enemies, padded to the max roster.
        team = state.cars[agent].team_num
        allies, enemies = [], []
        for other_id, other_car in state.cars.items():
            if other_id == agent:
                continue
            (allies if other_car.team_num == team else enemies).append(staleness.get(other_id, 1.0))
        allies = (allies + [1.0] * (MAX_TEAM_SIZE - 1))[:MAX_TEAM_SIZE - 1]
        enemies = (enemies + [1.0] * MAX_TEAM_SIZE)[:MAX_TEAM_SIZE]
        return np.asarray(allies + enemies, dtype=np.float32)

    def _get_aux_encoder(self, obs_size: int):
        if not AUX_ENCODER_CHECKPOINT or self._aux_encoder_load_attempted:
            return self._aux_encoder
        self._aux_encoder_load_attempted = True
        from Auxiliary import AuxiliaryEncoder
        try:
            self._aux_encoder = AuxiliaryEncoder(AUX_ENCODER_CHECKPOINT, obs_size)
        except Exception as e:
            # Let this propagate instead of falling back to None: get_obs_space()
            # already reports the ENCODED_DIM-larger size whenever
            # AUX_ENCODER_CHECKPOINT is set, so a silent fallback would return
            # an observation smaller than the policy's input layer. See
            # docs/ISSUES.md P2.
            raise RuntimeError(
                f"PartialInfoObs: failed to load AUX_ENCODER_CHECKPOINT="
                f"{AUX_ENCODER_CHECKPOINT!r} (obs_size={obs_size}). Check "
                f"that obs_size matches what the checkpoint was trained "
                f"with (Train_Auxiliary_Encoder.py's --data-dir)."
            ) from e
        return self._aux_encoder

    def _masked_state_for(self, agent: AgentID, state: GameState):
        """Returns (masked_state, staleness) where staleness maps each other
        car to 0.0 (visible), (0, 1) (remembered), or 1.0 (unknown)."""
        acting_physics = state.cars[agent].physics
        # PhysicsObject.forward is a @property (rotation_mtx[:, 0]), not a
        # method -- confirmed via source.
        forward = acting_physics.forward
        acting_pos = acting_physics.position

        masked_cars = dict(state.cars)
        masked_state = copy.copy(state)
        masked_state.cars = masked_cars
        memory = self._memory.setdefault(agent, {})
        staleness: Dict[AgentID, float] = {}

        for other_id, other_car in state.cars.items():
            if other_id == agent:
                continue

            to_other = other_car.physics.position - acting_pos
            dist = np.linalg.norm(to_other)
            in_view = dist < 1e-6 or np.dot(forward, to_other / dist) >= self._cos_fov

            if in_view:
                noise_scale = max(0.0, dist - NOISE_START_DIST_UU) / NOISE_START_DIST_UU
                if noise_scale > 0:
                    noisy_pos = other_car.physics.position + np.random.normal(
                        0, NOISE_STD_UU * noise_scale, size=3).astype(np.float32)
                    seen = _sighted_car(other_car, noisy_pos)
                    masked_cars[other_id] = seen
                else:
                    seen = _sighted_car(other_car, other_car.physics.position.copy())
                memory[other_id] = [seen, 0]
                staleness[other_id] = 0.0
            elif other_id in memory:
                entry = memory[other_id]
                entry[1] += 1
                if entry[1] > MAX_STALENESS_STEPS:
                    del memory[other_id]
                    masked_cars[other_id] = _blank_car(other_car)
                    staleness[other_id] = 1.0
                else:
                    masked_cars[other_id] = entry[0]
                    staleness[other_id] = entry[1] / (MAX_STALENESS_STEPS + 1)
            else:
                masked_cars[other_id] = _blank_car(other_car)
                staleness[other_id] = 1.0

        return masked_state, staleness
