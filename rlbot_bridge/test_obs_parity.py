"""
Checks that a real-game state reaches the policy as the numbers training gave it.

Plays random actions in the training simulator, rewrites each state as the
GamePacket RLBot would have sent, runs that through the bridge, and compares
the DefaultObs vector from both sides for every car, on both teams.

    .venv\\Scripts\\python.exe test_obs_parity.py
"""

import sys
from pathlib import Path

import numpy as np
from rlbot import flat
from rlgym.api import RLGym
from rlgym.rocket_league.common_values import BOOST_LOCATIONS
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, AnyCondition
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence
from rlgym_compat import GameState

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from Actions import HumanlikeAction  # noqa: E402
from Observation import EXTRA_OBS_DIM, PartialInfoObs  # noqa: E402
from compat_state import fix_boost_timers, pad_cooldowns, snapshot, to_controller  # noqa: E402

BIG_PADS = {i for i, loc in enumerate(BOOST_LOCATIONS) if loc[2] > 72}
STEPS = 400
TOLERANCE = 2e-3


def vec(a) -> flat.Vector3:
    return flat.Vector3(float(a[0]), float(a[1]), float(a[2]))


def physics(p) -> flat.Physics:
    pitch, yaw, roll = p.euler_angles
    return flat.Physics(vec(p.position), flat.Rotator(float(pitch), float(yaw), float(roll)),
                        vec(p.linear_velocity), vec(p.angular_velocity))


def air_state(car) -> flat.AirState:
    if car.on_ground:
        return flat.AirState.OnGround
    return flat.AirState.Jumping if car.is_jumping else flat.AirState.InAir


def to_packet(state, ids, pad_order, frame, velocity=None) -> flat.GamePacket:
    """The packet RLBot would send for this simulator state."""
    players = []
    for agent, car in state.cars.items():
        body = physics(car.physics)
        if velocity is not None:
            body = flat.Physics(body.location, body.rotation, velocity, body.angular_velocity)
        players.append(flat.PlayerInfo(
            physics=body,
            team=car.team_num,
            player_id=ids[agent],
            boost=car.boost_amount,
            air_state=air_state(car),
            has_jumped=car.has_jumped,
            has_double_jumped=car.has_double_jumped,
            has_dodged=car.has_flipped,
            demolished_timeout=car.demo_respawn_timer if car.demo_respawn_timer > 0 else -1,
            is_supersonic=car.is_supersonic,
        ))

    pads = []
    for sim_idx in pad_order:
        remaining = float(state.boost_pad_timers[sim_idx])
        cooldown = 10.0 if int(sim_idx) in BIG_PADS else 4.0
        pads.append(flat.BoostPadState(is_active=remaining == 0, timer=0.0 if remaining == 0 else cooldown - remaining))

    return flat.GamePacket(
        players=players,
        boost_pads=pads,
        balls=[flat.BallInfo(physics=physics(state.ball))],
        match_info=flat.MatchInfo(frame_num=frame, match_phase=flat.MatchPhase.Active),
    )


def main() -> None:
    env = RLGym(
        state_mutator=MutatorSequence(FixedTeamSizeMutator(4, 4), KickoffMutator()),
        obs_builder=PartialInfoObs(), action_parser=HumanlikeAction(), reward_fn=GoalReward(),
        termination_cond=GoalCondition(), truncation_cond=AnyCondition(NoTouchTimeoutCondition(30)),
        transition_engine=RocketSimEngine(),
    )
    obs = env.reset()
    ids = {agent: i + 1 for i, agent in enumerate(obs)}

    # RLBot lists pads in its own order; shuffling proves the mapping undoes that
    pad_order = np.random.default_rng(0).permutation(len(BOOST_LOCATIONS))
    field = flat.FieldInfo(boost_pads=[
        flat.BoostPad(location=vec(BOOST_LOCATIONS[i]), is_full_boost=int(i) in BIG_PADS) for i in pad_order
    ])
    cooldowns = pad_cooldowns(field)
    compat = GameState.create_compat_game_state(field, flat.MatchConfiguration())

    base = PartialInfoObs()._base
    # position, forward, up, velocity, angular velocity, boost for each car block
    compared = list(range(9)) + list(range(9, 43))
    for block_start in range(52, 212, 20):
        compared += list(range(block_start, block_start + 16))
    compared = np.array(compared)

    worst = 0.0
    cold_pads = 0
    for step in range(STEPS):
        actions = {agent: np.array([np.random.randint(90)]) for agent in obs}
        obs, _, terminated, truncated = env.step(actions)
        if any(terminated.values()) or any(truncated.values()):
            obs = env.reset()
            continue
        sim_state = env.state
        packet = to_packet(sim_state, ids, pad_order, step + 1)
        compat.update(packet)
        fix_boost_timers(compat, packet, cooldowns)
        frozen = snapshot(compat)
        cold_pads += int((sim_state.boost_pad_timers > 0).sum())

        for agent, player_id in ids.items():
            want = base.build_obs([agent], sim_state, {})[agent]
            got = base.build_obs([player_id], frozen, {})[player_id]
            diff = np.abs(want[compared] - got[compared])
            if diff.max() > TOLERANCE:
                raise SystemExit(f"step {step}, {agent}: index {compared[diff.argmax()]} is off by "
                                 f"{diff.max():.4f} (sim {want[compared[diff.argmax()]]:.4f}, "
                                 f"bridge {got[compared[diff.argmax()]]:.4f})")
            worst = max(worst, float(diff.max()))

    # memory must stay frozen: a remembered car can't change as the live one moves
    builder = PartialInfoObs()
    agents = list(ids.values())
    builder.reset(agents, compat, {})
    builder.build_obs(agents, snapshot(compat), {})
    remembered = [entry[0] for memory in builder._memory.values() for entry in memory.values()]
    before = [car.physics.linear_velocity.copy() for car in remembered]
    compat.update(to_packet(env.state, ids, pad_order, STEPS + 2, velocity=flat.Vector3(1234.0, -987.0, 55.0)))
    leaked = sum(not np.array_equal(old, car.physics.linear_velocity) for old, car in zip(before, remembered))
    if leaked:
        raise SystemExit(f"{leaked} remembered cars changed when the live state updated")

    # full observation, through the masking and pending-action layers
    obs_full = PartialInfoObs()
    parser = HumanlikeAction()
    shared = {}
    frozen = snapshot(compat)
    obs_full.reset(agents, frozen, shared)
    parser.reset(agents, frozen, shared)
    size = obs_full.build_obs(agents, frozen, shared)[agents[0]].shape[0]
    if size != 212 + EXTRA_OBS_DIM:
        raise SystemExit(f"observation is {size} wide, expected {212 + EXTRA_OBS_DIM}")
    parsed = parser.parse_actions({a: np.array([40], dtype=np.int32) for a in agents}, frozen, shared)
    to_controller(parsed[agents[0]][0])

    print(f"PASS: {STEPS} steps, 8 cars, worst difference {worst:.5f}; "
          f"{cold_pads} cold-pad readings; {len(remembered)} remembered cars stayed frozen; obs width {size}")


if __name__ == "__main__":
    main()
