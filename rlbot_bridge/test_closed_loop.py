"""
Runs bot.py's real decision loop in the simulator, with no game involved.

Both teams are a TrainedHivemind driving cars purely through packets in and
controller states out, exactly what RLBot does in a match. If the bridge is
right, the policy plays the way it did in training; a wrong control mapping,
observation or timing shows up as random-looking play (random: ~2.6 touches/min,
0.83 air time).

    .venv\\Scripts\\python.exe test_closed_loop.py <path to PPO_POLICY.pt>
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from rlbot import flat
from rlgym.api import ActionParser, RLGym
from rlgym.rocket_league.common_values import BOOST_LOCATIONS
from rlgym.rocket_league.done_conditions import AnyCondition, GoalCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from Observation import PartialInfoObs  # noqa: E402
from bot import TICKS_PER_DECISION, TrainedHivemind  # noqa: E402
from test_obs_parity import BIG_PADS, to_packet, vec  # noqa: E402

STEPS = 1500  # 100 seconds of game time
STEPS_PER_MINUTE = 15 * 60


class PassThrough(ActionParser):
    """Sends the bots' controller output straight to the physics engine."""

    def get_action_space(self, agent):
        return "real", 8

    def reset(self, agents, initial_state, shared_info):
        pass

    def parse_actions(self, actions, state, shared_info):
        return {agent: np.tile(controls, (TICKS_PER_DECISION, 1)) for agent, controls in actions.items()}


def make_bot(team, player_ids, indices, field, config):
    bot = TrainedHivemind("test")
    bot.team, bot.player_ids, bot.indices = team, player_ids, indices
    bot.field_info, bot.match_config = field, config
    bot.initialize()
    return bot


def controls_row(c: flat.ControllerState) -> np.ndarray:
    return np.array([c.throttle, c.steer, c.pitch, c.yaw, c.roll, c.jump, c.boost, c.handbrake], dtype=np.float32)


def main() -> None:
    policy = Path(sys.argv[1]).resolve()
    run_file = HERE / "active_run.json"
    with tempfile.TemporaryDirectory() as out:
        run_file.write_text(json.dumps({
            "policy": str(policy), "run_label": "closed_loop", "metrics_csv": os.path.join(out, "m.csv"),
        }), encoding="utf-8")

        env = RLGym(
            state_mutator=MutatorSequence(FixedTeamSizeMutator(4, 4), KickoffMutator()),
            obs_builder=PartialInfoObs(), action_parser=PassThrough(), reward_fn=GoalReward(),
            termination_cond=GoalCondition(), truncation_cond=AnyCondition(NoTouchTimeoutCondition(30)),
            transition_engine=RocketSimEngine(),
        )
        obs = env.reset()
        agents = list(obs)
        ids = {agent: i + 1 for i, agent in enumerate(agents)}
        index_of = {agent: i for i, agent in enumerate(agents)}
        blue = [a for a in agents if not env.state.cars[a].is_orange]
        orange = [a for a in agents if env.state.cars[a].is_orange]

        pad_order = np.arange(len(BOOST_LOCATIONS))
        field = flat.FieldInfo(boost_pads=[
            flat.BoostPad(location=vec(BOOST_LOCATIONS[i]), is_full_boost=i in BIG_PADS) for i in pad_order
        ])
        config = flat.MatchConfiguration()
        bots = [make_bot(0, [ids[a] for a in blue], [index_of[a] for a in blue], field, config),
                make_bot(1, [ids[a] for a in orange], [index_of[a] for a in orange], field, config)]

        touches = goals = air = cars_counted = 0
        frame = 1
        started = time.time()
        for step in range(STEPS):
            packet = to_packet(env.state, ids, pad_order, frame)
            outputs = {}
            for bot in bots:
                outputs.update(bot.get_outputs(packet))
            actions = {agent: controls_row(outputs[index_of[agent]]) for agent in agents}
            _, _, terminated, truncated = env.step(actions)
            frame += TICKS_PER_DECISION

            state = env.state
            touches += sum(car.ball_touches for car in state.cars.values())
            air += sum(not car.on_ground for car in state.cars.values())
            cars_counted += len(state.cars)
            if any(terminated.values()) or any(truncated.values()):
                goals += any(terminated.values())
                env.reset()
                for bot in bots:
                    bot.needs_reset = True

        elapsed = time.time() - started
        minutes = STEPS / STEPS_PER_MINUTE
        print(f"{STEPS} steps in {elapsed:.0f}s ({elapsed / STEPS * 1000:.0f} ms per decision for 8 cars)")
        print(f"touches/min {touches / minutes:.1f}  air time {air / cars_counted:.2f}  goals {goals}")
    run_file.unlink()


if __name__ == "__main__":
    main()
