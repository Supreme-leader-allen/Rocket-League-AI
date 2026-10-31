"""
Runs a trained policy in real Rocket League through RLBot v5.

One process controls all 4 cars (RLBot's "hivemind"), the same way training
puts one policy on every car. Observations and actions come from the training
code itself (Observation.PartialInfoObs, Actions.HumanlikeAction), so the view
cone, reaction delay and 15 decisions/sec are identical. Ball touches, spacing
and the other coordination numbers come from Metrics.CoordinationMetrics, one
row per kickoff-to-goal segment, tracked for this bot's team only.

Started by RLBot, not by hand -- run_matches.py picks the policy and output
file through active_run.json.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from rlbot import flat
from rlbot.managers import Hivemind
from rlgym_compat import GameState

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from Actions import HumanlikeAction  # noqa: E402
from Metrics import CoordinationMetrics  # noqa: E402
from Observation import PartialInfoObs  # noqa: E402
from compat_state import fix_boost_timers, neutral_controls, pad_cooldowns, snapshot, to_controller  # noqa: E402

TICKS_PER_DECISION = 8  # HumanlikeAction's repeats; 120 Hz / 8 = 15 decisions/sec
POLICY_LAYER_SIZES = [2048, 2048, 1024, 1024]  # Train_Ground.POLICY_LAYER_SIZES
OBS_SIZE = 243
N_ACTIONS = 90
PLAY_PHASES = (flat.MatchPhase.Kickoff, flat.MatchPhase.Active)
RESET_PHASES = (flat.MatchPhase.Countdown, flat.MatchPhase.Kickoff)


class PolicyNet(torch.nn.Module):
    """Same layers as rlgym_ppo's DiscreteFF, so its PPO_POLICY.pt loads
    directly without pulling rlgym_ppo and its dependencies into this venv."""

    def __init__(self):
        super().__init__()
        layers, previous = [], OBS_SIZE
        for size in POLICY_LAYER_SIZES:
            layers += [torch.nn.Linear(previous, size), torch.nn.ReLU()]
            previous = size
        layers += [torch.nn.Linear(previous, N_ACTIONS), torch.nn.Softmax(dim=-1)]
        self.model = torch.nn.Sequential(*layers)

    @torch.no_grad()
    def act(self, obs_batch: np.ndarray) -> np.ndarray:
        probs = self.model(torch.as_tensor(obs_batch, dtype=torch.float32))
        return torch.multinomial(probs, 1).flatten().numpy()  # sampled, as in training and evaluate_match


def load_run_config() -> dict:
    with open(HERE / "active_run.json", encoding="utf-8") as f:
        return json.load(f)


class TrainedHivemind(Hivemind):
    def initialize(self):
        config = load_run_config()
        self.policy = PolicyNet()
        self.policy.load_state_dict(torch.load(config["policy"], map_location="cpu"))
        self.policy.eval()

        self.game_state = GameState.create_compat_game_state(self.field_info, self.match_config)
        self.cooldowns = pad_cooldowns(self.field_info)
        self.obs_builder = PartialInfoObs()
        self.action_parser = HumanlikeAction()
        self.metrics = CoordinationMetrics(config["metrics_csv"], config["run_label"])
        self.shared_info = {}
        self.segment_open = False
        self.needs_reset = True
        self.next_decision_frame = -1
        self.previous_phase = None
        self.held_controls = neutral_controls(self.indices)

    def _reset_segment(self, state):
        """A kickoff is a fresh episode to the policy: empty memory and queue."""
        agents = list(self.player_ids)
        self.obs_builder.reset(agents, state, self.shared_info)
        self.action_parser.reset(agents, state, self.shared_info)
        self.metrics.reset(agents, state, self.shared_info)
        self.segment_open = True
        self.needs_reset = False

    def _decide(self, state):
        agents = list(self.player_ids)
        obs = self.obs_builder.build_obs(agents, state, self.shared_info)
        chosen = self.policy.act(np.stack([obs[agent] for agent in agents]))
        actions = {agent: np.array([chosen[i]], dtype=np.int32) for i, agent in enumerate(agents)}
        parsed = self.action_parser.parse_actions(actions, state, self.shared_info)
        self.held_controls = {
            self.indices[i]: to_controller(parsed[agent][0]) for i, agent in enumerate(agents)
        }

    def get_outputs(self, packet: flat.GamePacket) -> dict[int, flat.ControllerState]:
        phase = packet.match_info.match_phase
        self.game_state.update(packet)
        fix_boost_timers(self.game_state, packet, self.cooldowns)

        if phase in RESET_PHASES and self.previous_phase not in RESET_PHASES:
            self.needs_reset = True
        self.previous_phase = phase

        if phase == flat.MatchPhase.GoalScored and self.segment_open:
            # one last row so this segment records who scored
            self.metrics.get_rewards(list(self.player_ids), snapshot(self.game_state), {}, {}, self.shared_info)
            self.segment_open = False

        if phase not in PLAY_PHASES:
            self.next_decision_frame = -1
            self.held_controls = neutral_controls(self.indices)
            return self.held_controls

        frame = packet.match_info.frame_num
        if frame >= self.next_decision_frame:
            state = snapshot(self.game_state)
            if self.needs_reset:
                self._reset_segment(state)
            self.metrics.get_rewards(list(self.player_ids), state, {}, {}, self.shared_info)
            self.game_state.reset_car_ball_touches()
            self._decide(state)
            self.next_decision_frame = frame + TICKS_PER_DECISION

        return self.held_controls

    def retire(self):
        if self.segment_open:
            self.metrics.reset(list(self.player_ids), self.game_state, self.shared_info)  # flushes the last row


if __name__ == "__main__":
    TrainedHivemind(os.environ.get("RLBOT_AGENT_ID", "rocket-league-ai/trained")).run()
