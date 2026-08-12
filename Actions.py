"""
HumanlikeAction: full raw LookupTableAction action space (no abstraction --
the agent always picks its own raw controller action) with two "humanlike"
constraints folded in, per ROADMAP.md:

Reaction time -- a per-agent delay queue. The discrete action index chosen
at decision step t isn't applied to the sim until decision step
t + reaction_delay_ticks. Note reaction_delay_ticks counts *decision steps*
(env.step() calls), not raw physics ticks -- at repeats=8 (120Hz physics /
8 = 15 decisions/sec), reaction_delay_ticks=3 is ~200ms, in the human 150-250ms
reaction-time range.

Input rate -- repeats folds tick-skip/action-repeat in alongside the delay
queue: the chosen (delayed) action is held for `repeats` consecutive physics
ticks, which is what RocketSimEngine.step expects (an (repeats, 8) array per
agent, one row per physics tick simulated this env.step() call).

The queue is filled with a neutral no-input action before an agent has made
enough real decisions to fill the delay window. That neutral action is NOT
index 0 of LookupTableAction's table -- index 0 is a full reverse+left-turn
action ([-1, -1, 0, -1, 0, 0, 0, 0]). The actual all-zero neutral action is
found by searching the table for the all-zero row rather than assuming a
fixed index, so this stays correct even if make_lookup_table's ordering
changes.
"""

from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np
from rlgym.api import ActionParser, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.action_parsers import LookupTableAction


class HumanlikeAction(ActionParser[AgentID, np.ndarray, np.ndarray, GameState, Tuple[str, int]]):
    def __init__(self, repeats: int = 8, reaction_delay_ticks: int = 3):
        super().__init__()
        self._table_parser = LookupTableAction()
        self._lookup_table = self._table_parser.make_lookup_table()
        self._neutral_index = self._find_neutral_index(self._lookup_table)
        self._repeats = repeats
        self._delay_steps = reaction_delay_ticks
        self._queues: Dict[AgentID, deque] = {}

    @staticmethod
    def _find_neutral_index(lookup_table: np.ndarray) -> int:
        zero_rows = np.where(~lookup_table.any(axis=1))[0]
        if len(zero_rows) == 0:
            raise ValueError("LookupTableAction's table has no all-zero (neutral) action to use as delay-queue filler.")
        return int(zero_rows[0])

    def get_action_space(self, agent: AgentID) -> Tuple[str, int]:
        return self._table_parser.get_action_space(agent)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self._table_parser.reset(agents, initial_state, shared_info)
        self._queues = {
            agent: deque([self._neutral_index] * self._delay_steps, maxlen=self._delay_steps + 1)
            for agent in agents
        }

    def parse_actions(self, actions: Dict[AgentID, np.ndarray], state: GameState,
                       shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        parsed_actions = {}
        for agent, action in actions.items():
            chosen_index = int(np.asarray(action).reshape(-1)[0])

            queue = self._queues.setdefault(
                agent, deque([self._neutral_index] * self._delay_steps, maxlen=self._delay_steps + 1)
            )
            queue.append(chosen_index)
            delayed_index = queue[0]

            controls = self._lookup_table[delayed_index]
            parsed_actions[agent] = np.tile(controls, (self._repeats, 1))

        return parsed_actions
