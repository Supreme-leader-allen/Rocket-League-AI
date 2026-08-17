"""
HumanlikeAction: wraps LookupTableAction to add reaction delay and to fold
in the same input-repeat behavior as RepeatAction, so both humanlike
constraints (reaction time + input rate) live in one parser.

Reaction delay: the discrete action index chosen at decision step t isn't
handed to LookupTableAction until step t + delay_steps. Default delay is
tuned for ~150-200ms given tick_skip=8 (120Hz physics / 8 = 15 decisions/
sec, so ~66ms per decision step -- delay_steps=3 is ~200ms).

Input rate: each decision's resulting controls are held for `repeats`
physics ticks before the next decision is requested, same convention as
RepeatAction. Don't also wrap this in RepeatAction -- this class already
does that tiling, and double-repeating will silently slow decisions down
far more than intended.

Verified against the installed rlgym.rocket_league.action_parsers.LookupTableAction:
- Index 0 is NOT neutral -- it's [-1, -1, 0, -1, 0, 0, 0, 0] (full reverse
  + full left steer), since make_lookup_table()'s ground loop starts at
  throttle=-1, steer=-1. The actual all-zero row is index 8. Rather than
  hardcode 8 (fragile if the installed version's table generation order
  ever changes), _idle_action_index is found at construction time by
  searching the table for the fully-zero row.
- Real actions arrive from RLGymV2GymWrapper as a 1-element int32 numpy
  array per agent (confirmed via rlgym_ppo.batched_agents.batched_agent's
  env.step(action_buffer) call, which reshapes to (n_agents, action_dim)
  and indexes per-agent, giving shape (1,) since this is a discrete
  action space). The idle filler used to seed the delay queue must match
  that shape -- a bare Python int filler crashed with AttributeError
  ('int' object has no attribute 'shape') inside LookupTableAction.
  parse_actions's `len(action.shape)` check during the first
  reaction_delay_ticks steps of every episode, confirmed by running
  parse_actions() directly.
- reset()'s signature was missing the `agents` parameter that
  rlgym.api.ActionParser.reset (and RLGym.reset(), which calls
  action_parser.reset(agents, state, shared_info)) actually requires --
  confirmed by running env.reset(), which raised TypeError: reset() takes
  3 positional arguments but 4 were given.
- Tiling controls to shape (repeats, 8) matches LookupTableAction's
  parse_actions output (each parsed action is a length-8 control row);
  np.tile broadcasts that to (repeats, 8) as rlgym's ActionParser
  docstring specifies ("(ticks, actiondim=8)"), and this is fed straight
  to RocketSimEngine rather than through a separate RepeatAction, so no
  double-repeat risk.
"""

from collections import deque
from typing import Dict, Any, List

import numpy as np
from rlgym.api import ActionParser, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.action_parsers import LookupTableAction


class HumanlikeAction(ActionParser[AgentID, int, np.ndarray, GameState, int]):
    def __init__(self, repeats: int = 8, reaction_delay_ticks: int = 3):
        super().__init__()
        self._table_parser = LookupTableAction()
        self._repeats = repeats
        self._delay_steps = max(0, reaction_delay_ticks)

        neutral_rows = np.where((self._table_parser._lookup_table == 0).all(axis=1))[0]
        if len(neutral_rows) == 0:
            raise RuntimeError(
                "HumanlikeAction: no all-zero row found in LookupTableAction's "
                "table -- can't pick an idle filler action."
            )
        self._idle_action_index = np.array([neutral_rows[0]], dtype=np.int32)

        self._queues: Dict[AgentID, deque] = {}

    def get_action_space(self, agent: AgentID):
        return self._table_parser.get_action_space(agent)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self._table_parser.reset(agents, initial_state, shared_info)
        self._queues.clear()

    def parse_actions(self, actions: Dict[AgentID, int], state: GameState,
                       shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        delayed_indices = {}
        for agent, action in actions.items():
            queue = self._queues.setdefault(
                agent,
                deque([self._idle_action_index] * self._delay_steps, maxlen=self._delay_steps + 1),
            )
            queue.append(action)
            # oldest entry in the queue is the decision that actually executes now
            delayed_indices[agent] = queue[0]

        parsed = self._table_parser.parse_actions(delayed_indices, state, shared_info)

        return {agent: np.tile(controls, (self._repeats, 1)) for agent, controls in parsed.items()}