"""
Turns RLBot's live game data into the GameState training saw.

rlgym_compat does most of this, but two differences would silently damage the
bot's play, so they're corrected here. test_obs_parity.py checks the result
against the real training pipeline.

1. Boost pad timers. RocketSim's are seconds UNTIL a pad is ready again
   (rocketsim_engine.py: pad_state.cooldown). RLBot's are seconds SINCE it was
   taken (flat.BoostPadState.timer), and rlgym_compat copies that straight in.
   Fed through unchanged, a pad that just went cold looks about to respawn.

2. In-place updates. rlgym_compat writes new values into the same numpy
   arrays every tick, while PartialInfoObs keeps per-car memory by holding on
   to the arrays it was given (Observation._sighted_car). Handing it the live
   state would make a remembered, hidden car's velocity keep tracking the real
   car, leaking exactly what the view cone is meant to hide. snapshot() gives
   it a frozen copy each decision instead.
"""

import copy
from typing import Dict, List

import numpy as np
from rlbot import flat

BIG_PAD_COOLDOWN = 10.0
SMALL_PAD_COOLDOWN = 4.0


def pad_cooldowns(field_info: flat.FieldInfo) -> np.ndarray:
    """Respawn time of each pad, in RLBot's pad order."""
    return np.array(
        [BIG_PAD_COOLDOWN if pad.is_full_boost else SMALL_PAD_COOLDOWN for pad in field_info.boost_pads],
        dtype=np.float32,
    )


def fix_boost_timers(state, packet: flat.GamePacket, cooldowns: np.ndarray) -> None:
    """Rewrite state.boost_pad_timers as seconds until ready, training's meaning."""
    for rlbot_idx, pad in enumerate(packet.boost_pads):
        remaining = 0.0 if pad.is_active else max(0.0, float(cooldowns[rlbot_idx]) - pad.timer)
        state.boost_pad_timers[state._boost_pad_order_mapping[rlbot_idx]] = remaining


def snapshot(state):
    """An independent copy of the state that later ticks can't change."""
    return copy.deepcopy(state)


def to_controller(controls: np.ndarray) -> flat.ControllerState:
    """One 8-value control row, in the order rocketsim_engine.py reads it."""
    return flat.ControllerState(
        throttle=float(controls[0]),
        steer=float(controls[1]),
        pitch=float(controls[2]),
        yaw=float(controls[3]),
        roll=float(controls[4]),
        jump=bool(controls[5]),
        boost=bool(controls[6]),
        handbrake=bool(controls[7]),
    )


def neutral_controls(indices: List[int]) -> Dict[int, flat.ControllerState]:
    return {index: flat.ControllerState() for index in indices}
