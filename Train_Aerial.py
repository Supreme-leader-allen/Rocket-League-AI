"""
Model B: aerial, built on Model A (see ROADMAP.md). Same network
interface as Train_Ground.py -- POLICY_LAYER_SIZES/CRITIC_LAYER_SIZES,
zero_padding, LookupTableAction action space are all byte-identical --
so Model A's checkpoint loads as a warm start via checkpoint_load_folder
without a shape mismatch. Only the state distribution and reward
weighting change: the state mutator sometimes spawns the ball in the
air (RandomAirborneBallMutator, defined in this file -- no built-in
rlgym mutator does this, confirmed via
`dir(rlgym.rocket_league.state_mutators)` against the installed
version), and InAirReward's weight is raised instead of held near zero.

Fill in CHECKPOINT_TO_LOAD below with Model A's actual saved checkpoint
path (a directory containing PPO_POLICY.pt etc.) before running -- e.g.
via the AERIAL_CHECKPOINT_TO_LOAD env var, or by editing the constant
directly. Pbt.find_latest_checkpoint("checkpoints/model_a_ground") will
resolve it for you if Model A was trained with Train_Ground.py's
defaults:

    AERIAL_CHECKPOINT_TO_LOAD=checkpoints/model_a_ground-<ts>/<timesteps> python Train_Aerial.py

The Learner kwargs below (checkpoints_save_folder, checkpoint_load_folder,
render, render_delay, save_every_ts) were confirmed against
help(Learner.__init__) on the installed rlgym_ppo 1.3.13 in
Train_Ground.py and aren't re-verified separately here since this file
constructs Learner the same way.

To watch training live with RLViser:
    RLGYM_RENDER=1 python Train_Aerial.py
"""

import os
import random
from typing import Any, Dict

import numpy as np

CHECKPOINT_TO_LOAD = os.environ.get("AERIAL_CHECKPOINT_TO_LOAD", "")


def _loaded_ts_from_checkpoint_path(path: str) -> float:
    """Model A's cumulative_timesteps, read off the digit-named folder in
    its checkpoint path (Learner.save()'s own naming convention --
    confirmed against its source). Computed at true module top level,
    not inside `if __name__ == "__main__":`, because build_rlgym_v2_env
    needs it too (for AnnealedCombinedReward's initial_timesteps, so
    Model B's shaping anneal starts from Model A's actual progress
    instead of 0 -- see docs/ISSUES.md P3) and build_rlgym_v2_env runs
    inside rlgym_ppo's spawned worker processes. Windows multiprocessing
    only supports the "spawn" start method (confirmed via
    multiprocessing.get_all_start_methods()), which re-imports this
    module fresh in every worker -- re-running all true top-level code,
    but NOT the `if __name__ == "__main__":` block, which is why
    anything a worker-side function needs has to live out here, the same
    way CHECKPOINT_TO_LOAD/N_PROC/etc. already do. Returns 0.0 (rather
    than raising) for an empty or malformed path -- __main__ below does
    the actual "you forgot to set this" validation, since only the
    process that constructs Learner needs to fail loudly on that.
    """
    if not path:
        return 0.0
    basename = os.path.basename(os.path.normpath(path))
    return float(basename) if basename.isdigit() else 0.0


LOADED_TIMESTEPS = _loaded_ts_from_checkpoint_path(CHECKPOINT_TO_LOAD)

RENDER = os.environ.get("RLGYM_RENDER", "0") == "1"
N_PROC = int(os.environ.get("AERIAL_N_PROC", 32))
CHECKPOINTS_SAVE_FOLDER = os.environ.get("AERIAL_CHECKPOINT_DIR", "checkpoints/model_b_aerial")
# How many NEW timesteps to train Model B for, on top of whatever
# cumulative_timesteps Model A's checkpoint already carries -- NOT an
# absolute limit. See the timestep_limit derivation in __main__ below for
# why: Learner.load() restores cumulative_timesteps from the loaded
# checkpoint (confirmed against Learner.save()'s source -- the
# digit-named checkpoint subfolder IS str(cumulative_timesteps)), and its
# main loop is a plain `while cumulative_timesteps < timestep_limit`. If
# this were treated as an absolute limit (the previous behavior), warm-
# starting from a Model A checkpoint that already met or exceeded it
# would make the loop exit on iteration zero -- Model B would train
# nothing and silently become a renamed copy of Model A. See
# docs/ISSUES.md P1.
AERIAL_ADDITIONAL_TIMESTEPS = int(os.environ.get("AERIAL_ADDITIONAL_TIMESTEPS", 1_000_000_000))
SAVE_EVERY_TS = int(os.environ.get("AERIAL_SAVE_EVERY_TS", 1_000_000))
# Previously bare constants with no override -- confirmed by running the
# real experiments/02_aerial.sh pipeline: every other script's CSV
# run_label matches its own filename (01_ground.sh -> "01_ground", via
# Train_Ground.py's PBT_RUN_LABEL), but 02_aerial.sh had no equivalent
# env var to set, so its real output was unavoidably labeled
# "model_b_aerial" -- surprising to anyone following the "01_ground",
# "02_aerial" naming convention every other script actually produces,
# and something Plot_Results.py's --highlight-label default/examples
# would silently miss if pointed at "02_aerial" instead. Same
# AERIAL_*-prefixed env var pattern this file already uses elsewhere
# (AERIAL_CHECKPOINT_TO_LOAD, AERIAL_N_PROC, etc.), default unchanged so
# nothing breaks if unset.
RUN_LABEL = os.environ.get("AERIAL_RUN_LABEL", "model_b_aerial")
CSV_PATH = os.environ.get("AERIAL_CSV_PATH", "metrics/aerial_training.csv")
FITNESS_CSV_PATH = os.environ.get("AERIAL_FITNESS_CSV_PATH", "metrics/aerial_fitness.csv")

AIRBORNE_SPAWN_PROBABILITY = 0.5


class RandomAirborneBallMutator:
    """
    Fills a real gap: no built-in rlgym.rocket_league.state_mutators
    class spawns the ball in the air. Applied AFTER KickoffMutator in
    the MutatorSequence, so it only overrides the ball's state and
    leaves KickoffMutator's car placements untouched. With probability
    `probability`, replaces the grounded kickoff ball spot with a
    random airborne position/velocity; otherwise leaves the ordinary
    grounded kickoff as-is, so Model B still sees plenty of normal
    ground starts too -- widening the state distribution per
    ROADMAP.md, not replacing it outright.
    """

    def __init__(self, probability: float = 0.5, min_height: float = 300.0,
                 max_height: float = 1600.0, max_speed: float = 1500.0):
        self.probability = probability
        self.min_height = min_height
        self.max_height = max_height
        self.max_speed = max_speed

    def apply(self, state, shared_info: Dict[str, Any]) -> None:
        from rlgym.rocket_league import common_values

        if random.random() > self.probability:
            return

        margin = common_values.BALL_RADIUS * 2
        x = random.uniform(-common_values.SIDE_WALL_X + margin, common_values.SIDE_WALL_X - margin)
        y = random.uniform(-common_values.BACK_NET_Y + margin, common_values.BACK_NET_Y - margin)
        z = random.uniform(self.min_height, self.max_height)

        state.ball.position = np.array([x, y, z], dtype=np.float32)
        state.ball.linear_velocity = np.random.uniform(-self.max_speed, self.max_speed, size=3).astype(np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)


def build_rlgym_v2_env():
    from rlgym.api import RLGym
    from rlgym.rocket_league.done_conditions import (
        GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition,
    )
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    from rlgym_ppo.util import RLGymV2GymWrapper

    from Rewards import SpeedTowardBallReward, InAirReward, VelocityBallToGoalReward, AnnealedCombinedReward
    from Observation import PartialInfoObs
    from Actions import HumanlikeAction
    from Metrics import CoordinationMetrics, FitnessTracker

    no_touch_timeout_seconds = 30
    game_timeout_seconds = 300

    # Full raw action space -- identical to Train_Ground.py's, must stay
    # that way for the checkpoint to remain valid.
    action_parser = HumanlikeAction(repeats=8, reaction_delay_ticks=3)

    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds),
    )

    # Re-annealed from a higher starting point than Train_Ground.py: the
    # ground behaviors are already learned (warm-started from Model A),
    # but the aerial behaviors are new and need dense guidance again.
    # InAirReward's weight is raised (0.2 -> 0.05) instead of held near
    # zero (Model A used 0.002 -> 0.0) -- this is the one substantive
    # reward change between the two phases, per ROADMAP.md.
    #
    # Annealed over cumulative timesteps, not wall-clock time -- see
    # AnnealedCombinedReward's docstring in Rewards.py for why the
    # wall-clock version was a confirmed bug (docs/ISSUES.md P3), and
    # LOADED_TIMESTEPS's docstring above for why it's computed at module
    # top level. initial_timesteps is seeded from Model A's own
    # checkpoint progress (LOADED_TIMESTEPS) rather than 0, since Model
    # B's very first step already starts from a policy that isn't naive.
    ANNEAL_TIMESTEPS = 200_000_000
    shaping = AnnealedCombinedReward(
        weighted_rewards=[
            (SpeedTowardBallReward(), 0.01, 0.0),
            (VelocityBallToGoalReward(), 0.1, 0.02),
            (InAirReward(), 0.2, 0.05),
        ],
        anneal_timesteps=ANNEAL_TIMESTEPS,
        n_proc=N_PROC,
        initial_timesteps=LOADED_TIMESTEPS,
    )
    coordination_metrics = CoordinationMetrics(csv_path=CSV_PATH, run_label=RUN_LABEL)
    fitness_tracked_goal_reward = FitnessTracker(
        GoalReward(), csv_path=FITNESS_CSV_PATH, run_label=RUN_LABEL,
    )

    reward_fn = CombinedReward(
        (shaping, 1.0),
        (fitness_tracked_goal_reward, 10.0),
        (coordination_metrics, 0.0),
    )

    # zero_padding=4 -- MUST stay identical to Train_Ground.py (see that
    # file's comment on why 4, not 3) or the checkpoint's input layer
    # size won't match this env's observation size.
    obs_builder = PartialInfoObs(zero_padding=4)

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=4, orange_size=4),
        KickoffMutator(),
        RandomAirborneBallMutator(probability=AIRBORNE_SPAWN_PROBABILITY),
    )

    renderer = None
    if RENDER:
        from rlgym.rocket_league.rlviser import RLViserRenderer
        renderer = RLViserRenderer(tick_rate=120 / 8)

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=renderer,
    )

    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":
    from rlgym_ppo import Learner

    if not CHECKPOINT_TO_LOAD:
        raise SystemExit(
            "CHECKPOINT_TO_LOAD is empty -- set the AERIAL_CHECKPOINT_TO_LOAD env "
            "var (or edit the constant at the top of this file) to Model A's saved "
            "checkpoint directory before running Train_Aerial.py, e.g.:\n"
            "  AERIAL_CHECKPOINT_TO_LOAD=checkpoints/model_a_ground-<ts>/<timesteps> "
            "python Train_Aerial.py\n"
            "Use Pbt.find_latest_checkpoint('checkpoints/model_a_ground') to resolve "
            "it if Model A was trained with Train_Ground.py's defaults."
        )

    # LOADED_TIMESTEPS (computed at module top level -- see its docstring
    # above) is the loaded checkpoint's own cumulative_timesteps. See
    # AERIAL_ADDITIONAL_TIMESTEPS's comment above for why timestep_limit
    # must be derived relative to this rather than passed as an absolute
    # value. 0.0 here means the path didn't end in a digit-named
    # checkpoint subfolder, which is almost certainly a mistake given
    # CHECKPOINT_TO_LOAD is required to be set at all.
    if LOADED_TIMESTEPS == 0.0:
        raise SystemExit(
            f"CHECKPOINT_TO_LOAD={CHECKPOINT_TO_LOAD!r} doesn't end in a "
            f"digit-named checkpoint subfolder (e.g. '.../model_a_ground-"
            f"<ts>/100024') -- can't determine its cumulative_timesteps to "
            f"derive Model B's timestep_limit. Point it at a specific "
            f"numbered checkpoint directory, e.g. what "
            f"Pbt.find_latest_checkpoint('checkpoints/model_a_ground') resolves."
        )
    TIMESTEP_LIMIT = int(LOADED_TIMESTEPS) + AERIAL_ADDITIONAL_TIMESTEPS
    print(f"Warm-starting from {CHECKPOINT_TO_LOAD} (cumulative_timesteps={int(LOADED_TIMESTEPS)}); "
          f"training to timestep_limit={TIMESTEP_LIMIT} "
          f"(+{AERIAL_ADDITIONAL_TIMESTEPS} new steps).")

    min_inference_size = max(1, int(round(N_PROC * 0.9)))

    # Must stay byte-for-byte identical to Train_Ground.py's -- this is
    # what makes checkpoint_load_folder a valid warm start instead of a
    # shape-mismatch error.
    POLICY_LAYER_SIZES = [2048, 2048, 1024, 1024]
    CRITIC_LAYER_SIZES = [2048, 2048, 1024, 1024]

    learner = Learner(
        build_rlgym_v2_env,
        n_proc=N_PROC,
        min_inference_size=min_inference_size,
        metrics_logger=None,
        ppo_batch_size=100_000,
        policy_layer_sizes=POLICY_LAYER_SIZES,
        critic_layer_sizes=CRITIC_LAYER_SIZES,
        ts_per_iteration=100_000,
        exp_buffer_size=300_000,
        ppo_minibatch_size=50_000,
        ppo_ent_coef=0.01,
        policy_lr=1e-4,
        critic_lr=1e-4,
        ppo_epochs=2,
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=SAVE_EVERY_TS,
        checkpoints_save_folder=CHECKPOINTS_SAVE_FOLDER,
        timestep_limit=TIMESTEP_LIMIT,
        log_to_wandb=False,
        render=RENDER,
        render_delay=0,
        checkpoint_load_folder=CHECKPOINT_TO_LOAD,
    )
    learner.learn()
