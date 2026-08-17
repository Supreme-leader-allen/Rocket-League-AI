"""
Model A: ground, 4v4. This is the "for show" bot -- coordinated 4v4 play
on the ground, with the action space and network already sized for
everything Model B (aerial) will need later (see ROADMAP.md).

Run with:
    python train_ground.py

Confirmed against `help(Learner.__init__)` on the installed rlgym_ppo
1.3.13: `checkpoints_save_folder`, `checkpoint_load_folder`, `render`, and
`render_delay` are all real, correctly-spelled kwargs. One behavioral
gotcha found in the process: Learner's own default for
`checkpoint_load_folder` is the string `"latest"`, not `None` -- it means
"scan for a prior run's checkpoint and auto-resume if one exists", which
is why this file always passes `checkpoint_load_folder=CHECKPOINT_LOAD_FOLDER`
explicitly below rather than omitting the kwarg when unset (see that
line's comment).

Before a long run, still worth checking:
- That n_proc=32 and the batch sizes below are sane for your CPU core
  count -- these are the tutorial's defaults for a fairly beefy machine;
  scale down if training is memory-starved or CPU-bound.

To watch training live with RLViser, run:
    RLGYM_RENDER=1 python train_ground.py

RLViser is what requirements.txt's `rlgym[rl-rlviser]` extra installs.
Leave rendering off for real training runs -- it slows one of your
n_proc environments down to real-time speed on purpose so you can
actually watch it, which is fine for a quick sanity check but will tank
your overall steps/sec if left on for a long run. If a window doesn't
pop up when RLGYM_RENDER=1, check whether the rl-rlviser extra installed
a standalone RLViser program that needs to be running separately
alongside training rather than launching automatically -- this varies
by version, worth a quick check against your installed package.

PBT_* environment variables (all optional, default to the same values
this file used before PBT existed) let pbt.py launch this script as a
population member without duplicating it: PBT_POLICY_LR, PBT_CRITIC_LR,
PBT_ENT_COEF, PBT_N_PROC, PBT_CHECKPOINT_DIR (overrides
checkpoints_save_folder), PBT_CHECKPOINT_LOAD_DIR (warm start, unset =
fresh start), PBT_TIMESTEP_LIMIT, PBT_SAVE_EVERY_TS (must be smaller
than PBT_TIMESTEP_LIMIT or a short generation saves zero checkpoints --
see SAVE_EVERY_TS's comment below), PBT_RUN_LABEL, PBT_CSV_PATH,
PBT_FITNESS_CSV_PATH. See pbt.py.

Lucy-SKG-style auxiliary abstraction (see auxiliary.py): set AUX_LOGGING=1
to log (observation, reward) data for train_auxiliary_encoder.py to
train on, and AUX_ENCODER_CHECKPOINT=<path> (read by obs.py, not this
file) once you've trained an encoder to actually use it. Both default
off/unset -- nothing about this file's behavior changes unless you opt
in to one or both.
    AUX_LOGGING=1 python train_ground.py
"""

import os

import numpy as np

RENDER = os.environ.get("RLGYM_RENDER", "0") == "1"
AUX_LOGGING = os.environ.get("AUX_LOGGING", "0") == "1"
AUX_DATA_DIR = os.environ.get("AUX_DATA_DIR", "metrics/aux_data")

POLICY_LR = float(os.environ.get("PBT_POLICY_LR", 1e-4))
CRITIC_LR = float(os.environ.get("PBT_CRITIC_LR", 1e-4))
ENT_COEF = float(os.environ.get("PBT_ENT_COEF", 0.01))
N_PROC = int(os.environ.get("PBT_N_PROC", 32))
CHECKPOINTS_SAVE_FOLDER = os.environ.get("PBT_CHECKPOINT_DIR", "checkpoints/model_a_ground")
CHECKPOINT_LOAD_FOLDER = os.environ.get("PBT_CHECKPOINT_LOAD_DIR") or None
TIMESTEP_LIMIT = int(os.environ.get("PBT_TIMESTEP_LIMIT", 1_000_000_000))
# Confirmed against Learner._learn()'s source: it only calls self.save()
# when ts_since_last_save >= save_every_ts (periodic, checked once per
# epoch) or on a keyboard 'c'/'q' press -- there is NO save-on-exit when
# the loop ends naturally because cumulative_timesteps hit
# timestep_limit. A PBT generation whose PBT_TIMESTEP_LIMIT is smaller
# than save_every_ts would therefore save zero checkpoints, silently
# breaking pbt.py's exploit/evaluate steps (nothing for
# find_latest_checkpoint to find). pbt.py sets this smaller than
# GENERATION_TIMESTEPS to guarantee at least one save per generation.
SAVE_EVERY_TS = int(os.environ.get("PBT_SAVE_EVERY_TS", 1_000_000))
RUN_LABEL = os.environ.get("PBT_RUN_LABEL", "model_a_ground")
CSV_PATH = os.environ.get("PBT_CSV_PATH", "metrics/ground_training.csv")
FITNESS_CSV_PATH = os.environ.get("PBT_FITNESS_CSV_PATH", "metrics/ground_fitness.csv")


def build_rlgym_v2_env(run_label: str = None, csv_path: str = None, fitness_csv_path: str = None):
    """
    run_label/csv_path/fitness_csv_path are parameterized (rather than
    hardcoded below), defaulting to the PBT_RUN_LABEL/PBT_CSV_PATH/
    PBT_FITNESS_CSV_PATH env vars (themselves defaulting to the plain
    non-PBT values) so both run_baseline.py and pbt.py can reuse this
    exact env config -- same rewards, obs, action space, state mutator --
    while logging under different labels/files. Don't hardcode a
    different CoordinationMetrics/FitnessTracker call somewhere else;
    always go through these parameters.
    """
    run_label = run_label if run_label is not None else RUN_LABEL
    csv_path = csv_path if csv_path is not None else CSV_PATH
    fitness_csv_path = fitness_csv_path if fitness_csv_path is not None else FITNESS_CSV_PATH

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

    # Full raw action space, delay + repeat baked in for the humanlike
    # reaction-time / input-rate constraints. No abstraction -- every
    # decision is a real controller input, just delayed and held like a
    # human's would be.
    action_parser = HumanlikeAction(repeats=8, reaction_delay_ticks=3)

    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds),
    )

    # Dense shaping anneals toward (near-)zero over wall-clock training
    # time, per Liu et al. -- tune ANNEAL_SECONDS to your actual run
    # length; 2 hours is a starting guess, not a researched constant.
    # InAirReward is included but weighted toward 0 throughout: the agent
    # can still jump/fly any time (full action space), it's just not
    # rewarded for it in this phase.
    ANNEAL_SECONDS = 2 * 60 * 60
    shaping = AnnealedCombinedReward(
        weighted_rewards=[
            (SpeedTowardBallReward(), 0.01, 0.0),
            (VelocityBallToGoalReward(), 0.1, 0.02),
            (InAirReward(), 0.002, 0.0),
        ],
        anneal_seconds=ANNEAL_SECONDS,
    )
    # Zero-weight -- pure data collection, no effect on training. See
    # metrics.py for what each column means and why overcommit_rate /
    # simultaneous_air_rate were added on top of your original metric list.
    coordination_metrics = CoordinationMetrics(
        csv_path=csv_path,
        run_label=run_label,
    )

    # GoalReward wrapped in FitnessTracker: identical contribution to
    # training (weight 10.0, same as bare GoalReward would be) but also
    # logs episode_return for pbt.py's fitness ranking. See metrics.py for
    # why PBT fitness uses this instead of coordination_metrics' columns.
    fitness_tracked_goal_reward = FitnessTracker(
        GoalReward(), csv_path=fitness_csv_path, run_label=run_label,
    )

    reward_entries = [
        (shaping, 1.0),
        (fitness_tracked_goal_reward, 10.0),  # sparse, team-shared by construction -- the credit-assignment mechanism itself
        (coordination_metrics, 0.0),
    ]
    if AUX_LOGGING:
        # Zero-weight, same as the others -- logs data for the SR/RP
        # auxiliary networks (auxiliary.py), no effect on training.
        from Metrics import AuxiliaryDataLogger
        reward_entries.append((AuxiliaryDataLogger(data_dir=AUX_DATA_DIR), 0.0))

    reward_fn = CombinedReward(*reward_entries)

    # zero_padding is DefaultObs's "max cars per team" -- for this
    # project's fixed 4v4, that means zero_padding=4, giving 3 padded
    # ally slots (teammates besides self) + 4 padded enemy slots.
    # zero_padding=3 was confirmed (by direct execution) to be wrong:
    # the real 4v4 roster (3 allies, 4 enemies) already exceeds the
    # padding minimums implied by 3 (2 allies, 3 enemies), so no padding
    # ever triggers and the built observation comes out 40 elements
    # larger (212) than what get_obs_space declares (172) -- which
    # rlgym_ppo uses to size the policy network's input layer, so
    # training crashed immediately with a matmul shape-mismatch.
    # zero_padding=4 makes declared and actual sizes match exactly (both
    # 212), verified directly against DefaultObs with this project's
    # actual FixedTeamSizeMutator(blue_size=4, orange_size=4).
    obs_builder = PartialInfoObs(zero_padding=4)

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=4, orange_size=4),
        KickoffMutator(),
    )

    renderer = None
    if RENDER:
        from rlgym.rocket_league.rlviser import RLViserRenderer
        renderer = RLViserRenderer(tick_rate=120 / 8)  # matches HumanlikeAction's repeats=8

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

    min_inference_size = max(1, int(round(N_PROC * 0.9)))

    # POLICY_LAYER_SIZES / CRITIC_LAYER_SIZES must stay byte-for-byte
    # identical in train_aerial.py -- this is what makes checkpoint_load_folder
    # a valid warm start instead of a shape-mismatch error. NOT overridable
    # via PBT env vars on purpose: PBT evolves training hyperparameters
    # (learning rates, entropy coefficient), not architecture -- changing
    # layer sizes between population members would break checkpoint
    # exploit/copy entirely.
    POLICY_LAYER_SIZES = [2048, 2048, 1024, 1024]
    CRITIC_LAYER_SIZES = [2048, 2048, 1024, 1024]

    # Basic self-play only (both teams mirror the current live policy).
    # For a real checkpoint pool of frozen past opponents (the actual
    # autocurriculum mechanism from ROADMAP.md), you need to wire
    # self_play.CheckpointPool into build_rlgym_v2_env's orange-team
    # action selection -- not done here yet. See self_play.py.
    learner_kwargs = dict(
        n_proc=N_PROC,
        min_inference_size=min_inference_size,
        metrics_logger=None,
        ppo_batch_size=100_000,
        policy_layer_sizes=POLICY_LAYER_SIZES,
        critic_layer_sizes=CRITIC_LAYER_SIZES,
        ts_per_iteration=100_000,
        exp_buffer_size=300_000,
        ppo_minibatch_size=50_000,
        ppo_ent_coef=ENT_COEF,
        policy_lr=POLICY_LR,
        critic_lr=CRITIC_LR,
        ppo_epochs=2,
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=SAVE_EVERY_TS,
        checkpoints_save_folder=CHECKPOINTS_SAVE_FOLDER,  # verify exact param name -- see module docstring
        timestep_limit=TIMESTEP_LIMIT,
        log_to_wandb=False,
        render=RENDER,          # slows one env to real-time and pipes it to RLViser -- see module docstring
        render_delay=0,         # seconds between rendered frames; raise this to slow playback down further
        # Always passed explicitly (verified against Learner's source):
        # Learner's own default is checkpoint_load_folder="latest", which
        # is NOT the same as "fresh start" -- it makes Learner search
        # checkpoints_save_folder's parent directory for a sibling
        # timestamped folder from a PRIOR run and silently auto-resume
        # from it if one happens to exist. Passing None explicitly is a
        # real no-load (Learner only attempts a load when
        # `checkpoint_load_folder is not None`), which is what "fresh
        # start" is actually supposed to mean here -- important for PBT
        # generation 0 reusing the same CHECKPOINTS_SAVE_FOLDER name
        # across runs.
        checkpoint_load_folder=CHECKPOINT_LOAD_FOLDER,
    )

    learner = Learner(build_rlgym_v2_env, **learner_kwargs)
    learner.learn()