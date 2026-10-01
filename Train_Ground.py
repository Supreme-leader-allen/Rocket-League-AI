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
# Back to 0.01 (rlgym_ppo's tutorial value and ZealanL's guide value) after
# a brief drop to 0.001. The drop was based on reading the 300M-step run's
# flat Policy Entropy (4.4998 -> 4.4985) as "the entropy bonus is swamping
# the reward". It can't have been: 4.4998 is ln(90), i.e. the policy was
# exactly uniform, and the entropy bonus's gradient is exactly ZERO at a
# uniform distribution -- it cannot hold a policy there. The policy was
# flat because the reward gave it nothing to learn from (docs/ISSUES.md
# "P0 -- the policy never learned"), and the second run at 0.001 stayed
# flat too, which is consistent with that. Once there is a real reward
# gradient, 0.01 is what keeps the policy from collapsing early onto the
# first behavior that works.
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
# Per-iteration PPO stats (entropy, KL, ...) -- see Metrics.TrainingStatsLogger.
# Defaults to <CSV_PATH stem>_training_stats.csv so every experiment script
# gets one without passing anything new.
STATS_CSV_PATH = os.environ.get("PBT_STATS_CSV_PATH") or     os.path.splitext(CSV_PATH)[0] + "_training_stats.csv"
# Budget, in (estimated) cumulative environment timesteps, over which
# AnnealedCombinedReward decays dense shaping toward zero -- see that
# class's docstring in Rewards.py for the full mechanism and why it
# replaced a wall-clock version (docs/ISSUES.md P3). Exposed as an env
# var (not just a local constant) specifically so an ablation run can
# set it to something the run will never reach -- e.g.
# ANNEAL_TIMESTEPS=1000000000 with a much smaller PBT_TIMESTEP_LIMIT --
# to hold shaping constant and test whether annealing itself is what
# produces coordination (Liu et al.). AnnealedCombinedReward treats
# <=0 as "already fully annealed", which is the opposite of "never
# anneal" -- use a huge value, not 0, for that ablation.
ANNEAL_TIMESTEPS = float(os.environ.get("ANNEAL_TIMESTEPS", 200_000_000))


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

    from Rewards import build_shaping
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

    # Dense shaping anneals toward (near-)zero over cumulative training
    # timesteps, not wall-clock time -- see ANNEAL_TIMESTEPS above and
    # AnnealedCombinedReward's docstring in Rewards.py for why the
    # wall-clock version was a confirmed bug (docs/ISSUES.md P3: it
    # silently restarted at 0 on every checkpoint resume and never
    # completed on session-limited platforms). InAirReward keeps only a
    # small weight in this phase -- enough that the bot doesn't unlearn
    # jumping, not enough to reward flying around.
    #
    # If resuming from a checkpoint, recover its cumulative_timesteps
    # from the digit-named folder in its path (Learner.save()'s own
    # naming convention -- confirmed against its source, same technique
    # Pbt.find_latest_checkpoint's resolution and Train_Aerial.py's
    # timestep_limit fix use) so the anneal continues from wherever
    # training actually left off instead of restarting at 0.
    initial_timesteps = 0.0
    if CHECKPOINT_LOAD_FOLDER:
        basename = os.path.basename(os.path.normpath(CHECKPOINT_LOAD_FOLDER))
        if basename.isdigit():
            initial_timesteps = float(basename)
    # Terms and weights live in Rewards.SHAPING_SCHEDULES so Model A and
    # Model B can't silently drift apart -- see the comment there for why
    # they were rebalanced.
    shaping = build_shaping("ground", ANNEAL_TIMESTEPS, N_PROC, initial_timesteps)
    # Zero-weight -- pure data collection, no effect on training. See
    # metrics.py for what each column means and why overcommit_rate /
    # simultaneous_air_rate were added on top of your original metric list.
    coordination_metrics = CoordinationMetrics(
        csv_path=csv_path,
        run_label=run_label,
        n_proc=N_PROC,
        initial_timesteps=initial_timesteps,
    )

    # GoalReward wrapped in FitnessTracker: identical contribution to
    # training (weight 10.0, same as bare GoalReward would be) but also
    # logs episode_return for pbt.py's fitness ranking. See metrics.py for
    # why PBT fitness uses this instead of coordination_metrics' columns.
    fitness_tracked_goal_reward = FitnessTracker(
        GoalReward(), csv_path=fitness_csv_path, run_label=run_label,
        n_proc=N_PROC, initial_timesteps=initial_timesteps,
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

    # Observation size is 243, not 212: PartialInfoObs appends 31 features
    # (pending actions + per-car visibility) -- see Observation.py.
    #
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


# POLICY_LAYER_SIZES / CRITIC_LAYER_SIZES must stay byte-for-byte identical
# between Model A and Model B -- this is what makes checkpoint_load_folder a
# valid warm start instead of a shape-mismatch error. Train_Aerial.py imports
# these (and shared_learner_kwargs) from here rather than copying them. NOT
# overridable via PBT env vars on purpose: PBT evolves training
# hyperparameters, not architecture.
POLICY_LAYER_SIZES = [2048, 2048, 1024, 1024]
CRITIC_LAYER_SIZES = [2048, 2048, 1024, 1024]


def shared_learner_kwargs(n_proc: int) -> dict:
    """Learner settings common to Model A and Model B."""
    return dict(
        n_proc=n_proc,
        min_inference_size=max(1, int(round(n_proc * 0.9))),
        metrics_logger=None,
        ppo_batch_size=100_000,
        policy_layer_sizes=POLICY_LAYER_SIZES,
        critic_layer_sizes=CRITIC_LAYER_SIZES,
        ts_per_iteration=100_000,
        exp_buffer_size=300_000,
        ppo_minibatch_size=50_000,
        ppo_epochs=2,
        standardize_returns=True,
        standardize_obs=False,
        log_to_wandb=False,
        render=RENDER,          # slows one env to real-time and pipes it to RLViser -- see module docstring
        render_delay=0,         # seconds between rendered frames; raise this to slow playback down further
    )


if __name__ == "__main__":
    from rlgym_ppo import Learner
    from Metrics import TrainingStatsLogger

    # Basic self-play only (both teams mirror the current live policy).
    # Training against frozen past checkpoints needs rlgym_ppo changes --
    # see Self_Play.py's module docstring.
    learner_kwargs = shared_learner_kwargs(N_PROC)
    learner_kwargs.update(
        ppo_ent_coef=ENT_COEF,
        policy_lr=POLICY_LR,
        critic_lr=CRITIC_LR,
        save_every_ts=SAVE_EVERY_TS,
        checkpoints_save_folder=CHECKPOINTS_SAVE_FOLDER,
        timestep_limit=TIMESTEP_LIMIT,
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

    TrainingStatsLogger(
        STATS_CSV_PATH, RUN_LABEL, n_actions=90, ent_coef=ENT_COEF,
        policy_lr=POLICY_LR, critic_lr=CRITIC_LR, anneal_timesteps=ANNEAL_TIMESTEPS,
    ).install()

    learner = Learner(build_rlgym_v2_env, **learner_kwargs)
    learner.learn()
