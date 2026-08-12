"""
Model A: ground, 4v4. This is the "for show" bot -- coordinated 4v4 play
on the ground, with the action space and network already sized for
everything Model B (aerial) will need later (see ROADMAP.md).

Run with:
    python train_ground.py

Before a long run, verify two things against your installed rlgym_ppo
version (they're flagged inline below too):
1. That Learner accepts `checkpoints_save_folder` under that exact name --
   run `python -c "from rlgym_ppo import Learner; help(Learner.__init__)"`
   and check the actual parameter list. The tutorial doc's example doesn't
   show this parameter, only save_every_ts, so don't assume the spelling
   below is exactly right without checking.
2. That n_proc=32 and the batch sizes below are sane for your CPU core
   count -- these are the tutorial's defaults for a fairly beefy machine;
   scale down if training is memory-starved or CPU-bound.
"""

import numpy as np


def build_rlgym_v2_env(run_label: str = "model_a_ground", csv_path: str = "metrics/ground_training.csv"):
    """
    run_label/csv_path are parameterized (rather than hardcoded below) so
    run_baseline.py can reuse this exact env config -- same rewards, obs,
    action space, state mutator -- while logging under a different label,
    which matters for analyze.py to tell trained and baseline episodes
    apart. Don't hardcode a different CoordinationMetrics call somewhere
    else; always go through these two parameters.
    """
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
    from Metrics import CoordinationMetrics

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

    reward_fn = CombinedReward(
        (shaping, 1.0),
        (GoalReward(), 10.0),  # sparse, team-shared by construction -- the credit-assignment mechanism itself
        (coordination_metrics, 0.0),
    )

    # zero_padding is "max cars per team" for DefaultObs, not "other
    # agents excluding self" -- it must equal the actual team size (4)
    # for a 4v4, giving 3 padded ally slots (teammates besides self) + 4
    # padded enemy slots. zero_padding=3 undercounts this: with a real
    # 4v4, actual ally/enemy counts (3/4) already exceed the padding
    # minimums implied by 3 (2/3), so no padding ever triggers and the
    # built obs vector silently comes out 40 elements larger than the
    # declared obs space, which mismatches the policy network's input
    # layer. zero_padding=4 makes the declared and actual sizes match
    # exactly, and stays correct for Model B since team size never
    # changes in this project (always 4v4).
    obs_builder = PartialInfoObs(zero_padding=4)

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=4, orange_size=4),
        KickoffMutator(),
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
    )

    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":
    from rlgym_ppo import Learner

    n_proc = 32
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    # POLICY_LAYER_SIZES / CRITIC_LAYER_SIZES must stay byte-for-byte
    # identical in train_aerial.py -- this is what makes checkpoint_load_folder
    # a valid warm start instead of a shape-mismatch error.
    POLICY_LAYER_SIZES = [2048, 2048, 1024, 1024]
    CRITIC_LAYER_SIZES = [2048, 2048, 1024, 1024]

    # Basic self-play only (both teams mirror the current live policy).
    # For a real checkpoint pool of frozen past opponents (the actual
    # autocurriculum mechanism from ROADMAP.md), you need to wire
    # self_play.CheckpointPool into build_rlgym_v2_env's orange-team
    # action selection -- not done here yet. See self_play.py.
    learner = Learner(
        build_rlgym_v2_env,
        n_proc=n_proc,
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
        save_every_ts=1_000_000,
        checkpoints_save_folder="checkpoints/model_a_ground",  # verify exact param name -- see module docstring
        timestep_limit=1_000_000_000,
        log_to_wandb=False,
    )
    learner.learn()