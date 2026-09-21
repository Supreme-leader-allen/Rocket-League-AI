"""
CheckpointPool: sampling infrastructure for population-style self-play
(Liu et al. 2019 / FTW-style), where the opponent team is controlled by a
frozen snapshot of an earlier policy rather than the live policy being
trained.

evaluate_match/build_eval_env (bottom of this file): plays two DIFFERENT
frozen checkpoints against each other -- this is what pbt.py needs for a
fitness signal that actually distinguishes population members. Read why
below.

READ THIS BEFORE USING: this is the least "just works" file in the
scaffold. Stock rlgym_ppo.Learner, when spawn_opponents=True, mirrors the
*current live policy* onto both teams -- basic self-play, but not a
diverse pool of past versions. To actually sample from a pool of frozen
past checkpoints, you need to intercept action selection for the
opponent-side agents *inside* your environment-construction function
(build_rlgym_v2_env), before those actions ever reach the outer Learner.
That means loading a second, frozen copy of the policy network directly
in the env process and calling it yourself for orange-team agents.

Confirmed against the installed rlgym_ppo 1.3.13: the discrete policy
class is `rlgym_ppo.ppo.DiscreteFF`, constructed as
`DiscreteFF(input_shape, n_actions, layer_sizes, device)` (matches how
rlgym_ppo.ppo.ppo_learner.PPOLearner builds it internally for the live
policy, so a state_dict saved from that class loads straight into a
fresh DiscreteFF with the same layer_sizes). Checkpoint file naming
(PPO_POLICY.pt, PPO_VALUE_NET.pt, etc.) confirmed against
PPOLearner.save_to/load_from's source.

get_action(obs, deterministic=False) returns (action, log_prob); when
non-deterministic it's a flattened torch tensor of shape (1,) for a
single unbatched observation, when deterministic it's a bare numpy
scalar from .argmax() -- FrozenPolicy.act normalizes both to a (1,)
int array, which is what HumanlikeAction.parse_actions (actions.py)
expects per agent (confirmed by running it -- see actions.py).

WHY THIS MATTERS FOR PBT SPECIFICALLY: train_ground.py's self-play
mirrors ONE policy onto both blue and orange within a single population
member's own training run -- clone vs. clone, not member A vs. member B.
FitnessTracker's episode_return (metrics.py) measures how often that one
policy scores against a mirror of itself, which is a real but limited
signal (roughly: how decisively is this policy winning games in
general), NOT a head-to-head comparison between population members.
Two members with very different skill levels can both show similar
self-play episode_return, because each is only ever compared to itself.
evaluate_match below fixes this by actually loading two DIFFERENT
members' checkpoints onto opposite teams and playing them against each
other -- pbt.py uses this when FrozenPolicy is implemented, and falls
back to the self-play proxy (clearly logged as the weaker signal) when
it isn't yet.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class CheckpointPool:
    """
    Tracks saved policy checkpoints for opponent sampling. This class only
    manages *which checkpoint to use* -- loading the checkpoint into a
    runnable frozen policy (FrozenPolicy below) and actually routing
    orange-team actions through it is the part that has to be wired into
    build_rlgym_v2_env by hand.
    """

    def __init__(self, pool_dir: str, max_checkpoints: int = 30, recent_bias: float = 0.7):
        """
        pool_dir: directory to copy checkpoints into as they're archived.
        max_checkpoints: prune oldest checkpoints beyond this count.
        recent_bias: probability of sampling from the most-recent third of
            the pool rather than uniformly across all of it -- keeps most
            training matches close to current skill level (Liu et al. and
            FTW both bias toward recent opponents rather than sampling
            uniformly over the whole history) while still occasionally
            replaying older versions to prevent forgetting.
        """
        self._pool_dir = Path(pool_dir)
        self._pool_dir.mkdir(parents=True, exist_ok=True)
        self._max_checkpoints = max_checkpoints
        self._recent_bias = recent_bias
        # Monotonic, never reused -- fixes a real collision: naming by
        # len(list_checkpoints()) means that once the pool reaches
        # max_checkpoints, each archive() adds one and prunes one, so the
        # count stays constant and every subsequent archive() overwrote
        # the SAME directory instead of adding a new opponent, silently
        # freezing the pool. See docs/ISSUES.md P2. Seeded from any
        # existing pool contents so a freshly constructed instance
        # pointed at a non-empty pool_dir doesn't restart numbering at 0
        # and collide with what's already there.
        self._next_ckpt_id = self._infer_next_ckpt_id()

    def _infer_next_ckpt_id(self) -> int:
        ids = []
        for p in self.list_checkpoints():
            suffix = p.name[len("ckpt_"):]
            if p.name.startswith("ckpt_") and suffix.isdigit():
                ids.append(int(suffix))
        return (max(ids) + 1) if ids else 0

    def archive(self, checkpoint_dir: str) -> None:
        """Copy a Learner checkpoint directory into the pool under a
        monotonically increasing id, and prune old entries past
        max_checkpoints."""
        src = Path(checkpoint_dir)
        dest = self._pool_dir / f"ckpt_{self._next_ckpt_id:06d}"
        self._next_ckpt_id += 1
        shutil.copytree(src, dest, dirs_exist_ok=True)
        self._prune()

    def list_checkpoints(self):
        return sorted(p for p in self._pool_dir.iterdir() if p.is_dir())

    def _prune(self):
        checkpoints = self.list_checkpoints()
        excess = len(checkpoints) - self._max_checkpoints
        for old in checkpoints[:max(0, excess)]:
            shutil.rmtree(old, ignore_errors=True)

    def sample(self) -> Optional[Path]:
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        recent_n = max(1, len(checkpoints) // 3)
        if random.random() < self._recent_bias:
            return random.choice(checkpoints[-recent_n:])
        return random.choice(checkpoints)


class FrozenPolicy:
    """
    Loads a saved checkpoint's policy network for inference-only use as an
    opponent, with gradients disabled. Uses rlgym_ppo.ppo.DiscreteFF,
    confirmed against the installed rlgym_ppo version -- see module
    docstring.
    """

    def __init__(self, checkpoint_dir: str, policy_layer_sizes, obs_size: int, action_size: int, device: str = "cpu"):
        from rlgym_ppo.ppo import DiscreteFF

        self.device = device
        self.policy = DiscreteFF(obs_size, action_size, policy_layer_sizes, device)
        state_dict = torch.load(
            os.path.join(checkpoint_dir, "PPO_POLICY.pt"), map_location=device
        )
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        """Single-agent inference. evaluate_match no longer calls this
        (see act_batch, and docs/ISSUES.md P2 on why) -- kept for any
        other single-agent caller."""
        return self.act_batch(np.asarray(obs)[None, :], deterministic=deterministic)

    def act_batch(self, obs_batch: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        obs_batch: (n_agents, obs_size). Returns an (n_agents,) array of
        action indices from ONE forward pass, instead of one batch-1 call
        per agent. DiscreteFF.get_output reshapes to (batch, n_actions)
        regardless of input batch size (confirmed via source:
        `probs.view(-1, self.n_actions)`), so this produces the same
        per-agent distribution as calling act() once per agent -- just
        without the redundant per-agent Python/torch call overhead
        (previously 8 batch-1 CPU forward passes per step in
        evaluate_match; see docs/ISSUES.md P2).

        deterministic=True only supported for batch size 1: DiscreteFF.
        get_action's deterministic branch does
        `probs.cpu().numpy().argmax()` with no axis argument (confirmed
        by reading its source), which argmaxes over the FLATTENED
        (batch, n_actions) array -- a single global index, not one per
        row. That's an upstream rlgym_ppo limitation for batched greedy
        selection, not something to work around here; evaluate_match
        uses deterministic=False (the default), same as before.
        """
        if deterministic and obs_batch.shape[0] > 1:
            raise NotImplementedError(
                "FrozenPolicy.act_batch(deterministic=True) is not meaningful "
                "for batch size > 1 -- DiscreteFF.get_action's deterministic "
                "branch argmaxes over the flattened batch, not per row "
                "(rlgym_ppo 1.3.13's DiscreteFF.get_action source)."
            )
        with torch.no_grad():
            action, _ = self.policy.get_action(obs_batch, deterministic=deterministic)
        # get_action returns a flattened torch tensor of shape (batch,)
        # when stochastic, or a bare numpy scalar from .argmax() when
        # deterministic (batch size 1 only, per above) -- normalize both
        # to what HumanlikeAction.parse_actions expects per agent: an int
        # array (confirmed by running parse_actions -- see actions.py's
        # module docstring).
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        return np.atleast_1d(action).astype(np.int32)


def build_eval_env():
    """
    Minimal 4v4 env for pure-inference matches between two frozen
    policies -- no training, no shaping, no CoordinationMetrics/
    FitnessTracker logging. Reward is bare GoalReward, used only to
    detect which team scored at the terminal step of each episode (see
    evaluate_match) -- not used for any learning here. Same obs/action/
    state-mutator config as train_ground.py's build_rlgym_v2_env
    (including zero_padding=4 -- see that file's comment on why 3 was
    wrong) so a FrozenPolicy loaded from a train_ground.py checkpoint
    sees exactly the observations/actions it was trained on.

    Returns the RAW rlgym.api.RLGym env, NOT wrapped in
    RLGymV2GymWrapper. Confirmed by running it: RLGymV2GymWrapper
    collapses everything to positional arrays/lists with no AgentID
    access at all (reset() returns a plain ndarray, step() takes a
    positional array and returns an aggregated bool/list, not
    per-agent dicts) -- it's built for rlgym_ppo.Learner's internal
    batched-agent pipeline, not for a caller that needs to route
    different agents' actions to different policies by team. The raw
    RLGym env keeps the AgentID-keyed dict interface evaluate_match
    actually needs.
    """
    from rlgym.api import RLGym
    from rlgym.rocket_league.done_conditions import (
        GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition,
    )
    from rlgym.rocket_league.reward_functions import GoalReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator

    from Observation import PartialInfoObs
    from Actions import HumanlikeAction

    action_parser = HumanlikeAction(repeats=8, reaction_delay_ticks=3)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=30),
        TimeoutCondition(timeout_seconds=300),
    )
    obs_builder = PartialInfoObs(zero_padding=4)
    state_mutator = MutatorSequence(FixedTeamSizeMutator(blue_size=4, orange_size=4), KickoffMutator())

    return RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=GoalReward(),
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
    )


def _resolve_device() -> str:
    """Same resolution rlgym_ppo.Learner uses for device="auto" (confirmed
    against its source): CUDA if available, else CPU -- never MPS, even
    on Apple silicon (see docs/COMPUTE.md). evaluate_match defaults to
    this instead of hardcoding "cpu", so the PBT tournament runs on
    whatever device training actually used rather than always CPU."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def evaluate_match(blue_checkpoint_dir: str, orange_checkpoint_dir: str, n_episodes: int = 10,
                    policy_layer_sizes=(2048, 2048, 1024, 1024), device: Optional[str] = None) -> float:
    """
    Plays n_episodes of 4v4 with blue controlled by a frozen policy
    loaded from blue_checkpoint_dir and orange from orange_checkpoint_dir.
    Returns blue's win rate (draws -- episodes that time out with no
    goal -- count as 0.5). THIS is the fitness signal that actually
    distinguishes population members from each other, unlike the
    self-play episode_return proxy -- see module docstring.

    Winner per episode is read off GoalReward's reward dict at whichever
    step the episode terminates on: whichever team's agents receive
    positive reward that step scored (confirmed against GoalReward's
    source: +1 to state.scoring_team's agents, -1 to the other team,
    every step state.goal_scored is True).

    build_eval_env() returns the raw rlgym.api.RLGym env (not
    RLGymV2GymWrapper -- see that function's docstring for why), so obs/
    actions/rewards are all AgentID-keyed dicts and team_of_agent below
    is read directly off env.state.cars[agent].is_orange -- no guessing
    required, confirmed by running a match end to end.

    Each step, all of blue's agents are batched into ONE forward pass and
    all of orange's into another -- 2 forward passes per step instead of
    8 batch-1 passes (see FrozenPolicy.act_batch and docs/ISSUES.md P2).
    `device` defaults to _resolve_device()'s result (matching Learner's
    own "auto" resolution) rather than always CPU -- see docs/COMPUTE.md
    on why this tournament is not free and should run on the training
    device when one is available.
    """
    if device is None:
        device = _resolve_device()

    env = build_eval_env()
    obs = env.reset()

    obs_size = len(next(iter(obs.values())))
    action_size = list(env.action_spaces.values())[0][1]

    policy_blue = FrozenPolicy(blue_checkpoint_dir, policy_layer_sizes, obs_size, action_size, device=device)
    policy_orange = FrozenPolicy(orange_checkpoint_dir, policy_layer_sizes, obs_size, action_size, device=device)

    wins = 0.0
    for ep in range(n_episodes):
        if ep > 0:
            obs = env.reset()
        team_of_agent = _infer_teams(env, obs)
        blue_agents = [a for a in obs if team_of_agent[a] == "blue"]
        orange_agents = [a for a in obs if team_of_agent[a] == "orange"]

        terminated = truncated = False
        winner = None
        while not (terminated or truncated):
            actions = {}
            if blue_agents:
                blue_batch = np.stack([obs[a] for a in blue_agents])
                for agent, act_idx in zip(blue_agents, policy_blue.act_batch(blue_batch)):
                    actions[agent] = np.array([act_idx], dtype=np.int32)
            if orange_agents:
                orange_batch = np.stack([obs[a] for a in orange_agents])
                for agent, act_idx in zip(orange_agents, policy_orange.act_batch(orange_batch)):
                    actions[agent] = np.array([act_idx], dtype=np.int32)

            obs, reward, terminated_d, truncated_d = env.step(actions)
            terminated = all(terminated_d.values())
            truncated = all(truncated_d.values())

            if terminated:
                blue_reward = np_mean_or_zero([r for a, r in reward.items() if team_of_agent[a] == "blue"])
                winner = "blue" if blue_reward > 0 else "orange"

        if winner == "blue":
            wins += 1.0
        elif winner is None:
            wins += 0.5  # truncated with no goal -- draw

    return wins / n_episodes


def _infer_teams(env, obs: dict) -> dict:
    """AgentID -> 'blue'/'orange' mapping, read directly off the raw
    RLGym env's current GameState (env.state is a confirmed public
    property -- see rlgym.api.rlgym.RLGym.state's source). Only valid
    when build_eval_env()'s raw (unwrapped) env is used; a
    RLGymV2GymWrapper-wrapped env has no per-agent state access at all."""
    return {a: ("orange" if env.state.cars[a].is_orange else "blue") for a in obs}


def np_mean_or_zero(values):
    return float(np.mean(values)) if values else 0.0