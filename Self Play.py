"""
CheckpointPool: sampling infrastructure for population-style self-play
(Liu et al. 2019 / FTW-style), where the opponent team is controlled by a
frozen snapshot of an earlier policy rather than the live policy being
trained.

READ THIS BEFORE USING: this is the least "just works" file in the
scaffold. Stock rlgym_ppo.Learner, when spawn_opponents=True, mirrors the
*current live policy* onto both teams -- basic self-play, but not a
diverse pool of past versions. To actually sample from a pool of frozen
past checkpoints, you need to intercept action selection for the
opponent-side agents *inside* your environment-construction function
(build_rlgym_v2_env), before those actions ever reach the outer Learner.
That means loading a second, frozen copy of the policy network directly
in the env process and calling it yourself for orange-team agents.

The exact class to load depends on your installed rlgym_ppo version --
recent versions expose a discrete feedforward policy class somewhere
under rlgym_ppo.ppo (commonly referenced as DiscreteFF in the community,
alongside a continuous-action equivalent). Confirm the exact import path
and constructor signature against your installed version before relying
on FrozenPolicy.load below:

    python -c "import rlgym_ppo.ppo as p; print(dir(p))"

and inspect whichever policy class matches your action space (discrete,
since this project uses LookupTableAction).
"""

import os
import random
import shutil
from pathlib import Path
from typing import Optional

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

    def archive(self, checkpoint_dir: str) -> None:
        """Copy a Learner checkpoint directory into the pool, tagged by
        timestamp, and prune old entries past max_checkpoints."""
        src = Path(checkpoint_dir)
        dest = self._pool_dir / f"ckpt_{len(self.list_checkpoints()):06d}"
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
    opponent, with gradients disabled. Skeleton only -- fill in the actual
    policy class/constructor once you've confirmed it against your
    installed rlgym_ppo version (see module docstring).
    """

    def __init__(self, checkpoint_dir: str, policy_layer_sizes, obs_size: int, action_size: int, device: str = "cpu"):
        self.device = device
        # TODO: replace with your installed version's actual policy class,
        # constructed with the SAME policy_layer_sizes used in train_ground.py
        # / train_aerial.py so the saved state_dict actually matches.
        #
        # from rlgym_ppo.ppo import DiscreteFF  # verify this path
        # self.policy = DiscreteFF(obs_size, action_size, policy_layer_sizes, device)
        # self.policy.load_state_dict(torch.load(os.path.join(checkpoint_dir, "PPO_POLICY.pt"),
        #                                          map_location=device))
        # self.policy.eval()
        raise NotImplementedError(
            "Fill in the policy class/constructor for your installed rlgym_ppo "
            "version before using FrozenPolicy -- see module docstring."
        )

    def act(self, obs: torch.Tensor) -> int:
        with torch.no_grad():
            # action, _ = self.policy.get_action(obs, deterministic=False)
            # return int(action)
            raise NotImplementedError