"""
Population-based training (Liu et al. 2019 / FTW-style) over
Train_Ground.py. Each population member is its own independent training
run with its own learning rate / entropy coefficient / checkpoint
lineage, launched as a subprocess per generation via the PBT_* env vars
Train_Ground.py reads (see that file's module docstring for the full
list) -- this avoids duplicating the training script.

Fitness is deliberately NOT any of Metrics.py's CoordinationMetrics
columns -- ranking the population on overcommit_rate or teammate spacing
would make "coordination improved" true by construction rather than an
independent result, since you'd be selecting directly on the thing
you're trying to measure. CoordinationMetrics keeps logging throughout
PBT the same as any standalone run, so you still get coordination data
per member/generation -- PBT just doesn't use it to decide who survives.

Fitness signal: Self_Play.py's evaluate_match, a real cross-play
round-robin -- every pair of members plays EVAL_EPISODES_PER_MATCHUP
episodes each, and fitness is total win rate across all matchups. This
is what actually distinguishes population members' skill from each
other (see self_play.py's module docstring for why train_ground.py's
own self-play, mirroring one policy onto both teams within a single
member's run, CAN'T do this by itself: a member only ever plays a mirror
of itself). Falls back to FitnessTracker's episode_return proxy (loudly
logged as the weaker signal) only if a member has no checkpoint yet
(generation 0 before any run has saved one) or evaluate_match raises.

Each generation:
1. Train every member for GENERATION_TIMESTEPS (train_generation).
2. Evaluate the population via cross-play (evaluate_population).
3. Exploit + explore: the bottom BOTTOM_FRACTION of the population, by
   fitness, is told to warm-start its NEXT generation from a randomly
   chosen top performer's checkpoint (exploit) instead of its own, and
   has its learning rate / entropy coefficient randomly perturbed
   (explore) -- standard PBT. This does NOT copy files around: each
   member always trains into and saves under its own checkpoint
   lineage (PBT_CHECKPOINT_DIR), and exploit just overrides which
   checkpoint the NEXT generation's subprocess is told to load from via
   PBT_CHECKPOINT_LOAD_DIR.

Run with:
    python Pbt.py

Tune POPULATION_SIZE, N_GENERATIONS, GENERATION_TIMESTEPS, and
N_PROC_PER_MEMBER below to your hardware. Default is sequential (one
population member trains at a time) with N_PROC_PER_MEMBER=8, not
Train_Ground.py's standalone default of 32 -- running a whole population
in parallel multiplies environment count by population size, which
isn't realistic on most single machines.
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

POPULATION_SIZE = 4
N_GENERATIONS = 10
GENERATION_TIMESTEPS = 2_000_000
N_PROC_PER_MEMBER = 8
EVAL_EPISODES_PER_MATCHUP = 4
BOTTOM_FRACTION = 0.25  # fraction of the population replaced each generation
PERTURB_RANGE = (0.8, 1.2)

CHECKPOINT_ROOT = "checkpoints/pbt"
CSV_ROOT = "metrics/pbt"
GENERATION_LOG_PATH = os.path.join(CSV_ROOT, "generations.jsonl")

INITIAL_POLICY_LR = 1e-4
INITIAL_CRITIC_LR = 1e-4
INITIAL_ENT_COEF = 0.01

# GENERATION_TIMESTEPS must exceed this or a generation could complete
# without a single save -- see Train_Ground.py's SAVE_EVERY_TS comment
# for why (Learner only saves periodically, never on exit at
# timestep_limit). Kept well under GENERATION_TIMESTEPS so at least a
# few checkpoints land per generation, not just barely one.
SAVE_EVERY_TS = max(200_000, GENERATION_TIMESTEPS // 8)


class PopulationMember:
    def __init__(self, member_id: int, policy_lr: float, critic_lr: float, ent_coef: float):
        self.member_id = member_id
        self.policy_lr = policy_lr
        self.critic_lr = critic_lr
        self.ent_coef = ent_coef
        self.fitness: Optional[float] = None
        # Set by exploit_and_explore() the generation before it should
        # take effect; consumed (reset to None) by train_generation().
        self.pending_load_override: Optional[str] = None

    @property
    def run_label(self) -> str:
        return f"pbt_member_{self.member_id}"

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(CHECKPOINT_ROOT, self.run_label)

    def latest_checkpoint(self) -> Optional[str]:
        return find_latest_checkpoint(self.checkpoint_dir)


def find_latest_checkpoint(base_save_folder: str) -> Optional[str]:
    """
    Given a PBT_CHECKPOINT_DIR-style base path (e.g.
    "checkpoints/pbt/pbt_member_0"), find the most recent checkpoint any
    Train_Ground.py run under that name has saved, and return a literal
    path Learner's checkpoint_load_folder accepts directly.

    Deliberately not just passing checkpoint_load_folder="latest" and
    letting Learner resolve it itself: Learner's own "latest" logic (see
    its source) always resolves relative to *that process's own*
    checkpoints_save_folder, which only ever points at the CURRENT
    member's lineage. PBT's exploit step needs to point a member at a
    DIFFERENT member's checkpoints, so this has to be resolvable
    independently, for any member, from the orchestrating pbt.py
    process. Mirrors Learner.load()'s own "latest" resolution algorithm
    exactly (confirmed by reading its source): checkpoints_save_folder
    gets "-<unix_ns>" appended on every run (add_unix_timestamp defaults
    True, never overridden here), so a given base_save_folder
    accumulates one sibling directory per run; within the most recent
    one, checkpoints are saved as digit-named subfolders keyed by
    cumulative timestep count.
    """
    save_path = os.path.dirname(base_save_folder) or "."
    base_name = os.path.basename(base_save_folder)
    if not os.path.exists(save_path):
        return None

    highest_timestamp = -1
    best_run_folder = None
    for filename in os.listdir(save_path):
        full_path = os.path.join(save_path, filename)
        if not os.path.isdir(full_path) or not filename.startswith(base_name + "-"):
            continue
        suffix = filename[len(base_name) + 1:]
        if suffix.isdigit() and int(suffix) > highest_timestamp:
            highest_timestamp = int(suffix)
            best_run_folder = full_path

    if best_run_folder is None:
        return None

    highest_ts = -1
    best_ckpt = None
    for filename in os.listdir(best_run_folder):
        full_path = os.path.join(best_run_folder, filename)
        if not os.path.isdir(full_path) or not filename.isdigit():
            continue
        if int(filename) > highest_ts:
            highest_ts = int(filename)
            best_ckpt = full_path

    return best_ckpt


def train_generation(member: PopulationMember, generation: int) -> None:
    """
    Launches Train_Ground.py as a subprocess for this member/generation
    and blocks until it finishes. Confirmed by reading
    rlgym_ppo.learner.Learner._learn()'s source: its main loop is a
    plain `while cumulative_timesteps < timestep_limit`, so it returns
    on its own once PBT_TIMESTEP_LIMIT is hit -- subprocess.run() here
    is guaranteed to return without needing an external shutdown signal.
    """
    env = os.environ.copy()
    env["PBT_POLICY_LR"] = str(member.policy_lr)
    env["PBT_CRITIC_LR"] = str(member.critic_lr)
    env["PBT_ENT_COEF"] = str(member.ent_coef)
    env["PBT_N_PROC"] = str(N_PROC_PER_MEMBER)
    env["PBT_CHECKPOINT_DIR"] = member.checkpoint_dir
    env["PBT_TIMESTEP_LIMIT"] = str(GENERATION_TIMESTEPS)
    env["PBT_SAVE_EVERY_TS"] = str(SAVE_EVERY_TS)
    env["PBT_RUN_LABEL"] = member.run_label
    env["PBT_CSV_PATH"] = os.path.join(CSV_ROOT, f"{member.run_label}_coordination.csv")
    env["PBT_FITNESS_CSV_PATH"] = os.path.join(CSV_ROOT, f"{member.run_label}_fitness.csv")

    load_folder = member.pending_load_override or member.latest_checkpoint()
    if load_folder:
        env["PBT_CHECKPOINT_LOAD_DIR"] = load_folder
    else:
        env.pop("PBT_CHECKPOINT_LOAD_DIR", None)

    source = "exploited checkpoint" if member.pending_load_override else \
        ("own lineage" if load_folder else "fresh start")
    print(f"[gen {generation}] training member {member.member_id} "
          f"(policy_lr={member.policy_lr:.2e}, critic_lr={member.critic_lr:.2e}, "
          f"ent_coef={member.ent_coef:.4f}, warm start: {source})")

    result = subprocess.run([sys.executable, "Train_Ground.py"], env=env)
    member.pending_load_override = None
    if result.returncode != 0:
        raise RuntimeError(
            f"Train_Ground.py subprocess failed for member {member.member_id} "
            f"(exit code {result.returncode})"
        )


def evaluate_population(population: List[PopulationMember], generation: int) -> None:
    from Self_Play import evaluate_match

    checkpoints = {m.member_id: m.latest_checkpoint() for m in population}
    missing = [mid for mid, ckpt in checkpoints.items() if ckpt is None]
    if missing:
        print(f"[gen {generation}] members {missing} have no checkpoint yet -- "
              f"falling back to self-play episode_return proxy for fitness.")
        _evaluate_population_fallback(population)
        return

    wins = {m.member_id: 0.0 for m in population}
    try:
        for i, member_a in enumerate(population):
            for member_b in population[i + 1:]:
                win_rate_a = evaluate_match(
                    checkpoints[member_a.member_id], checkpoints[member_b.member_id],
                    n_episodes=EVAL_EPISODES_PER_MATCHUP,
                )
                wins[member_a.member_id] += win_rate_a
                wins[member_b.member_id] += (1.0 - win_rate_a)
    except Exception as e:
        print(f"[gen {generation}] evaluate_match failed ({e!r}) -- "
              f"falling back to self-play episode_return proxy for fitness.")
        _evaluate_population_fallback(population)
        return

    n_opponents = max(1, len(population) - 1)
    for member in population:
        member.fitness = wins[member.member_id] / n_opponents
    print(f"[gen {generation}] cross-play fitness: "
          f"{ {m.member_id: round(m.fitness, 3) for m in population} }")


def _evaluate_population_fallback(population: List[PopulationMember]) -> None:
    """
    WEAKER SIGNAL -- see module docstring. Each member's own recent
    episode_return from its FitnessTracker CSV shards (self-play vs. a
    mirror of itself), averaged over the last 20 logged episodes.
    """
    import csv as csv_module

    for member in population:
        candidates = sorted(Path(CSV_ROOT).glob(f"{member.run_label}_fitness.*.csv"))
        returns = []
        for path in candidates:
            with open(path, newline="") as f:
                for row in csv_module.DictReader(f):
                    try:
                        returns.append(float(row["episode_return"]))
                    except (KeyError, ValueError):
                        continue
        member.fitness = sum(returns[-20:]) / len(returns[-20:]) if returns else 0.0


def exploit_and_explore(population: List[PopulationMember], generation: int) -> None:
    ranked = sorted(population, key=lambda m: m.fitness, reverse=True)
    n_replace = max(1, int(round(len(population) * BOTTOM_FRACTION)))
    top_performers = ranked[:len(ranked) - n_replace]
    bottom_performers = ranked[len(ranked) - n_replace:]

    for loser in bottom_performers:
        winner = random.choice(top_performers)
        winner_ckpt = winner.latest_checkpoint()
        if winner_ckpt is None:
            continue  # nothing to exploit yet

        loser.pending_load_override = winner_ckpt
        loser.policy_lr = winner.policy_lr * random.uniform(*PERTURB_RANGE)
        loser.critic_lr = winner.critic_lr * random.uniform(*PERTURB_RANGE)
        loser.ent_coef = max(1e-4, winner.ent_coef * random.uniform(*PERTURB_RANGE))
        print(f"[gen {generation}] member {loser.member_id} (fitness={loser.fitness:.3f}) "
              f"exploits member {winner.member_id} (fitness={winner.fitness:.3f})")


def log_generation(population: List[PopulationMember], generation: int) -> None:
    os.makedirs(CSV_ROOT, exist_ok=True)
    with open(GENERATION_LOG_PATH, "a") as f:
        for member in population:
            f.write(json.dumps({
                "generation": generation,
                "member_id": member.member_id,
                "fitness": member.fitness,
                "policy_lr": member.policy_lr,
                "critic_lr": member.critic_lr,
                "ent_coef": member.ent_coef,
            }) + "\n")


def main() -> None:
    population = [
        PopulationMember(i, INITIAL_POLICY_LR, INITIAL_CRITIC_LR, INITIAL_ENT_COEF)
        for i in range(POPULATION_SIZE)
    ]

    for generation in range(N_GENERATIONS):
        for member in population:
            train_generation(member, generation)

        evaluate_population(population, generation)
        log_generation(population, generation)

        if generation < N_GENERATIONS - 1:
            exploit_and_explore(population, generation)

    best = max(population, key=lambda m: m.fitness)
    print(f"PBT complete. Best member: {best.member_id} "
          f"(fitness={best.fitness:.3f}, checkpoint={best.latest_checkpoint()})")


if __name__ == "__main__":
    main()
