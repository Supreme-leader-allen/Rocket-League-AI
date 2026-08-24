# CLAUDE.md

4v4 Rocket League agents trained with `rlgym_ppo`, studying whether team
coordination emerges from competition alone (Liu et al., *Emergent Coordination
through Competition*). Every Python module lives at the repo root. The shell
scripts that run the experiments live in `experiments/`, and this project's
own documentation lives in `docs/`.

## Where things are

- `docs/ISSUES.md` — **known defects, ranked by severity. Read before running a
  training job or trusting a result.**
- `README.md` — what each module does and the design decisions behind them.
  Describes intended behavior, some of which `docs/ISSUES.md` shows is not
  the actual behavior. Left uncorrected on purpose (see "Whose files are
  whose" below).
- `ROADMAP.md` — the research argument: what is being tested, and how this
  implementation differs from the papers it draws on. Same caveat as
  `README.md`.
- `docs/COMPUTE.md` — **hardware, throughput, and running a training job.**
- `Benchmark.py` — measures this machine's actual throughput.
- `experiments/` — the experiment matrix as runnable scripts. `run_all.sh` is
  the single entry point; `params.sh` holds every parameter in three layers
  (frozen / measured / yours). `ROUND` is the only knob that differs between a
  local smoke test and a full server run.

## Check docs/ISSUES.md before trusting that something works

Three of this project's five research pillars are currently broken or unwired,
and none of the defects raise — they run to completion and produce
normal-looking checkpoints and metrics. The highest-severity one silently
disables the partial-information layer, which invalidates the coordination
result rather than degrading it. Do not recommend a training run, interpret a
metric, or build on a module without checking whether `docs/ISSUES.md` lists it.

**`docs/ISSUES.md` supersedes every other description in this repo.**
`README.md`, `ROADMAP.md`, and several module docstrings still describe
behavior that
`docs/ISSUES.md` demonstrates is not happening — `PartialInfoObs`'s occlusion,
`FitnessTracker`'s `episode_return`, the two-model warm start, and PBT's
generations are the main ones. Those descriptions are the design intent, not a
record of what runs. Where they disagree with `docs/ISSUES.md`, that file is
right.

## Whose files are whose

This project was inherited. `.gitignore`, `Benchmark.py`, `CLAUDE.md`,
`docs/COMPUTE.md`, and `docs/ISSUES.md` are the current owner's; `README.md`,
`ROADMAP.md`, and every `*.py` module are the original author's and are left
untouched deliberately, so the boundary stays legible. That is why defects found
in the original author's files are recorded in `docs/ISSUES.md` rather than
corrected in place. Keep this split: put new findings and new documentation in
the owner's files, and change an original file only when actually fixing the
code in it.

## Before estimating compute, run the benchmark

```
python Benchmark.py
```

Do not estimate training time, recommend a GPU, size a cloud instance, or
optimize for speed from first principles. This project's compute profile is not
what general RL intuition predicts — the networks are ~30x `rlgym_ppo`'s
defaults, which moves the bottleneck onto the PPO update on most hardware, and
the correct optimization differs by machine. `Benchmark.py` measures it and says
which side is the bottleneck. `docs/COMPUTE.md` explains the traps, including
several that are easy to get wrong from reading the code alone.

## Conventions

- Modules are `PascalCase.py`; docs are `UPPERCASE.md`.
- Module docstrings carry the reasoning — including what was tried, verified
  against installed-package source, and rejected. When you change behavior that
  a docstring justifies, update the docstring's argument, not just the code.
  Their "confirmed by running it" claims are reliable for what they name —
  signatures, array shapes, attribute names — but they check single calls in
  isolation. They do not establish that two modules agree on a shared
  convention, and the P0 defect in `docs/ISSUES.md` lives in exactly that gap.
  Several docstrings currently assert behavior that `docs/ISSUES.md` shows is
  not happening; that file wins.
- `POLICY_LAYER_SIZES` / `CRITIC_LAYER_SIZES` must stay identical between
  `Train_Ground.py` and `Train_Aerial.py`, or warm-starting Model B from Model
  A's checkpoint fails on a shape mismatch.
- Never commit checkpoints or metrics output. See `.gitignore`.
