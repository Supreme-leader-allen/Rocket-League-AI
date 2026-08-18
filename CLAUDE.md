# CLAUDE.md

4v4 Rocket League agents trained with `rlgym_ppo`, studying whether team
coordination emerges from competition alone (Liu et al., *Emergent Coordination
through Competition*). Flat layout: every module lives at the repo root.

## Where things are

- `README.md` — what each module does and the design decisions behind them.
- `ROADMAP.md` — the research argument: what is being tested, and how this
  implementation differs from the papers it draws on.
- `COMPUTE.md` — **hardware, throughput, and running a training job.**
- `Benchmark.py` — measures this machine's actual throughput.

## Before estimating compute, run the benchmark

```
python Benchmark.py
```

Do not estimate training time, recommend a GPU, size a cloud instance, or
optimize for speed from first principles. This project's compute profile is not
what general RL intuition predicts — the networks are ~30x `rlgym_ppo`'s
defaults, which moves the bottleneck onto the PPO update on most hardware, and
the correct optimization differs by machine. `Benchmark.py` measures it and says
which side is the bottleneck. `COMPUTE.md` explains the traps, including several
that are easy to get wrong from reading the code alone.

## Conventions

- Modules are `PascalCase.py`; docs are `UPPERCASE.md`.
- Module docstrings carry the reasoning — including what was tried, verified
  against installed-package source, and rejected. When you change behavior that
  a docstring justifies, update the docstring's argument, not just the code.
- `POLICY_LAYER_SIZES` / `CRITIC_LAYER_SIZES` must stay identical between
  `Train_Ground.py` and `Train_Aerial.py`, or warm-starting Model B from Model
  A's checkpoint fails on a shape mismatch.
- Never commit checkpoints or metrics output. See `.gitignore`.
