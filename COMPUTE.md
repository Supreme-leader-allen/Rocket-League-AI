# COMPUTE.md

Hardware selection, throughput, and how to actually run a training job.

This document is written for whoever — or whatever — is about to spend money or
hours on a training run. If you are an agent working on this repo, read this
before recommending a GPU, sizing a cloud instance, estimating a runtime, or
optimizing anything for speed. The compute profile of this project is not what
general reinforcement-learning intuition predicts, and the specific ways it
differs are listed below.

## The one rule: measure, then estimate

```
python Benchmark.py
```

Two to four minutes. It measures this machine's environment throughput, its
multi-process saturation point, its PPO update cost, and its peak device
memory, then prints a `PBT_N_PROC` value, an end-to-end SPS estimate, a runtime
for the configured `timestep_limit`, and — most importantly — **which side is
the bottleneck**, because that determines which optimization is worth doing and
the answer differs from machine to machine.

Report those numbers before recommending hardware. Do not estimate throughput
from core counts and GPU spec sheets; the section below explains why that
particular shortcut fails here.

## Six things that are counterintuitive about this project

These are the mistakes worth guarding against — each one is easy to make from
reasonable priors, and each one changes the answer materially.

**1. This is not the usual CPU-bound RL workload.** The reflex — "RL training is
bottlenecked on environment simulation, the GPU barely matters" — is wrong here.
`Train_Ground.py` uses `POLICY_LAYER_SIZES = [2048, 2048, 1024, 1024]`, about
7.9M parameters per network. `rlgym_ppo`'s own default is `(256, 256, 256)`,
roughly 0.25M. That is a ~30x difference, and it moves the bottleneck onto the
update for most hardware. On some machines the update is the majority of wall
clock time. `Benchmark.py` will tell you which side yours lands on.

**2. `get_all_batches_shuffled` walks the entire experience buffer, not one
batch.** Reading `ppo_batch_size=100_000` and concluding that each epoch
processes 100k samples underestimates the update by 3x. Verified against
`rlgym_ppo/ppo/experience_buffer.py`: the method yields chunks of `batch_size`
until the buffer is exhausted. With `exp_buffer_size=300_000` and
`ppo_epochs=2`, one 100k-step iteration performs
`2 x 3 x (100_000/50_000) = 12` forward+backward passes over 50k-sample
minibatches, through both networks. Every collected timestep is trained on
about six times.

This is also the largest available lever. Lowering `exp_buffer_size` to
`100_000` cuts update cost by 3x. It is not a free win — it drops sample reuse
from ~6x to ~2x, which is a change to training dynamics, not just to speed. But
6x reuse of off-policy data is already aggressive for PPO, so the tradeoff is
usually worth making. `ppo_epochs: 2 -> 1` is the same lever at half strength.

**3. `copy.deepcopy` in `Observation.py` is a real hotspot but usually the wrong
thing to fix first.** `PartialInfoObs._masked_state_for` deep-copies the entire
`GameState` once per agent per step — eight full copies per environment step in
4v4. Profiling puts it at ~80% of environment step time, and the whole
partial-information layer costs roughly 4-5x versus `DefaultObs`.

That sounds like an obvious first target. It usually is not, because collection
may only be a small fraction of total loop time. Fixing it when the update
dominates produces a single-digit end-to-end improvement. Check
`Benchmark.py`'s bottleneck verdict first. When collection *is* the bottleneck
— many cores, fast GPU, or after lowering `exp_buffer_size` — this becomes the
highest-value change in the repo. The fix is to shallow-copy and replace only
the `physics` fields being masked, instead of deep-copying the whole state.

**4. `ANNEAL_SECONDS` is wall-clock time within a single process.** In
`Train_Ground.py`, `AnnealedCombinedReward` decays dense shaping over
`ANNEAL_SECONDS = 2 * 60 * 60`, measured from process start. Two consequences:

- On a run longer than two hours, shaping is at zero for most of training. The
  value is a placeholder — the file's own comment says so. Set it to something
  proportional to the real run length.
- Under checkpoint-resume training, the clock restarts from zero every session,
  so the anneal never completes. On any platform with session limits (notebook
  services, preemptible instances) the mechanism is effectively broken. The
  correct fix is to anneal on `cumulative_timesteps` instead of wall clock.

**5. `checkpoint_load_folder` defaults to the string `"latest"`, not `None`.**
Omitting it does not mean "fresh start" — `Learner` scans for a prior run's
checkpoint directory and silently resumes from it. `Train_Ground.py` already
passes it explicitly for this reason; preserve that if you refactor. Symptom of
getting it wrong: a "new" run that starts with a suspiciously competent policy.

**6. Peak device memory is about 3.2 GiB. Size for compute and vCPUs, not
VRAM.** Measured at `ppo_minibatch_size=50_000`; halving the minibatch roughly
halves it. Large layer sizes make this look like it should need a big card, and
it does not. When renting, **vCPU count matters more than the GPU model** — the
same GPU paired with 8 versus 16 vCPUs can differ by 2x end to end. Check the
vCPU allocation before deploying, and prefer more cores over more VRAM at equal
price.

## Reasoning about a machine you cannot benchmark

If you must estimate before you have access to the hardware, the update cost is
the predictable part. One iteration of `ts_per_iteration` steps costs
approximately:

```
slices        = ppo_epochs * (exp_buffer_size // ppo_batch_size) * (ppo_batch_size // ppo_minibatch_size)
flops         = slices * ppo_minibatch_size * 6 * (policy_params + critic_params)
update_secs   = flops / (device_fp32_flops * efficiency)     # efficiency ~= 0.6-0.7 on GPU, ~0.9 on CPU
collect_secs  = ts_per_iteration / (n_proc * single_proc_sps)
total_sps     = ts_per_iteration / (collect_secs + infer_secs + update_secs)
```

At the current settings that is ~56 TFLOPs per 100k steps. The `6 *` is
forward-plus-backward (2 FLOPs per parameter forward, roughly 2x that
backward).

Reference point, for calibration only — **one machine, not a prediction**:
Apple M4 Pro, 14 cores, no CUDA, measured with `Benchmark.py`.

| Quantity | Measured |
| --- | --- |
| Single-process env throughput (project config) | ~3,500-4,000 SPS |
| Same, with `DefaultObs` instead | ~13,500-19,800 SPS |
| Collection saturation point | ~12 processes, ~36,000 SPS |
| PPO update per 100k steps, CPU (2.76 TFLOP/s peak) | 21.2 s |
| PPO update per 100k steps, Apple GPU (6.85 TFLOP/s peak) | 11.3 s |
| Peak device memory at `ppo_minibatch_size=50_000` | 3.18 GiB |

Note the saturation point: it is well below the core count. Setting
`PBT_N_PROC` to the number of cores wastes memory without buying throughput.
`Train_Ground.py`'s default of 32 is inherited from the rlgym tutorial and is
not a recommendation for your machine.

On Apple silicon specifically: `Learner` resolves `device="auto"` to CUDA-or-CPU
and never selects MPS, so a Mac runs the update on CPU regardless of what
`Benchmark.py` reports for the GPU. Fine for smoke tests, not for real runs.

## Running a long job

Set the process count from `Benchmark.py`, and write everything to a path that
survives whatever your platform does to the machine:

```bash
PBT_N_PROC=<from Benchmark.py> \
PBT_CHECKPOINT_DIR=<persistent path>/checkpoints/model_a_ground \
PBT_CSV_PATH=<persistent path>/metrics/ground_training.csv \
PBT_FITNESS_CSV_PATH=<persistent path>/metrics/ground_fitness.csv \
PBT_SAVE_EVERY_TS=10000000 \
PBT_TIMESTEP_LIMIT=1000000000 \
python Train_Ground.py 2>&1 | tee -a <persistent path>/train.log
```

- **Run it under `tmux` or `nohup`.** Any run worth doing outlives the SSH
  session or browser tab that started it.
- **Check the printed Steps per Second in the first minute** and compare it to
  `Benchmark.py`'s estimate. A large shortfall means a misconfiguration — too
  few vCPUs, or the update running on CPU because CUDA is not visible. Catching
  that in minute one instead of hour six is the entire point of measuring.
  `python -c "import torch; print(torch.cuda.is_available())"` should print
  `True` on any GPU instance.
- **Never set `RLGYM_RENDER=1` for a real run.** It slows one environment to
  real time on purpose.
- `PBT_SAVE_EVERY_TS` must be smaller than the timestep limit or the run saves
  nothing: `Learner` has no save-on-exit path when the loop ends by hitting
  `timestep_limit`. `n_checkpoints_to_keep` defaults to 5, so disk use stays
  around 5 x 174 MB regardless of save frequency.
- **To resume**, pass `PBT_CHECKPOINT_LOAD_DIR=<path to a numbered checkpoint
  dir>`. Required on any platform with session limits. If you are doing this,
  fix the `ANNEAL_SECONDS` issue above first, or the shaping schedule is
  meaningless.

`Pbt.py` multiplies environment count by population size. Its defaults
(`POPULATION_SIZE=4`, `N_PROC_PER_MEMBER=8`, sequential generations) are
deliberately conservative; scale `N_PROC_PER_MEMBER` using the same saturation
number from `Benchmark.py`, divided by how many members you run concurrently.

## What not to do

- **Do not hardcode runtime estimates into documentation or code.** They are
  machine-specific and they rot. `Benchmark.py` exists so nobody has to.
- **Do not pick hardware from a GPU spec sheet alone.** Check vCPU count.
- **Do not commit `checkpoints/`, `data/checkpoints/`, or `metrics/`.** A single
  checkpoint directory is 174 MB.
- **Do not treat `timestep_limit = 1_000_000_000` as a requirement.** It is a
  default. For validating the coordination hypothesis in `ROADMAP.md`, a
  smaller budget — or 2v2 instead of 4v4, which cuts environment cost roughly
  4x — may answer the question at a fraction of the cost. Decide what result
  you need before buying compute for a number you inherited from a config file.
