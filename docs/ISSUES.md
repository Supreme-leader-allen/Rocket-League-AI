# ISSUES.md

Known defects, ranked by severity. **Read this before running a training job or
trusting a result.**

Every entry below was verified against the actual installed package sources
(`rlgym-rocket-league==2.0.1`, `rlgym-api==2.0.0`, `rlgym-ppo==1.3.13`), and the
P0 entry was reproduced by executing the real `PartialInfoObs` against a
hand-constructed 4v4 `GameState`. Nothing here is inferred from reading this
repo's code alone — that is precisely the failure mode that let P0 survive.

**Why the affected files still say otherwise.** This project was inherited, and
the original author's files (`README.md`, `ROADMAP.md`, and every `*.py` module)
are left untouched so the boundary between inherited and added work stays
legible — see `CLAUDE.md`. So `README.md` still describes the occlusion layer as
working and `ROADMAP.md` still rests the coordination argument on it. Those are
statements of design intent that were never corrected, not evidence against what
follows. This file is the authority; expect it to contradict them.

## The property that makes this list dangerous

**None of these raise.** Every one of them runs to completion, prints normal
progress, and writes normal checkpoints and CSVs. A crash costs an hour; a
silent defect costs the study. Rank severity accordingly:

| Severity | Issue | Consequence |
| --- | --- | --- |
| **P0** | Partial-information frame mismatch | Research conclusion invalid |
| **P1** | `timestep_limit` vs. restored `cumulative_timesteps` | Entire run trains zero steps |
| **P1** | `FitnessTracker.episode_return` is always 0 | PBT selection is arbitrary |
| **P2** | `CheckpointPool.archive` name collision | Opponent pool freezes |
| **P2** | Aux-encoder load failure | Observation size disagreement |
| **P2** | `evaluate_match` runs batch-1 on CPU | PBT costs hours more than budgeted |
| **P3** | `ANNEAL_SECONDS`, `"latest"`, `Environment.py` | See each entry |

---

## P0 — The partial-information layer does not work

`Observation.py:135` and `Observation.py:158`.

### What the code does

`PartialInfoObs._masked_state_for` selects a coordinate frame **per car, from
that car's own team**:

```python
acting_physics = acting_car.physics if acting_car.is_orange else acting_car.inverted_physics
masked_physics = masked_other.physics if masked_other.is_orange else masked_other.inverted_physics
```

`DefaultObs._build_obs` selects a frame **once, from the acting agent's team**,
then applies it uniformly to every car:

```python
if car.team_num == ORANGE_TEAM:
    inverted = True
    ...
car_obs = self._generate_car_obs(other_car, inverted)   # same `inverted` for every car
```

The two conventions disagree.

### What actually happens

Two distinct failures, both reproduced by execution:

**1. Teammate masking is a no-op.** For a blue acting agent, `DefaultObs` reads
`car.physics` for every car, but `PartialInfoObs` writes a blue teammate's
masked values into `car.inverted_physics` — an object nothing reads. (Orange
agents get the mirror-image of the same bug.) **Teammates are always fully
observed, at all distances, through walls, regardless of the view cone.**

**2. The opponent view cone is inverted.** `forward` comes from the inverted
frame while the opponent's position comes from the un-inverted frame, so
`other_physics.position - acting_pos` subtracts two vectors expressed in
different frames. The result is not a meaningful direction. Measured behavior is
an exact mirror: **the agent is blind to opponents in front of it and tracks the
ones behind it.**

Measured, with the acting car at the origin facing +x (observation length 212,
matching the value recorded in `README.md`):

| Test | Expected | Actual |
| --- | --- | --- |
| Teammate out of view, moved x=-1000 → -4000 | observation unchanged | **changed**, max diff 0.73 |
| Opponent in front, moved x=+2000 → +4000 | observation changed | **unchanged** |
| Opponent behind, moved x=-2000 → -4000 | observation unchanged | **changed**, max diff 0.50 |

### Why this is the most severe entry

This does not degrade training. It trains a perfectly functional agent that
answers a different research question than the one `ROADMAP.md` states.

The run completes. `CoordinationMetrics` emits `overcommit_rate` and
`simultaneous_air_rate` exactly as designed. The numbers look publishable. But
the agent learned under **full teammate observability and scrambled opponent
information**, so the "imperfect information" pillar was never active — and the
half that failed is the *teammate* half, which is the only half the coordination
question depends on. Every other issue in this file wastes a run at worst; this
one invalidates the conclusion drawn from a run that appeared to succeed.

Secondary cost: the `copy.deepcopy` in `_masked_state_for` — roughly 80% of
environment step time, see `COMPUTE.md` — is spent producing teammate masking
that is then discarded.

### Why the docstrings did not catch it

`Rewards.py` uses the identical-looking `physics if is_orange else
inverted_physics` idiom (it comes from the RLGym v2 tutorial) and **is correct
there**: the car and the ball go through the same rule, so the frame is
self-consistent, and the inversion is a 180° rotation about Z (`INV_VEC =
[-1,-1,1]`, a proper rotation) which preserves dot products. Copying that idiom
into the observation builder breaks it, because there the frame must match
`DefaultObs`'s convention rather than merely be internally consistent.

This is a cross-module convention mismatch. The verification recorded in
`README.md`'s "Confirmed, not just traced" section checked per-call API shapes
and signatures, which is a different class of question and would not have
surfaced this.

### How to reproduce

Build a 4v4 `GameState`, place a car outside the acting agent's view cone with
no memory of it, and check that moving that car does not change the acting
agent's observation:

```python
ob = PartialInfoObs(zero_padding=4)
ob.reset(agents, state_a, {})
a = ob.build_obs(["blue0"], state_a, {})["blue0"]
ob.reset(agents, state_b, {})          # identical except the hidden car moved
b = ob.build_obs(["blue0"], state_b, {})["blue0"]
assert np.allclose(a, b), "masking never reached DefaultObs"
```

### The fix

Select the frame once, from the acting agent's team, and apply it to every car —
matching `DefaultObs`:

```python
inverted = acting_car.is_orange
acting_physics = acting_car.inverted_physics if inverted else acting_car.physics
# ...and for each other car, including the write target:
masked_physics = masked_other.inverted_physics if inverted else masked_other.physics
```

Note this must be applied to the read (`other_physics`), the acting car's
`forward`/position, and the write target, or the cone and the mask will
disagree in a new way.

---

## P1 — `timestep_limit` is absolute, but `cumulative_timesteps` is restored

`Pbt.py:176` and `Train_Aerial.py:43`.

### What the code does

`Pbt.py` sets an absolute per-generation limit while warm-starting from a
checkpoint:

```python
env["PBT_TIMESTEP_LIMIT"] = str(GENERATION_TIMESTEPS)   # 2_000_000, every generation
```

`rlgym_ppo`'s `Learner.load()` restores `cumulative_timesteps` from the
checkpoint's `BOOK_KEEPING_VARS.json`, and its main loop is
`while self.agent.cumulative_timesteps < self.timestep_limit`.

### What actually happens

Generation 0 trains 0 → 2e6 and saves. Generation 1 loads that checkpoint, so
`cumulative_timesteps` is already 2e6, `2e6 < 2e6` is false, and **the loop
exits without a single iteration.** Generations 1–9 are no-ops: they re-run the
cross-play tournament against unchanged checkpoints, rank them, and perturb
hyperparameters that are never used.

### Why it matters

`subprocess.run()` returns 0. `Pbt.py` prints `[gen 1] training member 0 ...`
for every generation and finishes with `PBT complete. Best member: ...`. Nothing
in the output distinguishes this from a successful ten-generation run. What you
actually have is the generation-0 policy.

**The same trap on the main path is worse.** `Train_Aerial.py`'s
`AERIAL_TIMESTEP_LIMIT` defaults to `1_000_000_000`, and `Train_Ground.py`'s
`timestep_limit` defaults to the same value. Running the two-model plan in
`ROADMAP.md` as documented — train Model A to its default limit, then warm-start
Model B — means **Model B exits immediately having trained nothing**, and looks
like it finished. Model B would be Model A under a different name.

### The fix

Derive the limit from the loaded checkpoint's timestep count rather than passing
an absolute value: `limit = loaded_checkpoint_ts + GENERATION_TIMESTEPS`. The
checkpoint's cumulative count is already recoverable —
`Pbt.find_latest_checkpoint` resolves digit-named subfolders keyed by exactly
that number.

---

## P1 — `FitnessTracker.episode_return` is always exactly zero

`Metrics.py:253`.

### What the code does

```python
self._episode_sum += float(np.mean(list(rewards.values())))
```

`rewards` contains all 8 agents. `GoalReward` is antisymmetric — its source
returns `1 if state.scoring_team == state.cars[agent].team_num else -1`.

### What actually happens

In a balanced 4v4, `(4×(+1) + 4×(−1)) / 8 = 0`, on every step. The
`episode_return` column is a constant zero for the life of the project.

### Why it matters

`Pbt._evaluate_population_fallback` ranks members on this column. With every
member scoring 0, `sorted` is stable, so ranking degenerates to member index:
"replace the worst performers" becomes "replace the highest-numbered members."
The exploit/explore step — the entire evolutionary mechanism — is selecting at
random whenever the fallback path is taken.

This compounds with the P1 above: generations that train zero steps save no new
checkpoints, which is exactly the condition that triggers the fallback.

The docstring's justification ("GoalReward is identical across teammates, so
averaging rather than summing keeps this a per-agent return") is half right —
the reward is identical across teammates but *opposite across teams*, and the
mean is taken over all agents rather than one team.

### The fix

Average over the tracked member's own team only, not the full agent dict.

---

## P2 — `CheckpointPool.archive` collides with itself once the pool is full

`Self_Play.py:93`.

```python
dest = self._pool_dir / f"ckpt_{len(self.list_checkpoints()):06d}"
```

Once the pool reaches `max_checkpoints`, each `archive()` adds one and prunes
one, so the count is constant, so the computed name is constant. Combined with
`copytree(..., dirs_exist_ok=True)`, every subsequent archive overwrites the
same directory and **the pool stops taking in new opponents.**

Why it matters: this pool *is* the autocurriculum. A frozen pool means opponent
difficulty stops tracking the policy's improvement, which is the mechanism the
Liu et al. result depends on. It is P2 only because `CheckpointPool` is not yet
wired into `build_rlgym_v2_env` — see `ROADMAP.md`'s known-gaps section. On the
day that wiring lands, this becomes P1.

Fix: name by a monotonic counter or timestamp, not by current pool size.

---

## P2 — Aux-encoder load failure produces an observation-size disagreement

`Observation.py:127`.

`get_obs_space` adds `ENCODED_DIM` whenever `AUX_ENCODER_CHECKPOINT` is set. But
when the encoder fails to load, `_get_aux_encoder` prints a warning and
continues with `None`, so `build_obs` returns the un-concatenated observation.
`rlgym_ppo` sizes the policy input layer from the former and feeds it the
latter.

Why it matters: the module docstring claims this design makes a mismatch "fail
immediately on startup rather than something subtle later," and this
except-and-continue branch creates exactly the mismatch it claims to prevent. It
does eventually raise — a matmul shape error at the first forward pass, with
nothing in the message mentioning the encoder. Least severe of the P2s because
it is the only issue in this file that fails loudly at all.

Fix: let the load failure propagate, or fall back consistently in both
`get_obs_space` and `build_obs`.

---

## P2 — `evaluate_match` is a batch-1 CPU inference loop

`Self_Play.py:124`.

`FrozenPolicy` defaults to `device="cpu"`, and `evaluate_match` calls
`policy.act(obs[agent])` once per agent per step — 8 separate batch-1 forward
passes through a 7.9M-parameter network, for up to 4500 steps per episode, over
`POPULATION_SIZE * (POPULATION_SIZE-1) / 2` matchups per generation.

Why it matters: this can take longer than the training it is evaluating, and it
is not counted in any of the runtime estimates derived from `Benchmark.py` —
`Benchmark.py` measures collection and the PPO update, not the PBT tournament.
Budget for it separately.

Fix: batch all agents on one team into a single forward pass, and honor the
training device.

---

## P3 — Documented elsewhere, still worth knowing on handover

- **`ANNEAL_SECONDS` is per-process wall clock** (`Train_Ground.py`,
  `Train_Aerial.py`; discussed in `COMPUTE.md`). The anneal restarts from zero
  on every checkpoint resume and never completes on session-limited platforms.
  This matters more than a scheduling detail: per Liu et al., annealing dense
  shaping to zero *is* the mechanism that produces coordination rather than four
  selfish ball-chasers. It stacks with P0 — both remove the conditions under
  which coordination could emerge. Fix by annealing on `cumulative_timesteps`.
- **`checkpoint_load_folder` defaults to `"latest"`, not `None`.**
  `Train_Ground.py` and `Train_Aerial.py` already pass it explicitly; preserve
  that on any refactor. Symptom of losing it: a "fresh" run that starts
  suspiciously strong.
- **`Environment.py` is leftover first-draft tutorial code** — 1v1,
  `DefaultObs`, `zero_padding=None`, full information — and it has a complete
  `__main__` that really starts training. It is the only filename in the repo
  that reads like "the main environment." Running it by mistake gives a training
  job with no relationship to this project's research setup.

## Not issues

- **`analyze.py` does not exist.** Deliberate — see `README.md`. The metrics are
  collected; the t-test was intentionally left uncoded.
- **`copy.deepcopy` in `Observation.py` is ~80% of step time.** A performance
  characteristic, not a defect, and `COMPUTE.md` is right that fixing it first
  is usually wasted work while the PPO update dominates. Note that P0 changes
  the calculus: part of that cost currently buys nothing at all.

## What to fix before spending money on a run

P0, then `Train_Aerial.py`'s timestep limit, then `episode_return`. All three
are small edits. The full training program costs on the order of $10–20 of
rented GPU time (see `COMPUTE.md`), so compute is not the constraint here —
runs whose results turn out to be unusable are.
