# ISSUES.md

Known defects, ranked by severity. **Read this before running a training job or
trusting a result.**

**Status as of 2026-08-29: every entry below is now FIXED**, each one verified
by actually running the fix (not just reading the diff — the exact mistake
that let P0 survive the first time). The original description, reproduction
steps, and severity reasoning for each entry are left intact below as a record
of what was wrong and why it mattered; a "**FIXED**" block under each entry's
heading says what changed and how it was verified. Kept as a record rather than
deleted, per `CLAUDE.md`.

Every entry below was originally verified against the actual installed package
sources (`rlgym-rocket-league==2.0.1`, `rlgym-api==2.0.0`, `rlgym-ppo==1.3.13`),
and the P0 entry was reproduced by executing the real `PartialInfoObs` against a
hand-constructed 4v4 `GameState`. Nothing here was inferred from reading this
repo's code alone — that is precisely the failure mode that let P0 survive.

**Why the affected files still said otherwise.** This project was inherited,
and the original author's files (every `*.py` module) were fixed directly once
each defect below was actually being corrected — see `CLAUDE.md`'s "Whose
files are whose." `README.md` and `ROADMAP.md` are left as design-intent
documents and were NOT corrected to match the fixes below (per `CLAUDE.md`,
those two specifically stay untouched) — so they may still describe pre-fix
behavior in places. This file remains the authority on what the code actually
does now.

## The property that made this list dangerous

**None of these raised.** Every one of them ran to completion, printed normal
progress, and wrote normal checkpoints and CSVs. A crash costs an hour; a
silent defect costs the study. That is why each fix below was verified by
execution, not just by reading the corrected code.

| Severity | Issue | Consequence | Status |
| --- | --- | --- | --- |
| **P0** | Partial-information frame mismatch | Research conclusion invalid | **FIXED** |
| **P1** | `timestep_limit` vs. restored `cumulative_timesteps` | Entire run trains zero steps | **FIXED** |
| **P1** | `FitnessTracker.episode_return` is always 0 | PBT selection is arbitrary | **FIXED** |
| **P2** | `CheckpointPool.archive` name collision | Opponent pool freezes | **FIXED** |
| **P2** | Aux-encoder load failure | Observation size disagreement | **FIXED** |
| **P2** | `evaluate_match` runs batch-1 on CPU | PBT costs hours more than budgeted | **FIXED** |
| **P2** | PBT never wired into the experiment matrix | Evolution pillar silently unreachable via `run_all.sh` | **FIXED** |
| **P2** | `Train_Aerial.py` output mislabeled `model_b_aerial` | `--highlight-label 02_aerial` silently finds nothing | **FIXED** |
| **P3** | `ANNEAL_SECONDS`, `"latest"`, `Environment.py` | See each entry | **FIXED** |

---

## P0 — The partial-information layer does not work — **FIXED**

**Fixed 2026-08-29.** `_masked_state_for` now selects `inverted` once from
the acting car's team and applies it to the read (`other_physics`), the
acting car's `forward`/position, and the write target (`masked_physics`)
— matching `DefaultObs`'s convention exactly, per "The fix" below.
Verified by executing the exact reproduction check this entry specifies,
plus the two additional cases from the measurement table below, all
three now behaving as originally *expected* rather than as originally
*measured*:

| Test | Expected | Result after fix |
| --- | --- | --- |
| Hidden teammate moved, still out of view | unchanged | **unchanged**, max diff 0.0 |
| Opponent in front, moved further away | changed | **changed**, max diff 0.48–0.50 |
| Opponent behind, moved further away | unchanged | **unchanged**, max diff 0.0 |

(First run of the opponent-in-front/behind check showed small nonzero
diffs even on the "should be unchanged" case; traced to `PartialInfoObs`'s
own in-view Gaussian position noise being unseeded across two freshly
constructed instances — an unrelated, pre-existing, and correct feature,
not a masking bug. Re-run with `np.random.seed()` fixed across both
calls to isolate it: exact equality, as shown above.)

Original description of the defect (now historical) follows.

`Observation.py:135` and `Observation.py:158` at the time this was filed.

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

## P1 — `timestep_limit` is absolute, but `cumulative_timesteps` is restored — **FIXED**

**Fixed 2026-08-29.** `Pbt.py`'s `train_generation` now derives
`PBT_TIMESTEP_LIMIT` as `loaded_ts + GENERATION_TIMESTEPS`, reading
`loaded_ts` off the resolved checkpoint path's own digit-named folder
(`int(os.path.basename(load_folder))`, confirmed against
`Learner.save()`'s source — that folder name literally IS
`str(cumulative_timesteps)`). `Train_Aerial.py` got the equivalent fix:
`AERIAL_TIMESTEP_LIMIT` (absolute) was replaced with
`AERIAL_ADDITIONAL_TIMESTEPS` (relative to the loaded checkpoint), computed
via the same technique in a new top-level `LOADED_TIMESTEPS` (has to be
true module-level code, not inside `if __name__ == "__main__":`, because
`build_rlgym_v2_env` needs it too and Windows multiprocessing's "spawn"
start method re-imports the module fresh in every worker without ever
running the `__main__` guard).

Verified by running a real, scaled-down 2-generation PBT loop end to end
(`POPULATION_SIZE=2`, `GENERATION_TIMESTEPS=6000`, real `Train_Ground.py`
subprocesses, not a mock). Generation 0 trained both members fresh to a
saved checkpoint at `cumulative_timesteps=100000` (one full
`ts_per_iteration` batch, since that's hardcoded larger than the smoke
test's tiny budget — expected iteration-granularity overshoot, not a
bug). Generation 1 then logged
`warm start: own lineage, loaded_ts=100000, target_ts=106000` and
`warm start: exploited checkpoint, loaded_ts=100000, target_ts=106000`,
and — the actual proof — both members' `Learner` output showed
`Loading from checkpoint ... 100000` followed by a full new training
iteration ending at `Cumulative Timesteps: 200,000` with a new checkpoint
saved. Pre-fix, generation 1 would have loaded `cumulative_timesteps=
100000`, compared it against an absolute `PBT_TIMESTEP_LIMIT=6000`, and
exited on iteration zero.

This also caught a second, related defect worth recording: `Learner` has
no save-on-exit path, only a periodic one (`ts_since_last_save >=
save_every_ts`), so a generation whose timestep budget is smaller than
`save_every_ts` would save zero checkpoints regardless of the fix above.
`Train_Ground.py` now exposes `PBT_SAVE_EVERY_TS`, and `Pbt.py` sets it to
a fraction of `GENERATION_TIMESTEPS` to guarantee at least one save per
generation — this predates the current fix pass but is the reason the
verification above had anything to load in the first place.

`experiments/02_aerial.sh` had to be updated in the same pass: it was
written against `Train_Aerial.py`'s old absolute-`AERIAL_TIMESTEP_LIMIT`
interface (itself a shell-level workaround for this exact bug, added by
whoever wrote `params.sh` — see its `latest_checkpoint_ts`/
`latest_checkpoint_path` comments referencing this same P1 entry), which
the Python-level fix silently ignores. Now passes
`AERIAL_ADDITIONAL_TIMESTEPS="$LIMIT"` and lets `Train_Aerial.py` derive
the absolute limit itself.

Original description of the defect (now historical) follows.

`Pbt.py:176` and `Train_Aerial.py:43` at the time this was filed.

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

## P1 — `FitnessTracker.episode_return` is always exactly zero — **FIXED**

**Fixed 2026-08-29.** `get_rewards` now averages only over the tracked
(blue, canonically — self-play mirrors one policy onto both teams, so
which team is arbitrary, but it has to be exactly one, not both) team's
rewards, instead of `np.mean(list(rewards.values()))` over all 8 agents.

Verified by directly constructing a `FitnessTracker(GoalReward(), ...)`
against a real 4v4 `GameState` from `RocketSimEngine`, forcing
`state.goal_scored=True` with `state.ball.position[1]` set to derive
`scoring_team` to each team in turn (per `GameState.scoring_team`'s own
source), and confirming `episode_sum` came out `+1.0` when the tracked
team scored and `-1.0` when the opposing team did — previously it was
`0.0` in both cases, on every step, unconditionally.

This is also `Pbt._evaluate_population_fallback`'s fitness signal (used
when a member has no checkpoint yet, or `evaluate_match` raises), so the
fix directly un-breaks that path too, though the scaled-down PBT run used
to verify the P1 timestep-limit fix above always had checkpoints
available and so exercised the real cross-play path instead.

Original description of the defect (now historical) follows.

`Metrics.py:253` at the time this was filed.

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

## P2 — `CheckpointPool.archive` collides with itself once the pool is full — **FIXED**

**Fixed 2026-08-29.** Naming now uses a monotonic counter
(`self._next_ckpt_id`, never reused, incremented on every `archive()`
call) instead of `len(list_checkpoints())`. Seeded from any existing pool
contents on construction (`_infer_next_ckpt_id`), so a freshly constructed
`CheckpointPool` pointed at a non-empty `pool_dir` continues numbering
correctly rather than colliding with what's already there.

Verified by archiving 6 checkpoints into a pool with `max_checkpoints=3`:
the pool correctly rotated (`ckpt_000000..002`, then `...001..003`, ...,
ending at `...003..005`) instead of freezing at `ckpt_000000..002` after
the third archive. Also verified that a second `CheckpointPool` instance
constructed against that same non-empty directory resumes numbering at
the correct next id rather than restarting at 0.

Original description of the defect (now historical) follows.

`Self_Play.py:93` at the time this was filed.

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

## P2 — Aux-encoder load failure produces an observation-size disagreement — **FIXED**

**Fixed 2026-08-29.** `_get_aux_encoder`'s `except` block now raises
(`RuntimeError`, chained from the original exception) instead of printing
a warning and continuing with `self._aux_encoder = None`. This makes the
module's own docstring claim — a mismatch "fails loudly and immediately
on startup" — actually true instead of contradicted by this one code path.

Verified two ways: (1) set `AUX_ENCODER_CHECKPOINT` to a nonexistent
path, confirmed `get_obs_space` still reports the encoder-augmented size
(228 = 212 + `ENCODED_DIM`) while `build_obs` now raises immediately
rather than silently returning 212; (2) trained a real, throwaway
`StateRepresentationNet` checkpoint end to end via
`Train_Auxiliary_Encoder.py` and confirmed the success path is
unaffected — `get_obs_space` and `build_obs` still agree (both 228), no
regression.

Original description of the defect (now historical) follows.

`Observation.py:127` at the time this was filed.

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

## P2 — `evaluate_match` is a batch-1 CPU inference loop — **FIXED**

**Fixed 2026-08-29.** `FrozenPolicy` gained `act_batch(obs_batch)`, which
runs one forward pass over an entire `(n_agents, obs_size)` batch
(`DiscreteFF.get_output` already reshapes to `(batch, n_actions)`
regardless of input batch size — confirmed by reading its source, so this
produces identical per-agent output to the old call-once-per-agent loop,
just without 7 redundant Python/torch round trips per step). `evaluate_match`
now groups agents by team each step and calls `act_batch` twice (once per
team) instead of `act` eight times. `device` now defaults to a new
`_resolve_device()` helper that mirrors `Learner`'s own `device="auto"`
resolution (CUDA if available, else CPU — confirmed against its source),
instead of `FrozenPolicy`'s hardcoded `device="cpu"` default.

One upstream limitation surfaced and documented rather than worked
around: `DiscreteFF.get_action(deterministic=True)` does
`probs.cpu().numpy().argmax()` with no axis argument, which argmaxes over
the flattened batch instead of per row — not meaningful for batch size
> 1. `act_batch` raises `NotImplementedError` if called that way;
`evaluate_match` already only ever used `deterministic=False` (the
default), so this doesn't affect it.

Verified as part of the same scaled-down PBT run used for the P1 fix
above: `evaluate_population`'s cross-play tournament ran successfully for
both generations against real `FrozenPolicy`-loaded checkpoints, with no
`evaluate_match failed` fallback message in the output (which would have
appeared if the batched path raised).

Original description of the defect (now historical) follows.

`Self_Play.py:124` at the time this was filed.

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

## P2 — PBT (`Pbt.py`) was never wired into the experiment matrix — **FIXED**

`experiments/run_all.sh` and every ablation script `00_baseline.sh` through
`05_aux_encoder.sh`, at the time this was filed.

### What the code did

Every one of the six ablation scripts called `Train_Ground.py` or
`Train_Aerial.py` directly as a single training run. `Pbt.py` — this
project's whole population-based-training / evolution pillar (Liu et al.
2019 / FTW-style, see `Pbt.py`'s own module docstring) — was never
referenced anywhere in `experiments/`. It only ran if someone manually
typed `python Pbt.py`, which also bypassed `params.sh`'s
`$OUT`/`ROUND`/`$N_PROC` conventions entirely: `Pbt.py` hardcoded
`CHECKPOINT_ROOT = "checkpoints/pbt"` and `CSV_ROOT = "metrics/pbt"` as
literals relative to CWD, with no `ROUND` equivalent at all.

### Why it matters

`CLAUDE.md` already flags that several of this project's research
pillars are "broken or unwired" — PBT was one of the unwired ones, and
silently so: `run_all.sh` completing without error says nothing about
whether the evolution pillar was ever exercised, since it was never in
the queue to begin with. Anyone following the documented "ONLY COMMAND
YOU NEED" workflow would never produce PBT results, never generate
`generations.jsonl`, and never exercise `Self_Play.evaluate_match`'s
cross-play tournament in the pipeline context — with no indication from
`run_all.sh`'s output that any of that was missing.

### The fix

- Added `experiments/06_pbt.sh`, following the exact pattern every other
  script uses (source `params.sh`, `banner`, log to `$OUT/logs/`), and
  wired it into `run_all.sh`'s `EXPERIMENTS` queue behind a new
  `INCLUDE_PBT=1` opt-in flag (mirroring `CORE_ONLY`'s gating pattern) —
  off by default, since a full PBT pass costs meaningfully more compute
  than any single ablation here (population x generations x
  timesteps-per-generation, not just timesteps-per-run), and that cost
  should be a conscious choice rather than silently added to every
  invocation.
- `Pbt.py` now reads `PBT_OUT_DIR` (unset = original relative behavior,
  so standalone `python Pbt.py` is unaffected) and, when set, roots
  `CHECKPOINT_ROOT`/`CSV_ROOT`/`GENERATION_LOG_PATH` under it. `CSV_ROOT`
  is FLAT under `$OUT/metrics` in that case (not nested in a `pbt/`
  subfolder), so `pbt_member_N_{coordination,fitness}` CSVs sit alongside
  `00_baseline.csv` etc. for `Plot_Results.py` to pick up — only
  `generations.jsonl` stays nested at `metrics/pbt/` specifically, since
  `Plot_Results.py`'s `plot_pbt_generations` looks for it at exactly that
  path regardless of `PBT_OUT_DIR`.
- `Pbt.py`'s `POPULATION_SIZE`, `N_GENERATIONS`, `GENERATION_TIMESTEPS`,
  and `N_PROC_PER_MEMBER` are now also readable from
  `PBT_POPULATION_SIZE`/`PBT_N_GENERATIONS`/`PBT_GENERATION_TIMESTEPS`/
  `PBT_N_PROC_PER_MEMBER` env vars (checked against every env var
  `Train_Ground.py` already reads before picking these names — none
  collide with the inner `PBT_*` channel `Pbt.py` already uses to
  configure each `Train_Ground.py` subprocess, e.g. `PBT_N_PROC`,
  `PBT_POLICY_LR`). `06_pbt.sh` maps `ROUND` onto `GENERATION_TIMESTEPS`
  (`= ROUND * 100,000`, the same formula every other script uses for its
  timestep budget, just applied per member per generation instead of per
  whole run), and keeps `POPULATION_SIZE`/`N_GENERATIONS` small (2/2) at
  or below the documented local-smoke-test `ROUND` (5) so a bare
  `ROUND=5` invocation doesn't take population-times-generations longer
  than every other script; above that they step up to `Pbt.py`'s own
  real-run defaults (4/10). `N_PROC_PER_MEMBER` gets the full measured
  `$N_PROC`, not a fraction of it — `Pbt.py` trains population members
  sequentially, one at a time, never in parallel (confirmed by reading
  `main()`), so there's no concurrent-process split to account for.
- Caught and fixed one bug this surfaced before it could bite: `SAVE_EVERY_TS
  = max(200_000, GENERATION_TIMESTEPS // 8)`'s hardcoded floor was never
  binding at the old fixed `GENERATION_TIMESTEPS=2_000_000` default
  (`// 8` already exceeds 200,000), but would have silently broken
  checkpoint saving the moment `GENERATION_TIMESTEPS` became overridable
  to something smaller for a smoke test — at 20,000, the floor would
  make `SAVE_EVERY_TS=200,000 > GENERATION_TIMESTEPS`, so no save would
  ever trigger within the generation, exactly the failure this line
  exists to prevent. Changed to `max(1, GENERATION_TIMESTEPS // 8)` —
  unchanged at the default scale, safe at any scale, since the real
  floor on how few steps get collected before a save is even checked is
  `Train_Ground.py`'s hardcoded `ts_per_iteration` (100,000), not this
  value.

### Verified by execution

Ran `N_PROC=2 ROUND=1 bash experiments/06_pbt.sh` end to end (population
2, generations 2, generation_timesteps 100,000 — `ROUND<=5` triggers the
small defaults). Confirmed: it completed with exit code 0; real
cross-play fitness was computed both generations (`{0: 0.625, 1: 0.375}`
then `{0: 0.375, 1: 0.625}`, no fallback-to-proxy message anywhere in the
output); generation 1 correctly resumed from a `$OUT`-rooted checkpoint
path (`output\checkpoints\pbt\pbt_member_0-.../100000`) and trained on to
200,000, with the exploited member showing perturbed hyperparameters
(`policy_lr` 1.00e-04 → 8.80e-05); `output/metrics/pbt_member_{0,1}_
{coordination,fitness}.<pid>.csv` all exist and are non-empty (84 fitness
rows total across both members); `output/metrics/pbt/generations.jsonl`
has all 4 expected rows (2 members x 2 generations); `python
Plot_Results.py --results-dir output` ran without error and produced
`pbt_fitness_by_generation.png` (visually confirmed: member 0 starts at
0.625 and drops to 0.375 while member 1 mirrors it upward — consistent
with a 2-member zero-sum matchup after an exploit swap).

---

## P2 — `Train_Aerial.py`'s real output was mislabeled `model_b_aerial`, not `02_aerial` — **FIXED**

`Train_Aerial.py:85-87` and `experiments/02_aerial.sh`, at the time this was
filed.

### What the code did

`Train_Ground.py` reads `RUN_LABEL`/`CSV_PATH`/`FITNESS_CSV_PATH` from
`PBT_RUN_LABEL`/`PBT_CSV_PATH`/`PBT_FITNESS_CSV_PATH` env vars, which is
what lets `01_ground.sh` set `PBT_RUN_LABEL="01_ground"` and get output
consistently labeled to match its own filename, written straight into
`$OUT/metrics/`. `Train_Aerial.py` had no equivalent: `RUN_LABEL =
"model_b_aerial"`, `CSV_PATH = "metrics/aerial_training.csv"`,
`FITNESS_CSV_PATH = "metrics/aerial_fitness.csv"` were bare constants
with no override, and always relative to CWD rather than `$OUT`.
`02_aerial.sh` worked around the second half by writing to a
repo-relative `metrics/` dir and `mv`-ing the results into `$OUT/metrics/`
afterward, but had no way to work around the label at all.

### How this was found

Not from reading the code — from actually running the real
`experiments/run_all.sh` pipeline (`N_PROC=8 ROUND=5 CORE_ONLY=1`,
verifying `Plot_Results.py` against genuine training output rather than
the hand-written synthetic CSVs it had only been tested against before).
The synthetic test data used labels matching the intended convention
(`"01_ground"`, `"02_aerial"`) because whoever wrote it assumed that
convention held; it doesn't, for Model B specifically.

### Why it matters

Every other script's CSV `run_label` matches its own filename. Anyone
following that pattern — e.g. `python Plot_Results.py --results-dir
output --highlight-label 02_aerial`, the natural guess given
`--highlight-label 01_ground` is the documented default — silently gets
no data for that condition, since no row actually has `run_label ==
"02_aerial"`. `Plot_Results.py` itself doesn't crash (it groups by
whatever `run_label` values are actually present), so this is a
data-labeling trap, not a plotting bug — but it directly undermines
"the main ablation-vs-baseline pair" comparisons `Plot_Results.py` is
for, and it's exactly the kind of surprise that costs someone an hour of
"why is my highlight plot empty" before they think to grep the raw CSV
for what label actually got written.

### The fix

`Train_Aerial.py`'s `RUN_LABEL`/`CSV_PATH`/`FITNESS_CSV_PATH` now read
from `AERIAL_RUN_LABEL`/`AERIAL_CSV_PATH`/`AERIAL_FITNESS_CSV_PATH` env
vars (same `AERIAL_*` prefix this file already uses for
`AERIAL_CHECKPOINT_TO_LOAD` etc.; defaults unchanged, so nothing breaks
if unset). `experiments/02_aerial.sh` now sets all three explicitly
(`AERIAL_RUN_LABEL="02_aerial"`, `AERIAL_CSV_PATH="$OUT/metrics/
02_aerial.csv"`, `AERIAL_FITNESS_CSV_PATH="$OUT/metrics/
02_aerial_fitness.csv"`) and no longer needs the `mv` workaround --
output lands directly in `$OUT/metrics/`, consistently labeled, the same
way `01_ground.sh` already worked.

### Verified by execution

Re-ran `02_aerial.sh` alone (`N_PROC=8 ROUND=1`, warm-started from a real
`01_ground` checkpoint from the full smoke test) and confirmed the shards
land at `$OUT/metrics/02_aerial.<pid>.csv` /
`02_aerial_fitness.<pid>.csv` with `run_label,...` rows reading
`02_aerial,...`, not `model_b_aerial,...` -- and that no
`metrics/aerial_*.csv` gets written to the repo-relative CWD anymore.
Re-ran `Plot_Results.py --results-dir output` against the corrected
3-condition dataset (`baseline_random`, `01_ground`, `02_aerial`, 146
real episode rows from genuinely multiple worker processes -- 8 shards
each for `01_ground`/`02_aerial`) and confirmed all 19 expected plots
(7 metrics x {bar, histogram} + 4 coordination trend lines + 1 fitness
boxplot) generate correctly with the intended labels throughout, no
`--highlight-label` guessing required.

---

## P3 — Documented elsewhere, still worth knowing on handover — **FIXED**

- **`ANNEAL_SECONDS` is per-process wall clock** (`Train_Ground.py`,
  `Train_Aerial.py`; discussed in `COMPUTE.md`). The anneal restarts from zero
  on every checkpoint resume and never completes on session-limited platforms.
  This matters more than a scheduling detail: per Liu et al., annealing dense
  shaping to zero *is* the mechanism that produces coordination rather than four
  selfish ball-chasers. It stacks with P0 — both remove the conditions under
  which coordination could emerge. Fix by annealing on `cumulative_timesteps`.

  **Fixed 2026-08-29.** `AnnealedCombinedReward` (`Rewards.py`) now takes
  `anneal_timesteps`/`n_proc`/`initial_timesteps` instead of `anneal_seconds`,
  tracking `len(agents)`-per-`get_rewards()`-call as a local step count
  (confirmed against `rlgym_ppo`'s `BatchedAgentManager._collect_response`
  source: it counts `n_collected = prev_n_agents` for a multi-agent env, i.e.
  cumulative_timesteps advances by agent count, not by 1 per `env.step()`),
  scaled by `n_proc` to estimate the global total, seeded with
  `initial_timesteps` recovered from a resumed checkpoint's own digit-named
  path. `ANNEAL_TIMESTEPS` is now an env var on `Train_Ground.py` (default
  200M), which also unblocked `experiments/04_no_anneal.sh`'s pre-existing
  self-check (it was written expecting exactly this exposure and refused to
  run without it). Verified with a direct unit test: progress is 0 before any
  steps, scales correctly with `n_proc`, clamps to 1.0 past the budget, and —
  the actual bug — resuming with `initial_timesteps=900` against a 1000-budget
  anneal starts at `progress=0.9`, not 0. `Benchmark.py`'s mirror env
  construction and `Train_Aerial.py`'s equivalent (seeded from
  `LOADED_TIMESTEPS`, Model A's own progress) were updated to match.

- **`checkpoint_load_folder` defaults to `"latest"`, not `None`.**
  `Train_Ground.py` and `Train_Aerial.py` already pass it explicitly; preserve
  that on any refactor. Symptom of losing it: a "fresh" run that starts
  suspiciously strong.

  **Re-confirmed 2026-08-29, no drift found.** Grepped every `Learner(`
  construction site and every `checkpoint_load_folder=` usage across the repo:
  both `Train_Ground.py` and `Train_Aerial.py` still pass it explicitly. Not a
  fix, just a check requested alongside the others.

- **`Environment.py` is leftover first-draft tutorial code** — 1v1,
  `DefaultObs`, `zero_padding=None`, full information — and it has a complete
  `__main__` that really starts training. It is the only filename in the repo
  that reads like "the main environment." Running it by mistake gives a training
  job with no relationship to this project's research setup.

  **Fixed 2026-08-29.** Renamed (via `git mv`, history preserved) to
  `_legacy_tutorial_environment.py`, with a header docstring explaining its
  status, matching `train_phase1.py`/`src/`'s existing "left for reference,
  safe to ignore or delete" treatment.

## Not issues

- **`analyze.py` does not exist.** Deliberate — see `README.md`. The metrics are
  collected; the t-test was intentionally left uncoded.
- **`copy.deepcopy` in `Observation.py` is ~80% of step time.** A performance
  characteristic, not a defect, and `COMPUTE.md` is right that fixing it first
  is usually wasted work while the PPO update dominates. **Update, 2026-08-29:**
  with P0 fixed, that cost now actually buys the partial-information layer it
  was always meant to buy, rather than partly paying for masking that got
  discarded. Still not worth optimizing first — see `COMPUTE.md`'s bottleneck
  guidance and run `Benchmark.py` before touching it.

## What to fix before spending money on a run

**All fixed as of 2026-08-29 — see the FIXED blocks under each entry above for
what changed and how each was verified.** Before this pass, the priority was:
P0, then `Train_Aerial.py`'s timestep limit, then `episode_return`. The full
training program costs on the order of $10–20 of rented GPU time (see
`COMPUTE.md`), so compute was never the constraint here — runs whose results
turned out to be unusable were. Run `python Benchmark.py` before sizing
hardware or estimating runtime for an actual training job, per `COMPUTE.md`.
