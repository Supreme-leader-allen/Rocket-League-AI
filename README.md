# Rocket League AI — Starter Project

Two models: Model A (ground, 4v4) trains first, Model B (aerial) warm-starts from Model A's checkpoint. See `ROADMAP.md` for the full design rationale (fixed network across phases, no action abstraction, humanlike constraints, self-play-driven autocurriculum per Liu et al. 2019).

**The files below (`Train_Ground.py`, `Train_Aerial.py`, `Rewards.py`, `Observation.py`, `Actions.py`, `Self_Play.py`) are the current, correct ones — they use the RLGym v2 API (`rlgym.api` + `RocketSimEngine`), matching your tutorial code.**

`train_phase1.py` and everything under `src/` are from an earlier draft built on the older `rlgym-sim` package before you shared your tutorial code. That's a different, now-secondary API — don't mix it with the v2 files. Left in place for reference only; safe to ignore or delete locally.

## Setup

```
pip install -r requirements.txt
```

All API calls and file interactions below have been run against the actual installed versions (`rlgym==2.0.1`, `rlgym-ppo==1.3.13`, `torch==2.13.0`) in this environment, not just traced by hand — see "Confirmed, not just traced" at the bottom for what was specifically checked and what was found wrong.

## Files

`Rewards.py` — `SpeedTowardBallReward`, `InAirReward`, `VelocityBallToGoalReward` (dense, individual shaping) plus `AnnealedCombinedReward`, which decays shaping weight toward zero over wall-clock training time so the agent ends up optimizing mostly for the sparse, team-shared `GoalReward`. That anneal is the credit-assignment/coordination mechanism, per Liu et al.'s *Emergent Coordination through Competition* — see `ROADMAP.md`.

`Observation.py` — `PartialInfoObs`. Wraps `DefaultObs` but occludes other cars outside each agent's forward view cone, holding last-known position/velocity (with a staleness cutoff) instead of updating every tick. This is the "no perfect information" piece, and it doubles as a cheap memory substitute since `rlgym_ppo`'s stock networks aren't recurrent. Also handles the optional Lucy-SKG auxiliary-encoder concatenation (see below).

`Actions.py` — `HumanlikeAction`. Full raw `LookupTableAction` action space (no abstraction — the agent always picks its own action), with a reaction-delay queue and input-repeat baked in for the "humanlike reaction time / inputs per second" requirement.

`Self_Play.py` — `CheckpointPool` for archiving and sampling past checkpoints as opponents (**not fully wired into `Train_Ground.py`'s env construction yet** — stock `rlgym_ppo.Learner` self-play just mirrors the live policy on both teams; that's flagged as the biggest open engineering piece in `ROADMAP.md`). Also has `FrozenPolicy`/`evaluate_match`/`build_eval_env` — a working cross-play evaluator used by `Pbt.py` for fitness (see "Population-based training" below).

`Train_Ground.py` — Model A. 4v4, `zero_padding=4`, ground-based kickoffs, `InAirReward` weighted near zero. Run with:

```
python Train_Ground.py
```

`Train_Aerial.py` — Model B. Same network/obs/action dimensions as Model A (required for the checkpoint to load), state mutator widened to sometimes spawn the ball in the air (`RandomAirborneBallMutator`, defined in this file — no built-in rlgym mutator does this), `InAirReward` weight raised, shaping re-annealed from a higher starting point. Set `AERIAL_CHECKPOINT_TO_LOAD` to Model A's actual checkpoint path before running:

```
AERIAL_CHECKPOINT_TO_LOAD=checkpoints/model_a_ground-<ts>/<timesteps> python Train_Aerial.py
```

## Research data collection

`Metrics.py` — `CoordinationMetrics`, a zero-weight reward term wired into both `Train_Ground.py` and `Train_Aerial.py` that logs one CSV row per episode: three mechanical-skill metrics (distance to ball, velocity toward ball, air time) and four coordination metrics (teammate spacing, boost spread across teammates, and two added metrics — `overcommit_rate` and `simultaneous_air_rate` — that catch multiple teammates converging on the same ball/aerial at once, which raw teammate distance alone can't distinguish from "just spread out because nobody's near the ball"). Also has `FitnessTracker` (logs episode return for PBT, see below) and `AuxiliaryDataLogger` (logs data for the Lucy-SKG auxiliary nets, see below). Writes one shard per training process to `metrics/` (see the file for why it's sharded rather than one shared CSV).

`Run_Baseline.py` — runs episodes with random actions (no trained policy at all) through the exact same env config as `Train_Ground.py`, logging to `metrics/baseline_random.*.csv`. Useful if you want baseline data alongside the trained-agent data, same CSV format either way:

```
python Run_Baseline.py --episodes 200
```

`analyze.py` doesn't exist yet (a Welch's t-test script comparing two CSVs would go here) — not part of the current workflow, you said you just need the data, not the test coded.

## Population-based training with evolution

`Pbt.py` — a population of `Train_Ground.py` instances (default 4, `POPULATION_SIZE`) trained in generations. Each member has its own learning rate / entropy coefficient and its own checkpoint lineage; after each generation the bottom performers are told to warm-start their next generation from a top performer's checkpoint (exploit) and get their hyperparameters randomly perturbed (explore) — standard PBT, the same mechanism behind Liu et al.'s paper. Run with:

```
python Pbt.py
```

Default is sequential (one population member trains at a time) with `N_PROC_PER_MEMBER=8`, not `Train_Ground.py`'s standalone default of 32 — running a whole population in parallel multiplies environment count by population size, which isn't realistic on most single machines. Tune `POPULATION_SIZE`, `N_GENERATIONS`, `GENERATION_TIMESTEPS`, and `N_PROC_PER_MEMBER` at the top of the file to your hardware.

Fitness is deliberately **not** any of `CoordinationMetrics`' columns — ranking the population on `overcommit_rate` or teammate spacing would make "coordination improved" true by construction rather than an independent result, since you'd be selecting directly on the thing you're trying to measure. `Metrics.py`'s data collection keeps running throughout PBT the same as any other training run, so you still get coordination data per member/generation — PBT just doesn't use it to decide who survives.

**Why fitness is a real cross-play tournament, not self-play return:** `Train_Ground.py`'s self-play mirrors one policy onto both blue and orange within a single member's own training run — clone vs. clone. Using `FitnessTracker`'s `episode_return` (a member's scoring rate against a mirror of itself) as fitness can't actually distinguish one population member's skill from another's — there's no rival being beaten, just a policy playing itself. `Pbt.py` instead runs a real cross-play round-robin each generation via `Self_Play.evaluate_match` — it loads two *different* members' checkpoints onto opposite teams (via `Self_Play.FrozenPolicy`, which wraps `rlgym_ppo.ppo.DiscreteFF`, confirmed against the installed version) and plays them head-to-head; fitness is win rate across those matchups. That's what real PBT for competitive multi-agent RL (Liu et al., FTW, AlphaStar league) actually ranks on. It only falls back to the weaker self-play `episode_return` proxy (loudly logged when this happens) if a member has no checkpoint yet or `evaluate_match` raises.

`Pbt.py` reads/writes `Train_Ground.py`'s learning rate, entropy coefficient, `n_proc`, checkpoint save/load paths, save interval, and metrics CSV paths via `PBT_*` environment variables (defaulting to the same values `Train_Ground.py` used before PBT existed, so running it standalone is unaffected) — see its module docstring for the full list. `Pbt.py` launches it as a subprocess per population member per generation rather than duplicating the training script.

`Self_Play.py`'s `CheckpointPool` is a different, complementary mechanism — within-episode opponent diversity (who you play against this match), still not wired into `Train_Ground.py`'s env construction. `Pbt.py` evolves across-generation population hyperparameters and doesn't depend on that being finished first.

## Lucy-SKG-style auxiliary abstraction

`Auxiliary.py` — `StateRepresentationNet` (an autoencoder: encoder 128→32→16, mirrored decoder) and `RewardPredictionNet` (LSTM over a 20-step observation window, predicts negative/near-zero/positive reward) — matching the architecture from *Lucy-SKG: Learning to Play Rocket League Efficiently Using Deep Reinforcement Learning* (Moschopoulos et al., 2023), the paper that beat both Necto and Nexto. This is observation-side abstraction — a learned compressed representation gets concatenated onto what the policy sees — and never touches `Actions.py` or the action space.

**Read this before using it — it's not identical to the paper.** Lucy-SKG trains SR/RP jointly with PPO in a single backward pass, sharing the actor's own layers. `rlgym_ppo`'s stock `Learner` doesn't expose a hook to add extra loss terms to its training step, and patching that would mean forking rlgym_ppo internals. What's built instead: `Train_Auxiliary_Encoder.py` trains SR and RP **separately** on logged data, each with its own optimizer, completely outside `rlgym_ppo`'s loop — representation pretraining rather than the paper's joint training, and the two nets don't share weights (RP's LSTM takes the raw observation, not SR's encoded output). Say so explicitly in any writeup; it's a real methodological difference, not just an implementation detail.

Workflow:

```
AUX_LOGGING=1 python Train_Ground.py          # logs (obs, reward) data to metrics/aux_data/
python Train_Auxiliary_Encoder.py             # trains SR+RP, saves checkpoints/auxiliary_encoder.pt
AUX_ENCODER_CHECKPOINT=checkpoints/auxiliary_encoder.pt python Train_Ground.py   # uses it
```

All three env vars (`AUX_LOGGING`, `AUX_ENCODER_CHECKPOINT`, and `AUX_DATA_DIR` if you want a non-default location) default off/unset — nothing about existing behavior changes unless you opt in. `Metrics.py`'s `AuxiliaryDataLogger` (same zero-weight-reward pattern as `CoordinationMetrics`/`FitnessTracker`) does the logging; `Observation.py`'s `PartialInfoObs` loads the trained encoder and concatenates its 16-dim output onto every observation, growing `get_obs_space`'s reported size to match (`rlgym_ppo` sizes its policy input layer from that, so a mismatch here fails loudly and immediately rather than subtly — confirmed by actually running it with `AUX_ENCODER_CHECKPOINT` set: declared and actual sizes both come out to 228 = 212 + 16).

The smoke test below was actually run in this environment (not just traced by hand):

```
python -c "
from Auxiliary import StateRepresentationNet, RewardPredictionNet
import torch
sr = StateRepresentationNet(obs_size=50)
rp = RewardPredictionNet(obs_size=50)
print(sr.loss(torch.randn(4, 50)))
print(rp.loss(torch.randn(4, 20, 50), torch.randn(4)))
"
```

replacing `50` with your actual observation size — easiest way to find it is loading any logged shard and checking `np.load("metrics/aux_data/shard_....npz")["obs"].shape[1]` once `AUX_LOGGING=1` has produced at least one shard.

## Watching training

`Train_Ground.py` and `Train_Aerial.py` both support RLViser (installed via `rlgym[rl-rlviser]` in `requirements.txt`). Off by default since it slows one of your `n_proc` environments to real-time so you can watch it. Turn it on with an environment variable:

```
RLGYM_RENDER=1 python Train_Ground.py
```

If no window appears, `rl-rlviser` may expect a standalone RLViser program running alongside training rather than launching one itself — check your installed version's docs if this happens (not independently confirmed here).

## Confirmed, not just traced

Everything below was checked by actually running it against this environment's installed package versions, not inferred from documentation — several turned out to be wrong and are now fixed:

- **`Learner`'s checkpoint kwargs**: `checkpoints_save_folder`, `checkpoint_load_folder`, `render`, `render_delay` are all correctly spelled in `rlgym_ppo` 1.3.13. One real gotcha found: `checkpoint_load_folder`'s own default is `"latest"` (auto-resume-scan), not `None` — `Train_Ground.py` now always passes it explicitly so "no `PBT_CHECKPOINT_LOAD_DIR` set" reliably means a true fresh start.
- **`LookupTableAction` index 0 is NOT neutral** — it's `[-1, -1, 0, -1, 0, 0, 0, 0]` (full reverse + full left steer). The real neutral/all-zero row is index 8. `Actions.py` now finds it programmatically instead of hardcoding an index.
- **`Actions.py`'s `reset()` was missing the `agents` parameter** that `rlgym.api.ActionParser.reset` (and `RLGym.reset()`, which calls it with 3 positional args) actually requires — this crashed every `env.reset()` call. Fixed.
- **The reaction-delay queue's idle filler was a bare Python `int`**, but real actions arrive as `(1,)` numpy arrays — this crashed `LookupTableAction.parse_actions` (`'int' object has no attribute 'shape'`) during the first few ticks of every episode. Fixed.
- **`PhysicsObject.forward` is a `@property`, not a method** — `Observation.py` was calling it as `forward()`, which raised `TypeError: 'numpy.ndarray' object is not callable`. Fixed to `forward` (no parens).
- **`zero_padding=3` was wrong for this project's 4v4.** `DefaultObs`'s `zero_padding` means "max cars per team," not "other agents excluding self" — for a real 4v4 it needs to be 4 (giving 3 padded ally slots + 4 padded enemy slots). With 3, the real roster already exceeds the padding minimums, so no padding ever triggers and the built observation comes out 40 elements larger than what `get_obs_space` declares — which crashed policy network construction with a matmul shape mismatch as soon as real training started. Fixed everywhere it appears (`Train_Ground.py`, `Train_Aerial.py`, `Self_Play.py`'s `build_eval_env`).
- **`RLGymV2GymWrapper` does NOT use the agent-ID-dict Gymnasium multi-agent convention.** `reset()` returns a plain `(n_agents, obs_dim)` ndarray, `action_space` is a single shared attribute (not per-agent-callable), and `step()` takes a positional `(n_agents, ...)` array and returns an aggregated bool/list, not per-agent dicts. `Run_Baseline.py` was written assuming dict access throughout and has been rewritten to match the real interface. `Self_Play.py`'s `build_eval_env` returns the **raw** `rlgym.api.RLGym` env instead (which does use AgentID-keyed dicts) since `evaluate_match` genuinely needs per-agent routing that the wrapper can't provide.
- **`build_obs` runs before `get_rewards`** in every `rlgym.api.RLGym.step()` call (confirmed against its source) — `AuxiliaryDataLogger`'s dependence on `shared_info["aux_obs"]` being populated first is safe.
- **`Learner._learn()`'s loop is a plain `while cumulative_timesteps < timestep_limit`** — it returns cleanly when the limit is hit, no external shutdown signal needed, so `Pbt.py`'s blocking `subprocess.run()` per generation works. One gotcha this surfaced: `Learner` only saves a checkpoint *periodically* (`ts_since_last_save >= save_every_ts`), never automatically on exit — a PBT generation shorter than `save_every_ts` would save zero checkpoints. `Train_Ground.py` now exposes `PBT_SAVE_EVERY_TS` so `Pbt.py` can keep it well under `GENERATION_TIMESTEPS`.
- **The real self-play policy class is `rlgym_ppo.ppo.DiscreteFF`**, constructed as `DiscreteFF(input_shape, n_actions, layer_sizes, device)` — confirmed against how `rlgym_ppo.ppo.ppo_learner.PPOLearner` builds it internally, and checkpoint file naming (`PPO_POLICY.pt` etc.) confirmed against its `save_to`/`load_from`. `Self_Play.py`'s `FrozenPolicy` is implemented against this.
- **`_infer_teams()`** now reads team membership directly off `env.state.cars[agent].is_orange` on the raw `RLGym` env (a confirmed public property) rather than guessing at wrapper internals — no longer "best-effort."
- **`Pbt.find_latest_checkpoint()`** mirrors `Learner.load()`'s own `"latest"`-resolution algorithm (read from its source) and was tested against real checkpoint directories produced by an actual training run — it resolves to a literal path `checkpoint_load_folder` accepts.
- **Module import casing**: `Train_Ground.py` was importing `from metrics import ...` (lowercase) when the actual file is `Metrics.py`. This "worked" in the sense of not raising `ModuleNotFoundError` only because a `metrics/` *directory* (the CSV output folder) already existed and got picked up as an empty namespace package — silently shadowing the real module and failing with a confusing `ImportError: cannot import name ... (unknown location)`. Fixed to the correct case.

## Next steps

Get Model A training and watch for coherent 4v4 ground play (coordinated positioning, not just four cars chasing the ball) — a single `Train_Ground.py` run, not `Pbt.py`, is the right first step so you're not debugging PBT and basic training at the same time. Once that's stable, move to `Train_Aerial.py` and/or `Pbt.py`. Population self-play (`Self_Play.py`'s `CheckpointPool` wired into `Train_Ground.py`'s own env construction, for within-episode opponent diversity) is the one piece still intentionally left unfinished — see `ROADMAP.md`'s "known gaps" section for why.
