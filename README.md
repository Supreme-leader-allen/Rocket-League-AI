# Rocket League AI — Starter Project

Two models: Model A (ground, 4v4) trains first, Model B (aerial) warm-starts from Model A's checkpoint. See `ROADMAP.md` for the full design rationale (fixed network across phases, no action abstraction, humanlike constraints, self-play-driven autocurriculum per Liu et al. 2019).

**The files below (`train_ground.py`, `train_aerial.py`, `Rewards.py`, `obs.py`, `actions.py`, `self_play.py`) are the current, correct ones — they use the RLGym v2 API (`rlgym.api` + `RocketSimEngine`), matching your tutorial code.**

`train_phase1.py` and everything under `src/` are from an earlier draft built on the older `rlgym-sim` package before you shared your tutorial code. That's a different, now-secondary API — don't mix it with the v2 files. Left in place for reference only; safe to ignore or delete locally.

## Setup

```
pip install -r requirements.txt
```

## Files

`Rewards.py` — `SpeedTowardBallReward`, `InAirReward`, `VelocityBallToGoalReward` (dense, individual shaping) plus `AnnealedCombinedReward`, which decays shaping weight toward zero over wall-clock training time so the agent ends up optimizing mostly for the sparse, team-shared `GoalReward`. That anneal is the credit-assignment/coordination mechanism, per Liu et al.'s *Emergent Coordination through Competition* — see `ROADMAP.md`.

`obs.py` — `PartialInfoObs`. Wraps `DefaultObs` but occludes other cars outside each agent's forward view cone, holding last-known position/velocity (with a staleness cutoff) instead of updating every tick. This is the "no perfect information" piece, and it doubles as a cheap memory substitute since `rlgym_ppo`'s stock networks aren't recurrent.

`actions.py` — `HumanlikeAction`. Full raw `LookupTableAction` action space (no abstraction — the agent always picks its own action), with a reaction-delay queue and input-repeat baked in for the "humanlike reaction time / inputs per second" requirement.

`self_play.py` — `CheckpointPool` for archiving and sampling past checkpoints as opponents. **Not fully wired in yet** — stock `rlgym_ppo.Learner` self-play just mirrors the live policy on both teams. Getting genuine population self-play (frozen historical opponents, the actual autocurriculum mechanism) means finishing `FrozenPolicy` in this file against your installed `rlgym_ppo` version and hooking it into `build_rlgym_v2_env`'s orange-team action selection. This is flagged as the biggest open engineering piece in `ROADMAP.md`.

`train_ground.py` — Model A. 4v4, `zero_padding=3`, ground-based kickoffs, `InAirReward` weighted near zero. Run with:

```
python train_ground.py
```

`train_aerial.py` — Model B. Same network/obs/action dimensions as Model A (required for the checkpoint to load), state mutator widened to sometimes spawn the ball in the air, `InAirReward` weight raised. Fill in `CHECKPOINT_TO_LOAD` with Model A's actual checkpoint path before running:

```
python train_aerial.py
```

## Research data collection

`metrics.py` — `CoordinationMetrics`, a zero-weight reward term wired into both `train_ground.py` and `train_aerial.py` that logs one CSV row per episode: three mechanical-skill metrics (distance to ball, velocity toward ball, air time) and four coordination metrics (teammate spacing, boost spread across teammates, and two added metrics — `overcommit_rate` and `simultaneous_air_rate` — that catch multiple teammates converging on the same ball/aerial at once, which raw teammate distance alone can't distinguish from "just spread out because nobody's near the ball"). Writes one shard per training process to `metrics/` (see the file for why it's sharded rather than one shared CSV).

`run_baseline.py` — runs episodes with random actions (no trained policy at all) through the exact same env config as `train_ground.py`, logging to `metrics/baseline_random.*.csv`. Useful if you want baseline data alongside the trained-agent data, same CSV format either way:

```
python run_baseline.py --episodes 200
```

`analyze.py` exists (a Welch's t-test script comparing two CSVs) but isn't part of the current workflow — you said you just need the data, not the test coded. Left in place in case it's useful later; nothing here depends on it.

## Before a long run

A few things in the new files are flagged inline as needing verification against your installed package versions rather than asserted as fact — worth checking before you sink hours of compute into it:

- `Learner`'s exact checkpoint save/load parameter names (`checkpoints_save_folder` / `checkpoint_load_folder` as used here — confirm via `help(Learner.__init__)`).
- That index `0` in `LookupTableAction`'s table is a neutral/no-input action (used in `actions.py` as the reaction-delay queue's filler).
- The exact policy class path for `self_play.py`'s `FrozenPolicy`, if you build out the checkpoint-pool self-play.
- `RLGymV2GymWrapper`'s exact `reset()`/`step()` return signature, used directly (not through `Learner`) in `run_baseline.py` — flagged inline there too.

## Next steps

Get Model A training and watch for coherent 4v4 ground play (coordinated positioning, not just four cars chasing the ball). Once that's stable, move to `train_aerial.py`. Population self-play (`self_play.py`) and population-based training with evolution are the two pieces intentionally left unfinished — see `ROADMAP.md`'s "known gaps" section for why, and do them in that order.