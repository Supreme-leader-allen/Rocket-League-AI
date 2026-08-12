# Rocket League AI — Roadmap (v2)

This supersedes the first draft. It's rebuilt around the RLGym v2 API (`rlgym.api` + `RocketSimEngine` + `rlgym_ppo`, matching your tutorial code) and around five hard requirements you added: 4v4 coordination, a network that never needs architectural changes between phases, no action abstraction, humanlike constraints (reaction time, input rate, imperfect information), and an autocurriculum that emerges from self-play/competition rather than manual difficulty tuning, following Liu et al.'s *Emergent Coordination through Competition* (2019).

The old plan had an "abstraction" phase and a hand-tuned difficulty ramp. Both are dropped here — abstraction because you want the agent choosing raw actions, and the manual ramp because you want the curriculum to come from competition itself.

## Toolchain

`rlgym` (v2 API: `rlgym.api`, `rlgym.rocket_league`) with RocketSim (`RocketSimEngine`) for fast simulated physics, and `rlgym_ppo` for the PPO learner and vectorized rollout collection. This is exactly the stack your tutorial code uses, and it's the current standard — the older `rlgym-sim`/`rlgym_sim` packages I scaffolded in the first draft are a different, now-secondary API and shouldn't be mixed with v2 code. The `train_phase1.py`/`src/*.py` files from the first draft are left in place but deprecated; the new files below (`train_ground.py`, `train_aerial.py`, `Rewards.py`, `obs.py`, `actions.py`, `self_play.py`) are the ones to use.

## The one constraint that shapes everything else: a fixed network

You want to train a ground bot first, then grow it into an aerial bot without starting over. That's only possible if the policy and critic networks never change shape. A PyTorch network's weight matrices are sized by its input and output dimensions; if those dimensions differ between two checkpoints, the saved weights literally cannot be loaded into the new network (the matrix shapes won't multiply). So "no retraining from scratch" isn't really a training-technique question, it's an interface-freezing question: freeze the observation vector size, the action space, and the layer sizes on day one, and every later phase becomes a warm-start (`checkpoint_load_folder`) rather than a rebuild.

Concretely, three things are pinned from the very first run and never change across Model A (ground) and Model B (aerial):

**Observation size.** `DefaultObs` takes a `zero_padding` argument that pads the "other cars" section of the observation to a fixed count regardless of how many cars are actually on the field that episode. Its default is 3, which is exactly what a 4v4 needs (3 teammates + 3 opponents padded per side). Use `zero_padding=3` from the first run, even though early training only ever spawns full 4v4 games — the point is that the obs vector's length is fixed by the padding value, not by how many cars happen to be alive, so nothing about the vector shape changes later.

**Action space.** `LookupTableAction` (or a continuous parser, if you switch) defines a fixed set of outputs from the first run. Since you don't want abstraction, this stays the *full* action table throughout — the agent can pick any raw control combination (including jump/pitch/yaw/roll) starting in Phase 1, not a restricted "ground moves only" subset. See the next section for how ground-first training happens without touching the action space.

**Layer sizes.** `policy_layer_sizes` and `critic_layer_sizes` passed to `Learner` must be identical between the ground run and the aerial run. The tutorial's `[2048, 2048, 1024, 1024]` is already large enough to carry ground and aerial behavior in the same weights — keep it as-is rather than shrinking it for "just" ground play. A network can be given more capacity than it currently needs; it cannot be given more capacity later without invalidating the checkpoint.

## No abstraction, but still a ground-first curriculum

Here's the resolution to something that looks like a contradiction: you want raw action control from the start, but you also want to train ground behavior before aerial behavior. The trick is that "ground-first" doesn't have to mean restricting what the agent is *allowed* to do — it can mean shaping what situations it's *born into* and what get rewarded, while every action stays available every tick.

Practically: Model A uses `KickoffMutator` (or a similar ground-based state mutator) so episodes start on the ground, and its reward function weights aerial behavior near zero (`InAirReward` weight ~0 or omitted). The agent is free to jump, flip, and fly at any point — it just won't be rewarded for doing so yet, and starting states won't put it in the air. Model B, initialized from Model A's weights, widens the state mutator to include airborne ball/car spawns and raises the aerial-related reward weights. The action space and network never change; only the state distribution and reward weighting shift between the two runs.

## Credit assignment and autocurriculum: Liu et al., 2019

*Emergent Coordination through Competition* (Liu, Lever, Merel, Tunyasuvunakool, Heess, Graepel — DeepMind, ICLR 2019) trained 2v2 simulated soccer agents and found that coordinated teamwork emerged without any explicit multi-agent credit-assignment mechanism (no difference rewards, no counterfactual baselines). Two design choices did the work instead, and both map directly onto your 4v4 setup:

**Reward is shared at the team level, individual only in its dense-shaping component.** Sparse outcome reward (goals, wins) is identical for every teammate — nobody gets more or less credit for a goal than their teammates do, which is exactly how `GoalReward` already behaves in rlgym (it's keyed on team, not on which car touched the ball). Dense shaping reward (`SpeedTowardBallReward`, `VelocityBallToGoalReward`) stays individual, since it's meant to teach personal mechanics, not coordination. You don't need a custom credit-assignment algorithm; you need to make sure you're not accidentally making the sparse term individual (e.g. "reward only the car that scored") since that's what the paper's team-shared signal specifically avoids.

**Dense shaping is annealed toward zero over the course of training.** Liu et al. use shaping rewards heavily early on (when sparse win/loss signal is too rare to learn from) and anneal their weight down as training progresses, so that by the end the agent is optimizing almost purely for the sparse team outcome. This is what actually produces coordination: shaping rewards are inherently selfish/individual (chase the ball, hit it forward), and if they never decay, the agent has no incentive to ever pass, rotate, or make space for a teammate. `Rewards.py` below includes an `AnnealedCombinedReward` wrapper that does this.

**The autocurriculum is the self-play population itself, not a hand-built difficulty ramp.** Instead of you writing code that ramps up opponent difficulty, you maintain a pool of past checkpoints and sample opponents from it (mixing recent and older versions). As Model A's policy improves, its own checkpoint pool gets harder automatically — this is the same mechanism used by Liu et al., OpenAI Five, and every top open-source Rocket League bot (Nexto). `self_play.py` sketches this. Be aware this is the single biggest engineering gap between "what rlgym-ppo gives you out of the box" and "what the paper does": stock `rlgym_ppo.Learner` with `spawn_opponents=True` mirrors the *current* live policy onto both teams (basic self-play, not a diverse checkpoint pool). Getting genuine population self-play working means intercepting the opponent-side action selection inside your environment-construction function and routing it through a frozen, separately-loaded copy of an older checkpoint — `self_play.py` scaffolds the pieces but this needs real testing against your installed rlgym-ppo version before you trust it.

## Humanlike constraints

Three separate things were bundled under "humanlike," and they need three separate mechanisms:

**Reaction time.** Real players don't act on what they see instantly — there's roughly 150-250ms between perceiving something and reacting to it. `actions.py`'s `HumanlikeAction` implements this as a per-agent delay queue: the action chosen at decision step *t* isn't applied to the sim until step *t + delay_steps*. Default is tuned to roughly 150-200ms given your tick skip.

**Input rate.** This is really about decision frequency, which you already control via tick skip / action repeat — at `tick_skip=8` (120Hz physics / 8 = 15 decisions/sec), you're already in a plausible human-ish ballpark for meaningful input changes, well below the 120Hz the raw simulation runs at. `HumanlikeAction` folds this repeat logic in alongside the delay queue so both constraints live in one place.

**Imperfect information.** `obs.py`'s `PartialInfoObs` restricts each car's view of *other* cars to a forward-facing cone (a rough analogue of a player's screen/attention), and for cars outside that cone, holds their last-seen position/velocity instead of updating it every tick, with a staleness cutoff after which they're zeroed out entirely (fully forgotten). This is also doing double duty as a cheap substitute for a recurrent (LSTM) memory core, which is what Liu et al. actually used for partial observability — `rlgym_ppo`'s stock `Learner` builds plain feedforward networks, not recurrent ones, so giving the agent "memory" via the observation itself (last-known values persisting across ticks) is the practical option here without forking the PPO implementation. A true recurrent core would be strictly better and is a reasonable later upgrade, but it's a bigger lift (modifying rlgym-ppo's network/rollout code) than this project needs to start with.

Ball position/velocity stay fully observed — Liu et al.'s partial-observability setup and most competitive Rocket League bots still give full information about the ball, since occluding it would make the task closer to blind guessing than imperfect teamwork, which isn't the research question here.

## Two-model plan

**Model A — ground, 4v4.** `train_ground.py`. `FixedTeamSizeMutator(blue_size=4, orange_size=4)`, ground-based `KickoffMutator`, near-zero `InAirReward` weight, self-play checkpoint pool, full annealed shaping. Train this until it plays coherent, coordinated 4v4 on the ground — this is your "for show" bot.

**Model B — aerial, built on Model A.** `train_aerial.py`. Same `policy_layer_sizes`/`critic_layer_sizes`, same `zero_padding=3`, same `LookupTableAction` — loaded via `checkpoint_load_folder` pointed at Model A's saved checkpoint. State mutator widens to include airborne ball spawns and aerial-relevant kickoff variants; `InAirReward` weight raised; shaping re-annealed from a higher starting point since the aerial behaviors are new and need dense guidance again even though the ground behaviors are already learned. Self-play pool can either continue from Model A's pool or start fresh once aerial play is common enough to matter competitively.

Because the network interface never changed, Model B starts already knowing how to drive, rotate, and challenge on the ground — it only has to learn the incremental skill of leaving the ground, not relearn 4v4 positioning from zero.

## Research data: measuring coordination, not just skill

Your research plan lists five raw signals (velocity to ball, distance to ball, teammate distance, boost, air time) and a t-test comparing trained agents to a baseline. Three of those five (distance/velocity to ball, air time) are mechanical-skill signals — they tell you how well a car controls itself, not whether the team is coordinating. Teammate distance is the one genuine coordination signal in the original list, and on its own it's ambiguous: cars can be far apart because they're well-positioned across the field, or far apart because nobody's anywhere near the play. `metrics.py` keeps all five but adds two derived metrics that resolve that ambiguity directly — `overcommit_rate` (fraction of steps where 2+ teammates are simultaneously within challenge range of the ball) and `simultaneous_air_rate` (same idea for aerials). Both measure the literal opposite of role-based rotation (first man / second man / third man), which is the concrete, measurable form "coordination" takes in this game. If you're picking one or two numbers to lead with in a writeup, these two are the strongest coordination evidence in the set.

Data collection is wired into both training scripts as a zero-weight reward term (no effect on training, pure logging), and `run_baseline.py` gives you a genuine baseline — random actions, no learned policy — logged in the same CSV format, if you want baseline data alongside the trained-agent data. See `README.md`'s "Research data collection" section for exact commands.

## Known gaps to budget time for

Being direct about what's genuinely hard here, so you don't lose a week thinking something's broken when it's actually just unbuilt:

Population self-play (a real checkpoint pool with frozen opponents, not just live self-mirroring) requires custom code inside your environment-construction function, and the exact way to load a frozen policy snapshot depends on your installed `rlgym_ppo` version's internal policy class — `self_play.py` flags exactly where you'll need to check this against your version.

`PartialInfoObs`'s occlusion logic mutates car physics data before handing it to the underlying obs builder rather than slicing into `DefaultObs`'s output vector directly, specifically so it doesn't depend on guessing that vector's internal layout. It only masks position and linear velocity for now (the two fields directly confirmed from your reward code); extending it to rotation and boost amount is flagged in the file as a follow-up once you've confirmed those field names against your installed `rlgym.rocket_league.api`.

Population-based training with evolution (swapping weights/hyperparameters between population members) isn't scaffolded yet — it's a layer on top of a working self-play pool, not something to start before self-play itself is solid. Once `self_play.py` is working and Model A is training coherently, that's the next thing to build, not before.