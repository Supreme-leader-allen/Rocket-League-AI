"""
Watch trained bots play in RLViser, with no Rocket League needed.

    python Watch.py --policy <path to PPO_POLICY.pt>
    python Watch.py --policy <team spirit policy> --opponent <no team spirit policy>
    python Watch.py --policy <path> --speed 2

Without --opponent, the policy plays itself, as in training. Same view cone,
reaction delay and decision rate as training. Needs rlviser.exe in this folder
(rlviser_py launches ./rlviser.exe); close its window or press Ctrl+C to stop.
"""

import argparse
import os
import time

import numpy as np

SECONDS_PER_DECISION = 8 / 120


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, help="path to PPO_POLICY.pt (blue)")
    parser.add_argument("--opponent", help="path to PPO_POLICY.pt for orange; defaults to --policy")
    parser.add_argument("--episodes", type=int, default=10, help="kickoff-to-goal episodes to play")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time")
    args = parser.parse_args()

    from rlgym.rocket_league.rlviser import RLViserRenderer
    from Self_Play import FrozenPolicy, build_eval_env, _infer_teams, _resolve_device
    from Train_Ground import POLICY_LAYER_SIZES

    if not os.path.isfile("rlviser.exe"):
        raise SystemExit("rlviser.exe isn't in this folder. Get rlviser-x86_64-pc-windows-msvc.exe from "
                         "https://github.com/VirxEC/rlviser/releases, put it here, and rename it rlviser.exe.")

    env = build_eval_env(renderer=RLViserRenderer(tick_rate=120 / 8))
    obs = env.reset()
    obs_size = len(next(iter(obs.values())))
    action_size = list(env.action_spaces.values())[0][1]
    device = _resolve_device()
    blue = FrozenPolicy(os.path.dirname(args.policy) or ".", POLICY_LAYER_SIZES, obs_size, action_size, device)
    orange = blue
    if args.opponent:
        orange = FrozenPolicy(os.path.dirname(args.opponent) or ".", POLICY_LAYER_SIZES, obs_size, action_size, device)

    score = {"blue": 0, "orange": 0}
    frame_time = SECONDS_PER_DECISION / args.speed
    try:
        for episode in range(args.episodes):
            if episode > 0:
                obs = env.reset()
            teams = _infer_teams(env, obs)
            sides = [([a for a in obs if teams[a] == side], policy)
                     for side, policy in (("blue", blue), ("orange", orange))]

            done = False
            while not done:
                started = time.perf_counter()
                actions = {}
                for agents, policy in sides:
                    chosen = policy.act_batch(np.stack([obs[a] for a in agents]))
                    actions.update({a: np.array([c], dtype=np.int32) for a, c in zip(agents, chosen)})
                obs, reward, terminated, truncated = env.step(actions)
                env.render()
                done = any(terminated.values()) or any(truncated.values())
                time.sleep(max(0.0, frame_time - (time.perf_counter() - started)))

            if any(terminated.values()):
                side = teams[max(reward, key=reward.get)]
                score[side] += 1
            print(f"episode {episode + 1}: blue {score['blue']} - orange {score['orange']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
