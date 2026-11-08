"""
Play a match yourself against the trained bots.

    .venv\\Scripts\\python.exe play_vs_bot.py --policy <path to PPO_POLICY.pt>

Default is a full 4v4 the bots were trained for: you plus 3 Psyonix teammates
against 4 trained bots. --teammates sets your teammates' Psyonix skill, or
"none" for you alone against them. --opponents sets how many trained bots
(1-4); fewer than 4 is a situation they never trained in, so expect odder play.

Use a controller or keyboard as in normal Rocket League. The game is in RLBot's
offline mode. The bots' coordination metrics still go to results/play_<...>.csv.
"""

import argparse
import json
import time
from pathlib import Path

from rlbot import flat
from rlbot.config import get_human, load_player_config, load_psyonix_config
from rlbot.managers import MatchManager

from run_matches import HERE, LAUNCHERS, SKILLS, stop_leftover_servers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, help="path to PPO_POLICY.pt")
    parser.add_argument("--teammates", default="pro", choices=list(SKILLS) + ["none"])
    parser.add_argument("--opponents", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--launcher", default="epic", choices=list(LAUNCHERS))
    args = parser.parse_args()

    policy = Path(args.policy).resolve()
    if not policy.is_file():
        raise SystemExit(f"No policy file at {policy}")

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    label = f"play_{args.teammates}_vs_{args.opponents}"
    with open(HERE / "active_run.json", "w", encoding="utf-8") as f:
        json.dump({"policy": str(policy), "run_label": label,
                   "metrics_csv": str(results_dir / f"{label}.csv")}, f)

    players = [get_human(0)]
    if args.teammates != "none":
        players += [load_psyonix_config(0, SKILLS[args.teammates]) for _ in range(3)]
    players += [load_player_config(HERE / "bot.toml", 1) for _ in range(args.opponents)]
    config = flat.MatchConfiguration(
        launcher=LAUNCHERS[args.launcher],
        auto_start_agents=True,
        wait_for_agents=True,
        game_map_upk="Stadium_P",
        player_configurations=players,
        game_mode=flat.GameMode.Soccar,
        skip_replays=True,
        mutators=flat.MutatorSettings(match_length=flat.MatchLengthMutator.FiveMinutes),
        existing_match_behavior=flat.ExistingMatchBehavior.Restart,
    )

    stop_leftover_servers()
    with MatchManager() as manager:
        try:
            manager.start_match(config)
            print("Match started -- you're blue. Good luck.")
            last = None
            while True:
                packet = manager.packet
                if packet is not None and packet.teams:
                    last = packet
                    if packet.match_info.match_phase == flat.MatchPhase.Ended:
                        break
                time.sleep(0.5)
            scores = {team.team_index: team.score for team in last.teams}
            print(f"Final: you (blue) {scores[0]} - trained bots (orange) {scores[1]}")
            time.sleep(5)
        finally:
            manager.shut_down()


if __name__ == "__main__":
    main()
