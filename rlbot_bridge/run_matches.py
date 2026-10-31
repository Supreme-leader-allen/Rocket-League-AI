"""
Plays a trained policy against Psyonix bots, 4v4, in real Rocket League.

    .venv\\Scripts\\python.exe run_matches.py --policy <path to PPO_POLICY.pt> --label run6_ts

Plays --matches 5-minute matches at each skill, swapping sides every match so
neither side of the field is favored, and appends one row per match to
results/<label>_matches.csv as soon as it ends, so stopping early loses nothing.
Coordination metrics go to results/<label>_<skill>.<pid>.csv.

Needs RLBot v5 installed from rlbot.org and Rocket League closed beforehand.
"""

import argparse
import csv
import json
import time
from pathlib import Path

from rlbot import flat
from rlbot.config import load_player_config, load_psyonix_config
from rlbot.managers import MatchManager

HERE = Path(__file__).resolve().parent
RESULT_COLUMNS = ["label", "skill", "match", "our_team", "our_score", "opponent_score", "result"]
SKILLS = {"rookie": flat.PsyonixSkill.Rookie, "pro": flat.PsyonixSkill.Pro, "allstar": flat.PsyonixSkill.AllStar}
LAUNCHERS = {"epic": flat.Launcher.Epic, "steam": flat.Launcher.Steam}


def match_config(skill: flat.PsyonixSkill, our_team: int, launcher: flat.Launcher) -> flat.MatchConfiguration:
    players = [load_player_config(HERE / "bot.toml", our_team) for _ in range(4)]
    players += [load_psyonix_config(1 - our_team, skill) for _ in range(4)]
    return flat.MatchConfiguration(
        launcher=launcher,
        auto_start_agents=True,
        wait_for_agents=True,
        game_map_upk="Stadium_P",
        player_configurations=players,
        game_mode=flat.GameMode.Soccar,
        skip_replays=True,
        instant_start=False,
        mutators=flat.MutatorSettings(match_length=flat.MatchLengthMutator.FiveMinutes),
        existing_match_behavior=flat.ExistingMatchBehavior.Restart,
        enable_rendering=flat.DebugRendering.OffByDefault,
        enable_state_setting=False,
    )


def play_match(manager: MatchManager, config: flat.MatchConfiguration, our_team: int) -> tuple[int, int]:
    manager.start_match(config)
    last = None
    while True:
        packet = manager.packet
        if packet is not None and packet.teams:
            last = packet
            if packet.match_info.match_phase == flat.MatchPhase.Ended:
                break
        time.sleep(0.5)
    scores = {team.team_index: team.score for team in last.teams}
    manager.stop_match()
    time.sleep(5)  # let the bot process flush its metrics before the next match
    return scores[our_team], scores[1 - our_team]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, help="path to PPO_POLICY.pt")
    parser.add_argument("--label", required=True, help="name for this bot in the results, e.g. run6_ts")
    parser.add_argument("--skills", nargs="+", default=list(SKILLS), choices=list(SKILLS))
    parser.add_argument("--matches", type=int, default=10, help="matches per skill level")
    parser.add_argument("--launcher", default="epic", choices=list(LAUNCHERS))
    args = parser.parse_args()

    policy = Path(args.policy).resolve()
    if not policy.is_file():
        raise SystemExit(f"No policy file at {policy}")

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"{args.label}_matches.csv"
    if not results_path.exists():
        with open(results_path, "w", newline="") as f:
            csv.writer(f).writerow(RESULT_COLUMNS)

    tally = {skill: [0, 0, 0] for skill in args.skills}
    with MatchManager() as manager:
        for skill in args.skills:
            for match in range(args.matches):
                our_team = match % 2
                with open(HERE / "active_run.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "policy": str(policy),
                        "run_label": f"{args.label}_{skill}",
                        "metrics_csv": str(results_dir / f"{args.label}_{skill}.csv"),
                    }, f)

                print(f"{skill} match {match + 1}/{args.matches} (team {'orange' if our_team else 'blue'})")
                ours, theirs = play_match(manager, match_config(SKILLS[skill], our_team, LAUNCHERS[args.launcher]), our_team)
                result = "win" if ours > theirs else "loss" if ours < theirs else "draw"
                tally[skill][("win", "loss", "draw").index(result)] += 1
                print(f"  {ours}-{theirs} {result}")
                with open(results_path, "a", newline="") as f:
                    csv.writer(f).writerow([args.label, skill, match + 1, our_team, ours, theirs, result])

        manager.shut_down()

    for skill, (wins, losses, draws) in tally.items():
        print(f"{skill}: {wins} wins, {losses} losses, {draws} draws")


if __name__ == "__main__":
    main()
