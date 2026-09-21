"""
Plot_Results.py: reads the per-process CSV shards CoordinationMetrics /
FitnessTracker write during a run (see metrics.py) and generates comparison
plots across run_labels/conditions -- the "generate as many plots as
possible" step of experiments/run_all.sh's workflow, which had no plotting
step before this file existed (docs/ISSUES.md notes analyze.py -- the t-test
-- was left uncoded on purpose; this file is visualization only, no
statistical test, same deliberate scope).

Three kinds of plot, per coordination/mechanical metric in Metrics.py's
CSV_COLUMNS: a bar chart (mean +/- SEM by condition, final/aggregate
state), a histogram (trained vs. baseline distribution), and a trend
line (metric vs. episode_id, rolling-averaged, one line per condition --
this is the one that shows coordination *emerging* over a run rather
than only where it ends up; see plot_metric_trend's docstring for the
episode_id-as-ordinal-proxy caveat). Plus a fitness (episode_return)
boxplot and, when a PBT pass has run, a fitness-per-generation line
chart.

Reads:
    <results-dir>/metrics/*.csv            (CoordinationMetrics shards,
                                             CSV_COLUMNS from metrics.py)
    <results-dir>/metrics/*_fitness.*.csv  (FitnessTracker shards,
                                             FITNESS_CSV_COLUMNS from metrics.py)
    <results-dir>/metrics/pbt/generations.jsonl   (optional -- only present
                                             if a PBT pass has run, either
                                             `python Pbt.py` standalone or
                                             `INCLUDE_PBT=1` through
                                             experiments/run_all.sh, which
                                             now wires in 06_pbt.sh --
                                             docs/ISSUES.md)

Writes:
    <results-dir>/plots/*.png

Run after experiments/run_all.sh finishes:
    python Plot_Results.py --results-dir output
    python Plot_Results.py --results-dir output --baseline-label baseline_random --highlight-label 01_ground
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- safe on a remote box with no display
import matplotlib.pyplot as plt
import pandas as pd

COORDINATION_METRICS = [
    "avg_teammate_pairwise_dist",
    "boost_stddev_teammates",
    "overcommit_rate",
    "simultaneous_air_rate",
]
MECHANICAL_METRICS = [
    "avg_dist_to_ball",
    "avg_vel_toward_ball",
    "air_time_fraction",
]
ALL_METRICS = COORDINATION_METRICS + MECHANICAL_METRICS


def _load_coordination_csvs(metrics_dir: Path) -> pd.DataFrame:
    # Coordination shards are named "<label>.<pid>.csv" (fitness shards use
    # "<label>_fitness.<pid>.csv" -- excluded here by requiring the file NOT
    # end in "_fitness.<digits>.csv", since both live in the same directory
    # and glob("*.csv") would otherwise mix schemas).
    frames = []
    for path in sorted(metrics_dir.glob("*.csv")):
        if "_fitness." in path.name or path.name.startswith("pbt"):
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"skip {path.name}: {e}")
            continue
        if "run_label" not in df.columns:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["run_label"] + ALL_METRICS)
    df = pd.concat(frames, ignore_index=True)
    # Metrics.py's CoordinationMetrics._flush() writes a literal "" for
    # any metric that had zero samples in a short episode. Checked
    # whether this breaks aggregation/plotting by directly constructing
    # the worst cases pandas could plausibly mishandle -- a single-row
    # shard whose only value for a column is "", and concatenating an
    # all-empty shard with a real-valued one -- and it doesn't: pandas'
    # read_csv treats a bare empty CSV field as NaN and infers float64
    # regardless of how many real values surround it, in every case
    # tried, so .mean()/.sem()/ax.hist() below all already work
    # correctly on this data without any extra handling. The explicit
    # pd.to_numeric coercion here is kept anyway as cheap, harmless
    # insurance against a genuinely non-numeric value some other way
    # (not something confirmed to happen, just cheap enough not to skip).
    for metric in ALL_METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    if "episode_id" in df.columns:
        df["episode_id"] = pd.to_numeric(df["episode_id"], errors="coerce")
    return df


def _load_fitness_csvs(metrics_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(metrics_dir.glob("*_fitness.*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"skip {path.name}: {e}")
            continue
        if "run_label" not in df.columns:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["run_label", "episode_return"])
    df = pd.concat(frames, ignore_index=True)
    # FitnessTracker doesn't write "" (episode_return always has a value,
    # even if 0.0 -- see Metrics.py), but coerce anyway for the same
    # reason as _load_coordination_csvs: cheap, and defends against any
    # future change there without this file needing to know about it.
    if "episode_return" in df.columns:
        df["episode_return"] = pd.to_numeric(df["episode_return"], errors="coerce")
    return df


def plot_bar_comparison(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    if metric not in df.columns:
        return
    grouped = df.groupby("run_label")[metric].agg(["mean", "sem", "count"]).dropna(subset=["mean"])
    if grouped.empty:
        return
    grouped = grouped.sort_values("mean")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(grouped.index, grouped["mean"], yerr=grouped["sem"].fillna(0), capsize=4)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} — mean ± SEM by condition (n episodes shown per bar)")
    for i, (label, row) in enumerate(grouped.iterrows()):
        ax.text(i, row["mean"], f"n={int(row['count'])}", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / f"bar_{metric}.png", dpi=150)
    plt.close(fig)


def plot_histogram_vs_baseline(df: pd.DataFrame, metric: str, baseline_label: str,
                                highlight_label: str, out_dir: Path) -> None:
    if metric not in df.columns:
        return
    baseline = df[df["run_label"] == baseline_label][metric].dropna()
    trained = df[df["run_label"] == highlight_label][metric].dropna()
    if baseline.empty and trained.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if not baseline.empty:
        ax.hist(baseline, bins=30, alpha=0.5, label=baseline_label, density=True)
    if not trained.empty:
        ax.hist(trained, bins=30, alpha=0.5, label=highlight_label, density=True)
    ax.set_xlabel(metric)
    ax.set_ylabel("density")
    ax.set_title(f"{metric}: {highlight_label} vs {baseline_label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"hist_{metric}_{highlight_label}_vs_{baseline_label}.png", dpi=150)
    plt.close(fig)


def plot_metric_trend(df: pd.DataFrame, metric: str, out_dir: Path, window: int = 20) -> None:
    """
    metric vs. episode_id, one line per condition, rolling-mean smoothed
    (raw per-episode values are noisy). This is what shows a coordination
    metric actually *emerging* over the course of training, rather than
    only where it ends up -- plot_bar_comparison/plot_histogram_vs_baseline
    above are aggregate/final-state only and can't show that.

    episode_id as the x-axis is a within-run ORDINAL PROXY for training
    progress, not real cumulative timesteps, and the approximation is
    weaker than it looks: CoordinationMetrics._episode_id (Metrics.py)
    starts fresh at 1 in every one of Train_Ground.py's n_proc worker
    processes independently (confirmed against its source), and
    _load_coordination_csvs concatenates every process's shard under one
    run_label. So "episode_id=10" in the resulting frame mixes different
    workers' 10th episode, which can correspond to noticeably different
    amounts of actual training if episode lengths vary across workers
    (they do -- an episode ending in a quick goal vs. running to the
    no-touch timeout). Binning by episode_id and averaging across
    processes (done below via groupby) smooths over this somewhat but
    doesn't fix it. Real cumulative_timesteps isn't logged per-episode
    anywhere currently -- Learner's actual counter lives in the parent
    process, not reachable from inside an env subprocess without extra
    plumbing (same limitation Rewards.py's AnnealedCombinedReward already
    documents and works around with an n_proc-scaled local-step estimate
    for the anneal schedule). The same estimation technique could log an
    approximate cumulative_timesteps column here too, which would make
    this x-axis a genuine measure of training progress instead of a
    per-worker ordinal -- worth doing if these trends need to be more
    than roughly indicative.
    """
    if metric not in df.columns or "episode_id" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for label, sub in df.groupby("run_label"):
        sub = sub.dropna(subset=[metric, "episode_id"]).sort_values("episode_id")
        if sub.empty:
            continue
        # Average across processes sharing the same nominal episode_id,
        # then roll -- see docstring on why this is an approximation.
        by_episode = sub.groupby("episode_id")[metric].mean()
        rolled = by_episode.rolling(window=window, min_periods=1).mean()
        ax.plot(by_episode.index, rolled.values, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("episode_id (within-run ordinal proxy for training progress -- see docstring)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} over training (rolling mean, window={window} episodes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"trend_{metric}.png", dpi=150)
    plt.close(fig)


def plot_fitness_distribution(fitness_df: pd.DataFrame, out_dir: Path) -> None:
    if fitness_df.empty or "episode_return" not in fitness_df.columns:
        return
    labels = sorted(fitness_df["run_label"].dropna().unique())
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [fitness_df[fitness_df["run_label"] == label]["episode_return"].dropna() for label in labels]
    data = [d for d in data if len(d) > 0]
    kept_labels = [label for label, d in zip(labels, [fitness_df[fitness_df["run_label"] == l]["episode_return"].dropna() for l in labels]) if len(d) > 0]
    if not data:
        return
    try:
        ax.boxplot(data, tick_labels=kept_labels)  # matplotlib >= 3.9
    except TypeError:
        ax.boxplot(data, labels=kept_labels)       # matplotlib < 3.9
    ax.set_ylabel("episode_return (FitnessTracker, tracked team only)")
    ax.set_title("Fitness (episode_return) distribution by condition")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "fitness_distribution.png", dpi=150)
    plt.close(fig)


def plot_pbt_generations(results_dir: Path, out_dir: Path) -> None:
    # Only present if a PBT pass has actually run (standalone `python
    # Pbt.py`, or experiments/run_all.sh with INCLUDE_PBT=1 -- see
    # experiments/06_pbt.sh and docs/ISSUES.md). Always nested at
    # metrics/pbt/ regardless of PBT_OUT_DIR -- see Pbt.py's
    # GENERATION_LOG_PATH comment.
    gen_log = results_dir / "metrics" / "pbt" / "generations.jsonl"
    if not gen_log.exists():
        return

    rows = []
    with open(gen_log) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for member_id, sub in df.groupby("member_id"):
        sub = sub.sort_values("generation")
        ax.plot(sub["generation"], sub["fitness"], marker="o", label=f"member {member_id}")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness (cross-play win rate)")
    ax.set_title("PBT: fitness per generation, per population member")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pbt_fitness_by_generation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="output",
                         help="Same $OUT directory experiments/run_all.sh wrote to.")
    parser.add_argument("--baseline-label", default="baseline_random")
    parser.add_argument("--highlight-label", default="01_ground",
                         help="Condition to compare against baseline in histograms "
                              "(the main ablation-vs-baseline pair).")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics_dir = results_dir / "metrics"
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_dir.exists():
        raise SystemExit(f"No metrics directory at {metrics_dir} -- run experiments/run_all.sh first.")

    coord_df = _load_coordination_csvs(metrics_dir)
    fitness_df = _load_fitness_csvs(metrics_dir)

    if coord_df.empty:
        print(f"No coordination CSV data found under {metrics_dir} -- nothing to plot yet.")
    else:
        print(f"Loaded {len(coord_df)} episode rows across labels: "
              f"{sorted(coord_df['run_label'].dropna().unique())}")
        for metric in ALL_METRICS:
            plot_bar_comparison(coord_df, metric, out_dir)
            plot_histogram_vs_baseline(coord_df, metric, args.baseline_label, args.highlight_label, out_dir)
        for metric in COORDINATION_METRICS:
            plot_metric_trend(coord_df, metric, out_dir)

    if not fitness_df.empty:
        print(f"Loaded {len(fitness_df)} fitness rows across labels: "
              f"{sorted(fitness_df['run_label'].dropna().unique())}")
        plot_fitness_distribution(fitness_df, out_dir)

    plot_pbt_generations(results_dir, out_dir)

    n_plots = len(list(out_dir.glob("*.png")))
    print(f"Done. {n_plots} plot(s) written to {out_dir}")


if __name__ == "__main__":
    main()
