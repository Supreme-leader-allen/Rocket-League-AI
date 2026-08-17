"""
Trains Lucy-SKG-style auxiliary networks (Auxiliary.py's
StateRepresentationNet + RewardPredictionNet) on data logged by
Metrics.py's AuxiliaryDataLogger (a Train_Ground.py run with
AUX_LOGGING=1). Saves StateRepresentationNet's weights to
checkpoints/auxiliary_encoder.pt -- the only checkpoint
Observation.py's AuxiliaryEncoder actually loads back for inference --
and RewardPredictionNet's to checkpoints/auxiliary_reward_predictor.pt
for completeness (nothing downstream currently reads that one back; see
Auxiliary.py's module docstring on why the two nets train independently
rather than sharing weights).

Run with:
    AUX_LOGGING=1 python Train_Ground.py     # log data first; let it run
                                              # long enough to produce a
                                              # few shards under
                                              # metrics/aux_data/, then
                                              # stop it
    python Train_Auxiliary_Encoder.py

Caveat: AuxiliaryDataLogger flushes every `flush_every` steps regardless
of episode boundaries and doesn't record where episodes start/end within
a shard -- RewardPredictionNet's sliding windows below are built
contiguously within each shard without checking for episode boundaries,
so a small fraction of windows may straddle two unrelated episodes. Not
worth a bigger logging redesign for an auxiliary/observational task.
"""

import argparse
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from Auxiliary import StateRepresentationNet, RewardPredictionNet, RP_WINDOW

DEFAULT_DATA_DIR = "metrics/aux_data"
DEFAULT_SR_CHECKPOINT = "checkpoints/auxiliary_encoder.pt"
DEFAULT_RP_CHECKPOINT = "checkpoints/auxiliary_reward_predictor.pt"


def load_shards(data_dir: str):
    paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    if not paths:
        raise FileNotFoundError(
            f"No shards found under {data_dir!r} -- run "
            f"`AUX_LOGGING=1 python Train_Ground.py` first and let it log "
            f"for a while before training the encoder."
        )
    obs_list, reward_list = [], []
    for path in paths:
        with np.load(path) as data:
            obs_list.append(data["obs"])
            reward_list.append(data["reward"])
    return obs_list, reward_list  # kept per-shard so RP windowing doesn't cross shard files


def build_rp_windows(obs_list, reward_list, window: int):
    windows, targets = [], []
    for obs, reward in zip(obs_list, reward_list):
        n = obs.shape[0]
        for start in range(0, n - window):
            windows.append(obs[start:start + window])
            targets.append(reward[start + window - 1])
    if not windows:
        return None, None
    return np.stack(windows), np.array(targets, dtype=np.float32)


def train_sr(obs_size, all_obs, epochs, batch_size, lr, device):
    net = StateRepresentationNet(obs_size).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    dataset = TensorDataset(torch.as_tensor(all_obs, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = net.loss(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.shape[0]
        print(f"[SR] epoch {epoch + 1}/{epochs}  loss={total_loss / len(dataset):.5f}")
    return net


def train_rp(obs_size, windows, targets, epochs, batch_size, lr, device):
    net = RewardPredictionNet(obs_size).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    dataset = TensorDataset(
        torch.as_tensor(windows, dtype=torch.float32),
        torch.as_tensor(targets, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_windows, batch_targets in loader:
            batch_windows = batch_windows.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            loss = net.loss(batch_windows, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_windows.shape[0]
        print(f"[RP] epoch {epoch + 1}/{epochs}  loss={total_loss / len(dataset):.5f}")
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--sr-checkpoint", default=DEFAULT_SR_CHECKPOINT)
    parser.add_argument("--rp-checkpoint", default=DEFAULT_RP_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=RP_WINDOW)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    obs_list, reward_list = load_shards(args.data_dir)
    obs_size = obs_list[0].shape[1]
    all_obs = np.concatenate(obs_list, axis=0)
    print(f"Loaded {len(obs_list)} shard(s), {all_obs.shape[0]} observations, obs_size={obs_size}")

    sr_net = train_sr(obs_size, all_obs, args.epochs, args.batch_size, args.lr, args.device)
    os.makedirs(os.path.dirname(args.sr_checkpoint) or ".", exist_ok=True)
    torch.save(sr_net.state_dict(), args.sr_checkpoint)
    print(f"Saved StateRepresentationNet to {args.sr_checkpoint}")

    windows, targets = build_rp_windows(obs_list, reward_list, args.window)
    if windows is None:
        print(f"Not enough data for any {args.window}-step window -- skipping RewardPredictionNet.")
    else:
        rp_net = train_rp(obs_size, windows, targets, args.epochs, args.batch_size, args.lr, args.device)
        os.makedirs(os.path.dirname(args.rp_checkpoint) or ".", exist_ok=True)
        torch.save(rp_net.state_dict(), args.rp_checkpoint)
        print(f"Saved RewardPredictionNet to {args.rp_checkpoint}")


if __name__ == "__main__":
    main()
