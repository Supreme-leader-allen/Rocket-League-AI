"""
Lucy-SKG-style auxiliary networks (see README.md's "Lucy-SKG-style
auxiliary abstraction" section for the full workflow and how this
differs from the paper's joint-training approach).

StateRepresentationNet: autoencoder, encoder obs_size -> 128 -> 32 -> 16
(ENCODED_DIM), mirrored decoder 16 -> 32 -> 128 -> obs_size. Trained via
reconstruction MSE loss (.loss()). The 16-dim bottleneck (.encode()) is
what Observation.py's PartialInfoObs concatenates onto every observation
once a trained checkpoint is supplied via AUX_ENCODER_CHECKPOINT.

RewardPredictionNet: single-layer LSTM over a window of `window` (20 by
default) consecutive RAW observations, classifying the reward at the end
of the window into one of 3 buckets (negative / near-zero / positive),
matching Lucy-SKG's RP auxiliary task. Trained via cross-entropy loss
(.loss()) against reward values discretized by NEAR_ZERO_THRESHOLD. Its
input is the raw obs_size-dim observation, not StateRepresentationNet's
encoded output -- it's an independent network, not a probe on SR's
latent space, since rlgym_ppo's Learner gives no hook to backprop a
shared loss between the two (see README's "not identical to the paper"
note). Train_Auxiliary_Encoder.py saves its weights too, for
completeness/diagnostics, but nothing downstream currently loads them
back -- only StateRepresentationNet's checkpoint is consumed by
Observation.py's AuxiliaryEncoder.

AuxiliaryEncoder: thin inference-only wrapper around a trained
StateRepresentationNet, used by Observation.py.

Tensor shapes verified by actually running this environment's installed
torch (2.13.0) -- not traced by hand:

    python -c "
    from Auxiliary import StateRepresentationNet, RewardPredictionNet
    import torch
    sr = StateRepresentationNet(obs_size=50)
    rp = RewardPredictionNet(obs_size=50)
    print(sr.loss(torch.randn(4, 50)))
    print(rp.loss(torch.randn(4, 20, 50), torch.randn(4)))
    "
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODED_DIM = 16
SR_HIDDEN_SIZES = (128, 32)  # obs_size -> 128 -> 32 -> ENCODED_DIM, mirrored back out
RP_WINDOW = 20
RP_LSTM_HIDDEN = 64
RP_NEAR_ZERO_THRESHOLD = 0.05  # |reward| below this counts as "near-zero"


class StateRepresentationNet(nn.Module):
    def __init__(self, obs_size: int):
        super().__init__()
        h1, h2 = SR_HIDDEN_SIZES
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, ENCODED_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(ENCODED_DIM, h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, obs_size),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction = self.forward(x)
        return F.mse_loss(reconstruction, x)


class RewardPredictionNet(nn.Module):
    """3-class reward classifier (negative / near-zero / positive) over
    a `window`-step sequence of raw observations, per Lucy-SKG's RP
    auxiliary task."""

    NEGATIVE, NEAR_ZERO, POSITIVE = 0, 1, 2

    def __init__(self, obs_size: int, window: int = RP_WINDOW, hidden_size: int = RP_LSTM_HIDDEN):
        super().__init__()
        self.window = window
        self.lstm = nn.LSTM(input_size=obs_size, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 3)

    def forward(self, obs_window: torch.Tensor) -> torch.Tensor:
        # obs_window: (batch, window, obs_size) -> logits (batch, 3)
        _, (h_n, _) = self.lstm(obs_window)
        return self.classifier(h_n[-1])

    @staticmethod
    def bucket_rewards(reward: torch.Tensor) -> torch.Tensor:
        buckets = torch.full(reward.shape, RewardPredictionNet.NEAR_ZERO, dtype=torch.long, device=reward.device)
        buckets[reward > RP_NEAR_ZERO_THRESHOLD] = RewardPredictionNet.POSITIVE
        buckets[reward < -RP_NEAR_ZERO_THRESHOLD] = RewardPredictionNet.NEGATIVE
        return buckets

    def loss(self, obs_window: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        logits = self.forward(obs_window)
        targets = self.bucket_rewards(reward)
        return F.cross_entropy(logits, targets)


class AuxiliaryEncoder:
    """Inference-only wrapper around a trained StateRepresentationNet
    checkpoint, used by Observation.py's PartialInfoObs to augment
    observations once AUX_ENCODER_CHECKPOINT is set."""

    def __init__(self, checkpoint_path: str, obs_size: int, device: str = "cpu"):
        self.device = device
        self.net = StateRepresentationNet(obs_size).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def encode(self, raw_obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(raw_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            z = self.net.encode(x)
        return z.squeeze(0).cpu().numpy()
