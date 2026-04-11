from __future__ import annotations

import torch
import torch.nn as nn


class PixelRolloutBaseline(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.obs_gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.trans_gru = nn.GRUCell(latent_dim, hidden_dim)
        self.latent_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        bsz, steps, channels, height, width = frames.shape
        latents = self.encoder(frames.reshape(bsz * steps, channels, height, width))
        return latents.reshape(bsz, steps, -1)

    def forward(self, obs_frames: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Args:
            obs_frames: [B, Tobs, C, H, W]
            horizon: number of future frames
        Returns:
            pred_frames: [B, H, C, H, W]
        """
        latents = self.encode_frames(obs_frames)
        _, hidden = self.obs_gru(latents)
        hidden = hidden[-1]
        prev_latent = latents[:, -1]

        outputs = []
        for _ in range(horizon):
            hidden = self.trans_gru(prev_latent, hidden)
            prev_latent = self.latent_head(hidden)
            frame = self.decoder(prev_latent)
            outputs.append(frame)
        return torch.stack(outputs, dim=1)

