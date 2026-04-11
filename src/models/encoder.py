from __future__ import annotations

import torch
import torch.nn as nn


class PixelToStateEncoder(nn.Module):
    def __init__(
        self,
        max_objects: int = 4,
        state_dim: int = 10,
        cnn_dim: int = 96,
        rnn_dim: int = 128,
        image_size: int = 64,
    ):
        super().__init__()
        self.max_objects = max_objects
        self.state_dim = state_dim
        self.image_size = image_size
        self.rnn_dim = rnn_dim

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, cnn_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.temporal = nn.GRU(cnn_dim, rnn_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(rnn_dim, rnn_dim),
            nn.ReLU(),
            nn.Linear(rnn_dim, max_objects * state_dim),
        )

    def _postprocess_state(self, raw: torch.Tensor) -> torch.Tensor:
        out = raw.clone()
        out[..., 0:2] = torch.sigmoid(raw[..., 0:2])  # x/y
        out[..., 2:4] = torch.tanh(raw[..., 2:4]) * 0.2  # vx/vy
        vis = torch.sigmoid(raw[..., 4:5])
        out[..., 4:5] = vis
        out[..., 5:6] = 1.0 - vis  # occluded
        out[..., 6:] = torch.sigmoid(raw[..., 6:])  # static-ish features
        return out

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B, T, C, H, W]
        Returns:
            initial_state: [B, O, F]
        """
        bsz, steps, channels, height, width = frames.shape
        encoded = self.frame_encoder(frames.reshape(bsz * steps, channels, height, width))
        encoded = encoded.reshape(bsz, steps, -1)
        _, hidden = self.temporal(encoded)
        hidden = hidden[-1]  # [B, rnn_dim]
        raw_state = self.head(hidden).reshape(bsz, self.max_objects, self.state_dim)
        return self._postprocess_state(raw_state)

