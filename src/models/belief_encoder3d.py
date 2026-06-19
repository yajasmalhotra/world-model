from __future__ import annotations

import torch
import torch.nn as nn


class ImageToBeliefEncoder3D(nn.Module):
    """
    Small supervised encoder from observed RGB frames to per-slot Gaussian
    belief parameters for final observed 3D position and velocity.
    """

    def __init__(
        self,
        max_objects: int = 5,
        cnn_dim: int = 96,
        rnn_dim: int = 128,
        world_min: float = -1.0,
        world_max: float = 1.0,
        velocity_limit: float = 0.16,
        min_log_std: float = -5.0,
        max_log_std: float = -1.0,
    ):
        super().__init__()
        self.max_objects = max_objects
        self.world_min = float(world_min)
        self.world_max = float(world_max)
        self.velocity_limit = float(velocity_limit)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

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
            nn.Linear(rnn_dim, max_objects * 13),
        )

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            frames: [B, T, C, H, W]
        Returns:
            mean: [B, O, 6], log_std: [B, O, 6], object_logits: [B, O]
        """
        bsz, steps, channels, height, width = frames.shape
        encoded = self.frame_encoder(frames.reshape(bsz * steps, channels, height, width))
        encoded = encoded.reshape(bsz, steps, -1)
        _, hidden = self.temporal(encoded)
        hidden = hidden[-1]
        raw = self.head(hidden).reshape(bsz, self.max_objects, 13)

        raw_mean = raw[..., 0:6]
        raw_log_std = raw[..., 6:12]
        object_logits = raw[..., 12]

        center = 0.5 * (self.world_min + self.world_max)
        half_span = 0.5 * (self.world_max - self.world_min)
        pos = center + half_span * torch.tanh(raw_mean[..., 0:3])
        vel = self.velocity_limit * torch.tanh(raw_mean[..., 3:6])
        mean = torch.cat([pos, vel], dim=-1)
        log_std = raw_log_std.clamp(self.min_log_std, self.max_log_std)
        return {"mean": mean, "log_std": log_std, "object_logits": object_logits}


def gaussian_belief_loss(
    outputs: dict[str, torch.Tensor],
    target_state: torch.Tensor,
    object_mask: torch.Tensor,
    object_bce_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    target = target_state[..., 0:6]
    mean = outputs["mean"]
    log_std = outputs["log_std"]
    mask = object_mask.unsqueeze(-1)
    inv_var = torch.exp(-2.0 * log_std)
    nll = 0.5 * ((target - mean) ** 2 * inv_var + 2.0 * log_std)
    nll_loss = (nll * mask).sum() / mask.sum().clamp_min(1.0)

    pos_rmse = torch.sqrt(((mean[..., 0:3] - target[..., 0:3]) ** 2).sum(dim=-1).clamp_min(1e-12))
    pos_rmse = (pos_rmse * object_mask).sum() / object_mask.sum().clamp_min(1.0)
    vel_rmse = torch.sqrt(((mean[..., 3:6] - target[..., 3:6]) ** 2).sum(dim=-1).clamp_min(1e-12))
    vel_rmse = (vel_rmse * object_mask).sum() / object_mask.sum().clamp_min(1.0)

    bce = nn.functional.binary_cross_entropy_with_logits(outputs["object_logits"], object_mask)
    total = nll_loss + object_bce_weight * bce
    return {"total": total, "nll": nll_loss, "pos_rmse": pos_rmse, "vel_rmse": vel_rmse, "object_bce": bce}
