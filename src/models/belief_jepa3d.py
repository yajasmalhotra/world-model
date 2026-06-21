from __future__ import annotations

import torch
import torch.nn as nn


class BeliefJEPA3D(nn.Module):
    """
    Small JEPA-style latent predictor for hidden 3D belief.

    The context branch sees observed RGB or RGB-D frames. The target branch sees
    privileged future state during training only. The predictor maps context
    embeddings to future latent targets and a per-timestep Gaussian belief over
    (x, y, z, vx, vy, vz), without reconstructing pixels.
    """

    def __init__(
        self,
        max_objects: int = 5,
        horizon: int = 18,
        input_channels: int = 3,
        cnn_dim: int = 96,
        rnn_dim: int = 128,
        latent_dim: int = 64,
        world_min: float = -1.0,
        world_max: float = 1.0,
        velocity_limit: float = 0.16,
        min_log_std: float = -5.0,
        max_log_std: float = -0.8,
    ):
        super().__init__()
        self.max_objects = int(max_objects)
        self.horizon = int(horizon)
        self.latent_dim = int(latent_dim)
        self.world_min = float(world_min)
        self.world_max = float(world_max)
        self.velocity_limit = float(velocity_limit)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, cnn_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.temporal = nn.GRU(cnn_dim, rnn_dim, batch_first=True)
        self.context_proj = nn.Sequential(nn.Linear(rnn_dim, rnn_dim), nn.ReLU())
        self.predictor = nn.Sequential(
            nn.Linear(rnn_dim, rnn_dim),
            nn.ReLU(),
            nn.Linear(rnn_dim, self.horizon * self.max_objects * (latent_dim + 12)),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(7, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def _encode_context(self, frames: torch.Tensor) -> torch.Tensor:
        bsz, steps, channels, height, width = frames.shape
        encoded = self.frame_encoder(frames.reshape(bsz * steps, channels, height, width))
        encoded = encoded.reshape(bsz, steps, -1)
        _, hidden = self.temporal(encoded)
        return self.context_proj(hidden[-1])

    def forward(self, frames: torch.Tensor, future_state: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        bsz = frames.shape[0]
        context = self._encode_context(frames)
        raw = self.predictor(context).reshape(bsz, self.horizon, self.max_objects, self.latent_dim + 12)
        pred_latent = raw[..., : self.latent_dim]
        raw_mean = raw[..., self.latent_dim : self.latent_dim + 6]
        raw_log_std = raw[..., self.latent_dim + 6 :]

        center = 0.5 * (self.world_min + self.world_max)
        half_span = 0.5 * (self.world_max - self.world_min)
        pos = center + half_span * torch.tanh(raw_mean[..., 0:3])
        vel = self.velocity_limit * torch.tanh(raw_mean[..., 3:6])
        mean = torch.cat([pos, vel], dim=-1)
        log_std = raw_log_std.clamp(self.min_log_std, self.max_log_std)

        out = {"mean": mean, "log_std": log_std, "pred_latent": pred_latent}
        if future_state is not None:
            steps = min(self.horizon, future_state.shape[1])
            target_dyn = future_state[:, :steps, :, 0:6]
            target_occ = future_state[:, :steps, :, 7:8]
            target_input = torch.cat([target_dyn, target_occ], dim=-1)
            target_latent = self.target_encoder(target_input)
            if steps < self.horizon:
                pad = torch.zeros(
                    (bsz, self.horizon - steps, self.max_objects, self.latent_dim),
                    device=frames.device,
                    dtype=frames.dtype,
                )
                target_latent = torch.cat([target_latent, pad], dim=1)
            out["target_latent"] = target_latent.detach()
        return out


def belief_jepa_loss(
    outputs: dict[str, torch.Tensor],
    future_state: torch.Tensor,
    future_mask: torch.Tensor,
    latent_weight: float = 1.0,
    belief_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    steps = min(outputs["mean"].shape[1], future_state.shape[1])
    mask = future_mask[:, :steps].unsqueeze(-1)
    target = future_state[:, :steps, :, 0:6]
    mean = outputs["mean"][:, :steps]
    log_std = outputs["log_std"][:, :steps]
    inv_var = torch.exp(-2.0 * log_std)
    nll = 0.5 * ((target - mean) ** 2 * inv_var + 2.0 * log_std)
    belief_nll = (nll * mask).sum() / mask.sum().clamp_min(1.0)

    target_latent = outputs["target_latent"][:, :steps]
    pred_latent = outputs["pred_latent"][:, :steps]
    latent_mse = (((pred_latent - target_latent) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)

    pos_rmse = torch.sqrt(((mean[..., 0:3] - target[..., 0:3]) ** 2).sum(dim=-1).clamp_min(1e-12))
    pos_rmse = (pos_rmse * future_mask[:, :steps]).sum() / future_mask[:, :steps].sum().clamp_min(1.0)
    total = latent_weight * latent_mse + belief_weight * belief_nll
    return {"total": total, "latent_mse": latent_mse, "belief_nll": belief_nll, "pos_rmse": pos_rmse}
