from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.ema_target_encoder = copy.deepcopy(self.target_encoder)
        self.target_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 7),
        )
        self._set_ema_requires_grad(False)

    def _set_ema_requires_grad(self, requires_grad: bool) -> None:
        for param in self.ema_target_encoder.parameters():
            param.requires_grad = requires_grad

    @torch.no_grad()
    def sync_ema_target_encoder(self) -> None:
        self.ema_target_encoder.load_state_dict(self.target_encoder.state_dict())
        self._set_ema_requires_grad(False)

    @torch.no_grad()
    def update_ema_target_encoder(self, decay: float) -> None:
        decay = float(decay)
        for ema_param, online_param in zip(self.ema_target_encoder.parameters(), self.target_encoder.parameters()):
            ema_param.mul_(decay).add_(online_param, alpha=1.0 - decay)
        for ema_buffer, online_buffer in zip(self.ema_target_encoder.buffers(), self.target_encoder.buffers()):
            ema_buffer.copy_(online_buffer)
        self._set_ema_requires_grad(False)

    @torch.no_grad()
    def ema_online_drift(self) -> torch.Tensor:
        diffs = []
        for ema_param, online_param in zip(self.ema_target_encoder.parameters(), self.target_encoder.parameters()):
            diffs.append(torch.mean((ema_param - online_param) ** 2))
        if not diffs:
            return torch.tensor(0.0)
        return torch.sqrt(torch.stack(diffs).mean())

    def _encode_context(self, frames: torch.Tensor) -> torch.Tensor:
        bsz, steps, channels, height, width = frames.shape
        encoded = self.frame_encoder(frames.reshape(bsz * steps, channels, height, width))
        encoded = encoded.reshape(bsz, steps, -1)
        _, hidden = self.temporal(encoded)
        return self.context_proj(hidden[-1])

    def _target_input(self, future_state: torch.Tensor, steps: int) -> torch.Tensor:
        target_dyn = future_state[:, :steps, :, 0:6]
        target_occ = future_state[:, :steps, :, 7:8]
        return torch.cat([target_dyn, target_occ], dim=-1)

    def _pad_horizon(self, values: torch.Tensor, batch_size: int) -> torch.Tensor:
        if values.shape[1] >= self.horizon:
            return values
        pad = torch.zeros(
            (batch_size, self.horizon - values.shape[1], self.max_objects, values.shape[-1]),
            device=values.device,
            dtype=values.dtype,
        )
        return torch.cat([values, pad], dim=1)

    def forward(
        self,
        frames: torch.Tensor,
        future_state: torch.Tensor | None = None,
        use_ema_target: bool = True,
        include_target_reconstruction: bool = True,
    ) -> dict[str, torch.Tensor]:
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
            target_input = self._target_input(future_state, steps)
            with torch.no_grad():
                target_encoder = self.ema_target_encoder if use_ema_target else self.target_encoder
                target_latent = target_encoder(target_input)
            out["target_latent"] = self._pad_horizon(target_latent, bsz).detach()
            if include_target_reconstruction:
                online_target_latent = self.target_encoder(target_input)
                target_reconstruction = self.target_decoder(online_target_latent)
                out["target_input"] = self._pad_horizon(target_input, bsz)
                out["online_target_latent"] = self._pad_horizon(online_target_latent, bsz)
                out["target_reconstruction"] = self._pad_horizon(target_reconstruction, bsz)
        return out


def _deterministic_sketch_directions(
    latent_dim: int,
    num_sketches: int,
    device: torch.device,
    dtype: torch.dtype,
    sketch_scale: float,
) -> torch.Tensor:
    count = max(1, int(num_sketches))
    idx = torch.arange(count * latent_dim, device=device, dtype=dtype).reshape(count, latent_dim)
    directions = torch.sin(idx * 12.9898 + 78.233) + 0.5 * torch.cos(idx * 4.1414 + 19.19)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return directions * float(sketch_scale)


def sketched_isotropic_gaussian_regularizer(
    latents: torch.Tensor,
    mask: torch.Tensor,
    num_sketches: int = 16,
    sketch_scale: float = 1.0,
) -> torch.Tensor:
    """
    LeJEPA-style lightweight SIGReg: match random low-dimensional sketches of
    the latent distribution to the characteristic function of N(0, I).
    """
    flat = latents[mask > 0.5]
    if flat.shape[0] == 0:
        return latents.sum() * 0.0
    directions = _deterministic_sketch_directions(
        latent_dim=flat.shape[-1],
        num_sketches=num_sketches,
        device=flat.device,
        dtype=flat.dtype,
        sketch_scale=sketch_scale,
    )
    projections = flat @ directions.T
    target_real = torch.exp(-0.5 * torch.sum(directions * directions, dim=-1))
    real_loss = (torch.cos(projections).mean(dim=0) - target_real).pow(2)
    imag_loss = torch.sin(projections).mean(dim=0).pow(2)
    return (real_loss + imag_loss).mean()


def _masked_std(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask > 0.5]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.std(unbiased=False)


def belief_jepa_loss(
    outputs: dict[str, torch.Tensor],
    future_state: torch.Tensor,
    future_mask: torch.Tensor,
    latent_weight: float = 1.0,
    belief_weight: float = 0.5,
    target_recon_weight: float = 0.1,
    sigreg_weight: float = 0.0,
    sigreg_sketches: int = 16,
    sigreg_scale: float = 1.0,
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
    target_reconstruction = outputs["target_reconstruction"][:, :steps]
    target_input = outputs["target_input"][:, :steps]
    target_recon_mse = (((target_reconstruction - target_input) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)

    pos_rmse = torch.sqrt(((mean[..., 0:3] - target[..., 0:3]) ** 2).sum(dim=-1).clamp_min(1e-12))
    pos_rmse = (pos_rmse * future_mask[:, :steps]).sum() / future_mask[:, :steps].sum().clamp_min(1.0)
    pred_std = _masked_std(pred_latent, mask.expand_as(pred_latent))
    target_std = _masked_std(target_latent, mask.expand_as(target_latent))
    cosine = F.cosine_similarity(pred_latent, target_latent, dim=-1)
    pred_target_cosine = (cosine * future_mask[:, :steps]).sum() / future_mask[:, :steps].sum().clamp_min(1.0)
    pred_sigreg = sketched_isotropic_gaussian_regularizer(
        pred_latent,
        future_mask[:, :steps],
        num_sketches=sigreg_sketches,
        sketch_scale=sigreg_scale,
    )
    online_target_latent = outputs.get("online_target_latent", outputs["target_latent"])[:, :steps]
    target_sigreg = sketched_isotropic_gaussian_regularizer(
        online_target_latent,
        future_mask[:, :steps],
        num_sketches=sigreg_sketches,
        sketch_scale=sigreg_scale,
    )
    sigreg = 0.5 * (pred_sigreg + target_sigreg)
    total = (
        latent_weight * latent_mse
        + belief_weight * belief_nll
        + target_recon_weight * target_recon_mse
        + float(sigreg_weight) * sigreg
    )
    return {
        "total": total,
        "latent_mse": latent_mse,
        "belief_nll": belief_nll,
        "target_recon_mse": target_recon_mse,
        "sigreg": sigreg,
        "pred_sigreg": pred_sigreg,
        "target_sigreg": target_sigreg,
        "pos_rmse": pos_rmse,
        "target_latent_std": target_std,
        "pred_latent_std": pred_std,
        "pred_target_cosine": pred_target_cosine,
    }


@torch.no_grad()
def belief_jepa_diagnostics(
    outputs: dict[str, torch.Tensor],
    future_state: torch.Tensor,
    future_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    steps = min(outputs["mean"].shape[1], future_state.shape[1])
    mask = future_mask[:, :steps].unsqueeze(-1)
    pred_latent = outputs["pred_latent"][:, :steps]
    target_latent = outputs["target_latent"][:, :steps]
    latent_mse = (((pred_latent - target_latent) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
    cosine = F.cosine_similarity(pred_latent, target_latent, dim=-1)
    pred_target_cosine = (cosine * future_mask[:, :steps]).sum() / future_mask[:, :steps].sum().clamp_min(1.0)
    return {
        "latent_mse": latent_mse,
        "pred_target_cosine": pred_target_cosine,
        "target_latent_std": _masked_std(target_latent, mask.expand_as(target_latent)),
        "pred_latent_std": _masked_std(pred_latent, mask.expand_as(pred_latent)),
    }
