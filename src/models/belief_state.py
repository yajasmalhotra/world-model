from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from src.data3d.scene_generator3d import STATE_INDEX_3D


@dataclass
class ParticleBeliefConfig:
    num_particles: int = 256
    init_pos_noise: float = 0.02
    init_vel_noise: float = 0.008
    process_pos_noise: float = 0.003
    process_vel_noise: float = 0.002
    world_min: float = -1.0
    world_max: float = 1.0

    @staticmethod
    def from_config(belief_cfg: Dict[str, object], data_cfg: Dict[str, object]) -> "ParticleBeliefConfig":
        return ParticleBeliefConfig(
            num_particles=int(belief_cfg.get("num_particles", 256)),
            init_pos_noise=float(belief_cfg.get("init_pos_noise", 0.02)),
            init_vel_noise=float(belief_cfg.get("init_vel_noise", 0.008)),
            process_pos_noise=float(belief_cfg.get("process_pos_noise", 0.003)),
            process_vel_noise=float(belief_cfg.get("process_vel_noise", 0.002)),
            world_min=float(data_cfg.get("world_min", -1.0)),
            world_max=float(data_cfg.get("world_max", 1.0)),
        )


def _bounce_particles(pos: torch.Tensor, vel: torch.Tensor, size: torch.Tensor, cfg: ParticleBeliefConfig) -> tuple[torch.Tensor, torch.Tensor]:
    lo = cfg.world_min + size.unsqueeze(-1) + 0.02
    hi = cfg.world_max - size.unsqueeze(-1) - 0.02
    low_hit = pos < lo
    high_hit = pos > hi
    hit = low_hit | high_hit
    vel = torch.where(hit, -vel, vel)
    pos = torch.minimum(torch.maximum(pos, lo), hi)
    return pos, vel


def initialize_particles(
    initial_state: torch.Tensor,
    object_mask: torch.Tensor,
    cfg: ParticleBeliefConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        initial_state: [B, O, F]
        object_mask: [B, O]
    Returns:
        particles: [B, O, P, 6] for x/y/z/vx/vy/vz
        weights: [B, O, P]
    """
    bsz, n_obj, _ = initial_state.shape
    device = initial_state.device
    dtype = initial_state.dtype
    num_particles = cfg.num_particles
    pos = initial_state[..., STATE_INDEX_3D["x"] : STATE_INDEX_3D["z"] + 1]
    vel = initial_state[..., STATE_INDEX_3D["vx"] : STATE_INDEX_3D["vz"] + 1]
    pos_noise = torch.randn((bsz, n_obj, num_particles, 3), device=device, dtype=dtype, generator=generator)
    vel_noise = torch.randn((bsz, n_obj, num_particles, 3), device=device, dtype=dtype, generator=generator)
    particle_pos = pos.unsqueeze(2) + pos_noise * cfg.init_pos_noise
    particle_vel = vel.unsqueeze(2) + vel_noise * cfg.init_vel_noise
    size = initial_state[..., STATE_INDEX_3D["size"]].unsqueeze(-1).expand(-1, -1, num_particles)
    particle_pos, particle_vel = _bounce_particles(particle_pos, particle_vel, size, cfg)
    particles = torch.cat([particle_pos, particle_vel], dim=-1)
    weights = torch.ones((bsz, n_obj, num_particles), device=device, dtype=dtype) / float(num_particles)
    weights = weights * object_mask.unsqueeze(-1)
    return particles, weights


def initialize_particles_from_gaussian(
    belief_mean: torch.Tensor,
    belief_log_std: torch.Tensor,
    template_state: torch.Tensor,
    object_mask: torch.Tensor,
    cfg: ParticleBeliefConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize particles from predicted Gaussian belief parameters.

    Args:
        belief_mean: [B, O, 6] for x/y/z/vx/vy/vz
        belief_log_std: [B, O, 6]
        template_state: [B, O, F], supplies object size for wall constraints
        object_mask: [B, O]
    Returns:
        particles: [B, O, P, 6]
        weights: [B, O, P]
    """
    bsz, n_obj, _ = belief_mean.shape
    device = belief_mean.device
    dtype = belief_mean.dtype
    num_particles = cfg.num_particles
    eps = torch.randn((bsz, n_obj, num_particles, 6), device=device, dtype=dtype, generator=generator)
    std = torch.exp(belief_log_std).clamp_min(1e-4)
    particles = belief_mean.unsqueeze(2) + eps * std.unsqueeze(2)
    size = template_state[..., STATE_INDEX_3D["size"]].unsqueeze(-1).expand(-1, -1, num_particles)
    pos, vel = _bounce_particles(particles[..., 0:3], particles[..., 3:6], size, cfg)
    particles = torch.cat([pos, vel], dim=-1)
    weights = torch.ones((bsz, n_obj, num_particles), device=device, dtype=dtype) / float(num_particles)
    weights = weights * object_mask.unsqueeze(-1)
    return particles, weights


def rollout_particle_belief_from_gaussian(
    belief_mean: torch.Tensor,
    belief_log_std: torch.Tensor,
    template_state: torch.Tensor,
    object_mask: torch.Tensor,
    horizon: int,
    cfg: ParticleBeliefConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    particles, weights = initialize_particles_from_gaussian(
        belief_mean,
        belief_log_std,
        template_state,
        object_mask,
        cfg,
        generator=generator,
    )
    size = template_state[..., STATE_INDEX_3D["size"]].unsqueeze(-1).expand(-1, -1, cfg.num_particles)
    outputs = []
    weight_outputs = []
    for _ in range(horizon):
        pos = particles[..., 0:3]
        vel = particles[..., 3:6]
        pos_noise = torch.randn(pos.shape, device=pos.device, dtype=pos.dtype, generator=generator) * cfg.process_pos_noise
        vel_noise = torch.randn(vel.shape, device=vel.device, dtype=vel.dtype, generator=generator) * cfg.process_vel_noise
        vel = vel + vel_noise
        pos = pos + vel + pos_noise
        pos, vel = _bounce_particles(pos, vel, size, cfg)
        particles = torch.cat([pos, vel], dim=-1)
        outputs.append(particles)
        weight_outputs.append(weights)
    return torch.stack(outputs, dim=1), torch.stack(weight_outputs, dim=1)


def rollout_particle_belief(
    initial_state: torch.Tensor,
    object_mask: torch.Tensor,
    horizon: int,
    cfg: ParticleBeliefConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Physics-only belief baseline.

    Returns:
        particles: [B, H, O, P, 6]
        weights: [B, H, O, P]
    """
    particles, weights = initialize_particles(initial_state, object_mask, cfg, generator=generator)
    size = initial_state[..., STATE_INDEX_3D["size"]].unsqueeze(-1).expand(-1, -1, cfg.num_particles)
    outputs = []
    weight_outputs = []
    for _ in range(horizon):
        pos = particles[..., 0:3]
        vel = particles[..., 3:6]
        pos_noise = torch.randn(pos.shape, device=pos.device, dtype=pos.dtype, generator=generator) * cfg.process_pos_noise
        vel_noise = torch.randn(vel.shape, device=vel.device, dtype=vel.dtype, generator=generator) * cfg.process_vel_noise
        vel = vel + vel_noise
        pos = pos + vel + pos_noise
        pos, vel = _bounce_particles(pos, vel, size, cfg)
        particles = torch.cat([pos, vel], dim=-1)
        outputs.append(particles)
        weight_outputs.append(weights)
    return torch.stack(outputs, dim=1), torch.stack(weight_outputs, dim=1)
