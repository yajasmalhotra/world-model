from __future__ import annotations

from typing import Dict

import torch


def move_boxes_to_far_corner(
    boxes: torch.Tensor,
    world_min: float,
    world_max: float,
    margin: float = 0.04,
) -> torch.Tensor:
    """
    Move active 3D boxes to a far world corner while preserving their extent.

    This creates a physical-geometry counterfactual that changes motion
    constraints without changing the target state sequence.
    """
    if boxes.numel() == 0:
        return boxes.clone()
    moved = boxes.clone()
    active = torch.any(boxes != 0, dim=-1)
    if not bool(active.any()):
        return moved

    lo = torch.minimum(boxes[..., 0:3], boxes[..., 3:6])
    hi = torch.maximum(boxes[..., 0:3], boxes[..., 3:6])
    extent = (hi - lo).clamp_min(1e-4)
    max_corner = torch.full_like(lo, float(world_max) - float(margin))
    min_corner = torch.maximum(max_corner - extent, torch.full_like(lo, float(world_min) + float(margin)))
    new_box = torch.cat([min_corner, max_corner], dim=-1)
    return torch.where(active.unsqueeze(-1), new_box, moved)


def weighted_position_mean(particles: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.sum(weights.unsqueeze(-1) * particles[..., 0:3], dim=-2)


def hidden_object_mask(target_state: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
    return object_mask * (target_state[..., 7] > 0.5).float()


def counterfactual_delta_metrics(
    base_particles: torch.Tensor,
    base_weights: torch.Tensor,
    cf_particles: torch.Tensor,
    cf_weights: torch.Tensor,
    target_state: torch.Tensor,
    object_mask: torch.Tensor,
    prefix: str,
) -> Dict[str, float]:
    """
    Compare hidden-interval belief means between base and counterfactual rollouts.
    """
    active = hidden_object_mask(target_state, object_mask) > 0.5
    if active.sum() <= 0:
        return {
            f"{prefix}_belief_delta": float("nan"),
            f"{prefix}_hidden_count": 0.0,
        }

    base_mean = weighted_position_mean(base_particles, base_weights)
    cf_mean = weighted_position_mean(cf_particles, cf_weights)
    delta = torch.linalg.norm(base_mean - cf_mean, dim=-1)
    return {
        f"{prefix}_belief_delta": float(delta[active].mean().item()),
        f"{prefix}_hidden_count": float(active.sum().item()),
    }
