from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = (pred - target) ** 2
    if mask.dim() == pred.dim() - 1:
        mask = mask.unsqueeze(-1)
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def state_rollout_loss(
    pred_state: torch.Tensor,
    target_state: torch.Tensor,
    object_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pos_loss = masked_mse(pred_state[..., 0:2], target_state[..., 0:2], object_mask)
    vel_loss = masked_mse(pred_state[..., 2:4], target_state[..., 2:4], object_mask)
    vis_loss = masked_mse(pred_state[..., 4:6], target_state[..., 4:6], object_mask)
    static_loss = masked_mse(pred_state[..., 6:], target_state[..., 6:], object_mask)
    total = pos_loss + 0.5 * vel_loss + 0.25 * vis_loss + 0.1 * static_loss
    return {
        "total": total,
        "pos": pos_loss,
        "vel": vel_loss,
        "vis": vis_loss,
        "static": static_loss,
    }


def encoder_state_loss(
    pred_init: torch.Tensor,
    target_init: torch.Tensor,
    object_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pos_loss = masked_mse(pred_init[..., 0:2], target_init[..., 0:2], object_mask)
    vel_loss = masked_mse(pred_init[..., 2:4], target_init[..., 2:4], object_mask)
    vis_loss = masked_mse(pred_init[..., 4:6], target_init[..., 4:6], object_mask)
    static_loss = masked_mse(pred_init[..., 6:], target_init[..., 6:], object_mask)
    total = pos_loss + 0.5 * vel_loss + 0.25 * vis_loss + 0.1 * static_loss
    return {
        "total": total,
        "pos": pos_loss,
        "vel": vel_loss,
        "vis": vis_loss,
        "static": static_loss,
    }


def pixel_baseline_loss(pred_frames: torch.Tensor, target_frames: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_frames, target_frames)

