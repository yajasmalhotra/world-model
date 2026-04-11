from __future__ import annotations

from typing import Dict

import torch


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num / den.clamp_min(1e-8)


def rollout_position_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    err2 = (pred[..., 0:2] - target[..., 0:2]) ** 2
    mse = _safe_div((err2.sum(dim=-1) * mask).sum(), mask.sum())
    return float(torch.sqrt(mse).item())


def occluded_position_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    occ_mask = mask * target[..., 5]
    err2 = (pred[..., 0:2] - target[..., 0:2]) ** 2
    mse = _safe_div((err2.sum(dim=-1) * occ_mask).sum(), occ_mask.sum())
    return float(torch.sqrt(mse).item())


def reappearance_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    prev_occ = torch.zeros_like(target[..., 5])
    prev_occ[:, 1:] = target[:, :-1, :, 5]
    reappear = mask * (target[..., 4] > 0.5).float() * (prev_occ > 0.5).float()
    err2 = (pred[..., 0:2] - target[..., 0:2]) ** 2
    mse = _safe_div((err2.sum(dim=-1) * reappear).sum(), reappear.sum())
    return float(torch.sqrt(mse).item())


def identity_consistency(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    # Approximate swap metric: after reappearance, nearest predicted object to each GT object
    # should preserve slot index.
    bsz, steps, n_obj, _ = pred.shape
    correct = 0.0
    total = 0.0
    for b in range(bsz):
        for t in range(1, steps):
            for o in range(n_obj):
                if mask[b, t, o] < 0.5:
                    continue
                was_occ = target[b, t - 1, o, 5] > 0.5
                now_vis = target[b, t, o, 4] > 0.5
                if not (was_occ and now_vis):
                    continue
                gt_pos = target[b, t, o, 0:2]
                pred_pos = pred[b, t, :, 0:2]
                d = ((pred_pos - gt_pos.unsqueeze(0)) ** 2).sum(dim=-1)
                nearest = int(torch.argmin(d).item())
                correct += 1.0 if nearest == o else 0.0
                total += 1.0
    if total == 0:
        return 1.0
    return float(correct / total)


def frame_mse(pred_frames: torch.Tensor, target_frames: torch.Tensor) -> float:
    return float(torch.mean((pred_frames - target_frames) ** 2).item())


def counterfactual_locality(
    base_pred: torch.Tensor,
    cf_pred: torch.Tensor,
    object_mask: torch.Tensor,
    object_idx: int,
) -> float:
    delta = torch.sqrt(((cf_pred[..., 0:2] - base_pred[..., 0:2]) ** 2).sum(dim=-1))
    target_shift = (delta[..., object_idx] * object_mask[..., object_idx]).sum()

    non_target = []
    for idx in range(delta.shape[-1]):
        if idx == object_idx:
            continue
        non_target.append((delta[..., idx] * object_mask[..., idx]).sum())
    non_target_shift = torch.stack(non_target).mean() if non_target else torch.tensor(0.0, device=delta.device)
    locality = _safe_div(target_shift, non_target_shift + 1e-6)
    return float(locality.item())


def summarize_metrics(metric_rows: Dict[str, list[float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for k, values in metric_rows.items():
        if not values:
            continue
        summary[k] = float(sum(values) / len(values))
    return summary

