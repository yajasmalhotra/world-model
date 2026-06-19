from __future__ import annotations

from typing import Dict, Iterable

import torch


def _safe_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(values.mean().item())


def _weighted_quantile(values: torch.Tensor, weights: torch.Tensor, q: float) -> torch.Tensor:
    order = torch.argsort(values, dim=-1)
    sorted_values = torch.gather(values, -1, order)
    sorted_weights = torch.gather(weights, -1, order)
    cdf = torch.cumsum(sorted_weights, dim=-1)
    idx = torch.argmax((cdf >= q).to(torch.int64), dim=-1)
    return torch.gather(sorted_values, -1, idx.unsqueeze(-1)).squeeze(-1)


def particle_belief_metrics(
    particles: torch.Tensor,
    weights: torch.Tensor,
    target_state: torch.Tensor,
    object_mask: torch.Tensor,
    density_sigma: float,
    mass_radius: float,
    credible_levels: Iterable[float] = (0.5, 0.7, 0.9),
) -> Dict[str, float]:
    """
    Args:
        particles: [B, H, O, P, 6]
        weights: [B, H, O, P]
        target_state: [B, H, O, F]
        object_mask: [B, H, O]

    Metrics are computed on hidden/occluded object timesteps only.
    """
    true_pos = target_state[..., 0:3]
    hidden_mask = object_mask * (target_state[..., 7] > 0.5).float()
    if hidden_mask.sum() <= 0:
        return {
            "hidden_nll": float("nan"),
            "hidden_mass_radius": float("nan"),
            "hidden_expected_distance": float("nan"),
            "hidden_mean_error": float("nan"),
            "hidden_count": 0.0,
        }

    pos_particles = particles[..., 0:3]
    diff = pos_particles - true_pos.unsqueeze(-2)
    dist = torch.sqrt(torch.sum(diff * diff, dim=-1).clamp_min(1e-12))
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    sigma = float(density_sigma)
    norm_const = (2.0 * torch.pi) ** (-1.5) * (sigma ** -3)
    density = norm_const * torch.sum(weights * torch.exp(-0.5 * (dist / sigma) ** 2), dim=-1)
    nll = -torch.log(density.clamp_min(1e-12))
    mass = torch.sum(weights * (dist <= float(mass_radius)).float(), dim=-1)
    expected_distance = torch.sum(weights * dist, dim=-1)
    belief_mean = torch.sum(weights.unsqueeze(-1) * pos_particles, dim=-2)
    mean_error = torch.sqrt(torch.sum((belief_mean - true_pos) ** 2, dim=-1).clamp_min(1e-12))

    active = hidden_mask > 0.5
    out: Dict[str, float] = {
        "hidden_nll": _safe_mean(nll[active]),
        "hidden_mass_radius": _safe_mean(mass[active]),
        "hidden_expected_distance": _safe_mean(expected_distance[active]),
        "hidden_mean_error": _safe_mean(mean_error[active]),
        "hidden_count": float(hidden_mask.sum().item()),
    }

    # Radial credible-region approximation around the belief mean. This is not
    # a full HPD region, but it catches overconfident wrong beliefs in v1.
    center_dist = torch.sqrt(torch.sum((pos_particles - belief_mean.unsqueeze(-2)) ** 2, dim=-1).clamp_min(1e-12))
    truth_center_dist = mean_error
    for level in credible_levels:
        radius = _weighted_quantile(center_dist, weights, float(level))
        contained = (truth_center_dist <= radius).float()
        key = f"coverage_{int(round(float(level) * 100))}"
        out[key] = _safe_mean(contained[active])
    return out


def summarize_metric_rows(rows: list[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row.keys()})
    summary: Dict[str, float] = {}
    for key in keys:
        values = [row[key] for row in rows if key in row and row[key] == row[key]]
        if not values:
            continue
        summary[key] = float(sum(values) / len(values))
    return summary
