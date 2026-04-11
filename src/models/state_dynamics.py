from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class ObjectCentricDynamics(nn.Module):
    """
    Object-centric recurrent dynamics model over fixed slots.

    State layout:
    [x, y, vx, vy, visible, occluded, size, shape_id, color_id, object_id]
    """

    def __init__(
        self,
        state_dim: int = 10,
        dynamic_dim: int = 4,
        hidden_dim: int = 128,
        interaction_dim: int = 64,
        max_occluders: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dynamic_dim = dynamic_dim
        self.hidden_dim = hidden_dim
        self.interaction_dim = interaction_dim
        self.max_occluders = max_occluders
        self.static_dim = state_dim - 6

        self.state_to_hidden = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, interaction_dim),
            nn.ReLU(),
            nn.Linear(interaction_dim, interaction_dim),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(max_occluders * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_cell = nn.GRUCell(hidden_dim + interaction_dim + hidden_dim, hidden_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dynamic_dim),
        )

    @staticmethod
    def compute_visibility(positions: torch.Tensor, occluders: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [B, O, 2] in [0, 1]
            occluders: [B, K, 4] as x0/y0/x1/y1
            mask: [B, O]
        Returns:
            visibility: [B, O]
        """
        bsz, n_obj, _ = positions.shape
        visibility = torch.zeros((bsz, n_obj), dtype=positions.dtype, device=positions.device)
        for b in range(bsz):
            for o in range(n_obj):
                if mask[b, o] < 0.5:
                    continue
                x = positions[b, o, 0]
                y = positions[b, o, 1]
                occ = occluders[b]
                in_x = (x >= occ[:, 0]) & (x <= occ[:, 2])
                in_y = (y >= occ[:, 1]) & (y <= occ[:, 3])
                is_occ = torch.any(in_x & in_y)
                visibility[b, o] = torch.where(
                    is_occ, torch.zeros_like(x, device=x.device), torch.ones_like(x, device=x.device)
                )
        return visibility

    def _interaction_messages(self, hidden: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
        bsz, n_obj, hidden_dim = hidden.shape
        hi = hidden.unsqueeze(2).expand(bsz, n_obj, n_obj, hidden_dim)
        hj = hidden.unsqueeze(1).expand(bsz, n_obj, n_obj, hidden_dim)
        pair = torch.cat([hi, hj], dim=-1)
        messages = self.pair_mlp(pair)

        device = hidden.device
        eye = torch.eye(n_obj, device=device).unsqueeze(0).unsqueeze(-1)
        messages = messages * (1.0 - eye)

        sender_mask = object_mask.unsqueeze(1).unsqueeze(-1)
        receiver_mask = object_mask.unsqueeze(2).unsqueeze(-1)
        messages = messages * sender_mask * receiver_mask
        return messages.sum(dim=2)

    def rollout(
        self,
        initial_state: torch.Tensor,
        object_mask: torch.Tensor,
        occluders: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """
        Args:
            initial_state: [B, O, F]
            object_mask: [B, O]
            occluders: [B, K, 4]
            horizon: number of future timesteps
        Returns:
            predicted_states: [B, H, O, F]
        """
        bsz, n_obj, _ = initial_state.shape
        hidden = self.state_to_hidden(initial_state)
        dynamic = initial_state[:, :, : self.dynamic_dim]
        static = initial_state[:, :, 6:]
        context = self.context_mlp(occluders.reshape(bsz, -1))

        outputs = []
        for _ in range(horizon):
            msg = self._interaction_messages(hidden, object_mask)
            context_expanded = context.unsqueeze(1).expand(bsz, n_obj, self.hidden_dim)
            rnn_in = torch.cat([hidden, msg, context_expanded], dim=-1)

            hidden = self.update_cell(
                rnn_in.reshape(bsz * n_obj, -1),
                hidden.reshape(bsz * n_obj, -1),
            ).reshape(bsz, n_obj, -1)
            delta = self.delta_head(hidden)

            dynamic_next = dynamic + delta
            dynamic = torch.cat(
                [
                    dynamic_next[..., 0:2].clamp(0.0, 1.0),
                    dynamic_next[..., 2:4].clamp(-0.2, 0.2),
                ],
                dim=-1,
            )

            vis = self.compute_visibility(dynamic[..., 0:2], occluders, object_mask)
            occ = 1.0 - vis
            state = torch.cat([dynamic, vis.unsqueeze(-1), occ.unsqueeze(-1), static], dim=-1)
            state = state * object_mask.unsqueeze(-1)
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    def forward(
        self,
        initial_state: torch.Tensor,
        object_mask: torch.Tensor,
        occluders: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        return self.rollout(initial_state, object_mask, occluders, horizon)


def apply_counterfactual(
    initial_state: torch.Tensor,
    object_idx: int,
    intervention: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    intervention = intervention or {}
    out = initial_state.clone()
    if "x" in intervention:
        out[:, object_idx, 0] = out[:, object_idx, 0] + float(intervention["x"])
    if "y" in intervention:
        out[:, object_idx, 1] = out[:, object_idx, 1] + float(intervention["y"])
    if "vx" in intervention:
        out[:, object_idx, 2] = out[:, object_idx, 2] + float(intervention["vx"])
    if "vy" in intervention:
        out[:, object_idx, 3] = out[:, object_idx, 3] + float(intervention["vy"])
    out[..., 0:2] = out[..., 0:2].clamp(0.0, 1.0)
    out[..., 2:4] = out[..., 2:4].clamp(-0.2, 0.2)
    return out
