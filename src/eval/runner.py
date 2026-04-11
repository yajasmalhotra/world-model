from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from src.data.scene_generator import SceneConfig, SyntheticSceneGenerator
from src.eval.metrics import (
    counterfactual_locality,
    frame_mse,
    identity_consistency,
    occluded_position_rmse,
    reappearance_rmse,
    rollout_position_rmse,
    summarize_metrics,
)
from src.models.state_dynamics import apply_counterfactual


SCENE_CONFIG_KEYS = {
    "image_size",
    "seq_len",
    "obs_len",
    "min_objects",
    "max_objects",
    "min_occluders",
    "max_occluders",
    "velocity_scale",
    "object_size_min",
    "object_size_max",
    "occluder_layout",
}


def _scene_cfg_from_data_cfg(data_cfg: Dict[str, Any]) -> SceneConfig:
    subset = {k: v for k, v in data_cfg.items() if k in SCENE_CONFIG_KEYS}
    return SceneConfig(**subset)


def _render_predicted_frames(
    generator: SyntheticSceneGenerator,
    pred_state: torch.Tensor,
    future_mask: torch.Tensor,
    occluders: torch.Tensor,
) -> torch.Tensor:
    pred_np = pred_state.detach().cpu().numpy()
    mask_np = future_mask.detach().cpu().numpy()
    occ_np = occluders.detach().cpu().numpy()
    rendered = []
    for b in range(pred_np.shape[0]):
        seq = generator.render_sequence(pred_np[b], mask_np[b], occ_np[b])
        rendered.append(seq)
    frames = np.stack(rendered, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(frames).permute(0, 1, 4, 2, 3)


@torch.no_grad()
def evaluate_world_model(
    dynamics: torch.nn.Module,
    dataloader,
    device: torch.device,
    data_cfg: Dict[str, Any],
    encoder: Optional[torch.nn.Module] = None,
) -> Dict[str, float]:
    dynamics.eval()
    if encoder is not None:
        encoder.eval()
    scene_cfg = _scene_cfg_from_data_cfg(data_cfg)
    generator = SyntheticSceneGenerator(scene_cfg)

    rows: Dict[str, list[float]] = {
        "rollout_rmse": [],
        "occluded_rmse": [],
        "reappearance_rmse": [],
        "identity_consistency": [],
        "frame_mse": [],
        "counterfactual_locality": [],
    }

    for batch in dataloader:
        obs_frames = batch["obs_frames"].to(device)
        obs_state = batch["obs_state"].to(device)
        future_state = batch["future_state"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        future_mask = batch["future_mask"].to(device)
        occluders = batch["occluders"].to(device)

        object_mask = obs_mask[:, -1]
        horizon = future_state.shape[1]
        if encoder is None:
            init_state = obs_state[:, -1]
        else:
            init_state = encoder(obs_frames)
            # Keep static object traits from observation target for more stable rollouts.
            init_state[..., 6:] = obs_state[:, -1, :, 6:]

        pred_state = dynamics(init_state, object_mask, occluders, horizon=horizon)
        rows["rollout_rmse"].append(rollout_position_rmse(pred_state, future_state, future_mask))
        rows["occluded_rmse"].append(occluded_position_rmse(pred_state, future_state, future_mask))
        rows["reappearance_rmse"].append(reappearance_rmse(pred_state, future_state, future_mask))
        rows["identity_consistency"].append(identity_consistency(pred_state, future_state, future_mask))

        pred_frames = _render_predicted_frames(generator, pred_state, future_mask, occluders).to(device)
        rows["frame_mse"].append(frame_mse(pred_frames, batch["future_frames"].to(device)))

        cf_state = apply_counterfactual(init_state, object_idx=0, intervention={"vx": 0.04})
        cf_pred = dynamics(cf_state, object_mask, occluders, horizon=horizon)
        rows["counterfactual_locality"].append(counterfactual_locality(pred_state, cf_pred, future_mask, 0))

    return summarize_metrics(rows)


@torch.no_grad()
def evaluate_pixel_baseline(model: torch.nn.Module, dataloader, device: torch.device) -> Dict[str, float]:
    model.eval()
    mses: list[float] = []
    for batch in dataloader:
        obs_frames = batch["obs_frames"].to(device)
        future_frames = batch["future_frames"].to(device)
        pred = model(obs_frames, horizon=future_frames.shape[1])
        mse = torch.mean((pred - future_frames) ** 2).item()
        mses.append(float(mse))
    return {"pixel_rollout_mse": float(sum(mses) / max(len(mses), 1))}

