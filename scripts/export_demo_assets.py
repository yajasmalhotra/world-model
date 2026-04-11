#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.scene_generator import SceneConfig, SyntheticSceneGenerator
from src.models.encoder import PixelToStateEncoder
from src.models.state_dynamics import ObjectCentricDynamics, apply_counterfactual
from src.train.utils import get_device, load_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GIF assets for report/demo.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--joint-ckpt", type=str, default=None)
    parser.add_argument("--dynamics-ckpt", type=str, default=None)
    parser.add_argument("--encoder-ckpt", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--output-dir", type=str, default="results/demo_assets")
    return parser.parse_args()


def latest_checkpoint(pattern: str) -> str | None:
    candidates = sorted(Path("runs").glob(pattern))
    if not candidates:
        return None
    return str(candidates[-1])


def to_scene_config(data_cfg: dict) -> SceneConfig:
    keys = {
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
    subset = {k: v for k, v in data_cfg.items() if k in keys}
    return SceneConfig(**subset)


def load_models(config: dict, device: torch.device, args: argparse.Namespace):
    model_cfg = config["model"]
    dynamics = ObjectCentricDynamics(
        state_dim=int(model_cfg["state_dim"]),
        dynamic_dim=int(model_cfg["dynamic_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        interaction_dim=int(model_cfg["interaction_dim"]),
        max_occluders=int(model_cfg["max_occluders"]),
    ).to(device)
    encoder = PixelToStateEncoder(
        max_objects=int(model_cfg["max_objects"]),
        state_dim=int(model_cfg["state_dim"]),
        cnn_dim=int(model_cfg["encoder_cnn_dim"]),
        rnn_dim=int(model_cfg["encoder_rnn_dim"]),
        image_size=int(config["data"]["image_size"]),
    ).to(device)

    joint_ckpt = args.joint_ckpt or latest_checkpoint("*_train_joint/checkpoints/best.pt")
    dynamics_ckpt = args.dynamics_ckpt or latest_checkpoint("*_train_dynamics/checkpoints/best.pt")
    encoder_ckpt = args.encoder_ckpt or latest_checkpoint("*_train_encoder/checkpoints/best.pt")

    if joint_ckpt:
        joint = load_checkpoint(joint_ckpt, device)
        dynamics.load_state_dict(joint["dynamics_state"], strict=False)
        encoder.load_state_dict(joint["encoder_state"], strict=False)
    else:
        if dynamics_ckpt:
            dynamics_state = load_checkpoint(dynamics_ckpt, device)
            dynamics.load_state_dict(dynamics_state.get("model_state", dynamics_state.get("dynamics_state")), strict=False)
        if encoder_ckpt:
            encoder_state = load_checkpoint(encoder_ckpt, device)
            encoder.load_state_dict(encoder_state.get("model_state", encoder_state.get("encoder_state")), strict=False)
    dynamics.eval()
    encoder.eval()
    return encoder, dynamics


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate([left, right], axis=1)


def stack_triplet(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.concatenate([a, b, c], axis=1)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device(config["project"].get("device", "auto"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_cfg = to_scene_config(config["data"])
    generator = SyntheticSceneGenerator(scene_cfg)
    encoder, dynamics = load_models(config, device, args)

    for seed in args.seeds:
        sample = generator.generate(seed=seed)
        obs_len = scene_cfg.obs_len

        obs_frames = torch.from_numpy(sample["frames"][:obs_len].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        obs_frames = obs_frames.unsqueeze(0).to(device)
        obs_state = torch.from_numpy(sample["state"][:obs_len]).unsqueeze(0).to(device)
        object_mask = torch.from_numpy(sample["object_mask"][obs_len - 1]).unsqueeze(0).to(device)
        occluders = torch.from_numpy(sample["occluders"]).unsqueeze(0).to(device)
        horizon = sample["state"].shape[0] - obs_len

        with torch.no_grad():
            init_state = encoder(obs_frames)
            init_state[..., 6:] = obs_state[:, -1, :, 6:]
            pred = dynamics(init_state, object_mask, occluders, horizon=horizon)

            cf_init = apply_counterfactual(init_state, object_idx=0, intervention={"vx": 0.04, "vy": -0.02})
            cf_pred = dynamics(cf_init, object_mask, occluders, horizon=horizon)

        mask_future = sample["object_mask"][obs_len:]
        pred_frames = generator.render_sequence(pred.squeeze(0).cpu().numpy(), mask_future, sample["occluders"])
        cf_frames = generator.render_sequence(cf_pred.squeeze(0).cpu().numpy(), mask_future, sample["occluders"])
        gt_frames = sample["frames"][obs_len:]

        preview_frames: List[np.ndarray] = []
        for t in range(horizon):
            preview_frames.append(stack_triplet(gt_frames[t], pred_frames[t], cf_frames[t]))

        gif_path = out_dir / f"seed_{seed}_gt_pred_cf.gif"
        imageio.mimsave(gif_path, preview_frames, fps=6)
        print(f"Wrote {gif_path}")

        compare_frames = [side_by_side(gt_frames[t], pred_frames[t]) for t in range(horizon)]
        compare_path = out_dir / f"seed_{seed}_gt_pred.gif"
        imageio.mimsave(compare_path, compare_frames, fps=6)
        print(f"Wrote {compare_path}")


if __name__ == "__main__":
    main()

