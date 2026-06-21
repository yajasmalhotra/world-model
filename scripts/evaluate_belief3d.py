#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data3d.dataset3d import SyntheticScene3DDataset, collate_scenes3d
from src.eval.belief_metrics import particle_belief_metrics, summarize_metric_rows
from src.models.belief_encoder3d import ImageToBeliefEncoder3D
from src.models.belief_state import ParticleBeliefConfig, rollout_particle_belief, rollout_particle_belief_from_gaussian
from src.train.utils import append_metrics, get_device, init_run_dir, load_checkpoint, load_config, save_summary, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 3D hidden-trajectory belief calibration.")
    parser.add_argument("--config", type=str, default="configs/belief3d.yaml")
    parser.add_argument("--split", type=str, default=None, help="Optional single split override.")
    parser.add_argument("--mode", type=str, default="physics", choices=["physics", "image", "all"])
    parser.add_argument("--encoder-ckpt", type=str, default=None)
    return parser.parse_args()


def make_loader(manifest_path: Path, data_cfg: Dict, batch_size: int) -> DataLoader:
    dataset = SyntheticScene3DDataset(manifest_path, data_cfg)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_scenes3d)


def latest_checkpoint(pattern: str) -> Optional[str]:
    candidates = sorted(Path("runs").glob(pattern))
    if not candidates:
        return None
    return str(candidates[-1])


def load_image_encoder(config: Dict, device: torch.device, ckpt_path: Optional[str]) -> ImageToBeliefEncoder3D:
    model_cfg = config["model3d"]
    data_cfg = config["data3d"]
    model = ImageToBeliefEncoder3D(
        max_objects=int(model_cfg["max_objects"]),
        cnn_dim=int(model_cfg["encoder_cnn_dim"]),
        rnn_dim=int(model_cfg["encoder_rnn_dim"]),
        world_min=float(data_cfg["world_min"]),
        world_max=float(data_cfg["world_max"]),
        velocity_limit=float(model_cfg["velocity_limit"]),
        min_log_std=float(model_cfg["min_log_std"]),
        max_log_std=float(model_cfg["max_log_std"]),
    ).to(device)
    if ckpt_path:
        ckpt = load_checkpoint(ckpt_path, device)
        model.load_state_dict(ckpt["model_state"], strict=False)
    return model


def target_object_mask(batch: Dict, future_mask: torch.Tensor) -> Optional[torch.Tensor]:
    metadata = batch.get("metadata")
    if not metadata:
        return None
    target_mask = torch.zeros_like(future_mask)
    found = False
    for b_idx, item in enumerate(metadata):
        target = item.get("target") if isinstance(item, dict) else None
        if not target:
            continue
        obj_idx = int(target.get("object_index", -1))
        if 0 <= obj_idx < future_mask.shape[-1]:
            target_mask[b_idx, :, obj_idx] = future_mask[b_idx, :, obj_idx]
            found = True
    return target_mask if found else None


@torch.no_grad()
def evaluate_split(
    loader: DataLoader,
    config: Dict,
    device: torch.device,
    mode: str,
    encoder: Optional[ImageToBeliefEncoder3D] = None,
) -> Dict[str, float]:
    data_cfg = config["data3d"]
    belief_cfg = config["belief"]
    particle_cfg = ParticleBeliefConfig.from_config(belief_cfg, data_cfg)
    rows: list[Dict[str, float]] = []
    if encoder is not None:
        encoder.eval()

    for batch in loader:
        obs_frames = batch["obs_frames"].to(device)
        obs_state = batch["obs_state"].to(device)
        future_state = batch["future_state"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        future_mask = batch["future_mask"].to(device)
        init_state = obs_state[:, -1]
        object_mask = obs_mask[:, -1]
        horizon = future_state.shape[1]
        if mode == "image":
            if encoder is None:
                raise RuntimeError("Image mode requires an encoder checkpoint.")
            outputs = encoder(obs_frames)
            particles, weights = rollout_particle_belief_from_gaussian(
                outputs["mean"],
                outputs["log_std"],
                init_state,
                object_mask,
                horizon=horizon,
                cfg=particle_cfg,
            )
        else:
            particles, weights = rollout_particle_belief(init_state, object_mask, horizon=horizon, cfg=particle_cfg)
        metrics = particle_belief_metrics(
            particles,
            weights,
            future_state,
            future_mask,
            density_sigma=float(belief_cfg["density_sigma"]),
            mass_radius=float(belief_cfg["mass_radius"]),
            credible_levels=belief_cfg.get("credible_levels", [0.5, 0.7, 0.9]),
        )
        target_mask = target_object_mask(batch, future_mask)
        if target_mask is not None:
            target_metrics = particle_belief_metrics(
                particles,
                weights,
                future_state,
                target_mask,
                density_sigma=float(belief_cfg["density_sigma"]),
                mass_radius=float(belief_cfg["mass_radius"]),
                credible_levels=belief_cfg.get("credible_levels", [0.5, 0.7, 0.9]),
            )
            metrics.update({f"target_{key}": value for key, value in target_metrics.items()})
        rows.append(metrics)
    return summarize_metric_rows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    device = get_device(config["project"].get("device", "auto"))
    if args.mode in ("image", "all") and args.encoder_ckpt is None:
        args.encoder_ckpt = latest_checkpoint("*_train_belief3d_encoder/checkpoints/best.pt")
    manifest_dir = Path(config["data3d"]["manifest_dir"])
    splits = [args.split] if args.split else list(config["eval"]["slices"])
    batch_size = int(config["eval"].get("batch_size", 8))

    run_dir = init_run_dir(config["project"]["output_root"], "evaluate_belief3d", config)
    summary: Dict[str, object] = {
        "run_type": "evaluate_belief3d",
        "device": str(device),
        "encoder_checkpoint": args.encoder_ckpt,
        "splits": {},
    }
    image_encoder = (
        load_image_encoder(config, device, args.encoder_ckpt)
        if args.mode in ("image", "all") and args.encoder_ckpt is not None
        else None
    )

    for split in splits:
        manifest_path = manifest_dir / f"{split}.jsonl"
        if not manifest_path.exists():
            print(f"Skipping missing split manifest: {manifest_path}")
            continue
        loader = make_loader(manifest_path, config["data3d"], batch_size=batch_size)
        if args.mode in ("physics", "all"):
            metrics = evaluate_split(loader, config, device, mode="physics")
            row = {"split": split, "mode": "physics_particle_belief", **metrics}
            append_metrics(run_dir, row)
            summary["splits"][f"physics::{split}"] = metrics
            print(row)
        if args.mode in ("image", "all"):
            if image_encoder is None:
                print("Skipping image mode because no encoder checkpoint was found.")
            else:
                metrics = evaluate_split(loader, config, device, mode="image", encoder=image_encoder)
                row = {"split": split, "mode": "image_to_belief", **metrics}
                append_metrics(run_dir, row)
                summary["splits"][f"image::{split}"] = metrics
                print(row)

    save_summary(run_dir, summary)
    print(f"Done. 3D belief evaluation artifacts in: {run_dir}")


if __name__ == "__main__":
    main()
