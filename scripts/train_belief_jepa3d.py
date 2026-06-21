#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data3d.dataset3d import SyntheticScene3DDataset, collate_scenes3d
from src.models.belief_jepa3d import BeliefJEPA3D, belief_jepa_loss
from src.train.utils import append_metrics, get_device, init_run_dir, load_config, save_checkpoint, save_summary, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Belief-JEPA 3D latent future/belief predictor.")
    parser.add_argument("--config", type=str, default="configs/belief3d_smoke.yaml")
    parser.add_argument("--rgbd", action="store_true", help="Use RGB-D context frames instead of RGB only.")
    return parser.parse_args()


def make_loader(manifest_path: Path, data_cfg: Dict, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = SyntheticScene3DDataset(manifest_path, data_cfg)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_scenes3d)


def context_frames(batch: Dict, device: torch.device, rgbd: bool) -> torch.Tensor:
    frames = batch["obs_frames"].to(device)
    if not rgbd:
        return frames
    return torch.cat([frames, batch["obs_depth"].to(device)], dim=2)


def build_model(config: Dict, device: torch.device, rgbd: bool) -> BeliefJEPA3D:
    model_cfg = config["model3d"]
    data_cfg = config["data3d"]
    horizon = max(int(data_cfg["seq_len"]), int(data_cfg["obs_len"]) + 14, 24) - int(data_cfg["obs_len"])
    return BeliefJEPA3D(
        max_objects=int(model_cfg["max_objects"]),
        horizon=horizon,
        input_channels=4 if rgbd else 3,
        cnn_dim=int(model_cfg["encoder_cnn_dim"]),
        rnn_dim=int(model_cfg["encoder_rnn_dim"]),
        latent_dim=int(model_cfg.get("jepa_latent_dim", 64)),
        world_min=float(data_cfg["world_min"]),
        world_max=float(data_cfg["world_max"]),
        velocity_limit=float(model_cfg["velocity_limit"]),
        min_log_std=float(model_cfg["min_log_std"]),
        max_log_std=float(model_cfg.get("jepa_max_log_std", -0.8)),
    ).to(device)


@torch.no_grad()
def evaluate_val(model: BeliefJEPA3D, loader: DataLoader, device: torch.device, rgbd: bool) -> Dict[str, float]:
    model.eval()
    rows: list[Dict[str, float]] = []
    for batch in loader:
        frames = context_frames(batch, device, rgbd)
        future_state = batch["future_state"].to(device)
        future_mask = batch["future_mask"].to(device)
        outputs = model(frames, future_state=future_state)
        losses = belief_jepa_loss(outputs, future_state, future_mask)
        rows.append({key: float(value.item()) for key, value in losses.items()})
    if not rows:
        return {}
    keys = rows[0].keys()
    return {f"val_{key}": float(sum(row[key] for row in rows) / len(rows)) for key in keys}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    device = get_device(config["project"].get("device", "auto"))
    data_cfg = config["data3d"]
    train_cfg = config["train_belief3d"]
    manifest_dir = Path(data_cfg["manifest_dir"])
    batch_size = int(data_cfg.get("batch_size", config["eval"].get("batch_size", 8)))

    train_loader = make_loader(manifest_dir / "train.jsonl", data_cfg, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(manifest_dir / "val.jsonl", data_cfg, batch_size=batch_size, shuffle=False)
    model = build_model(config, device, rgbd=args.rgbd)
    optimizer = Adam(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    epochs = int(train_cfg["epochs"])
    grad_clip = float(train_cfg["grad_clip"])
    run_name = "train_belief_jepa3d_rgbd" if args.rgbd else "train_belief_jepa3d"
    run_dir = init_run_dir(config["project"]["output_root"], run_name, config)
    best_metric = math.inf
    best_ckpt = run_dir / "checkpoints" / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            frames = context_frames(batch, device, args.rgbd)
            future_state = batch["future_state"].to(device)
            future_mask = batch["future_mask"].to(device)
            outputs = model(frames, future_state=future_state)
            losses = belief_jepa_loss(outputs, future_state, future_mask)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(losses["total"].item()))

        val_metrics = evaluate_val(model, val_loader, device, args.rgbd)
        row = {"epoch": epoch, "train_loss": float(sum(train_losses) / max(len(train_losses), 1)), **val_metrics}
        append_metrics(run_dir, row)
        print(row)
        metric = float(val_metrics.get("val_pos_rmse", math.inf))
        if metric < best_metric:
            best_metric = metric
            save_checkpoint(
                best_ckpt,
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "rgbd": bool(args.rgbd),
                },
            )

    save_summary(
        run_dir,
        {
            "run_type": run_name,
            "best_checkpoint": str(best_ckpt),
            "best_val_pos_rmse": best_metric,
            "device": str(device),
            "rgbd": bool(args.rgbd),
        },
    )
    print(f"Done. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
