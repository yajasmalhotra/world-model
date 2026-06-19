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
from src.models.belief_encoder3d import ImageToBeliefEncoder3D, gaussian_belief_loss
from src.train.utils import (
    append_metrics,
    get_device,
    init_run_dir,
    load_config,
    save_checkpoint,
    save_summary,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised 3D image-to-belief encoder.")
    parser.add_argument("--config", type=str, default="configs/belief3d.yaml")
    return parser.parse_args()


def make_loader(manifest_path: Path, data_cfg: Dict, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = SyntheticScene3DDataset(manifest_path, data_cfg)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_scenes3d)


def build_model(config: Dict, device: torch.device) -> ImageToBeliefEncoder3D:
    model_cfg = config["model3d"]
    data_cfg = config["data3d"]
    return ImageToBeliefEncoder3D(
        max_objects=int(model_cfg["max_objects"]),
        cnn_dim=int(model_cfg["encoder_cnn_dim"]),
        rnn_dim=int(model_cfg["encoder_rnn_dim"]),
        world_min=float(data_cfg["world_min"]),
        world_max=float(data_cfg["world_max"]),
        velocity_limit=float(model_cfg["velocity_limit"]),
        min_log_std=float(model_cfg["min_log_std"]),
        max_log_std=float(model_cfg["max_log_std"]),
    ).to(device)


@torch.no_grad()
def evaluate_val(model: ImageToBeliefEncoder3D, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    rows: list[Dict[str, float]] = []
    for batch in loader:
        obs_frames = batch["obs_frames"].to(device)
        obs_state = batch["obs_state"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        target = obs_state[:, -1]
        object_mask = obs_mask[:, -1]
        outputs = model(obs_frames)
        loss_dict = gaussian_belief_loss(outputs, target, object_mask)
        rows.append({key: float(value.item()) for key, value in loss_dict.items()})
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
    model = build_model(config, device)
    optimizer = Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(train_cfg["epochs"])
    grad_clip = float(train_cfg["grad_clip"])
    run_dir = init_run_dir(config["project"]["output_root"], "train_belief3d_encoder", config)
    best_metric = math.inf
    best_ckpt = run_dir / "checkpoints" / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            obs_frames = batch["obs_frames"].to(device)
            obs_state = batch["obs_state"].to(device)
            obs_mask = batch["obs_mask"].to(device)
            target = obs_state[:, -1]
            object_mask = obs_mask[:, -1]
            outputs = model(obs_frames)
            loss_dict = gaussian_belief_loss(outputs, target, object_mask)
            loss = loss_dict["total"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_metrics = evaluate_val(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(sum(train_losses) / max(len(train_losses), 1)),
            **val_metrics,
        }
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
                },
            )

    summary = {
        "run_type": "train_belief3d_encoder",
        "best_checkpoint": str(best_ckpt),
        "best_val_pos_rmse": best_metric,
        "device": str(device),
    }
    save_summary(run_dir, summary)
    print(f"Done. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
