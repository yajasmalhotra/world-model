#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.optim import Adam

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.pixel_baseline import PixelRolloutBaseline
from src.train.losses import pixel_baseline_loss
from src.train.pipeline import make_train_val_loaders
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
    parser = argparse.ArgumentParser(description="Train direct pixel baseline.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


@torch.no_grad()
def evaluate_val(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    for batch in loader:
        obs_frames = batch["obs_frames"].to(device)
        future_frames = batch["future_frames"].to(device)
        pred = model(obs_frames, horizon=future_frames.shape[1])
        losses.append(float(pixel_baseline_loss(pred, future_frames).item()))
    return {"val_mse": float(sum(losses) / max(len(losses), 1))}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    device = get_device(config["project"].get("device", "auto"))

    train_loader, val_loader = make_train_val_loaders(config)
    model = PixelRolloutBaseline().to(device)
    train_cfg = config["train"]
    optimizer = Adam(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    epochs = int(train_cfg["epochs_pixel_baseline"])
    grad_clip = float(train_cfg["grad_clip"])

    run_dir = init_run_dir(config["project"]["output_root"], "train_pixel_baseline", config)
    best_metric = math.inf
    best_ckpt = run_dir / "checkpoints" / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            obs_frames = batch["obs_frames"].to(device)
            future_frames = batch["future_frames"].to(device)
            pred = model(obs_frames, horizon=future_frames.shape[1])
            loss = pixel_baseline_loss(pred, future_frames)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))

        val_metrics = evaluate_val(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_mse": float(sum(losses) / max(len(losses), 1)),
            "val_mse": val_metrics["val_mse"],
        }
        append_metrics(run_dir, row)
        print(row)

        if val_metrics["val_mse"] < best_metric:
            best_metric = val_metrics["val_mse"]
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
        "run_type": "train_pixel_baseline",
        "best_checkpoint": str(best_ckpt),
        "best_val_mse": best_metric,
        "device": str(device),
    }
    save_summary(run_dir, summary)
    print(f"Done. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()

