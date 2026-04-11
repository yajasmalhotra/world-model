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

from src.models.encoder import PixelToStateEncoder
from src.train.losses import encoder_state_loss
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
    parser = argparse.ArgumentParser(description="Train pixel-to-state encoder.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


@torch.no_grad()
def evaluate_val(model: PixelToStateEncoder, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    rmses = []
    for batch in loader:
        obs_frames = batch["obs_frames"].to(device)
        obs_state = batch["obs_state"].to(device)
        obs_mask = batch["obs_mask"].to(device)

        target = obs_state[:, -1]
        object_mask = obs_mask[:, -1]
        pred = model(obs_frames)
        loss_dict = encoder_state_loss(pred, target, object_mask)
        losses.append(float(loss_dict["total"].item()))
        rmse = torch.sqrt(((pred[..., 0:2] - target[..., 0:2]) ** 2).sum(dim=-1))
        rmse = (rmse * object_mask).sum() / object_mask.sum().clamp_min(1.0)
        rmses.append(float(rmse.item()))
    return {"val_loss": float(sum(losses) / max(len(losses), 1)), "val_pos_rmse": float(sum(rmses) / max(len(rmses), 1))}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    device = get_device(config["project"].get("device", "auto"))

    train_loader, val_loader = make_train_val_loaders(config)
    model_cfg = config["model"]
    model = PixelToStateEncoder(
        max_objects=int(model_cfg["max_objects"]),
        state_dim=int(model_cfg["state_dim"]),
        cnn_dim=int(model_cfg["encoder_cnn_dim"]),
        rnn_dim=int(model_cfg["encoder_rnn_dim"]),
        image_size=int(config["data"]["image_size"]),
    ).to(device)

    train_cfg = config["train"]
    optimizer = Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(train_cfg["epochs_encoder"])
    grad_clip = float(train_cfg["grad_clip"])

    run_dir = init_run_dir(config["project"]["output_root"], "train_encoder", config)
    best_metric = math.inf
    best_ckpt = run_dir / "checkpoints" / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            obs_frames = batch["obs_frames"].to(device)
            obs_state = batch["obs_state"].to(device)
            obs_mask = batch["obs_mask"].to(device)

            target = obs_state[:, -1]
            object_mask = obs_mask[:, -1]
            pred = model(obs_frames)
            loss_dict = encoder_state_loss(pred, target, object_mask)
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
            "val_loss": val_metrics["val_loss"],
            "val_pos_rmse": val_metrics["val_pos_rmse"],
        }
        append_metrics(run_dir, row)
        print(row)

        if val_metrics["val_pos_rmse"] < best_metric:
            best_metric = val_metrics["val_pos_rmse"]
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
        "run_type": "train_encoder",
        "best_checkpoint": str(best_ckpt),
        "best_val_pos_rmse": best_metric,
        "device": str(device),
    }
    save_summary(run_dir, summary)
    print(f"Done. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()

