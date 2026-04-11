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
from src.models.state_dynamics import ObjectCentricDynamics
from src.train.losses import encoder_state_loss, state_rollout_loss
from src.train.pipeline import make_train_val_loaders
from src.train.utils import (
    append_metrics,
    get_device,
    init_run_dir,
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_summary,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train joint encoder + dynamics model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--encoder-ckpt", type=str, default=None)
    parser.add_argument("--dynamics-ckpt", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate_val(encoder, dynamics, loader, device: torch.device) -> dict[str, float]:
    encoder.eval()
    dynamics.eval()
    losses = []
    rmses = []
    for batch in loader:
        obs_frames = batch["obs_frames"].to(device)
        obs_state = batch["obs_state"].to(device)
        future_state = batch["future_state"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        future_mask = batch["future_mask"].to(device)
        occluders = batch["occluders"].to(device)

        target_init = obs_state[:, -1]
        object_mask = obs_mask[:, -1]
        pred_init = encoder(obs_frames)
        pred_rollout = dynamics(pred_init, object_mask, occluders, horizon=future_state.shape[1])

        init_loss = encoder_state_loss(pred_init, target_init, object_mask)["total"]
        rollout_loss = state_rollout_loss(pred_rollout, future_state, future_mask)["total"]
        losses.append(float((init_loss + rollout_loss).item()))

        rmse = torch.sqrt(((pred_rollout[..., 0:2] - future_state[..., 0:2]) ** 2).sum(dim=-1))
        rmse = (rmse * future_mask).sum() / future_mask.sum().clamp_min(1.0)
        rmses.append(float(rmse.item()))
    return {"val_loss": float(sum(losses) / max(len(losses), 1)), "val_rollout_rmse": float(sum(rmses) / max(len(rmses), 1))}


def maybe_load_weights(module, ckpt_path: str | None, device: torch.device, key: str) -> None:
    if not ckpt_path:
        return
    ckpt = load_checkpoint(ckpt_path, device)
    state = ckpt.get(key) or ckpt.get("model_state")
    module.load_state_dict(state, strict=False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    device = get_device(config["project"].get("device", "auto"))

    train_loader, val_loader = make_train_val_loaders(config)
    model_cfg = config["model"]
    encoder = PixelToStateEncoder(
        max_objects=int(model_cfg["max_objects"]),
        state_dim=int(model_cfg["state_dim"]),
        cnn_dim=int(model_cfg["encoder_cnn_dim"]),
        rnn_dim=int(model_cfg["encoder_rnn_dim"]),
        image_size=int(config["data"]["image_size"]),
    ).to(device)
    dynamics = ObjectCentricDynamics(
        state_dim=int(model_cfg["state_dim"]),
        dynamic_dim=int(model_cfg["dynamic_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        interaction_dim=int(model_cfg["interaction_dim"]),
        max_occluders=int(model_cfg["max_occluders"]),
    ).to(device)

    maybe_load_weights(encoder, args.encoder_ckpt, device, "encoder_state")
    maybe_load_weights(dynamics, args.dynamics_ckpt, device, "dynamics_state")

    params = list(encoder.parameters()) + list(dynamics.parameters())
    train_cfg = config["train"]
    optimizer = Adam(params, lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    epochs = int(train_cfg["epochs_joint"])
    grad_clip = float(train_cfg["grad_clip"])

    run_dir = init_run_dir(config["project"]["output_root"], "train_joint", config)
    best_metric = math.inf
    best_ckpt = run_dir / "checkpoints" / "best.pt"

    for epoch in range(1, epochs + 1):
        encoder.train()
        dynamics.train()
        train_losses = []
        for batch in train_loader:
            obs_frames = batch["obs_frames"].to(device)
            obs_state = batch["obs_state"].to(device)
            future_state = batch["future_state"].to(device)
            obs_mask = batch["obs_mask"].to(device)
            future_mask = batch["future_mask"].to(device)
            occluders = batch["occluders"].to(device)

            target_init = obs_state[:, -1]
            object_mask = obs_mask[:, -1]

            pred_init = encoder(obs_frames)
            pred_rollout = dynamics(pred_init, object_mask, occluders, horizon=future_state.shape[1])

            init_loss = encoder_state_loss(pred_init, target_init, object_mask)["total"]
            rollout_loss = state_rollout_loss(pred_rollout, future_state, future_mask)["total"]
            loss = init_loss + rollout_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_metrics = evaluate_val(encoder, dynamics, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(sum(train_losses) / max(len(train_losses), 1)),
            "val_loss": val_metrics["val_loss"],
            "val_rollout_rmse": val_metrics["val_rollout_rmse"],
        }
        append_metrics(run_dir, row)
        print(row)

        if val_metrics["val_rollout_rmse"] < best_metric:
            best_metric = val_metrics["val_rollout_rmse"]
            save_checkpoint(
                best_ckpt,
                {
                    "encoder_state": encoder.state_dict(),
                    "dynamics_state": dynamics.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_metric": best_metric,
                },
            )

    summary = {
        "run_type": "train_joint",
        "best_checkpoint": str(best_ckpt),
        "best_val_rollout_rmse": best_metric,
        "device": str(device),
        "encoder_checkpoint_used": args.encoder_ckpt,
        "dynamics_checkpoint_used": args.dynamics_ckpt,
    }
    save_summary(run_dir, summary)
    print(f"Done. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()

