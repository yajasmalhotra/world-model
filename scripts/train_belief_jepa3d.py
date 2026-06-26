#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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
from src.eval.counterfactual import move_boxes_to_far_corner
from src.models.belief_jepa3d import BeliefJEPA3D, belief_jepa_loss
from src.train.utils import append_metrics, get_device, init_run_dir, load_config, save_checkpoint, save_summary, set_seed


TARGET_ENCODER_KIND = "bidirectional_temporal_state6_occ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Belief-JEPA 3D latent future/belief predictor.")
    parser.add_argument("--config", type=str, default="configs/belief3d_smoke.yaml")
    parser.add_argument("--rgbd", action="store_true", help="Use RGB-D context frames instead of RGB only.")
    parser.add_argument("--no-ema", action="store_true", help="Use online target latents as a no-EMA ablation.")
    parser.add_argument("--sigreg-weight", type=float, default=None, help="Override train_belief3d.sigreg_weight.")
    parser.add_argument("--sigreg-sketches", type=int, default=None, help="Override train_belief3d.sigreg_sketches.")
    parser.add_argument("--sigreg-scale", type=float, default=None, help="Override train_belief3d.sigreg_scale.")
    return parser.parse_args()


def make_loader(manifest_path: Path, data_cfg: Dict, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = SyntheticScene3DDataset(manifest_path, data_cfg)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_scenes3d)


def context_frames(batch: Dict, device: torch.device, rgbd: bool) -> torch.Tensor:
    frames = batch["obs_frames"].to(device)
    if not rgbd:
        return frames
    return torch.cat([frames, batch["obs_depth"].to(device)], dim=2)


def structured_context(batch: Dict, device: torch.device, enabled: bool) -> Dict[str, torch.Tensor] | None:
    if not enabled:
        return None
    return {
        "obs_state": batch["obs_state"].to(device),
        "obs_mask": batch["obs_mask"].to(device),
        "visual_occluders": batch["visual_occluders"].to(device),
        "physical_obstacles": batch["physical_obstacles"].to(device),
        "solid_screens": batch["solid_screens"].to(device),
    }


def visual_counterfactual_structured_context(
    base: Dict[str, torch.Tensor] | None,
    world_min: float,
    world_max: float,
) -> Dict[str, torch.Tensor] | None:
    if base is None:
        return None
    changed = {key: value.clone() for key, value in base.items()}
    changed["visual_occluders"] = move_boxes_to_far_corner(
        changed["visual_occluders"],
        world_min=world_min,
        world_max=world_max,
    )
    return changed


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
        mixture_components=int(model_cfg.get("jepa_mixture_components", 3)),
        structured_context=bool(model_cfg.get("jepa_structured_context", True)),
        structured_dim=int(model_cfg.get("jepa_structured_dim", 64)),
        visual_geometry_weight=float(model_cfg.get("jepa_visual_geometry_weight", 1.0)),
        world_min=float(data_cfg["world_min"]),
        world_max=float(data_cfg["world_max"]),
        velocity_limit=float(model_cfg["velocity_limit"]),
        min_log_std=float(model_cfg["min_log_std"]),
        max_log_std=float(model_cfg.get("jepa_max_log_std", -0.8)),
    ).to(device)


def loss_weights(train_cfg: Dict) -> Dict[str, float | int]:
    return {
        "latent_weight": float(train_cfg.get("latent_weight", 1.0)),
        "belief_weight": float(train_cfg.get("belief_weight", 0.5)),
        "mixture_belief_weight": float(train_cfg.get("mixture_belief_weight", 0.25)),
        "target_recon_weight": float(train_cfg.get("target_recon_weight", 0.1)),
        "sigreg_weight": float(train_cfg.get("sigreg_weight", 0.0)),
        "sigreg_sketches": int(train_cfg.get("sigreg_sketches", 16)),
        "sigreg_scale": float(train_cfg.get("sigreg_scale", 1.0)),
    }


def visual_invariance_loss(
    base_outputs: Dict[str, torch.Tensor],
    visual_outputs: Dict[str, torch.Tensor],
    future_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    steps = min(base_outputs["mean"].shape[1], visual_outputs["mean"].shape[1], future_mask.shape[1])
    mask = future_mask[:, :steps].unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    latent_mse = (((base_outputs["pred_latent"][:, :steps] - visual_outputs["pred_latent"][:, :steps]) ** 2) * mask).sum() / denom
    mean_mse = (((base_outputs["mean"][:, :steps] - visual_outputs["mean"][:, :steps]) ** 2) * mask).sum() / denom
    std_mse = (((base_outputs["log_std"][:, :steps] - visual_outputs["log_std"][:, :steps]) ** 2) * mask).sum() / denom
    mixture_mean_mse = (
        ((base_outputs["mixture_mean"][:, :steps] - visual_outputs["mixture_mean"][:, :steps]) ** 2)
        * mask.unsqueeze(-2)
    ).sum() / denom
    mixture_std_mse = (
        ((base_outputs["mixture_log_std"][:, :steps] - visual_outputs["mixture_log_std"][:, :steps]) ** 2)
        * mask.unsqueeze(-2)
    ).sum() / denom
    mixture_logits_mse = (
        ((base_outputs["mixture_logits"][:, :steps] - visual_outputs["mixture_logits"][:, :steps]) ** 2)
        * future_mask[:, :steps].unsqueeze(-1)
    ).sum() / future_mask[:, :steps].sum().clamp_min(1.0)
    total = latent_mse + mean_mse + std_mse + 0.25 * (mixture_mean_mse + mixture_std_mse + mixture_logits_mse)
    return {
        "visual_invariance": total,
        "visual_invariance_latent": latent_mse,
        "visual_invariance_mean": mean_mse,
    }


def apply_cli_overrides(config: Dict, args: argparse.Namespace) -> Dict:
    updated = copy.deepcopy(config)
    train_cfg = updated["train_belief3d"]
    if args.sigreg_weight is not None:
        train_cfg["sigreg_weight"] = float(args.sigreg_weight)
    if args.sigreg_sketches is not None:
        train_cfg["sigreg_sketches"] = int(args.sigreg_sketches)
    if args.sigreg_scale is not None:
        train_cfg["sigreg_scale"] = float(args.sigreg_scale)
    return updated


def scalar_losses(losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().cpu().item()) for key, value in losses.items()}


def average_rows(rows: list[Dict[str, float]], prefix: str = "") -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row.keys()})
    averages: Dict[str, float] = {}
    for key in keys:
        values = [row[key] for row in rows if key in row]
        averages[f"{prefix}{key}"] = float(sum(values) / max(len(values), 1))
    return averages


def context_encoder_name(rgbd: bool, structured_enabled: bool) -> str:
    visual = "rgbd" if rgbd else "rgb"
    return f"{visual}_state_geometry" if structured_enabled else visual


@torch.no_grad()
def evaluate_val(
    model: BeliefJEPA3D,
    loader: DataLoader,
    device: torch.device,
    rgbd: bool,
    ema_enabled: bool,
    train_cfg: Dict,
) -> Dict[str, float]:
    model.eval()
    rows: list[Dict[str, float]] = []
    for batch in loader:
        frames = context_frames(batch, device, rgbd)
        struct = structured_context(batch, device, bool(model.use_structured_context))
        future_state = batch["future_state"].to(device)
        future_mask = batch["future_mask"].to(device)
        outputs = model(frames, future_state=future_state, structured_context=struct, use_ema_target=ema_enabled)
        losses = belief_jepa_loss(outputs, future_state, future_mask, **loss_weights(train_cfg))
        rows.append(scalar_losses(losses))
    metrics = average_rows(rows, prefix="val_")
    metrics["val_ema_online_drift"] = float(model.ema_online_drift().detach().cpu().item())
    return metrics


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    set_seed(int(config["project"]["seed"]))
    device = get_device(config["project"].get("device", "auto"))
    data_cfg = config["data3d"]
    train_cfg = config["train_belief3d"]
    ema_enabled = not bool(args.no_ema)
    ema_decay = float(train_cfg.get("ema_decay", 0.99))
    ema_update_after_step = int(train_cfg.get("ema_update_after_step", 0))
    loss_cfg = loss_weights(train_cfg)
    manifest_dir = Path(data_cfg["manifest_dir"])
    batch_size = int(data_cfg.get("batch_size", config["eval"].get("batch_size", 8)))

    train_loader = make_loader(manifest_dir / "train.jsonl", data_cfg, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(manifest_dir / "val.jsonl", data_cfg, batch_size=batch_size, shuffle=False)
    model = build_model(config, device, rgbd=args.rgbd)
    model.sync_ema_target_encoder()
    optimizer = Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(train_cfg["epochs"])
    grad_clip = float(train_cfg["grad_clip"])
    run_name = "train_belief_jepa3d_rgbd" if args.rgbd else "train_belief_jepa3d"
    if not ema_enabled:
        run_name = f"{run_name}_noema"
    if float(loss_cfg["sigreg_weight"]) <= 0.0:
        run_name = f"{run_name}_nosigreg"
    run_dir = init_run_dir(config["project"]["output_root"], run_name, config)
    best_metric = math.inf
    best_ckpt = run_dir / "checkpoints" / "best.pt"
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_rows: list[Dict[str, float]] = []
        for batch in train_loader:
            frames = context_frames(batch, device, args.rgbd)
            struct = structured_context(batch, device, bool(model.use_structured_context))
            future_state = batch["future_state"].to(device)
            future_mask = batch["future_mask"].to(device)
            outputs = model(frames, future_state=future_state, structured_context=struct, use_ema_target=ema_enabled)
            losses = belief_jepa_loss(outputs, future_state, future_mask, **loss_cfg)
            visual_invariance_weight = float(train_cfg.get("visual_invariance_weight", 0.0))
            if visual_invariance_weight > 0.0 and bool(model.use_structured_context):
                visual_struct = visual_counterfactual_structured_context(
                    struct,
                    world_min=float(data_cfg.get("world_min", -1.0)),
                    world_max=float(data_cfg.get("world_max", 1.0)),
                )
                visual_outputs = model(
                    frames,
                    structured_context=visual_struct,
                    use_ema_target=ema_enabled,
                    include_target_reconstruction=False,
                )
                invariance_losses = visual_invariance_loss(outputs, visual_outputs, future_mask)
                losses.update(invariance_losses)
                losses["total"] = losses["total"] + visual_invariance_weight * invariance_losses["visual_invariance"]
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            global_step += 1
            if ema_enabled and global_step > ema_update_after_step:
                model.update_ema_target_encoder(ema_decay)
            batch_metrics = scalar_losses(losses)
            batch_metrics["ema_online_drift"] = float(model.ema_online_drift().detach().cpu().item())
            train_rows.append(batch_metrics)

        val_metrics = evaluate_val(model, val_loader, device, args.rgbd, ema_enabled, train_cfg)
        train_metrics = average_rows(train_rows)
        row = {
            "epoch": epoch,
            "ema_enabled": float(ema_enabled),
            "ema_decay": ema_decay,
            "sigreg_weight": float(loss_cfg["sigreg_weight"]),
            "sigreg_sketches": float(loss_cfg["sigreg_sketches"]),
            "sigreg_scale": float(loss_cfg["sigreg_scale"]),
            "mixture_belief_weight": float(loss_cfg["mixture_belief_weight"]),
            "visual_invariance_weight": float(train_cfg.get("visual_invariance_weight", 0.0)),
            "mixture_components": float(model.mixture_components),
            "structured_context": float(model.use_structured_context),
            "global_step": float(global_step),
            "train_loss": train_metrics.get("total", math.nan),
            **train_metrics,
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
                    "rgbd": bool(args.rgbd),
                    "ema_enabled": bool(ema_enabled),
                    "ema_decay": ema_decay,
                    "sigreg_weight": float(loss_cfg["sigreg_weight"]),
                    "sigreg_sketches": int(loss_cfg["sigreg_sketches"]),
                    "sigreg_scale": float(loss_cfg["sigreg_scale"]),
                    "mixture_belief_weight": float(loss_cfg["mixture_belief_weight"]),
                    "visual_invariance_weight": float(train_cfg.get("visual_invariance_weight", 0.0)),
                    "target_encoder": TARGET_ENCODER_KIND,
                    "mixture_components": int(model.mixture_components),
                    "belief_head": f"gaussian_mixture_{int(model.mixture_components)}",
                    "structured_context": bool(model.use_structured_context),
                    "visual_geometry_weight": float(model.visual_geometry_weight),
                    "context_encoder": context_encoder_name(bool(args.rgbd), bool(model.use_structured_context)),
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
            "ema_enabled": bool(ema_enabled),
            "ema_decay": ema_decay,
            "ema_update_after_step": ema_update_after_step,
            "sigreg_weight": float(loss_cfg["sigreg_weight"]),
            "sigreg_sketches": int(loss_cfg["sigreg_sketches"]),
            "sigreg_scale": float(loss_cfg["sigreg_scale"]),
            "mixture_belief_weight": float(loss_cfg["mixture_belief_weight"]),
            "visual_invariance_weight": float(train_cfg.get("visual_invariance_weight", 0.0)),
            "target_encoder": TARGET_ENCODER_KIND,
            "mixture_components": int(model.mixture_components),
            "belief_head": f"gaussian_mixture_{int(model.mixture_components)}",
            "structured_context": bool(model.use_structured_context),
            "visual_geometry_weight": float(model.visual_geometry_weight),
            "context_encoder": context_encoder_name(bool(args.rgbd), bool(model.use_structured_context)),
        },
    )
    print(f"Done. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
