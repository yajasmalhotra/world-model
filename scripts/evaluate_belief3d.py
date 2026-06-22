#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data3d.dataset3d import SyntheticScene3DDataset, collate_scenes3d
from src.eval.belief_metrics import particle_belief_metrics, summarize_metric_rows
from src.eval.counterfactual import counterfactual_delta_metrics, move_boxes_to_far_corner
from src.models.belief_encoder3d import ImageToBeliefEncoder3D
from src.models.belief_jepa3d import BeliefJEPA3D, belief_jepa_diagnostics
from src.models.belief_state import (
    ParticleBeliefConfig,
    particles_from_gaussian_sequence,
    particles_from_gaussian_mixture_sequence,
    rollout_geometry_aware_particle_belief,
    rollout_particle_belief,
    rollout_particle_belief_from_gaussian,
)
from src.train.utils import append_metrics, get_device, init_run_dir, load_checkpoint, load_config, save_summary, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 3D hidden-trajectory belief calibration.")
    parser.add_argument("--config", type=str, default="configs/belief3d.yaml")
    parser.add_argument("--split", type=str, default=None, help="Optional single split override.")
    parser.add_argument(
        "--mode",
        type=str,
        default="constant",
        choices=["constant", "geometry", "image", "jepa", "all"],
    )
    parser.add_argument("--encoder-ckpt", type=str, default=None)
    parser.add_argument("--jepa-ckpt", type=str, default=None)
    return parser.parse_args()


def make_loader(manifest_path: Path, data_cfg: Dict, batch_size: int) -> DataLoader:
    dataset = SyntheticScene3DDataset(manifest_path, data_cfg)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_scenes3d)


def latest_checkpoint(pattern: str) -> Optional[str]:
    candidates = sorted(Path("runs").glob(pattern))
    if not candidates:
        return None
    return str(candidates[-1])


def select_preferred_jepa_checkpoint(candidates: Iterable[Path]) -> Optional[Path]:
    candidates = sorted(candidates)
    if not candidates:
        return None
    ema_sigreg_candidates = [path for path in candidates if "noema" not in str(path) and "nosigreg" not in str(path)]
    ema_candidates = [path for path in candidates if "noema" not in str(path)]
    return (ema_sigreg_candidates or ema_candidates or candidates)[-1]


def latest_jepa_checkpoint() -> Optional[str]:
    preferred = select_preferred_jepa_checkpoint(Path("runs").glob("*_train_belief_jepa3d*/checkpoints/best.pt"))
    return str(preferred) if preferred is not None else None


def load_image_encoder(config: Dict, device: torch.device, ckpt_path: Optional[str]) -> tuple[ImageToBeliefEncoder3D, bool]:
    ckpt = load_checkpoint(ckpt_path, device) if ckpt_path else None
    ckpt_config = ckpt.get("config", config) if ckpt else config
    model_cfg = ckpt_config["model3d"]
    data_cfg = ckpt_config["data3d"]
    rgbd = bool(ckpt.get("rgbd", False)) if ckpt else False
    model = ImageToBeliefEncoder3D(
        max_objects=int(model_cfg["max_objects"]),
        input_channels=4 if rgbd else 3,
        cnn_dim=int(model_cfg["encoder_cnn_dim"]),
        rnn_dim=int(model_cfg["encoder_rnn_dim"]),
        world_min=float(data_cfg["world_min"]),
        world_max=float(data_cfg["world_max"]),
        velocity_limit=float(model_cfg["velocity_limit"]),
        min_log_std=float(model_cfg["min_log_std"]),
        max_log_std=float(model_cfg["max_log_std"]),
    ).to(device)
    if ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    return model, rgbd


def load_belief_jepa(config: Dict, device: torch.device, ckpt_path: str) -> tuple[BeliefJEPA3D, bool, bool]:
    ckpt = load_checkpoint(ckpt_path, device)
    ckpt_config = ckpt.get("config", config)
    model_cfg = ckpt_config["model3d"]
    data_cfg = ckpt_config["data3d"]
    rgbd = bool(ckpt.get("rgbd", False))
    ema_enabled = bool(ckpt.get("ema_enabled", False))
    horizon = max(int(data_cfg["seq_len"]), int(data_cfg["obs_len"]) + 14, 24) - int(data_cfg["obs_len"])
    model = BeliefJEPA3D(
        max_objects=int(model_cfg["max_objects"]),
        horizon=horizon,
        input_channels=4 if rgbd else 3,
        cnn_dim=int(model_cfg["encoder_cnn_dim"]),
        rnn_dim=int(model_cfg["encoder_rnn_dim"]),
        latent_dim=int(model_cfg.get("jepa_latent_dim", 64)),
        mixture_components=int(ckpt.get("mixture_components", model_cfg.get("jepa_mixture_components", 3))),
        world_min=float(data_cfg["world_min"]),
        world_max=float(data_cfg["world_max"]),
        velocity_limit=float(model_cfg["velocity_limit"]),
        min_log_std=float(model_cfg["min_log_std"]),
        max_log_std=float(model_cfg.get("jepa_max_log_std", -0.8)),
    ).to(device)
    incompatible = model.load_state_dict(ckpt["model_state"], strict=False)
    ema_prefixes = ("ema_target_encoder.", "ema_target_temporal.", "ema_target_temporal_proj.")
    if any(key.startswith(ema_prefixes) for key in incompatible.missing_keys):
        model.sync_ema_target_encoder()
    missing_mixture = any(key.startswith("mixture_head.") for key in incompatible.missing_keys)
    model.mixture_enabled = bool(int(ckpt.get("mixture_components", 0)) > 1 and not missing_mixture)
    model.belief_head = str(ckpt.get("belief_head", "single_gaussian"))
    return model, rgbd, ema_enabled


def jepa_diagnostic_outputs(outputs: Dict[str, torch.Tensor], mixture_enabled: bool) -> Dict[str, torch.Tensor]:
    if mixture_enabled:
        return outputs
    return {key: value for key, value in outputs.items() if not key.startswith("mixture_")}


def metric_token(value: object) -> str:
    text = str(value or "unknown").strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    token = "_".join(part for part in "".join(chars).split("_") if part)
    return token or "unknown"


def context_frames(batch: Dict, device: torch.device, rgbd: bool) -> torch.Tensor:
    frames = batch["obs_frames"].to(device)
    if not rgbd:
        return frames
    return torch.cat([frames, batch["obs_depth"].to(device)], dim=2)


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


def target_path_mode_masks(batch: Dict, future_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    metadata = batch.get("metadata")
    if not metadata:
        return {}
    masks: Dict[str, torch.Tensor] = {}
    for b_idx, item in enumerate(metadata):
        target = item.get("target") if isinstance(item, dict) else None
        if not target:
            continue
        obj_idx = int(target.get("object_index", target.get("target_object_index", -1)))
        if not 0 <= obj_idx < future_mask.shape[-1]:
            continue
        path_mode = metric_token(target.get("path_mode", "unknown"))
        if path_mode not in masks:
            masks[path_mode] = torch.zeros_like(future_mask)
        masks[path_mode][b_idx, :, obj_idx] = future_mask[b_idx, :, obj_idx]
    return {
        path_mode: mask
        for path_mode, mask in masks.items()
        if float(mask.sum().detach().cpu().item()) > 0.0
    }


def seeded_geometry_rollout(
    init_state: torch.Tensor,
    object_mask: torch.Tensor,
    obstacles: torch.Tensor,
    horizon: int,
    particle_cfg: ParticleBeliefConfig,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return rollout_geometry_aware_particle_belief(
        init_state,
        object_mask,
        obstacles,
        horizon=horizon,
        cfg=particle_cfg,
    )


@torch.no_grad()
def evaluate_split(
    loader: DataLoader,
    config: Dict,
    device: torch.device,
    mode: str,
    encoder: Optional[ImageToBeliefEncoder3D] = None,
    image_rgbd: bool = False,
    jepa: Optional[BeliefJEPA3D] = None,
    jepa_rgbd: bool = False,
    jepa_ema_enabled: bool = False,
) -> Dict[str, float]:
    data_cfg = config["data3d"]
    belief_cfg = config["belief"]
    particle_cfg = ParticleBeliefConfig.from_config(belief_cfg, data_cfg)
    rows: list[Dict[str, float]] = []
    if encoder is not None:
        encoder.eval()
    if jepa is not None:
        jepa.eval()

    for batch_idx, batch in enumerate(loader):
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
            outputs = encoder(context_frames(batch, device, image_rgbd))
            particles, weights = rollout_particle_belief_from_gaussian(
                outputs["mean"],
                outputs["log_std"],
                init_state,
                object_mask,
                horizon=horizon,
                cfg=particle_cfg,
            )
        elif mode == "jepa":
            if jepa is None:
                raise RuntimeError("JEPA mode requires a Belief-JEPA checkpoint.")
            outputs = jepa(
                context_frames(batch, device, jepa_rgbd),
                future_state=future_state,
                use_ema_target=jepa_ema_enabled,
                include_target_reconstruction=False,
            )
            steps = min(outputs["mean"].shape[1], horizon)
            if bool(getattr(jepa, "mixture_enabled", False)):
                particles, weights = particles_from_gaussian_mixture_sequence(
                    outputs["mixture_logits"][:, :steps],
                    outputs["mixture_mean"][:, :steps],
                    outputs["mixture_log_std"][:, :steps],
                    object_mask,
                    cfg=particle_cfg,
                )
            else:
                particles, weights = particles_from_gaussian_sequence(
                    outputs["mean"][:, :steps],
                    outputs["log_std"][:, :steps],
                    object_mask,
                    cfg=particle_cfg,
                )
            future_state = future_state[:, :steps]
            future_mask = future_mask[:, :steps]
            diagnostics = belief_jepa_diagnostics(
                jepa_diagnostic_outputs(outputs, bool(getattr(jepa, "mixture_enabled", False))),
                future_state,
                future_mask,
            )
        elif mode == "geometry":
            rollout_seed = int(config["project"].get("seed", 0)) + 10_007 * int(batch_idx + 1)
            obstacles = batch["obstacles"].to(device)
            particles, weights = seeded_geometry_rollout(
                init_state,
                object_mask,
                obstacles,
                horizon,
                particle_cfg,
                rollout_seed,
            )
            moved_obstacles = move_boxes_to_far_corner(
                obstacles,
                world_min=float(data_cfg.get("world_min", -1.0)),
                world_max=float(data_cfg.get("world_max", 1.0)),
            )
            physical_cf_particles, physical_cf_weights = seeded_geometry_rollout(
                init_state,
                object_mask,
                moved_obstacles,
                horizon,
                particle_cfg,
                rollout_seed,
            )
            visual_cf_particles, visual_cf_weights = seeded_geometry_rollout(
                init_state,
                object_mask,
                obstacles,
                horizon,
                particle_cfg,
                rollout_seed,
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
        path_mode_masks = target_path_mode_masks(batch, future_mask)
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
        for path_mode, path_mask in path_mode_masks.items():
            path_metrics = particle_belief_metrics(
                particles,
                weights,
                future_state,
                path_mask,
                density_sigma=float(belief_cfg["density_sigma"]),
                mass_radius=float(belief_cfg["mass_radius"]),
                credible_levels=belief_cfg.get("credible_levels", [0.5, 0.7, 0.9]),
            )
            metrics.update({f"path_mode_{path_mode}_target_{key}": value for key, value in path_metrics.items()})
        if mode == "geometry":
            metrics.update(
                counterfactual_delta_metrics(
                    particles,
                    weights,
                    physical_cf_particles,
                    physical_cf_weights,
                    future_state,
                    future_mask,
                    prefix="counterfactual_physical",
                )
            )
            metrics.update(
                counterfactual_delta_metrics(
                    particles,
                    weights,
                    visual_cf_particles,
                    visual_cf_weights,
                    future_state,
                    future_mask,
                    prefix="counterfactual_visual",
                )
            )
            metrics["counterfactual_selectivity"] = (
                metrics["counterfactual_physical_belief_delta"] - metrics["counterfactual_visual_belief_delta"]
            )
            if target_mask is not None:
                metrics.update(
                    counterfactual_delta_metrics(
                        particles,
                        weights,
                        physical_cf_particles,
                        physical_cf_weights,
                        future_state,
                        target_mask,
                        prefix="target_counterfactual_physical",
                    )
                )
                metrics.update(
                    counterfactual_delta_metrics(
                        particles,
                        weights,
                        visual_cf_particles,
                        visual_cf_weights,
                        future_state,
                        target_mask,
                        prefix="target_counterfactual_visual",
                    )
                )
                metrics["target_counterfactual_selectivity"] = (
                    metrics["target_counterfactual_physical_belief_delta"]
                    - metrics["target_counterfactual_visual_belief_delta"]
                )
            for path_mode, path_mask in path_mode_masks.items():
                physical_prefix = f"path_mode_{path_mode}_target_counterfactual_physical"
                visual_prefix = f"path_mode_{path_mode}_target_counterfactual_visual"
                metrics.update(
                    counterfactual_delta_metrics(
                        particles,
                        weights,
                        physical_cf_particles,
                        physical_cf_weights,
                        future_state,
                        path_mask,
                        prefix=physical_prefix,
                    )
                )
                metrics.update(
                    counterfactual_delta_metrics(
                        particles,
                        weights,
                        visual_cf_particles,
                        visual_cf_weights,
                        future_state,
                        path_mask,
                        prefix=visual_prefix,
                    )
                )
                metrics[f"path_mode_{path_mode}_target_counterfactual_selectivity"] = (
                    metrics[f"{physical_prefix}_belief_delta"] - metrics[f"{visual_prefix}_belief_delta"]
                )
        if mode == "jepa":
            metrics.update({f"jepa_{key}": float(value.detach().cpu().item()) for key, value in diagnostics.items()})
        rows.append(metrics)
    return summarize_metric_rows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    device = get_device(config["project"].get("device", "auto"))
    if args.mode in ("image", "all") and args.encoder_ckpt is None:
        args.encoder_ckpt = latest_checkpoint("*_train_belief3d_encoder*/checkpoints/best.pt")
    if args.mode in ("jepa", "all") and args.jepa_ckpt is None:
        args.jepa_ckpt = latest_jepa_checkpoint()
    manifest_dir = Path(config["data3d"]["manifest_dir"])
    splits = [args.split] if args.split else list(config["eval"]["slices"])
    batch_size = int(config["eval"].get("batch_size", 8))

    run_dir = init_run_dir(config["project"]["output_root"], "evaluate_belief3d", config)
    summary: Dict[str, object] = {
        "run_type": "evaluate_belief3d",
        "device": str(device),
        "encoder_checkpoint": args.encoder_ckpt,
        "encoder_rgbd": False,
        "jepa_checkpoint": args.jepa_ckpt,
        "splits": {},
    }
    image_encoder: Optional[ImageToBeliefEncoder3D] = None
    image_rgbd = False
    if args.mode in ("image", "all") and args.encoder_ckpt is not None:
        image_encoder, image_rgbd = load_image_encoder(config, device, args.encoder_ckpt)
        summary["encoder_rgbd"] = bool(image_rgbd)
    belief_jepa: Optional[BeliefJEPA3D] = None
    jepa_rgbd = False
    jepa_ema_enabled = False
    if args.mode in ("jepa", "all") and args.jepa_ckpt is not None:
        belief_jepa, jepa_rgbd, jepa_ema_enabled = load_belief_jepa(config, device, args.jepa_ckpt)

    for split in splits:
        manifest_path = manifest_dir / f"{split}.jsonl"
        if not manifest_path.exists():
            print(f"Skipping missing split manifest: {manifest_path}")
            continue
        loader = make_loader(manifest_path, config["data3d"], batch_size=batch_size)
        if args.mode in ("constant", "all"):
            metrics = evaluate_split(loader, config, device, mode="constant")
            row = {"split": split, "mode": "constant_velocity_particle_belief", **metrics}
            append_metrics(run_dir, row)
            summary["splits"][f"constant::{split}"] = metrics
            print(row)
        if args.mode in ("geometry", "all"):
            metrics = evaluate_split(loader, config, device, mode="geometry")
            row = {"split": split, "mode": "geometry_aware_particle_belief", **metrics}
            append_metrics(run_dir, row)
            summary["splits"][f"geometry::{split}"] = metrics
            print(row)
        if args.mode in ("image", "all"):
            if image_encoder is None:
                print("Skipping image mode because no encoder checkpoint was found.")
            else:
                metrics = evaluate_split(loader, config, device, mode="image", encoder=image_encoder, image_rgbd=image_rgbd)
                metrics["image_rgbd"] = float(image_rgbd)
                row = {"split": split, "mode": "image_to_belief", **metrics}
                append_metrics(run_dir, row)
                summary["splits"][f"image::{split}"] = metrics
                print(row)
        if args.mode in ("jepa", "all"):
            if belief_jepa is None:
                print("Skipping Belief-JEPA mode because no checkpoint was found.")
            else:
                metrics = evaluate_split(
                    loader,
                    config,
                    device,
                    mode="jepa",
                    jepa=belief_jepa,
                    jepa_rgbd=jepa_rgbd,
                    jepa_ema_enabled=jepa_ema_enabled,
                )
                metrics["jepa_ema_enabled"] = float(jepa_ema_enabled)
                metrics["jepa_mixture_enabled"] = float(bool(getattr(belief_jepa, "mixture_enabled", False)))
                row = {"split": split, "mode": "belief_jepa_latent_predictor", **metrics}
                append_metrics(run_dir, row)
                summary["splits"][f"jepa::{split}"] = metrics
                print(row)

    save_summary(run_dir, summary)
    print(f"Done. 3D belief evaluation artifacts in: {run_dir}")


if __name__ == "__main__":
    main()
