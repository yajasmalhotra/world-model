#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.runner import evaluate_pixel_baseline, evaluate_world_model
from src.models.encoder import PixelToStateEncoder
from src.models.pixel_baseline import PixelRolloutBaseline
from src.models.state_dynamics import ObjectCentricDynamics
from src.train.pipeline import make_loader
from src.train.utils import (
    append_metrics,
    get_device,
    init_run_dir,
    load_checkpoint,
    load_config,
    save_summary,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dynamics-ckpt", type=str, default=None)
    parser.add_argument("--encoder-ckpt", type=str, default=None)
    parser.add_argument("--joint-ckpt", type=str, default=None)
    parser.add_argument("--pixel-baseline-ckpt", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "state", "pixel", "joint", "pixel_baseline"],
    )
    return parser.parse_args()


def latest_checkpoint(pattern: str) -> Optional[str]:
    candidates = sorted(Path("runs").glob(pattern))
    if not candidates:
        return None
    return str(candidates[-1])


def load_dynamics(config: Dict, device: torch.device, ckpt_path: Optional[str]) -> ObjectCentricDynamics:
    model_cfg = config["model"]
    dynamics = ObjectCentricDynamics(
        state_dim=int(model_cfg["state_dim"]),
        dynamic_dim=int(model_cfg["dynamic_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        interaction_dim=int(model_cfg["interaction_dim"]),
        max_occluders=int(model_cfg["max_occluders"]),
    ).to(device)
    if ckpt_path:
        ckpt = load_checkpoint(ckpt_path, device)
        state = ckpt.get("dynamics_state") or ckpt.get("model_state")
        dynamics.load_state_dict(state, strict=False)
    return dynamics


def load_encoder(config: Dict, device: torch.device, ckpt_path: Optional[str]) -> PixelToStateEncoder:
    model_cfg = config["model"]
    encoder = PixelToStateEncoder(
        max_objects=int(model_cfg["max_objects"]),
        state_dim=int(model_cfg["state_dim"]),
        cnn_dim=int(model_cfg["encoder_cnn_dim"]),
        rnn_dim=int(model_cfg["encoder_rnn_dim"]),
        image_size=int(config["data"]["image_size"]),
    ).to(device)
    if ckpt_path:
        ckpt = load_checkpoint(ckpt_path, device)
        state = ckpt.get("encoder_state") or ckpt.get("model_state")
        encoder.load_state_dict(state, strict=False)
    return encoder


def load_pixel_baseline(device: torch.device, ckpt_path: Optional[str]) -> PixelRolloutBaseline:
    model = PixelRolloutBaseline().to(device)
    if ckpt_path:
        ckpt = load_checkpoint(ckpt_path, device)
        model.load_state_dict(ckpt["model_state"], strict=False)
    return model


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    device = get_device(config["project"].get("device", "auto"))

    if args.joint_ckpt is None:
        args.joint_ckpt = latest_checkpoint("*_train_joint/checkpoints/best.pt")
    if args.dynamics_ckpt is None:
        args.dynamics_ckpt = latest_checkpoint("*_train_dynamics/checkpoints/best.pt")
    if args.encoder_ckpt is None:
        args.encoder_ckpt = latest_checkpoint("*_train_encoder/checkpoints/best.pt")
    if args.pixel_baseline_ckpt is None:
        args.pixel_baseline_ckpt = latest_checkpoint("*_train_pixel_baseline/checkpoints/best.pt")

    manifest_dir = Path(config["data"]["manifest_dir"])
    eval_cfg = config["eval"]
    eval_batch_size = int(eval_cfg["batch_size"])
    splits = eval_cfg["slices"]

    run_dir = init_run_dir(config["project"]["output_root"], "evaluate", config)
    summary: Dict[str, object] = {
        "run_type": "evaluate",
        "device": str(device),
        "checkpoints": {
            "joint": args.joint_ckpt,
            "dynamics": args.dynamics_ckpt,
            "encoder": args.encoder_ckpt,
            "pixel_baseline": args.pixel_baseline_ckpt,
        },
        "results": {},
    }

    use_state = args.mode in ("all", "state")
    use_pixel = args.mode in ("all", "pixel")
    use_joint = args.mode in ("all", "joint")
    use_pixel_baseline = args.mode in ("all", "pixel_baseline")

    state_dynamics = load_dynamics(config, device, args.dynamics_ckpt) if (use_state and args.dynamics_ckpt) else None
    pixel_dynamics = load_dynamics(config, device, args.dynamics_ckpt) if (use_pixel and args.dynamics_ckpt) else None
    pixel_encoder = load_encoder(config, device, args.encoder_ckpt) if (use_pixel and args.encoder_ckpt) else None

    joint_dynamics = load_dynamics(config, device, args.joint_ckpt) if (use_joint and args.joint_ckpt) else None
    joint_encoder = load_encoder(config, device, args.joint_ckpt) if (use_joint and args.joint_ckpt) else None

    pixel_baseline = (
        load_pixel_baseline(device, args.pixel_baseline_ckpt)
        if (use_pixel_baseline and args.pixel_baseline_ckpt)
        else None
    )

    for split in splits:
        manifest_path = manifest_dir / f"{split}.jsonl"
        if not manifest_path.exists():
            print(f"Skipping missing split manifest: {manifest_path}")
            continue
        loader = make_loader(manifest_path, config["data"], batch_size=eval_batch_size, shuffle=False)

        if state_dynamics is not None:
            metrics = evaluate_world_model(state_dynamics, loader, device, config["data"], encoder=None)
            summary["results"][f"state::{split}"] = metrics
            row = {"split": split, "mode": "state", **metrics}
            append_metrics(run_dir, row)
            print(row)

        if pixel_dynamics is not None and pixel_encoder is not None:
            metrics = evaluate_world_model(pixel_dynamics, loader, device, config["data"], encoder=pixel_encoder)
            summary["results"][f"pixel::{split}"] = metrics
            row = {"split": split, "mode": "pixel", **metrics}
            append_metrics(run_dir, row)
            print(row)

        if joint_dynamics is not None and joint_encoder is not None:
            metrics = evaluate_world_model(joint_dynamics, loader, device, config["data"], encoder=joint_encoder)
            summary["results"][f"joint::{split}"] = metrics
            row = {"split": split, "mode": "joint", **metrics}
            append_metrics(run_dir, row)
            print(row)

        if pixel_baseline is not None:
            metrics = evaluate_pixel_baseline(pixel_baseline, loader, device)
            summary["results"][f"pixel_baseline::{split}"] = metrics
            row = {"split": split, "mode": "pixel_baseline", **metrics}
            append_metrics(run_dir, row)
            print(row)

    save_summary(run_dir, summary)
    print(f"Done. Evaluation artifacts in: {run_dir}")


if __name__ == "__main__":
    main()

