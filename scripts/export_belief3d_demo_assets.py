#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data3d.dataset3d import scene3d_config_from_data_cfg
from src.data3d.scene_generator3d import STATE_INDEX_3D, Scene3DConfig, SyntheticScene3DGenerator
from src.models.belief_encoder3d import ImageToBeliefEncoder3D
from src.models.belief_jepa3d import BeliefJEPA3D
from src.models.belief_state import (
    ParticleBeliefConfig,
    initialize_particles,
    particles_from_gaussian_sequence,
    particles_from_gaussian_mixture_sequence,
    rollout_geometry_aware_particle_belief,
    rollout_particle_belief,
    rollout_particle_belief_from_gaussian,
)
from src.train.utils import get_device, load_checkpoint, load_config, set_seed


RGB_YELLOW = (250, 204, 21)
RGB_MAGENTA = (217, 70, 239)
RGB_GREEN = (34, 197, 94)
RGB_BLUE = (59, 130, 246)
RGB_DARK = (17, 24, 39)
RGB_PANEL = (248, 250, 252)
RGB_GRID = (203, 213, 225)
RGB_ORANGE = (249, 115, 22)

METHOD_LABELS = {
    "constant": "constant velocity",
    "geometry": "geometry aware",
    "image": "image to belief",
    "jepa": "Belief-JEPA",
}
METHOD_SHORT_LABELS = {
    "constant": "constant",
    "geometry": "geometry",
    "image": "image",
    "jepa": "JEPA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export 3D belief demo GIFs and per-frame metrics.")
    parser.add_argument("--config", type=str, default="configs/belief3d_smoke.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--output-dir", type=str, default="results/belief3d_demo_assets")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--max-particles", type=int, default=192)
    parser.add_argument("--panel-scale", type=int, default=2)
    parser.add_argument(
        "--scenario",
        type=str,
        default="structured_occlusion",
        choices=["random", "targeted_occlusion", "structured_occlusion", "impossible_reappearance"],
    )
    parser.add_argument("--mode", type=str, default="compare", choices=["constant", "geometry", "compare", "compare_all"])
    parser.add_argument(
        "--primary-method",
        type=str,
        default="auto",
        choices=["auto", "constant", "geometry", "image", "jepa"],
        help="Which belief trace to render in the main panels. 'auto' picks the lowest finite expected distance.",
    )
    parser.add_argument("--encoder-ckpt", type=str, default=None, help="Optional image-to-belief checkpoint for compare_all.")
    parser.add_argument("--jepa-ckpt", type=str, default=None, help="Optional Belief-JEPA checkpoint for compare_all.")
    parser.add_argument("--skip-mp4", action="store_true", help="Only write GIF/PNG/JSON artifacts.")
    return parser.parse_args()


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


def load_image_encoder(config: Dict, device: torch.device, ckpt_path: str) -> tuple[ImageToBeliefEncoder3D, bool]:
    ckpt = load_checkpoint(ckpt_path, device)
    ckpt_config = ckpt.get("config", config)
    model_cfg = ckpt_config["model3d"]
    data_cfg = ckpt_config["data3d"]
    rgbd = bool(ckpt.get("rgbd", False))
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
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model, rgbd


def load_belief_jepa(config: Dict, device: torch.device, ckpt_path: str) -> tuple[BeliefJEPA3D, bool, bool]:
    ckpt = load_checkpoint(ckpt_path, device)
    ckpt_config = ckpt.get("config", config)
    model_cfg = ckpt_config["model3d"]
    data_cfg = ckpt_config["data3d"]
    rgbd = bool(ckpt.get("rgbd", False))
    ema_enabled = bool(ckpt.get("ema_enabled", False))
    model_state = ckpt["model_state"]
    state_has_structured = any(str(key).startswith("structured_") for key in model_state.keys())
    structured_enabled = bool(ckpt.get("structured_context", state_has_structured))
    horizon = max(int(data_cfg["seq_len"]), int(data_cfg["obs_len"]) + 14, 24) - int(data_cfg["obs_len"])
    model = BeliefJEPA3D(
        max_objects=int(model_cfg["max_objects"]),
        horizon=horizon,
        input_channels=4 if rgbd else 3,
        cnn_dim=int(model_cfg["encoder_cnn_dim"]),
        rnn_dim=int(model_cfg["encoder_rnn_dim"]),
        latent_dim=int(model_cfg.get("jepa_latent_dim", 64)),
        mixture_components=int(ckpt.get("mixture_components", model_cfg.get("jepa_mixture_components", 3))),
        structured_context=structured_enabled,
        structured_dim=int(model_cfg.get("jepa_structured_dim", 64)),
        visual_geometry_weight=float(ckpt.get("visual_geometry_weight", model_cfg.get("jepa_visual_geometry_weight", 1.0))),
        world_min=float(data_cfg["world_min"]),
        world_max=float(data_cfg["world_max"]),
        velocity_limit=float(model_cfg["velocity_limit"]),
        min_log_std=float(model_cfg["min_log_std"]),
        max_log_std=float(model_cfg.get("jepa_max_log_std", -0.8)),
    ).to(device)
    incompatible = model.load_state_dict(model_state, strict=False)
    ema_prefixes = ("ema_target_encoder.", "ema_target_temporal.", "ema_target_temporal_proj.")
    if any(key.startswith(ema_prefixes) for key in incompatible.missing_keys):
        model.sync_ema_target_encoder()
    missing_mixture = any(key.startswith("mixture_head.") for key in incompatible.missing_keys)
    missing_structured = any(key.startswith("structured_") for key in incompatible.missing_keys)
    if missing_structured:
        model.use_structured_context = False
    model.mixture_enabled = bool(int(ckpt.get("mixture_components", 0)) > 1 and not missing_mixture)
    model.belief_head = str(ckpt.get("belief_head", "single_gaussian"))
    model.eval()
    return model, rgbd, ema_enabled


def sample_context_tensor(sample: Dict[str, np.ndarray], obs_len: int, device: torch.device, rgbd: bool) -> torch.Tensor:
    frames = torch.from_numpy(sample["frames"][:obs_len].astype(np.float32) / 255.0)
    frames = frames.permute(0, 3, 1, 2).unsqueeze(0).contiguous().to(device)
    if not rgbd:
        return frames
    depth = torch.from_numpy(sample["depth"][:obs_len].astype(np.float32))
    depth = depth.unsqueeze(1).unsqueeze(0).contiguous().to(device)
    return torch.cat([frames, depth], dim=2)


def sample_structured_context(
    sample: Dict[str, np.ndarray],
    obs_len: int,
    device: torch.device,
    enabled: bool,
) -> Dict[str, torch.Tensor] | None:
    if not enabled:
        return None
    return {
        "obs_state": torch.from_numpy(sample["state"][:obs_len].astype(np.float32)).unsqueeze(0).to(device),
        "obs_mask": torch.from_numpy(sample["object_mask"][:obs_len].astype(np.float32)).unsqueeze(0).to(device),
        "visual_occluders": torch.from_numpy(sample["visual_occluders"].astype(np.float32)).unsqueeze(0).to(device),
        "physical_obstacles": torch.from_numpy(sample["physical_obstacles"].astype(np.float32)).unsqueeze(0).to(device),
        "solid_screens": torch.from_numpy(sample["solid_screens"].astype(np.float32)).unsqueeze(0).to(device),
    }


def project_points(points: np.ndarray, cfg: Scene3DConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.maximum(cfg.camera_z - points[:, 2], 1e-4)
    scale = cfg.focal_length / depth
    center = (cfg.image_size - 1) * 0.5
    px = center + points[:, 0] * scale * center
    py = center - points[:, 1] * scale * center
    return px, py, depth


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    near_bright = (1.0 - np.clip(depth, 0.0, 1.0)) * 255.0
    return np.repeat(near_bright.astype(np.uint8)[..., None], 3, axis=-1)


def add_caption(image: Image.Image, caption: str) -> Image.Image:
    font = ImageFont.load_default()
    width, height = image.size
    caption_h = 16
    out = Image.new("RGB", (width, height + caption_h), "white")
    out.paste(image, (0, caption_h))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, width, caption_h], fill=(15, 23, 42))
    draw.text((5, 3), caption, fill="white", font=font)
    return out


def make_panel(array: np.ndarray, caption: str, scale: int) -> np.ndarray:
    image = Image.fromarray(array)
    if scale > 1:
        image = image.resize((image.size[0] * scale, image.size[1] * scale), Image.Resampling.NEAREST)
    return np.asarray(add_caption(image, caption))


def draw_camera_belief_overlay(
    rgb: np.ndarray,
    cfg: Scene3DConfig,
    particles: np.ndarray,
    true_pos: np.ndarray,
    hidden: bool,
    max_particles: int,
    show_particles: bool = True,
    secondary_particles: np.ndarray | None = None,
) -> np.ndarray:
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if show_particles:
        if secondary_particles is not None and secondary_particles.shape[0] > 0:
            secondary = secondary_particles
            if secondary.shape[0] > max_particles:
                stride = max(1, secondary.shape[0] // max_particles)
                secondary = secondary[::stride][:max_particles]
            px, py, depth = project_points(secondary[:, 0:3], cfg)
            order = np.argsort(depth)[::-1]
            for idx in order:
                x = float(px[idx])
                y = float(py[idx])
                if 0 <= x < cfg.image_size and 0 <= y < cfg.image_size:
                    draw.ellipse([x - 1.4, y - 1.4, x + 1.4, y + 1.4], fill=(*RGB_BLUE, 36))
        if particles.shape[0] > max_particles:
            # Deterministic stride keeps GIFs stable from run to run.
            stride = max(1, particles.shape[0] // max_particles)
            particles = particles[::stride][:max_particles]

        px, py, depth = project_points(particles[:, 0:3], cfg)
        order = np.argsort(depth)[::-1]
        for idx in order:
            x = float(px[idx])
            y = float(py[idx])
            if 0 <= x < cfg.image_size and 0 <= y < cfg.image_size:
                draw.ellipse([x - 1.2, y - 1.2, x + 1.2, y + 1.2], fill=(*RGB_YELLOW, 42))

    tx, ty, _ = project_points(true_pos.reshape(1, 3), cfg)
    x = float(tx[0])
    y = float(ty[0])
    marker = RGB_MAGENTA if hidden else RGB_GREEN
    if 0 <= x < cfg.image_size and 0 <= y < cfg.image_size:
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(*marker, 255), width=2)
        draw.line([x - 5, y, x + 5, y], fill=(*marker, 255), width=1)
        draw.line([x, y - 5, x, y + 5], fill=(*marker, 255), width=1)

    return np.asarray(Image.alpha_composite(base, overlay).convert("RGB"))


def iso_project(points: np.ndarray, cfg: Scene3DConfig, size: int) -> np.ndarray:
    span = cfg.world_max - cfg.world_min
    scale = 0.62 * size / max(span, 1e-6)
    center_x = size * 0.5
    center_y = size * 0.56
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    sx = center_x + (x - y) * 0.72 * scale
    sy = center_y - (z * 1.02 + (x + y) * 0.28) * scale
    return np.stack([sx, sy], axis=-1)


def draw_polyline(draw: ImageDraw.ImageDraw, points: np.ndarray, fill: tuple[int, int, int], width: int = 1) -> None:
    if points.shape[0] < 2:
        return
    draw.line([tuple(p.tolist()) for p in points], fill=fill, width=width)


def draw_3d_belief_panel(
    cfg: Scene3DConfig,
    particles: np.ndarray,
    true_history: np.ndarray,
    current_true: np.ndarray,
    hidden: bool,
    max_particles: int,
    show_particles: bool = True,
    secondary_particles: np.ndarray | None = None,
    occluders: np.ndarray | None = None,
    obstacles: np.ndarray | None = None,
) -> np.ndarray:
    size = cfg.image_size
    image = Image.new("RGB", (size, size), RGB_PANEL)
    draw = ImageDraw.Draw(image)

    corners = np.array(
        [
            [cfg.world_min, cfg.world_min, cfg.world_min],
            [cfg.world_max, cfg.world_min, cfg.world_min],
            [cfg.world_max, cfg.world_max, cfg.world_min],
            [cfg.world_min, cfg.world_max, cfg.world_min],
            [cfg.world_min, cfg.world_min, cfg.world_max],
            [cfg.world_max, cfg.world_min, cfg.world_max],
            [cfg.world_max, cfg.world_max, cfg.world_max],
            [cfg.world_min, cfg.world_max, cfg.world_max],
        ],
        dtype=np.float32,
    )
    projected = iso_project(corners, cfg, size)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        draw.line([tuple(projected[a]), tuple(projected[b])], fill=RGB_GRID, width=1)

    def draw_boxes(boxes: np.ndarray | None, fill: tuple[int, int, int], width: int) -> None:
        if boxes is None:
            return
        for box in boxes:
            if not np.any(box):
                continue
            x0, y0, z0, x1, y1, z1 = box.tolist()
            corners = np.array(
                [
                    [x0, y0, z0],
                    [x1, y0, z0],
                    [x1, y1, z0],
                    [x0, y1, z0],
                    [x0, y0, z1],
                    [x1, y0, z1],
                    [x1, y1, z1],
                    [x0, y1, z1],
                ],
                dtype=np.float32,
            )
            p = iso_project(corners, cfg, size)
            for a, b in edges:
                draw.line([tuple(p[a]), tuple(p[b])], fill=fill, width=width)

    draw_boxes(occluders, (100, 116, 139), 1)
    draw_boxes(obstacles, RGB_ORANGE, 2)

    if show_particles:
        if secondary_particles is not None and secondary_particles.shape[0] > 0:
            secondary = secondary_particles
            if secondary.shape[0] > max_particles:
                stride = max(1, secondary.shape[0] // max_particles)
                secondary = secondary[::stride][:max_particles]
            p2 = iso_project(secondary[:, 0:3], cfg, size)
            for point in p2:
                x, y = point.tolist()
                if -4 <= x <= size + 4 and -4 <= y <= size + 4:
                    draw.ellipse([x - 1.2, y - 1.2, x + 1.2, y + 1.2], fill=(59, 130, 246))
        if particles.shape[0] > max_particles:
            stride = max(1, particles.shape[0] // max_particles)
            particles = particles[::stride][:max_particles]
        p2 = iso_project(particles[:, 0:3], cfg, size)
        z_norm = (particles[:, 2] - cfg.world_min) / max(cfg.world_max - cfg.world_min, 1e-6)
        for point, zn in zip(p2, z_norm):
            x, y = point.tolist()
            if -4 <= x <= size + 4 and -4 <= y <= size + 4:
                shade = int(120 + 100 * float(np.clip(zn, 0.0, 1.0)))
                draw.ellipse([x - 1.2, y - 1.2, x + 1.2, y + 1.2], fill=(shade, 145, 28))

    if true_history.shape[0] >= 2:
        draw_polyline(draw, iso_project(true_history, cfg, size), fill=RGB_BLUE, width=2)

    marker = RGB_MAGENTA if hidden else RGB_GREEN
    cx, cy = iso_project(current_true.reshape(1, 3), cfg, size)[0].tolist()
    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], outline=marker, width=2)
    draw.line([cx - 5, cy, cx + 5, cy], fill=marker, width=1)
    draw.line([cx, cy - 5, cx, cy + 5], fill=marker, width=1)
    return np.asarray(image)


def draw_metric_panel(
    metrics: Dict[str, List[float]],
    frame_idx: int,
    width: int,
    height: int = 136,
    comparison: Dict[str, Dict[str, List[float]]] | None = None,
) -> np.ndarray:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, width, height], outline=RGB_GRID)
    names = [
        ("expected_distance", "dist", RGB_BLUE),
        ("mass_radius", "mass", RGB_GREEN),
        ("surprise", "surprise", RGB_MAGENTA),
        ("entropy", "entropy", RGB_ORANGE),
        ("coverage_90", "cov90", RGB_DARK),
    ]
    y = 8
    for key, label, color in names:
        series = np.asarray(metrics[key], dtype=np.float32)
        upto = series[: frame_idx + 1]
        valid = upto[np.isfinite(upto)]
        value = float(upto[-1]) if np.isfinite(upto[-1]) else float("nan")
        draw.text((8, y), f"{label}: {value:.3f}" if value == value else f"{label}: n/a", fill=RGB_DARK, font=font)
        plot_x0 = 86
        plot_y0 = y + 2
        compare_w = 220 if comparison else 0
        plot_w = max(80, width - plot_x0 - compare_w - 16)
        plot_h = 13
        draw.rectangle([plot_x0, plot_y0, plot_x0 + plot_w, plot_y0 + plot_h], outline=(226, 232, 240))
        if valid.size >= 2:
            lo = float(np.min(valid))
            hi = float(np.max(valid))
            if abs(hi - lo) < 1e-6:
                hi = lo + 1.0
            xs = np.linspace(plot_x0, plot_x0 + plot_w, len(upto))
            ys = plot_y0 + plot_h - ((upto - lo) / (hi - lo)) * plot_h
            pts = [(float(x), float(yv)) for x, yv in zip(xs, ys) if np.isfinite(yv)]
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=2)
        y += 20
    if comparison:
        draw.text((width - 220, 8), "expected distance", fill=RGB_DARK, font=font)
        y_cmp = 22
        for method in ("constant", "geometry", "image", "jepa"):
            if method not in comparison:
                continue
            value = comparison[method]["expected_distance"][frame_idx]
            text_value = f"{float(value):.3f}" if np.isfinite(value) else "n/a"
            draw.text((width - 220, y_cmp), f"{METHOD_SHORT_LABELS[method]}: {text_value}", fill=RGB_BLUE, font=font)
            y_cmp += 13
    phase = metrics["phase"][frame_idx]
    draw.text((8, height - 18), f"phase: {phase}", fill=RGB_ORANGE if phase == "hidden rollout" else RGB_DARK, font=font)
    return np.asarray(image)


def hstack_with_border(images: Iterable[np.ndarray], border: int = 4) -> np.ndarray:
    pil_images = [Image.fromarray(img) for img in images]
    widths, heights = zip(*(img.size for img in pil_images))
    out = Image.new("RGB", (sum(widths) + border * (len(pil_images) - 1), max(heights)), "white")
    x = 0
    for img in pil_images:
        out.paste(img, (x, 0))
        x += img.size[0] + border
    return np.asarray(out)


def vstack(images: Iterable[np.ndarray]) -> np.ndarray:
    pil_images = [Image.fromarray(img) for img in images]
    width = max(img.size[0] for img in pil_images)
    height = sum(img.size[1] for img in pil_images)
    out = Image.new("RGB", (width, height), "white")
    y = 0
    for img in pil_images:
        out.paste(img, (0, y))
        y += img.size[1]
    return np.asarray(out)


def write_mp4(path: Path, frames: List[np.ndarray], fps: int) -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for frame_idx, frame in enumerate(frames):
                Image.fromarray(frame).save(tmp_path / f"frame_{frame_idx:05d}.png")
            command = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(max(1, fps)),
                "-i",
                str(tmp_path / "frame_%05d.png"),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return None
            return (result.stderr or result.stdout or f"ffmpeg exited with code {result.returncode}").strip()
    try:
        imageio.mimsave(path, frames, fps=max(1, fps), macro_block_size=1)
    except Exception as exc:  # pragma: no cover - depends on optional ffmpeg backend.
        return str(exc)
    return None


def choose_focus_object(future_state: np.ndarray, future_mask: np.ndarray) -> int:
    hidden = future_mask * (future_state[..., STATE_INDEX_3D["occluded"]] > 0.5).astype(np.float32)
    counts = hidden.sum(axis=0)
    if float(counts.max()) > 0:
        return int(np.argmax(counts))
    visible_counts = future_mask.sum(axis=0)
    return int(np.argmax(visible_counts))


def demo_overrides(scene_cfg: Scene3DConfig, scenario: str) -> Dict[str, object]:
    if scenario == "random":
        return {"scenario": "random"}
    path_mode = "linear"
    if scenario == "impossible_reappearance":
        path_mode = "impossible_jump"
    return {
        "scenario": scenario,
        "path_mode": path_mode,
        "seq_len": max(scene_cfg.seq_len, scene_cfg.obs_len + 14, 24),
        "obs_len": scene_cfg.obs_len,
    }


def per_frame_metrics(
    particles: np.ndarray,
    weights: np.ndarray,
    true_state: np.ndarray,
    obj_idx: int,
    density_sigma: float,
    mass_radius: float,
    credible_levels: Iterable[float] = (0.5, 0.7, 0.9),
) -> Dict[str, List[float]]:
    obj_particles = particles[:, obj_idx, :, 0:3]
    obj_weights = weights[:, obj_idx, :]
    obj_weights = obj_weights / np.maximum(obj_weights.sum(axis=-1, keepdims=True), 1e-8)
    true_pos = true_state[:, obj_idx, 0:3]
    hidden = true_state[:, obj_idx, STATE_INDEX_3D["occluded"]] > 0.5
    diff = obj_particles - true_pos[:, None, :]
    dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 1e-12))
    belief_mean = np.sum(obj_weights[..., None] * obj_particles, axis=1)
    mean_error = np.sqrt(np.maximum(np.sum((belief_mean - true_pos) ** 2, axis=-1), 1e-12))
    expected_distance = np.sum(obj_weights * dist, axis=-1)
    mass = np.sum(obj_weights * (dist <= mass_radius), axis=-1)
    norm_const = (2.0 * np.pi) ** (-1.5) * (density_sigma**-3)
    density = norm_const * np.sum(obj_weights * np.exp(-0.5 * (dist / density_sigma) ** 2), axis=-1)
    density_nll = -np.log(np.maximum(density, 1e-12))
    surprise = np.maximum(-np.log(np.maximum(mass, 1e-8)), 0.0)
    entropy = -np.sum(obj_weights * np.log(np.maximum(obj_weights, 1e-8)), axis=-1)
    center_dist = np.sqrt(np.maximum(np.sum((obj_particles - belief_mean[:, None, :]) ** 2, axis=-1), 1e-12))
    order = np.argsort(center_dist, axis=-1)
    sorted_center_dist = np.take_along_axis(center_dist, order, axis=-1)
    sorted_weights = np.take_along_axis(obj_weights, order, axis=-1)
    cdf = np.cumsum(sorted_weights, axis=-1)

    metrics = {
        "expected_distance": expected_distance.astype(float).tolist(),
        "mean_error": mean_error.astype(float).tolist(),
        "mass_radius": mass.astype(float).tolist(),
        "density_nll": density_nll.astype(float).tolist(),
        "surprise": surprise.astype(float).tolist(),
        "entropy": entropy.astype(float).tolist(),
        "hidden": hidden.astype(bool).tolist(),
    }
    for level in credible_levels:
        idx = np.argmax(cdf >= float(level), axis=-1)
        radius = sorted_center_dist[np.arange(sorted_center_dist.shape[0]), idx]
        contained = (mean_error <= radius).astype(np.float32)
        key = f"coverage_{int(round(float(level) * 100))}"
        metrics[key] = contained.astype(float).tolist()
        metrics[f"calibration_error_{int(round(float(level) * 100))}"] = np.abs(contained - float(level)).astype(float).tolist()
    return metrics


def visible_prefix_metrics(length: int) -> Dict[str, List[float]]:
    return {
        "expected_distance": [float("nan")] * length,
        "mean_error": [float("nan")] * length,
        "mass_radius": [float("nan")] * length,
        "density_nll": [float("nan")] * length,
        "surprise": [float("nan")] * length,
        "entropy": [float("nan")] * length,
        "coverage_50": [float("nan")] * length,
        "coverage_70": [float("nan")] * length,
        "coverage_90": [float("nan")] * length,
        "calibration_error_50": [float("nan")] * length,
        "calibration_error_70": [float("nan")] * length,
        "calibration_error_90": [float("nan")] * length,
        "hidden": [False] * length,
    }


def combine_metrics(
    prefix: Dict[str, List[float]],
    rollout: Dict[str, List[float]],
    obs_len: int,
    target_metadata: Dict[str, object] | None = None,
) -> Dict[str, List[float]]:
    combined: Dict[str, List[float]] = {}
    for key in prefix.keys():
        combined[key] = prefix[key] + rollout[key]
    phases = ["observed target"] * max(0, obs_len - 1) + ["belief initialized"]
    impossible = bool((target_metadata or {}).get("is_impossible_event"))
    reappearance_frame = (target_metadata or {}).get("reappearance_frame")
    for offset, hidden in enumerate(rollout["hidden"]):
        frame_idx = obs_len + offset
        if impossible and isinstance(reappearance_frame, int) and frame_idx == reappearance_frame:
            phases.append("impossible event")
        else:
            phases.append("hidden rollout" if hidden else "reappearance / visible")
    combined["phase"] = phases
    return combined


def finite_mean(values: List[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return float("inf")
    return float(sum(finite) / len(finite))


def choose_primary_trace(comparison: Dict[str, Dict[str, List[float]]], fallback: str = "geometry") -> str:
    if not comparison:
        return fallback
    scores = []
    for method, method_metrics in comparison.items():
        scores.append((finite_mean(method_metrics.get("expected_distance", [])), method))
    scores.sort(key=lambda item: (item[0], item[1]))
    if scores and np.isfinite(scores[0][0]):
        return scores[0][1]
    if fallback in comparison:
        return fallback
    return next(iter(comparison.keys()))


def choose_demo_primary_method(
    mode: str,
    traces: Dict[str, Dict[str, object]],
    comparison: Optional[Dict[str, Dict[str, List[float]]]],
    requested: str = "auto",
) -> str:
    if requested != "auto":
        if requested not in traces:
            available = ", ".join(sorted(traces.keys()))
            raise RuntimeError(f"Requested primary method {requested!r} is unavailable. Available methods: {available}")
        return requested
    if mode == "constant":
        return "constant"
    if mode == "geometry":
        return "geometry"
    if mode == "compare_all":
        return choose_primary_trace(comparison or {}, fallback="geometry")
    return "geometry"


def build_demo_for_seed(
    seed: int,
    config: Dict,
    scene_cfg: Scene3DConfig,
    output_dir: Path,
    device: torch.device,
    fps: int,
    max_particles: int,
    panel_scale: int,
    scenario: str,
    mode: str,
    image_encoder: Optional[ImageToBeliefEncoder3D] = None,
    image_encoder_ckpt: Optional[str] = None,
    image_encoder_rgbd: bool = False,
    belief_jepa: Optional[BeliefJEPA3D] = None,
    belief_jepa_ckpt: Optional[str] = None,
    jepa_rgbd: bool = False,
    jepa_ema_enabled: bool = False,
    primary_method_override: str = "auto",
    write_video: bool = True,
) -> None:
    generator = SyntheticScene3DGenerator(scene_cfg)
    overrides = demo_overrides(scene_cfg, scenario)
    effective_cfg = generator.resolve_config(overrides)
    sample = generator.generate(seed=seed, overrides=overrides, tags=["demo", scenario])
    obs_len = effective_cfg.obs_len
    future_state = sample["state"][obs_len:]
    future_mask = sample["object_mask"][obs_len:]
    horizon = future_state.shape[0]
    target_metadata = sample["metadata"].get("target", {})
    focus_obj = int(target_metadata.get("object_index", choose_focus_object(future_state, future_mask)))

    init_state = torch.from_numpy(sample["state"][obs_len - 1 : obs_len]).to(device)
    object_mask = torch.from_numpy(sample["object_mask"][obs_len - 1 : obs_len]).to(device)
    particle_cfg = ParticleBeliefConfig.from_config(config["belief"], sample["metadata"]["config"])
    init_particles, _init_weights = initialize_particles(init_state, object_mask, particle_cfg)
    constant_particles, constant_weights = rollout_particle_belief(init_state, object_mask, horizon=horizon, cfg=particle_cfg)
    geometry_particles, geometry_weights = rollout_geometry_aware_particle_belief(
        init_state,
        object_mask,
        torch.from_numpy(sample["obstacles"]).unsqueeze(0).to(device),
        horizon=horizon,
        cfg=particle_cfg,
    )
    init_particles_np = init_particles.squeeze(0).detach().cpu().numpy()
    constant_particles_np = constant_particles.squeeze(0).detach().cpu().numpy()
    constant_weights_np = constant_weights.squeeze(0).detach().cpu().numpy()
    geometry_particles_np = geometry_particles.squeeze(0).detach().cpu().numpy()
    geometry_weights_np = geometry_weights.squeeze(0).detach().cpu().numpy()
    traces: Dict[str, Dict[str, object]] = {}

    def add_trace(method: str, particles: np.ndarray, weights: np.ndarray, checkpoint: Optional[str] = None) -> None:
        rollout_metrics = per_frame_metrics(
            particles=particles,
            weights=weights,
            true_state=future_state,
            obj_idx=focus_obj,
            density_sigma=float(config["belief"]["density_sigma"]),
            mass_radius=float(config["belief"]["mass_radius"]),
            credible_levels=config["belief"].get("credible_levels", [0.5, 0.7, 0.9]),
        )
        traces[method] = {
            "label": METHOD_LABELS[method],
            "checkpoint": checkpoint,
            "particles": particles,
            "weights": weights,
            "metrics": combine_metrics(
                visible_prefix_metrics(obs_len),
                rollout_metrics,
                obs_len=obs_len,
                target_metadata=target_metadata,
            ),
        }

    add_trace("constant", constant_particles_np, constant_weights_np)
    add_trace("geometry", geometry_particles_np, geometry_weights_np)

    if mode == "compare_all" and image_encoder is not None:
        with torch.no_grad():
            image_outputs = image_encoder(sample_context_tensor(sample, obs_len, device, rgbd=image_encoder_rgbd))
            image_particles, image_weights = rollout_particle_belief_from_gaussian(
                image_outputs["mean"],
                image_outputs["log_std"],
                init_state,
                object_mask,
                horizon=horizon,
                cfg=particle_cfg,
            )
        add_trace(
            "image",
            image_particles.squeeze(0).detach().cpu().numpy(),
            image_weights.squeeze(0).detach().cpu().numpy(),
            checkpoint=image_encoder_ckpt,
        )

    if mode == "compare_all" and belief_jepa is not None:
        with torch.no_grad():
            future_state_tensor = torch.from_numpy(future_state.astype(np.float32)).unsqueeze(0).to(device)
            jepa_outputs = belief_jepa(
                sample_context_tensor(sample, obs_len, device, rgbd=jepa_rgbd),
                future_state=future_state_tensor,
                structured_context=sample_structured_context(
                    sample,
                    obs_len,
                    device,
                    bool(getattr(belief_jepa, "use_structured_context", False)),
                ),
                use_ema_target=jepa_ema_enabled,
                include_target_reconstruction=False,
            )
            steps = min(horizon, jepa_outputs["mean"].shape[1])
            if steps == horizon:
                if bool(getattr(belief_jepa, "mixture_enabled", False)):
                    jepa_particles, jepa_weights = particles_from_gaussian_mixture_sequence(
                        jepa_outputs["mixture_logits"][:, :horizon],
                        jepa_outputs["mixture_mean"][:, :horizon],
                        jepa_outputs["mixture_log_std"][:, :horizon],
                        object_mask,
                        cfg=particle_cfg,
                    )
                else:
                    jepa_particles, jepa_weights = particles_from_gaussian_sequence(
                        jepa_outputs["mean"][:, :horizon],
                        jepa_outputs["log_std"][:, :horizon],
                        object_mask,
                        cfg=particle_cfg,
                    )
                add_trace(
                    "jepa",
                    jepa_particles.squeeze(0).detach().cpu().numpy(),
                    jepa_weights.squeeze(0).detach().cpu().numpy(),
                    checkpoint=belief_jepa_ckpt,
                )

    comparison = None
    if mode in ("compare", "compare_all"):
        comparison = {method: trace["metrics"] for method, trace in traces.items()}

    primary_method = choose_demo_primary_method(mode, traces, comparison, requested=primary_method_override)
    particles_np = traces[primary_method]["particles"]
    weights_np = traces[primary_method]["weights"]
    metrics = traces[primary_method]["metrics"]
    mode_label = {
        "constant": "constant_velocity_particle_belief",
        "geometry": "geometry_aware_particle_belief",
        "compare": "constant_vs_geometry_particle_belief",
        "compare_all": "constant_geometry_image_jepa_belief_comparison",
    }[mode]

    frames: List[np.ndarray] = []
    total_steps = obs_len + horizon
    empty_particles = np.zeros((0, 6), dtype=np.float32)
    for frame_idx in range(total_steps):
        current_true = sample["state"][frame_idx, focus_obj, 0:3]
        history = sample["state"][: frame_idx + 1, focus_obj, 0:3]
        hidden = bool(sample["state"][frame_idx, focus_obj, STATE_INDEX_3D["occluded"]] > 0.5)
        if frame_idx < obs_len - 1:
            particle_cloud = empty_particles
            secondary_cloud = None
            show_particles = False
        elif frame_idx == obs_len - 1:
            particle_cloud = init_particles_np[focus_obj]
            secondary_cloud = None
            show_particles = True
        else:
            particle_cloud = particles_np[frame_idx - obs_len, focus_obj]
            secondary_cloud = (
                constant_particles_np[frame_idx - obs_len, focus_obj]
                if mode in ("compare", "compare_all") and primary_method != "constant"
                else None
            )
            show_particles = True

        rgb_panel = make_panel(sample["frames"][frame_idx], "RGB camera", panel_scale)
        depth_panel = make_panel(depth_to_rgb(sample["depth"][frame_idx]), "Depth", panel_scale)
        overlay = draw_camera_belief_overlay(
            rgb=sample["frames"][frame_idx],
            cfg=effective_cfg,
            particles=particle_cloud,
            true_pos=current_true,
            hidden=hidden,
            max_particles=max_particles,
            show_particles=show_particles,
            secondary_particles=secondary_cloud,
        )
        overlay_panel = make_panel(overlay, f"{METHOD_SHORT_LABELS[primary_method]} belief + truth", panel_scale)
        world = draw_3d_belief_panel(
            cfg=effective_cfg,
            particles=particle_cloud,
            true_history=history,
            current_true=current_true,
            hidden=hidden,
            max_particles=max_particles,
            show_particles=show_particles,
            secondary_particles=secondary_cloud,
            occluders=sample["occluders"],
            obstacles=sample["obstacles"],
        )
        world_panel = make_panel(world, f"3D {METHOD_SHORT_LABELS[primary_method]} belief", panel_scale)
        top = hstack_with_border([rgb_panel, depth_panel, overlay_panel, world_panel])
        metrics_panel = draw_metric_panel(metrics, frame_idx, width=top.shape[1], comparison=comparison)
        frames.append(vstack([top, metrics_panel]))

    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"seed_{seed}_belief3d.gif"
    imageio.mimsave(gif_path, frames, duration=1000.0 / max(1, fps))
    mp4_path = output_dir / f"seed_{seed}_belief3d.mp4"
    mp4_error = write_mp4(mp4_path, frames, fps=fps) if write_video else None

    preview_path = output_dir / f"seed_{seed}_belief3d_preview.png"
    Image.fromarray(frames[min(len(frames) - 1, max(0, len(frames) // 2))]).save(preview_path)

    requested_methods = ["constant", "geometry"]
    if mode == "compare_all":
        requested_methods.extend(["image", "jepa"])
    method_metadata = {}

    def method_rgbd_flag(method: str) -> Optional[bool]:
        if method == "image":
            return bool(image_encoder_rgbd)
        if method == "jepa":
            return bool(jepa_rgbd)
        return None

    def method_belief_head(method: str) -> Optional[str]:
        if method == "jepa" and belief_jepa is not None:
            return str(getattr(belief_jepa, "belief_head", "single_gaussian"))
        return None

    def method_mixture_enabled(method: str) -> Optional[bool]:
        if method == "jepa" and belief_jepa is not None:
            return bool(getattr(belief_jepa, "mixture_enabled", False))
        return None

    def method_structured_context(method: str) -> Optional[bool]:
        if method == "jepa" and belief_jepa is not None:
            return bool(getattr(belief_jepa, "use_structured_context", False))
        return None

    for method in requested_methods:
        trace = traces.get(method)
        if trace is None:
            checkpoint = image_encoder_ckpt if method == "image" else belief_jepa_ckpt if method == "jepa" else None
            method_metadata[method] = {
                "label": METHOD_LABELS[method],
                "available": False,
                "checkpoint": checkpoint,
                "rgbd": method_rgbd_flag(method),
                "belief_head": method_belief_head(method),
                "mixture_enabled": method_mixture_enabled(method),
                "structured_context": method_structured_context(method),
                "mean_expected_distance": None,
                "mean_surprise": None,
                "mean_entropy": None,
                "mean_coverage_90": None,
                "mean_calibration_error_90": None,
                "primary": False,
            }
            continue
        trace_metrics = trace["metrics"]
        method_metadata[method] = {
            "label": trace["label"],
            "available": True,
            "checkpoint": trace["checkpoint"],
            "rgbd": method_rgbd_flag(method),
            "belief_head": method_belief_head(method),
            "mixture_enabled": method_mixture_enabled(method),
            "structured_context": method_structured_context(method),
            "mean_expected_distance": finite_mean(trace_metrics["expected_distance"]),
            "mean_surprise": finite_mean(trace_metrics["surprise"]),
            "mean_entropy": finite_mean(trace_metrics["entropy"]),
            "mean_coverage_90": finite_mean(trace_metrics.get("coverage_90", [])),
            "mean_calibration_error_90": finite_mean(trace_metrics.get("calibration_error_90", [])),
            "primary": method == primary_method,
        }

    serializable = {
        "seed": seed,
        "focus_object": focus_obj,
        "mode": mode_label,
        "primary_method": primary_method,
        "primary_method_requested": primary_method_override,
        "scenario": sample["metadata"]["scenario"],
        "obs_len": obs_len,
        "horizon": horizon,
        "total_frames": total_steps,
        "target": target_metadata,
        "metrics": metrics,
        "comparison_metrics": comparison,
        "phase_timeline": metrics.get("phase", []),
        "phase_labels": list(dict.fromkeys(metrics.get("phase", []))),
        "method_metadata": method_metadata,
        "jepa_ema_enabled": bool(jepa_ema_enabled) if "jepa" in traces else None,
        "artifacts": {
            "gif": str(gif_path),
            "mp4": str(mp4_path) if write_video and mp4_error is None else None,
            "mp4_error": mp4_error,
            "preview": str(preview_path),
        },
    }
    metrics_path = output_dir / f"seed_{seed}_belief3d_metrics.json"
    metrics_path.write_text(json.dumps(serializable, indent=2))
    print(f"Wrote {gif_path}")
    if write_video and mp4_error is None:
        print(f"Wrote {mp4_path}")
    elif write_video:
        print(f"Skipped MP4 export: {mp4_error}")
    print(f"Wrote {preview_path}")
    print(f"Wrote {metrics_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    device = get_device(config["project"].get("device", "auto"))
    scene_cfg = scene3d_config_from_data_cfg(config["data3d"])
    output_dir = Path(args.output_dir)
    image_encoder = None
    image_encoder_rgbd = False
    belief_jepa = None
    jepa_rgbd = False
    jepa_ema_enabled = False
    if args.mode == "compare_all":
        if args.encoder_ckpt is None:
            args.encoder_ckpt = latest_checkpoint("*_train_belief3d_encoder*/checkpoints/best.pt")
        if args.jepa_ckpt is None:
            args.jepa_ckpt = latest_jepa_checkpoint()
        if args.encoder_ckpt is not None:
            image_encoder, image_encoder_rgbd = load_image_encoder(config, device, args.encoder_ckpt)
            modality = "RGB-D" if image_encoder_rgbd else "RGB"
            print(f"Loaded {modality} image-to-belief checkpoint: {args.encoder_ckpt}")
        else:
            print("No image-to-belief checkpoint found; compare_all will skip image mode.")
        if args.jepa_ckpt is not None:
            belief_jepa, jepa_rgbd, jepa_ema_enabled = load_belief_jepa(config, device, args.jepa_ckpt)
            print(f"Loaded Belief-JEPA checkpoint: {args.jepa_ckpt}")
        else:
            print("No Belief-JEPA checkpoint found; compare_all will skip JEPA mode.")
    for seed in args.seeds:
        build_demo_for_seed(
            seed=seed,
            config=config,
            scene_cfg=scene_cfg,
            output_dir=output_dir,
            device=device,
            fps=int(args.fps),
            max_particles=int(args.max_particles),
            panel_scale=max(1, int(args.panel_scale)),
            scenario=args.scenario,
            mode=args.mode,
            image_encoder=image_encoder,
            image_encoder_ckpt=args.encoder_ckpt,
            image_encoder_rgbd=image_encoder_rgbd,
            belief_jepa=belief_jepa,
            belief_jepa_ckpt=args.jepa_ckpt,
            jepa_rgbd=jepa_rgbd,
            jepa_ema_enabled=jepa_ema_enabled,
            primary_method_override=args.primary_method,
            write_video=not bool(args.skip_mp4),
        )


if __name__ == "__main__":
    main()
