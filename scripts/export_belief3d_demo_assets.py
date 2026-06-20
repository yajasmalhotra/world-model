#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data3d.dataset3d import scene3d_config_from_data_cfg
from src.data3d.scene_generator3d import STATE_INDEX_3D, Scene3DConfig, SyntheticScene3DGenerator
from src.models.belief_state import ParticleBeliefConfig, rollout_particle_belief
from src.train.utils import get_device, load_config, set_seed


RGB_YELLOW = (250, 204, 21)
RGB_MAGENTA = (217, 70, 239)
RGB_GREEN = (34, 197, 94)
RGB_BLUE = (59, 130, 246)
RGB_DARK = (17, 24, 39)
RGB_PANEL = (248, 250, 252)
RGB_GRID = (203, 213, 225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export 3D belief demo GIFs and per-frame metrics.")
    parser.add_argument("--config", type=str, default="configs/belief3d_smoke.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--output-dir", type=str, default="results/belief3d_demo_assets")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--max-particles", type=int, default=192)
    parser.add_argument("--panel-scale", type=int, default=2)
    parser.add_argument("--mode", type=str, default="physics", choices=["physics"])
    return parser.parse_args()


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
) -> np.ndarray:
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

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


def draw_metric_panel(metrics: Dict[str, List[float]], frame_idx: int, width: int, height: int = 76) -> np.ndarray:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, width, height], outline=RGB_GRID)
    names = [
        ("expected_distance", "dist", RGB_BLUE),
        ("mass_radius", "mass", RGB_GREEN),
        ("surprise", "surprise", RGB_MAGENTA),
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
        plot_w = width - plot_x0 - 10
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
        y += 22
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


def choose_focus_object(future_state: np.ndarray, future_mask: np.ndarray) -> int:
    hidden = future_mask * (future_state[..., STATE_INDEX_3D["occluded"]] > 0.5).astype(np.float32)
    counts = hidden.sum(axis=0)
    if float(counts.max()) > 0:
        return int(np.argmax(counts))
    visible_counts = future_mask.sum(axis=0)
    return int(np.argmax(visible_counts))


def per_frame_metrics(
    particles: np.ndarray,
    weights: np.ndarray,
    true_state: np.ndarray,
    obj_idx: int,
    density_sigma: float,
    mass_radius: float,
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
    return {
        "expected_distance": expected_distance.astype(float).tolist(),
        "mean_error": mean_error.astype(float).tolist(),
        "mass_radius": mass.astype(float).tolist(),
        "density_nll": density_nll.astype(float).tolist(),
        "surprise": surprise.astype(float).tolist(),
        "hidden": hidden.astype(bool).tolist(),
    }


def build_demo_for_seed(
    seed: int,
    config: Dict,
    scene_cfg: Scene3DConfig,
    output_dir: Path,
    device: torch.device,
    fps: int,
    max_particles: int,
    panel_scale: int,
) -> None:
    generator = SyntheticScene3DGenerator(scene_cfg)
    sample = generator.generate(seed=seed)
    obs_len = scene_cfg.obs_len
    future_state = sample["state"][obs_len:]
    future_mask = sample["object_mask"][obs_len:]
    horizon = future_state.shape[0]
    focus_obj = choose_focus_object(future_state, future_mask)

    init_state = torch.from_numpy(sample["state"][obs_len - 1 : obs_len]).to(device)
    object_mask = torch.from_numpy(sample["object_mask"][obs_len - 1 : obs_len]).to(device)
    particle_cfg = ParticleBeliefConfig.from_config(config["belief"], config["data3d"])
    particles, weights = rollout_particle_belief(init_state, object_mask, horizon=horizon, cfg=particle_cfg)
    particles_np = particles.squeeze(0).detach().cpu().numpy()
    weights_np = weights.squeeze(0).detach().cpu().numpy()

    metrics = per_frame_metrics(
        particles=particles_np,
        weights=weights_np,
        true_state=future_state,
        obj_idx=focus_obj,
        density_sigma=float(config["belief"]["density_sigma"]),
        mass_radius=float(config["belief"]["mass_radius"]),
    )

    frames: List[np.ndarray] = []
    true_history = sample["state"][:obs_len, focus_obj, 0:3].copy()
    for t in range(horizon):
        current_true = future_state[t, focus_obj, 0:3]
        history = np.concatenate([true_history, future_state[: t + 1, focus_obj, 0:3]], axis=0)
        hidden = bool(metrics["hidden"][t])
        rgb_panel = make_panel(sample["frames"][obs_len + t], "RGB camera", panel_scale)
        depth_panel = make_panel(depth_to_rgb(sample["depth"][obs_len + t]), "Depth", panel_scale)
        overlay = draw_camera_belief_overlay(
            rgb=sample["frames"][obs_len + t],
            cfg=scene_cfg,
            particles=particles_np[t, focus_obj],
            true_pos=current_true,
            hidden=hidden,
            max_particles=max_particles,
        )
        overlay_panel = make_panel(overlay, "Belief particles + truth", panel_scale)
        world = draw_3d_belief_panel(
            cfg=scene_cfg,
            particles=particles_np[t, focus_obj],
            true_history=history,
            current_true=current_true,
            hidden=hidden,
            max_particles=max_particles,
        )
        world_panel = make_panel(world, "3D belief / trajectory", panel_scale)
        top = hstack_with_border([rgb_panel, depth_panel, overlay_panel, world_panel])
        metrics_panel = draw_metric_panel(metrics, t, width=top.shape[1])
        frames.append(vstack([top, metrics_panel]))

    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"seed_{seed}_belief3d.gif"
    imageio.mimsave(gif_path, frames, fps=fps)

    preview_path = output_dir / f"seed_{seed}_belief3d_preview.png"
    Image.fromarray(frames[min(len(frames) - 1, max(0, len(frames) // 2))]).save(preview_path)

    serializable = {
        "seed": seed,
        "focus_object": focus_obj,
        "mode": "physics_particle_belief",
        "obs_len": obs_len,
        "horizon": horizon,
        "metrics": metrics,
        "artifacts": {"gif": str(gif_path), "preview": str(preview_path)},
    }
    metrics_path = output_dir / f"seed_{seed}_belief3d_metrics.json"
    metrics_path.write_text(json.dumps(serializable, indent=2))
    print(f"Wrote {gif_path}")
    print(f"Wrote {preview_path}")
    print(f"Wrote {metrics_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    device = get_device(config["project"].get("device", "auto"))
    scene_cfg = scene3d_config_from_data_cfg(config["data3d"])
    output_dir = Path(args.output_dir)
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
        )


if __name__ == "__main__":
    main()
