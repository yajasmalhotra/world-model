from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw


STATE_FIELDS_3D = [
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "visible",
    "occluded",
    "size",
    "shape_id",
    "color_id",
    "object_id",
]
STATE_INDEX_3D = {name: idx for idx, name in enumerate(STATE_FIELDS_3D)}
STATE_DIM_3D = len(STATE_FIELDS_3D)

SHAPES_3D = ("sphere", "cube")
COLORS_3D = (
    (228, 87, 46),
    (41, 128, 185),
    (39, 174, 96),
    (241, 196, 15),
    (155, 89, 182),
    (230, 126, 34),
)
BACKGROUND_COLOR_3D = (248, 249, 250)
OCCLUDER_COLOR_3D = (74, 79, 87)


@dataclass
class Scene3DConfig:
    image_size: int = 64
    seq_len: int = 24
    obs_len: int = 8
    min_objects: int = 2
    max_objects: int = 5
    min_occluders: int = 1
    max_occluders: int = 3
    velocity_scale: float = 0.045
    object_size_min: float = 0.07
    object_size_max: float = 0.15
    world_min: float = -1.0
    world_max: float = 1.0
    occluder_layout: str = "center_bias"
    camera_z: float = 3.2
    focal_length: float = 2.0
    ambient_light: float = 0.28
    light_dir_x: float = -0.35
    light_dir_y: float = 0.45
    light_dir_z: float = 0.82
    scenario: str = "random"
    path_mode: str = "linear"
    target_min_hidden: int = 5
    target_max_hidden: int = 14
    target_min_visible_tail: int = 5
    target_occluder_count_min: int = 1
    target_occluder_count_max: int = 3


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _normalize_id(idx: int, max_count: int) -> float:
    if max_count <= 1:
        return 0.0
    return idx / float(max_count - 1)


def _project_point(x: float, y: float, z: float, cfg: Scene3DConfig) -> tuple[float, float, float, float]:
    depth = max(cfg.camera_z - z, 1e-4)
    scale = cfg.focal_length / depth
    center = (cfg.image_size - 1) * 0.5
    px = center + x * scale * center
    py = center - y * scale * center
    return float(px), float(py), float(depth), float(scale)


def _light_vector(cfg: Scene3DConfig) -> np.ndarray:
    light = np.array([cfg.light_dir_x, cfg.light_dir_y, cfg.light_dir_z], dtype=np.float32)
    return light / max(float(np.linalg.norm(light)), 1e-6)


def _shade_color(color: tuple[int, int, int], shade: float) -> np.ndarray:
    rgb = np.asarray(color, dtype=np.float32)
    return np.clip(rgb * shade, 0.0, 255.0)


def _box_contains_xy(point: np.ndarray, box: np.ndarray) -> bool:
    x, y = point[:2].tolist()
    x0, y0, _z0, x1, y1, _z1 = box.tolist()
    return x0 <= x <= x1 and y0 <= y <= y1


def _nonzero_boxes(boxes: np.ndarray) -> list[np.ndarray]:
    return [box for box in boxes if np.any(box)]


def merge_boxes(*box_groups: np.ndarray, max_count: int) -> np.ndarray:
    merged = np.zeros((max_count, 6), dtype=np.float32)
    cursor = 0
    for group in box_groups:
        for box in _nonzero_boxes(group):
            if cursor >= max_count:
                return merged
            merged[cursor] = box.astype(np.float32)
            cursor += 1
    return merged


def point_occluded(point: np.ndarray, occluders: np.ndarray) -> bool:
    """
    Camera convention: camera is at positive z and looks along -z, so larger z
    is closer. A point is occluded when its x/y lies inside an occluder whose
    front slab is closer to the camera than the point.
    """
    for occ in occluders:
        if not np.any(occ):
            continue
        if not _box_contains_xy(point, occ):
            continue
        occ_front_z = float(max(occ[2], occ[5]))
        if occ_front_z > float(point[2]):
            return True
    return False


class SyntheticScene3DGenerator:
    def __init__(self, config: Scene3DConfig):
        self.config = config

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "SyntheticScene3DGenerator":
        return SyntheticScene3DGenerator(Scene3DConfig(**config_dict))

    def resolve_config(self, overrides: Optional[Dict[str, Any]] = None) -> Scene3DConfig:
        if not overrides:
            return self.config
        cfg = self.config
        for key, value in overrides.items():
            if hasattr(cfg, key):
                cfg = replace(cfg, **{key: value})
        return cfg

    def _sample_occluders(self, rng: np.random.Generator, cfg: Scene3DConfig) -> np.ndarray:
        num_occluders = int(rng.integers(cfg.min_occluders, cfg.max_occluders + 1))
        occluders = np.zeros((cfg.max_occluders, 6), dtype=np.float32)
        span = cfg.world_max - cfg.world_min
        for idx in range(num_occluders):
            w = float(rng.uniform(0.18, 0.34) * span)
            h = float(rng.uniform(0.18, 0.34) * span)
            d = float(rng.uniform(0.05, 0.12) * span)
            if cfg.occluder_layout == "edge_bias":
                cx = float(rng.choice([rng.uniform(-0.75, -0.35), rng.uniform(0.35, 0.75)]))
                cy = float(rng.choice([rng.uniform(-0.75, -0.35), rng.uniform(0.35, 0.75)]))
            else:
                cx = float(rng.uniform(-0.35, 0.35))
                cy = float(rng.uniform(-0.35, 0.35))
            # Place occluders between camera and many objects, but inside bounds.
            cz = float(rng.uniform(0.15, 0.75))
            x0 = _clamp(cx - w / 2.0, cfg.world_min + 0.05, cfg.world_max - 0.05)
            y0 = _clamp(cy - h / 2.0, cfg.world_min + 0.05, cfg.world_max - 0.05)
            z0 = _clamp(cz - d / 2.0, cfg.world_min + 0.05, cfg.world_max - 0.05)
            x1 = _clamp(x0 + w, cfg.world_min + 0.05, cfg.world_max - 0.05)
            y1 = _clamp(y0 + h, cfg.world_min + 0.05, cfg.world_max - 0.05)
            z1 = _clamp(z0 + d, cfg.world_min + 0.05, cfg.world_max - 0.05)
            occluders[idx] = np.array([x0, y0, z0, x1, y1, z1], dtype=np.float32)
        return occluders

    def _sample_object_state(
        self, rng: np.random.Generator, cfg: Scene3DConfig, occluders: np.ndarray, object_idx: int
    ) -> np.ndarray:
        size = float(rng.uniform(cfg.object_size_min, cfg.object_size_max))
        speed = float(rng.uniform(0.45, 1.35) * cfg.velocity_scale)
        direction = rng.normal(size=(3,))
        direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
        velocity = direction * speed
        margin = size + 0.06

        for _ in range(128):
            pos = rng.uniform(cfg.world_min + margin, cfg.world_max - margin, size=(3,))
            if not point_occluded(pos.astype(np.float32), occluders):
                break
        else:
            pos = rng.uniform(cfg.world_min + margin, cfg.world_max - margin, size=(3,))

        shape_id = int(rng.integers(0, len(SHAPES_3D)))
        color_id = int(rng.integers(0, len(COLORS_3D)))
        state = np.zeros((STATE_DIM_3D,), dtype=np.float32)
        state[STATE_INDEX_3D["x"] : STATE_INDEX_3D["z"] + 1] = pos.astype(np.float32)
        state[STATE_INDEX_3D["vx"] : STATE_INDEX_3D["vz"] + 1] = velocity.astype(np.float32)
        state[STATE_INDEX_3D["visible"]] = 1.0
        state[STATE_INDEX_3D["occluded"]] = 0.0
        state[STATE_INDEX_3D["size"]] = size
        state[STATE_INDEX_3D["shape_id"]] = _normalize_id(shape_id, len(SHAPES_3D))
        state[STATE_INDEX_3D["color_id"]] = _normalize_id(color_id, len(COLORS_3D))
        state[STATE_INDEX_3D["object_id"]] = _normalize_id(object_idx, cfg.max_objects)
        return state

    def _make_object_state(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        size: float,
        shape_id: int,
        color_id: int,
        object_idx: int,
        cfg: Scene3DConfig,
    ) -> np.ndarray:
        state = np.zeros((STATE_DIM_3D,), dtype=np.float32)
        state[STATE_INDEX_3D["x"] : STATE_INDEX_3D["z"] + 1] = pos.astype(np.float32)
        state[STATE_INDEX_3D["vx"] : STATE_INDEX_3D["vz"] + 1] = vel.astype(np.float32)
        state[STATE_INDEX_3D["visible"]] = 1.0
        state[STATE_INDEX_3D["occluded"]] = 0.0
        state[STATE_INDEX_3D["size"]] = float(size)
        state[STATE_INDEX_3D["shape_id"]] = _normalize_id(shape_id, len(SHAPES_3D))
        state[STATE_INDEX_3D["color_id"]] = _normalize_id(color_id, len(COLORS_3D))
        state[STATE_INDEX_3D["object_id"]] = _normalize_id(object_idx, cfg.max_objects)
        return state

    def _target_hidden_window(self, rng: np.random.Generator, cfg: Scene3DConfig) -> tuple[int, int, int]:
        min_tail = max(1, min(int(cfg.target_min_visible_tail), max(1, cfg.seq_len - cfg.obs_len - 2)))
        start_low = min(max(1, cfg.obs_len), max(1, cfg.seq_len - min_tail - 2))
        start_high = max(start_low, min(cfg.obs_len + 3, cfg.seq_len - min_tail - 2))
        hidden_start = int(rng.integers(start_low, start_high + 1))
        max_hidden = max(1, min(int(cfg.target_max_hidden), cfg.seq_len - hidden_start - min_tail))
        min_hidden = max(1, min(int(cfg.target_min_hidden), max_hidden))
        hidden_duration = int(rng.integers(min_hidden, max_hidden + 1))
        hidden_end = min(cfg.seq_len - min_tail - 1, hidden_start + hidden_duration - 1)
        return hidden_start, hidden_end, hidden_end + 1

    def _target_path(
        self,
        rng: np.random.Generator,
        cfg: Scene3DConfig,
        hidden_start: int,
        hidden_end: int,
        size: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        margin = size + 0.08
        lo = cfg.world_min + margin
        hi = cfg.world_max - margin
        center_xy = rng.uniform(-0.18, 0.18, size=(2,))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        direction_xy = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        hidden_mid = 0.5 * float(hidden_start + hidden_end)

        max_speed = float("inf")
        for axis in range(2):
            for t in (0.0, float(cfg.seq_len - 1)):
                coeff = float(direction_xy[axis]) * (t - hidden_mid)
                if coeff > 1e-6:
                    max_speed = min(max_speed, (hi - float(center_xy[axis])) / coeff)
                elif coeff < -1e-6:
                    max_speed = min(max_speed, (lo - float(center_xy[axis])) / coeff)
        if not np.isfinite(max_speed) or max_speed <= 0.0:
            max_speed = cfg.velocity_scale
        base_speed = cfg.velocity_scale * float(rng.uniform(0.85, 1.45))
        speed_xy = min(max_speed * 0.9, base_speed)
        if speed_xy < min(max_speed * 0.45, cfg.velocity_scale * 0.55):
            speed_xy = max_speed * float(rng.uniform(0.45, 0.75))

        z_center = float(rng.uniform(-0.55, 0.05))
        max_z_speed = min((hi - z_center) / max(float(cfg.seq_len - 1) - hidden_mid, 1.0), (z_center - lo) / max(hidden_mid, 1.0))
        z_speed = float(rng.uniform(-0.35, 0.35) * max(0.0, max_z_speed))
        velocity = np.array([direction_xy[0] * speed_xy, direction_xy[1] * speed_xy, z_speed], dtype=np.float32)
        pos0 = np.array(
            [
                center_xy[0] - velocity[0] * hidden_mid,
                center_xy[1] - velocity[1] * hidden_mid,
                z_center - velocity[2] * hidden_mid,
            ],
            dtype=np.float32,
        )
        times = np.arange(cfg.seq_len, dtype=np.float32)[:, None]
        positions = pos0[None, :] + velocity[None, :] * times
        positions = np.clip(positions, lo, hi).astype(np.float32)
        return positions, pos0, velocity

    def _target_path_plan(
        self,
        rng: np.random.Generator,
        cfg: Scene3DConfig,
        hidden_start: int,
        hidden_end: int,
        size: float,
        path_mode: str,
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
        positions, _pos0, base_velocity = self._target_path(rng, cfg, hidden_start, hidden_end, size)
        velocities = np.repeat(base_velocity.reshape(1, 3), cfg.seq_len, axis=0).astype(np.float32)
        physical_obstacles = np.zeros((cfg.max_occluders, 6), dtype=np.float32)
        metadata: Dict[str, Any] = {
            "path_mode": path_mode,
            "collision_or_turn_frames": [],
            "valid_route_id": None,
        }
        if path_mode == "linear":
            return positions, velocities, metadata, physical_obstacles

        margin = size + 0.08
        lo = cfg.world_min + margin
        hi = cfg.world_max - margin
        turn_frame = int((hidden_start + hidden_end) // 2)
        turn_frame = max(hidden_start + 1, min(hidden_end, turn_frame))

        if path_mode == "curved":
            direction = base_velocity[:2]
            norm = max(float(np.linalg.norm(direction)), 1e-6)
            perp = np.array([-direction[1] / norm, direction[0] / norm], dtype=np.float32)
            phase = np.linspace(0.0, np.pi, cfg.seq_len, dtype=np.float32)
            amplitude = float(rng.uniform(0.16, 0.26))
            curve = np.sin(phase)[:, None] * perp[None, :] * amplitude
            curve[:hidden_start] *= np.linspace(0.0, 1.0, hidden_start, dtype=np.float32)[:, None]
            curve[hidden_end + 1 :] *= np.linspace(1.0, 0.0, cfg.seq_len - hidden_end - 1, dtype=np.float32)[:, None]
            positions[:, 0:2] = np.clip(positions[:, 0:2] + curve, lo, hi)
            metadata["collision_or_turn_frames"] = [turn_frame]
            metadata["valid_route_id"] = "curved_hidden_arc"
        else:
            axis = 0 if abs(float(base_velocity[0])) >= abs(float(base_velocity[1])) else 1
            reflected = base_velocity.copy()
            reflected[axis] *= -1.0
            pre_velocity = base_velocity.copy()
            turn_pos = positions[turn_frame].copy()
            for t in range(cfg.seq_len):
                if t <= turn_frame:
                    positions[t] = positions[0] + pre_velocity * float(t)
                    velocities[t] = pre_velocity
                else:
                    positions[t] = turn_pos + reflected * float(t - turn_frame)
                    velocities[t] = reflected
            positions = np.clip(positions, lo, hi).astype(np.float32)
            half = np.array([0.08, 0.28, 0.32], dtype=np.float32)
            half[axis] = 0.045
            obstacle_center = turn_pos.copy()
            obstacle_center[axis] += np.sign(float(pre_velocity[axis]) or 1.0) * float(size * 0.7)
            box0 = np.maximum(obstacle_center - half, cfg.world_min + 0.04)
            box1 = np.minimum(obstacle_center + half, cfg.world_max - 0.04)
            physical_obstacles[0] = np.concatenate([box0, box1]).astype(np.float32)
            metadata["collision_or_turn_frames"] = [turn_frame]
            metadata["valid_route_id"] = f"bounce_axis_{axis}"

        if path_mode == "impossible_jump":
            jump_frame = hidden_end + 1
            side = -1.0 if float(positions[jump_frame - 1, 0]) > 0.0 else 1.0
            impossible_pos = positions[jump_frame - 1].copy()
            impossible_pos[0] = side * float(rng.uniform(0.62, 0.84))
            impossible_pos[1] = float(rng.uniform(-0.74, 0.74))
            impossible_pos[2] = float(rng.uniform(-0.56, 0.12))
            positions[jump_frame:] = impossible_pos[None, :]
            velocities[jump_frame:] = 0.0
            metadata["collision_or_turn_frames"] = [jump_frame]
            metadata["valid_route_id"] = "teleport_reappearance"

        velocities[1:] = positions[1:] - positions[:-1]
        velocities[0] = velocities[1] if cfg.seq_len > 1 else base_velocity
        return positions.astype(np.float32), velocities.astype(np.float32), metadata, physical_obstacles

    def _target_occluders(
        self,
        rng: np.random.Generator,
        cfg: Scene3DConfig,
        target_positions: np.ndarray,
        target_size: float,
        hidden_start: int,
        hidden_end: int,
    ) -> tuple[np.ndarray, list[int]]:
        max_count = max(1, min(cfg.max_occluders, int(cfg.target_occluder_count_max)))
        min_count = max(1, min(max_count, int(cfg.target_occluder_count_min)))
        num_occluders = int(rng.integers(min_count, max_count + 1))
        hidden_frames = np.arange(hidden_start, hidden_end + 1)
        segments = [seg for seg in np.array_split(hidden_frames, num_occluders) if len(seg) > 0]
        occluders = np.zeros((cfg.max_occluders, 6), dtype=np.float32)
        target_indices: list[int] = []
        span = cfg.world_max - cfg.world_min
        for idx, segment in enumerate(segments[: cfg.max_occluders]):
            seg_pos = target_positions[segment]
            padding = float(target_size * rng.uniform(0.35, 0.75) + rng.uniform(0.01, 0.03) * span)
            x0 = float(np.min(seg_pos[:, 0]) - padding)
            x1 = float(np.max(seg_pos[:, 0]) + padding)
            y0 = float(np.min(seg_pos[:, 1]) - padding)
            y1 = float(np.max(seg_pos[:, 1]) + padding)
            jitter = rng.uniform(-0.035, 0.035, size=(2,))
            x0 += float(jitter[0])
            x1 += float(jitter[0])
            y0 += float(jitter[1])
            y1 += float(jitter[1])
            min_box_span = 0.08 * span
            x0 = _clamp(x0, cfg.world_min + 0.04, cfg.world_max - 0.04 - min_box_span)
            y0 = _clamp(y0, cfg.world_min + 0.04, cfg.world_max - 0.04 - min_box_span)
            x1 = _clamp(x1, x0 + min_box_span, cfg.world_max - 0.04)
            y1 = _clamp(y1, y0 + min_box_span, cfg.world_max - 0.04)
            thickness = float(rng.uniform(0.05, 0.14) * span)
            target_front_min = float(np.max(seg_pos[:, 2]) + target_size + 0.08)
            front_low = min(max(target_front_min, 0.12), cfg.world_max - 0.08)
            front_z = float(rng.uniform(front_low, cfg.world_max - 0.04))
            z0 = _clamp(front_z - thickness, cfg.world_min + 0.04, cfg.world_max - 0.04)
            z1 = _clamp(front_z, z0 + 0.03 * span, cfg.world_max - 0.04)
            occluders[idx] = np.array([x0, y0, z0, x1, y1, z1], dtype=np.float32)
            target_indices.append(idx)
        return occluders, target_indices

    def _advance_one_step(self, state: np.ndarray, cfg: Scene3DConfig) -> np.ndarray:
        updated = state.copy()
        pos = state[STATE_INDEX_3D["x"] : STATE_INDEX_3D["z"] + 1].astype(np.float32)
        vel = state[STATE_INDEX_3D["vx"] : STATE_INDEX_3D["vz"] + 1].astype(np.float32)
        size = float(state[STATE_INDEX_3D["size"]])
        lo = cfg.world_min + size + 0.02
        hi = cfg.world_max - size - 0.02
        pos = pos + vel
        for axis in range(3):
            if float(pos[axis]) < lo or float(pos[axis]) > hi:
                vel[axis] *= -1.0
                pos[axis] = _clamp(float(pos[axis]), lo, hi)
        updated[STATE_INDEX_3D["x"] : STATE_INDEX_3D["z"] + 1] = pos
        updated[STATE_INDEX_3D["vx"] : STATE_INDEX_3D["vz"] + 1] = vel
        return updated

    def _background(self, cfg: Scene3DConfig) -> np.ndarray:
        height = cfg.image_size
        top = np.array([238, 243, 247], dtype=np.float32)
        bottom = np.array(BACKGROUND_COLOR_3D, dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        image = top[None, None, :] * (1.0 - ramp) + bottom[None, None, :] * ramp
        return np.repeat(image, cfg.image_size, axis=1).astype(np.float32)

    def _normalise_depth(self, zbuffer: np.ndarray, cfg: Scene3DConfig, far_depth: float) -> np.ndarray:
        near_depth = max(cfg.camera_z - cfg.world_max - cfg.object_size_max, 1e-4)
        scene_far_depth = cfg.camera_z - cfg.world_min + cfg.object_size_max
        depth = np.ones_like(zbuffer, dtype=np.float32)
        valid = zbuffer < far_depth
        if np.any(valid):
            depth[valid] = np.clip(
                (zbuffer[valid] - near_depth) / max(scene_far_depth - near_depth, 1e-6),
                0.0,
                1.0,
            )
        return depth

    def _draw_projected_polygon(
        self,
        image: np.ndarray,
        zbuffer: np.ndarray,
        points: list[tuple[float, float]],
        depth: float,
        color: np.ndarray,
        outline: Optional[np.ndarray] = None,
    ) -> None:
        if len(points) < 3:
            return
        mask_img = Image.new("L", (self.config.image_size, self.config.image_size), 0)
        mask_draw = ImageDraw.Draw(mask_img)
        mask_draw.polygon(points, fill=255)
        mask = np.asarray(mask_img) > 0
        visible = mask & (depth < zbuffer)
        if np.any(visible):
            image[visible] = color
            zbuffer[visible] = depth

        if outline is None:
            return
        line_img = Image.new("L", (self.config.image_size, self.config.image_size), 0)
        line_draw = ImageDraw.Draw(line_img)
        line_draw.line(points + [points[0]], fill=255, width=1)
        line_mask = np.asarray(line_img) > 0
        line_visible = line_mask & (depth <= zbuffer + 1e-3)
        if np.any(line_visible):
            image[line_visible] = outline
            zbuffer[line_visible] = np.minimum(zbuffer[line_visible], depth)

    def _project_face(
        self, corners: list[tuple[float, float, float]], cfg: Scene3DConfig
    ) -> tuple[list[tuple[float, float]], float]:
        projected: list[tuple[float, float]] = []
        depths: list[float] = []
        for x, y, z in corners:
            px, py, depth, _scale = _project_point(x, y, z, cfg)
            projected.append((px, py))
            depths.append(depth)
        return projected, float(np.mean(depths))

    def _draw_sphere(
        self,
        image: np.ndarray,
        zbuffer: np.ndarray,
        state: np.ndarray,
        cfg: Scene3DConfig,
        color: tuple[int, int, int],
    ) -> None:
        x = float(state[STATE_INDEX_3D["x"]])
        y = float(state[STATE_INDEX_3D["y"]])
        z = float(state[STATE_INDEX_3D["z"]])
        size = float(state[STATE_INDEX_3D["size"]])
        cx, cy, center_depth, scale = _project_point(x, y, z, cfg)
        radius = max(2.0, size * scale * cfg.image_size)
        x0 = max(0, int(np.floor(cx - radius - 1)))
        y0 = max(0, int(np.floor(cy - radius - 1)))
        x1 = min(cfg.image_size - 1, int(np.ceil(cx + radius + 1)))
        y1 = min(cfg.image_size - 1, int(np.ceil(cy + radius + 1)))
        if x0 > x1 or y0 > y1:
            return

        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        dx = (xx.astype(np.float32) - cx) / radius
        dy = (yy.astype(np.float32) - cy) / radius
        r2 = dx * dx + dy * dy
        mask = r2 <= 1.0
        if not np.any(mask):
            return

        nz = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
        normals = np.stack([dx, -dy, nz], axis=-1)
        light = _light_vector(cfg)
        lambert = np.clip(np.sum(normals * light[None, None, :], axis=-1), 0.0, 1.0)
        shade = cfg.ambient_light + (1.0 - cfg.ambient_light) * lambert
        specular = np.power(np.clip(lambert, 0.0, 1.0), 24.0) * 42.0
        base = np.asarray(color, dtype=np.float32)
        shaded = np.clip(base[None, None, :] * shade[..., None] + specular[..., None], 0.0, 255.0)
        pixel_depth = center_depth - size * nz

        local_z = zbuffer[y0 : y1 + 1, x0 : x1 + 1]
        local_img = image[y0 : y1 + 1, x0 : x1 + 1]
        visible = mask & (pixel_depth < local_z)
        if np.any(visible):
            local_img[visible] = shaded[visible]
            local_z[visible] = pixel_depth[visible]

    def _draw_cube(
        self,
        image: np.ndarray,
        zbuffer: np.ndarray,
        state: np.ndarray,
        cfg: Scene3DConfig,
        color: tuple[int, int, int],
    ) -> None:
        x = float(state[STATE_INDEX_3D["x"]])
        y = float(state[STATE_INDEX_3D["y"]])
        z = float(state[STATE_INDEX_3D["z"]])
        size = float(state[STATE_INDEX_3D["size"]])
        x0, x1 = x - size, x + size
        y0, y1 = y - size, y + size
        z0, z1 = z - size, z + size
        front = [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        side_x = (
            [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)]
            if x >= 0.0
            else [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]
        )
        side_y = (
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]
            if y >= 0.0
            else [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]
        )

        outline = _shade_color(color, 0.38)
        for corners, shade in ((side_x, 0.62), (side_y, 0.76), (front, 1.0)):
            poly, depth = self._project_face(corners, cfg)
            self._draw_projected_polygon(
                image=image,
                zbuffer=zbuffer,
                points=poly,
                depth=depth,
                color=_shade_color(color, shade),
                outline=outline,
            )

    def _draw_object(
        self,
        image: np.ndarray,
        zbuffer: np.ndarray,
        state: np.ndarray,
        cfg: Scene3DConfig,
    ) -> None:
        shape_norm = float(state[STATE_INDEX_3D["shape_id"]])
        color_norm = float(state[STATE_INDEX_3D["color_id"]])
        shape_id = int(round(shape_norm * (len(SHAPES_3D) - 1)))
        color_id = int(round(color_norm * (len(COLORS_3D) - 1)))
        color = COLORS_3D[color_id % len(COLORS_3D)]
        if SHAPES_3D[shape_id % len(SHAPES_3D)] == "cube":
            self._draw_cube(image, zbuffer, state, cfg, color)
        else:
            self._draw_sphere(image, zbuffer, state, cfg, color)

    def _draw_occluder(
        self,
        image: np.ndarray,
        zbuffer: np.ndarray,
        occ: np.ndarray,
        cfg: Scene3DConfig,
    ) -> None:
        x0, y0, z0, x1, y1, z1 = occ.tolist()
        front_z = max(float(z0), float(z1))
        corners = [(x0, y0, front_z), (x1, y0, front_z), (x1, y1, front_z), (x0, y1, front_z)]
        poly, depth = self._project_face(corners, cfg)
        z_factor = (front_z - cfg.world_min) / max(cfg.world_max - cfg.world_min, 1e-6)
        shade = 0.76 + 0.14 * z_factor
        color = _shade_color(OCCLUDER_COLOR_3D, shade)
        outline = _shade_color(OCCLUDER_COLOR_3D, 0.45)
        self._draw_projected_polygon(image, zbuffer, poly, depth, color, outline=outline)

    def render_state_with_depth(
        self, state_t: np.ndarray, object_mask_t: np.ndarray, occluders: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        image = self._background(cfg)
        far_depth = cfg.camera_z - cfg.world_min + cfg.object_size_max + 1.0
        zbuffer = np.full((cfg.image_size, cfg.image_size), far_depth, dtype=np.float32)
        for o in range(cfg.max_objects):
            if object_mask_t[o] >= 0.5:
                self._draw_object(image, zbuffer, state_t[o], cfg)
        for occ in occluders:
            if np.any(occ):
                self._draw_occluder(image, zbuffer, occ, cfg)
        depth = self._normalise_depth(zbuffer, cfg, far_depth)
        return np.clip(image, 0.0, 255.0).astype(np.uint8), depth

    def render_state(self, state_t: np.ndarray, object_mask_t: np.ndarray, occluders: np.ndarray) -> np.ndarray:
        image, _depth = self.render_state_with_depth(state_t, object_mask_t, occluders)
        return image

    def render_sequence(self, states: np.ndarray, object_mask: np.ndarray, occluders: np.ndarray) -> np.ndarray:
        frames = np.zeros((states.shape[0], self.config.image_size, self.config.image_size, 3), dtype=np.uint8)
        for t in range(states.shape[0]):
            frames[t] = self.render_state(states[t], object_mask[t], occluders)
        return frames

    def render_sequence_with_depth(
        self, states: np.ndarray, object_mask: np.ndarray, occluders: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        frames = np.zeros((states.shape[0], self.config.image_size, self.config.image_size, 3), dtype=np.uint8)
        depth = np.zeros((states.shape[0], self.config.image_size, self.config.image_size), dtype=np.float32)
        for t in range(states.shape[0]):
            frames[t], depth[t] = self.render_state_with_depth(states[t], object_mask[t], occluders)
        return frames, depth

    def _apply_visibility(self, states: np.ndarray, object_mask: np.ndarray, occluders: np.ndarray, cfg: Scene3DConfig) -> None:
        for t in range(cfg.seq_len):
            for i in range(cfg.max_objects):
                if object_mask[t, i] < 0.5:
                    continue
                pos = states[t, i, STATE_INDEX_3D["x"] : STATE_INDEX_3D["z"] + 1]
                occluded = point_occluded(pos, occluders)
                states[t, i, STATE_INDEX_3D["visible"]] = 0.0 if occluded else 1.0
                states[t, i, STATE_INDEX_3D["occluded"]] = 1.0 if occluded else 0.0

    def _target_metadata(
        self,
        states: np.ndarray,
        target_idx: int,
        scenario: str,
        path_mode: str,
        hidden_start: int,
        hidden_end: int,
        reappearance_frame: int,
        target_positions: np.ndarray,
        target_velocities: np.ndarray,
        target_occluder_indices: list[int],
        obstacle_indices: list[int],
        collision_or_turn_frames: list[int],
        valid_route_id: Optional[str],
        is_impossible_event: bool,
    ) -> Dict[str, Any]:
        occluded = states[:, target_idx, STATE_INDEX_3D["occluded"]] > 0.5
        hidden_frames = np.flatnonzero(occluded).astype(int).tolist()
        actual_start = int(hidden_frames[0]) if hidden_frames else None
        actual_end = int(hidden_frames[-1]) if hidden_frames else None
        actual_reappearance = None
        if hidden_frames:
            for t in range(hidden_frames[-1] + 1, states.shape[0]):
                if not bool(occluded[t]):
                    actual_reappearance = int(t)
                    break
        return {
            "target_object_index": int(target_idx),
            "object_index": int(target_idx),
            "scenario": scenario,
            "path_mode": path_mode,
            "planned_occlusion_start": int(hidden_start),
            "planned_occlusion_end": int(hidden_end),
            "planned_reappearance_frame": int(reappearance_frame),
            "occlusion_start": actual_start,
            "occlusion_end": actual_end,
            "reappearance_frame": actual_reappearance,
            "hidden_frames": hidden_frames,
            "visible_before": int(np.sum(~occluded[: max(hidden_start, 0)])),
            "visible_after": int(np.sum(~occluded[min(reappearance_frame, states.shape[0]) :])),
            "path_start": target_positions[0].astype(float).tolist(),
            "path_end": target_positions[-1].astype(float).tolist(),
            "velocity": target_velocities[min(hidden_start, target_velocities.shape[0] - 1)].astype(float).tolist(),
            "trajectory_velocities": target_velocities.astype(float).tolist(),
            "occluder_ids": [int(idx) for idx in target_occluder_indices],
            "occluder_indices": [int(idx) for idx in target_occluder_indices],
            "obstacle_ids": [int(idx) for idx in obstacle_indices],
            "collision_or_turn_frames": [int(idx) for idx in collision_or_turn_frames],
            "valid_route_id": valid_route_id,
            "is_impossible_event": bool(is_impossible_event),
            "num_target_occluders": int(len(target_occluder_indices)),
        }

    def _generate_random_states(
        self, rng: np.random.Generator, cfg: Scene3DConfig
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        num_objects = int(rng.integers(cfg.min_objects, cfg.max_objects + 1))
        visual_occluders = self._sample_occluders(rng, cfg)
        physical_obstacles = np.zeros((cfg.max_occluders, 6), dtype=np.float32)
        solid_screens = np.zeros((cfg.max_occluders, 6), dtype=np.float32)
        occluders = merge_boxes(visual_occluders, solid_screens, max_count=cfg.max_occluders)
        state_t = np.zeros((cfg.max_objects, STATE_DIM_3D), dtype=np.float32)
        object_mask_t = np.zeros((cfg.max_objects,), dtype=np.float32)
        for i in range(num_objects):
            state_t[i] = self._sample_object_state(rng, cfg, occluders, object_idx=i)
            object_mask_t[i] = 1.0

        states = np.zeros((cfg.seq_len, cfg.max_objects, STATE_DIM_3D), dtype=np.float32)
        object_mask = np.zeros((cfg.seq_len, cfg.max_objects), dtype=np.float32)
        for t in range(cfg.seq_len):
            if t > 0:
                for i in range(num_objects):
                    state_t[i] = self._advance_one_step(state_t[i], cfg)
            states[t] = state_t
            object_mask[t] = object_mask_t
        self._apply_visibility(states, object_mask, occluders, cfg)
        return states, object_mask, visual_occluders, physical_obstacles, solid_screens, {"num_objects": int(num_objects)}

    def _target_episode_is_valid(self, target: Dict[str, Any], cfg: Scene3DConfig) -> bool:
        start = target.get("occlusion_start")
        reappearance = target.get("reappearance_frame")
        hidden_frames = target.get("hidden_frames", [])
        if start is None or reappearance is None:
            return False
        if int(start) < int(cfg.obs_len):
            return False
        if len(hidden_frames) < max(1, int(cfg.target_min_hidden)):
            return False
        if int(target.get("visible_after", 0)) < max(1, int(cfg.target_min_visible_tail)):
            return False
        return True

    def _path_mode_for_scenario(self, cfg: Scene3DConfig, rng: np.random.Generator) -> str:
        if cfg.path_mode != "linear":
            return cfg.path_mode
        if cfg.scenario == "test_structured_occlusion" or cfg.scenario == "structured_occlusion":
            return str(rng.choice(["bounce", "curved"]))
        if cfg.scenario == "test_impossible_reappearance" or cfg.scenario == "impossible_reappearance":
            return "impossible_jump"
        return "linear"

    def _generate_targeted_occlusion_attempt(
        self, rng: np.random.Generator, cfg: Scene3DConfig
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        num_objects = int(rng.integers(max(1, cfg.min_objects), cfg.max_objects + 1))
        target_idx = 0
        target_size = float(rng.uniform(cfg.object_size_min, cfg.object_size_max))
        hidden_start, hidden_end, reappearance_frame = self._target_hidden_window(rng, cfg)
        path_mode = self._path_mode_for_scenario(cfg, rng)
        target_positions, target_velocities, path_meta, physical_obstacles = self._target_path_plan(
            rng, cfg, hidden_start, hidden_end, target_size, path_mode
        )
        visual_occluders, target_occluder_indices = self._target_occluders(
            rng,
            cfg,
            target_positions=target_positions,
            target_size=target_size,
            hidden_start=hidden_start,
            hidden_end=hidden_end,
        )
        solid_screens = np.zeros((cfg.max_occluders, 6), dtype=np.float32)
        if path_mode == "impossible_jump" and cfg.max_occluders > 1:
            # A solid screen gives the impossible split one object that is both
            # camera-occluding and motion-blocking, while the reappearance jump
            # remains physically implausible by construction.
            solid_screens[0] = visual_occluders[0]
        occluders = merge_boxes(visual_occluders, solid_screens, max_count=cfg.max_occluders)

        state_t = np.zeros((cfg.max_objects, STATE_DIM_3D), dtype=np.float32)
        object_mask_t = np.zeros((cfg.max_objects,), dtype=np.float32)
        target_shape = int(rng.integers(0, len(SHAPES_3D)))
        target_color = int(rng.integers(0, len(COLORS_3D)))
        state_t[target_idx] = self._make_object_state(
            target_positions[0],
            target_velocities[0],
            target_size,
            target_shape,
            target_color,
            object_idx=target_idx,
            cfg=cfg,
        )
        object_mask_t[target_idx] = 1.0
        for i in range(1, num_objects):
            state_t[i] = self._sample_object_state(rng, cfg, occluders, object_idx=i)
            object_mask_t[i] = 1.0

        states = np.zeros((cfg.seq_len, cfg.max_objects, STATE_DIM_3D), dtype=np.float32)
        object_mask = np.zeros((cfg.seq_len, cfg.max_objects), dtype=np.float32)
        for t in range(cfg.seq_len):
            if t > 0:
                for i in range(1, num_objects):
                    state_t[i] = self._advance_one_step(state_t[i], cfg)
            state_t[target_idx, STATE_INDEX_3D["x"] : STATE_INDEX_3D["z"] + 1] = target_positions[t]
            state_t[target_idx, STATE_INDEX_3D["vx"] : STATE_INDEX_3D["vz"] + 1] = target_velocities[t]
            states[t] = state_t
            object_mask[t] = object_mask_t
        self._apply_visibility(states, object_mask, occluders, cfg)
        obstacle_indices = [idx for idx, box in enumerate(physical_obstacles) if np.any(box)]
        target_meta = self._target_metadata(
            states=states,
            target_idx=target_idx,
            scenario=cfg.scenario,
            path_mode=path_mode,
            hidden_start=hidden_start,
            hidden_end=hidden_end,
            reappearance_frame=reappearance_frame,
            target_positions=target_positions,
            target_velocities=target_velocities,
            target_occluder_indices=target_occluder_indices,
            obstacle_indices=obstacle_indices,
            collision_or_turn_frames=path_meta.get("collision_or_turn_frames", []),
            valid_route_id=path_meta.get("valid_route_id"),
            is_impossible_event=path_mode == "impossible_jump",
        )
        return (
            states,
            object_mask,
            visual_occluders,
            physical_obstacles,
            solid_screens,
            {"num_objects": int(num_objects), "target": target_meta},
        )

    def _generate_targeted_occlusion_states(
        self, rng: np.random.Generator, cfg: Scene3DConfig
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        last_result = self._generate_targeted_occlusion_attempt(rng, cfg)
        if self._target_episode_is_valid(last_result[5]["target"], cfg):
            return last_result
        for _ in range(63):
            candidate = self._generate_targeted_occlusion_attempt(rng, cfg)
            if self._target_episode_is_valid(candidate[5]["target"], cfg):
                return candidate
            last_result = candidate
        return last_result

    def generate(
        self,
        seed: int,
        overrides: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        cfg = self.resolve_config(overrides)
        old_cfg = self.config
        self.config = cfg
        try:
            rng = np.random.default_rng(seed)
            if cfg.scenario in {"targeted_occlusion", "structured_occlusion", "impossible_reappearance"}:
                (
                    states,
                    object_mask,
                    visual_occluders,
                    physical_obstacles,
                    solid_screens,
                    scenario_metadata,
                ) = self._generate_targeted_occlusion_states(rng, cfg)
            else:
                (
                    states,
                    object_mask,
                    visual_occluders,
                    physical_obstacles,
                    solid_screens,
                    scenario_metadata,
                ) = self._generate_random_states(rng, cfg)

            occluders = merge_boxes(visual_occluders, solid_screens, max_count=cfg.max_occluders)
            obstacles = merge_boxes(physical_obstacles, solid_screens, max_count=cfg.max_occluders)
            frames, depth = self.render_sequence_with_depth(states, object_mask, occluders)
            return {
                "frames": frames,
                "depth": depth,
                "state": states,
                "object_mask": object_mask,
                "occluders": occluders.astype(np.float32),
                "visual_occluders": visual_occluders.astype(np.float32),
                "physical_obstacles": physical_obstacles.astype(np.float32),
                "solid_screens": solid_screens.astype(np.float32),
                "obstacles": obstacles.astype(np.float32),
                "metadata": {
                    "seed": int(seed),
                    "scenario": cfg.scenario,
                    "num_objects": int(scenario_metadata["num_objects"]),
                    "tags": tags or [],
                    "geometry": {
                        "visual_occluders": [idx for idx, box in enumerate(visual_occluders) if np.any(box)],
                        "physical_obstacles": [idx for idx, box in enumerate(physical_obstacles) if np.any(box)],
                        "solid_screens": [idx for idx, box in enumerate(solid_screens) if np.any(box)],
                    },
                    "config": asdict(cfg),
                    **{k: v for k, v in scenario_metadata.items() if k != "num_objects"},
                },
            }
        finally:
            self.config = old_cfg
