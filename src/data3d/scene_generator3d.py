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


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _normalize_id(idx: int, max_count: int) -> float:
    if max_count <= 1:
        return 0.0
    return idx / float(max_count - 1)


def _to_px(x: float, y: float, cfg: Scene3DConfig) -> tuple[float, float]:
    lo = cfg.world_min
    span = cfg.world_max - cfg.world_min
    u = (x - lo) / span
    v = (y - lo) / span
    px = u * (cfg.image_size - 1)
    py = (1.0 - v) * (cfg.image_size - 1)
    return float(px), float(py)


def _box_contains_xy(point: np.ndarray, box: np.ndarray) -> bool:
    x, y = point[:2].tolist()
    x0, y0, _z0, x1, y1, _z1 = box.tolist()
    return x0 <= x <= x1 and y0 <= y <= y1


def point_occluded(point: np.ndarray, occluders: np.ndarray) -> bool:
    """
    Orthographic camera convention: camera looks along -z, so larger z is closer.
    A point is occluded when its projected x/y lies inside an occluder whose front
    slab is closer to the camera than the point.
    """
    for occ in occluders:
        if not np.any(occ):
            continue
        if not _box_contains_xy(point, occ):
            continue
        occ_z0 = float(min(occ[2], occ[5]))
        if occ_z0 > float(point[2]):
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

    def _draw_object(self, draw: ImageDraw.ImageDraw, state: np.ndarray, cfg: Scene3DConfig) -> None:
        x = float(state[STATE_INDEX_3D["x"]])
        y = float(state[STATE_INDEX_3D["y"]])
        z = float(state[STATE_INDEX_3D["z"]])
        size = float(state[STATE_INDEX_3D["size"]])
        shape_norm = float(state[STATE_INDEX_3D["shape_id"]])
        color_norm = float(state[STATE_INDEX_3D["color_id"]])
        shape_id = int(round(shape_norm * (len(SHAPES_3D) - 1)))
        color_id = int(round(color_norm * (len(COLORS_3D) - 1)))
        color = COLORS_3D[color_id % len(COLORS_3D)]
        cx, cy = _to_px(x, y, cfg)
        depth_scale = 0.75 + 0.35 * ((z - cfg.world_min) / (cfg.world_max - cfg.world_min))
        radius = max(2.0, size * cfg.image_size * 0.5 * depth_scale)
        left, top, right, bottom = cx - radius, cy - radius, cx + radius, cy + radius
        if SHAPES_3D[shape_id % len(SHAPES_3D)] == "cube":
            draw.rectangle([left, top, right, bottom], fill=color)
        else:
            draw.ellipse([left, top, right, bottom], fill=color)

    def _draw_occluder(self, draw: ImageDraw.ImageDraw, occ: np.ndarray, cfg: Scene3DConfig) -> None:
        x0, y0, _z0, x1, y1, _z1 = occ.tolist()
        px0, py1 = _to_px(x0, y0, cfg)
        px1, py0 = _to_px(x1, y1, cfg)
        draw.rectangle([px0, py0, px1, py1], fill=OCCLUDER_COLOR_3D)

    def render_state(self, state_t: np.ndarray, object_mask_t: np.ndarray, occluders: np.ndarray) -> np.ndarray:
        cfg = self.config
        image = Image.new("RGB", (cfg.image_size, cfg.image_size), BACKGROUND_COLOR_3D)
        draw = ImageDraw.Draw(image)
        drawables: list[tuple[float, str, int]] = []
        for o in range(cfg.max_objects):
            if object_mask_t[o] >= 0.5:
                drawables.append((float(state_t[o, STATE_INDEX_3D["z"]]), "object", o))
        for k, occ in enumerate(occluders):
            if np.any(occ):
                drawables.append((float((occ[2] + occ[5]) * 0.5), "occluder", k))
        # Draw from far to near because camera is at positive z.
        for _depth, kind, idx in sorted(drawables, key=lambda item: item[0]):
            if kind == "object":
                self._draw_object(draw, state_t[idx], cfg)
            else:
                self._draw_occluder(draw, occluders[idx], cfg)
        return np.asarray(image, dtype=np.uint8)

    def render_sequence(self, states: np.ndarray, object_mask: np.ndarray, occluders: np.ndarray) -> np.ndarray:
        frames = np.zeros((states.shape[0], self.config.image_size, self.config.image_size, 3), dtype=np.uint8)
        for t in range(states.shape[0]):
            frames[t] = self.render_state(states[t], object_mask[t], occluders)
        return frames

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
            num_objects = int(rng.integers(cfg.min_objects, cfg.max_objects + 1))
            occluders = self._sample_occluders(rng, cfg)
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
                for i in range(num_objects):
                    pos = state_t[i, STATE_INDEX_3D["x"] : STATE_INDEX_3D["z"] + 1]
                    occluded = point_occluded(pos, occluders)
                    state_t[i, STATE_INDEX_3D["visible"]] = 0.0 if occluded else 1.0
                    state_t[i, STATE_INDEX_3D["occluded"]] = 1.0 if occluded else 0.0
                states[t] = state_t
                object_mask[t] = object_mask_t

            frames = self.render_sequence(states, object_mask, occluders)
            return {
                "frames": frames,
                "state": states,
                "object_mask": object_mask,
                "occluders": occluders.astype(np.float32),
                "metadata": {
                    "seed": int(seed),
                    "num_objects": int(num_objects),
                    "tags": tags or [],
                    "config": asdict(cfg),
                },
            }
        finally:
            self.config = old_cfg
