from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

STATE_FIELDS = [
    "x",
    "y",
    "vx",
    "vy",
    "visible",
    "occluded",
    "size",
    "shape_id",
    "color_id",
    "object_id",
]
STATE_INDEX = {name: idx for idx, name in enumerate(STATE_FIELDS)}
STATE_DIM = len(STATE_FIELDS)

SHAPES = ("circle", "square", "triangle")
COLORS = (
    (228, 87, 46),
    (41, 128, 185),
    (39, 174, 96),
    (241, 196, 15),
    (155, 89, 182),
    (230, 126, 34),
)
BACKGROUND_COLOR = (250, 250, 250)
OCCLUDER_COLOR = (90, 90, 90)


@dataclass
class SceneConfig:
    image_size: int = 64
    seq_len: int = 20
    obs_len: int = 8
    min_objects: int = 2
    max_objects: int = 4
    min_occluders: int = 1
    max_occluders: int = 2
    velocity_scale: float = 0.035
    object_size_min: float = 0.06
    object_size_max: float = 0.11
    occluder_layout: str = "center_bias"


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _to_px(v: float, image_size: int) -> float:
    return float(v * (image_size - 1))


def _normalize_id(idx: int, max_count: int) -> float:
    if max_count <= 1:
        return 0.0
    return idx / float(max_count - 1)


def point_in_rect(x: float, y: float, rect: np.ndarray) -> bool:
    x0, y0, x1, y1 = rect.tolist()
    return x0 <= x <= x1 and y0 <= y <= y1


def compute_visibility(
    positions: np.ndarray, occluders: np.ndarray, object_mask: np.ndarray
) -> np.ndarray:
    """
    Compute visibility from positions and occluders.

    Args:
        positions: [B, O, 2] normalized x/y
        occluders: [B, K, 4] normalized x0/y0/x1/y1
        object_mask: [B, O] (0 or 1)
    Returns:
        visibility: [B, O] (0 or 1)
    """
    bsz, num_objects, _ = positions.shape
    visibility = np.zeros((bsz, num_objects), dtype=np.float32)
    for b in range(bsz):
        for o in range(num_objects):
            if object_mask[b, o] < 0.5:
                continue
            x, y = positions[b, o].tolist()
            is_occluded = False
            for occ in occluders[b]:
                if point_in_rect(x, y, occ):
                    is_occluded = True
                    break
            visibility[b, o] = 0.0 if is_occluded else 1.0
    return visibility


class SyntheticSceneGenerator:
    def __init__(self, config: SceneConfig):
        self.config = config

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "SyntheticSceneGenerator":
        return SyntheticSceneGenerator(SceneConfig(**config_dict))

    def resolve_config(self, overrides: Optional[Dict[str, Any]] = None) -> SceneConfig:
        if not overrides:
            return self.config
        cfg = self.config
        for key, value in overrides.items():
            if hasattr(cfg, key):
                cfg = replace(cfg, **{key: value})
        return cfg

    def _sample_occluders(self, rng: np.random.Generator, cfg: SceneConfig) -> np.ndarray:
        num_occluders = int(rng.integers(cfg.min_occluders, cfg.max_occluders + 1))
        occluders = np.zeros((cfg.max_occluders, 4), dtype=np.float32)
        for idx in range(num_occluders):
            w = float(rng.uniform(0.18, 0.35))
            h = float(rng.uniform(0.18, 0.35))
            if cfg.occluder_layout == "edge_bias":
                cx = float(rng.choice([rng.uniform(0.2, 0.35), rng.uniform(0.65, 0.8)]))
                cy = float(rng.choice([rng.uniform(0.2, 0.35), rng.uniform(0.65, 0.8)]))
            else:
                cx = float(rng.uniform(0.35, 0.65))
                cy = float(rng.uniform(0.35, 0.65))
            x0 = _clamp(cx - w / 2.0, 0.05, 0.95)
            y0 = _clamp(cy - h / 2.0, 0.05, 0.95)
            x1 = _clamp(x0 + w, 0.05, 0.95)
            y1 = _clamp(y0 + h, 0.05, 0.95)
            occluders[idx] = np.array([x0, y0, x1, y1], dtype=np.float32)
        return occluders

    def _sample_object_state(
        self, rng: np.random.Generator, cfg: SceneConfig, occluders: np.ndarray, object_idx: int
    ) -> np.ndarray:
        size = float(rng.uniform(cfg.object_size_min, cfg.object_size_max))
        speed = float(rng.uniform(0.5, 1.5) * cfg.velocity_scale)
        theta = float(rng.uniform(0.0, 2 * np.pi))
        vx = speed * np.cos(theta)
        vy = speed * np.sin(theta)
        margin = size + 0.05

        for _ in range(128):
            x = float(rng.uniform(margin, 1.0 - margin))
            y = float(rng.uniform(margin, 1.0 - margin))
            occluded = any(point_in_rect(x, y, occ) for occ in occluders if np.any(occ))
            if not occluded:
                break
        else:
            x = float(rng.uniform(margin, 1.0 - margin))
            y = float(rng.uniform(margin, 1.0 - margin))

        shape_id = int(rng.integers(0, len(SHAPES)))
        color_id = int(rng.integers(0, len(COLORS)))

        state = np.zeros((STATE_DIM,), dtype=np.float32)
        state[STATE_INDEX["x"]] = x
        state[STATE_INDEX["y"]] = y
        state[STATE_INDEX["vx"]] = vx
        state[STATE_INDEX["vy"]] = vy
        state[STATE_INDEX["visible"]] = 1.0
        state[STATE_INDEX["occluded"]] = 0.0
        state[STATE_INDEX["size"]] = size
        state[STATE_INDEX["shape_id"]] = _normalize_id(shape_id, len(SHAPES))
        state[STATE_INDEX["color_id"]] = _normalize_id(color_id, len(COLORS))
        state[STATE_INDEX["object_id"]] = _normalize_id(object_idx, cfg.max_objects)
        return state

    def _advance_one_step(self, state: np.ndarray, cfg: SceneConfig) -> np.ndarray:
        updated = state.copy()
        x = float(state[STATE_INDEX["x"]] + state[STATE_INDEX["vx"]])
        y = float(state[STATE_INDEX["y"]] + state[STATE_INDEX["vy"]])
        vx = float(state[STATE_INDEX["vx"]])
        vy = float(state[STATE_INDEX["vy"]])
        size = float(state[STATE_INDEX["size"]])
        lo = size + 0.02
        hi = 1.0 - (size + 0.02)
        if x < lo or x > hi:
            vx *= -1.0
            x = _clamp(x, lo, hi)
        if y < lo or y > hi:
            vy *= -1.0
            y = _clamp(y, lo, hi)
        updated[STATE_INDEX["x"]] = x
        updated[STATE_INDEX["y"]] = y
        updated[STATE_INDEX["vx"]] = vx
        updated[STATE_INDEX["vy"]] = vy
        return updated

    def render_state(
        self,
        state_t: np.ndarray,
        object_mask_t: np.ndarray,
        occluders: np.ndarray,
        cfg: Optional[SceneConfig] = None,
    ) -> np.ndarray:
        cfg = cfg or self.config
        image = Image.new("RGB", (cfg.image_size, cfg.image_size), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(image)

        for o in range(cfg.max_objects):
            if object_mask_t[o] < 0.5:
                continue
            x = float(state_t[o, STATE_INDEX["x"]])
            y = float(state_t[o, STATE_INDEX["y"]])
            size = float(state_t[o, STATE_INDEX["size"]])
            shape_norm = float(state_t[o, STATE_INDEX["shape_id"]])
            color_norm = float(state_t[o, STATE_INDEX["color_id"]])
            shape_id = int(round(shape_norm * (len(SHAPES) - 1)))
            color_id = int(round(color_norm * (len(COLORS) - 1)))
            color = COLORS[color_id % len(COLORS)]

            cx = _to_px(x, cfg.image_size)
            cy = _to_px(y, cfg.image_size)
            radius = max(2.0, size * cfg.image_size * 0.5)
            left = cx - radius
            top = cy - radius
            right = cx + radius
            bottom = cy + radius

            shape_name = SHAPES[shape_id % len(SHAPES)]
            if shape_name == "square":
                draw.rectangle([left, top, right, bottom], fill=color)
            elif shape_name == "triangle":
                draw.polygon(
                    [(cx, top), (left, bottom), (right, bottom)],
                    fill=color,
                )
            else:
                draw.ellipse([left, top, right, bottom], fill=color)

        for occ in occluders:
            if not np.any(occ):
                continue
            x0, y0, x1, y1 = occ.tolist()
            draw.rectangle(
                [
                    _to_px(x0, cfg.image_size),
                    _to_px(y0, cfg.image_size),
                    _to_px(x1, cfg.image_size),
                    _to_px(y1, cfg.image_size),
                ],
                fill=OCCLUDER_COLOR,
            )
        return np.asarray(image, dtype=np.uint8)

    def render_sequence(
        self,
        states: np.ndarray,
        object_mask: np.ndarray,
        occluders: np.ndarray,
        cfg: Optional[SceneConfig] = None,
    ) -> np.ndarray:
        cfg = cfg or self.config
        frames = np.zeros((states.shape[0], cfg.image_size, cfg.image_size, 3), dtype=np.uint8)
        for t in range(states.shape[0]):
            frames[t] = self.render_state(states[t], object_mask[t], occluders, cfg=cfg)
        return frames

    def generate(
        self,
        seed: int,
        overrides: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        cfg = self.resolve_config(overrides)
        rng = np.random.default_rng(seed)

        num_objects = int(rng.integers(cfg.min_objects, cfg.max_objects + 1))
        occluders = self._sample_occluders(rng, cfg)
        state_t = np.zeros((cfg.max_objects, STATE_DIM), dtype=np.float32)
        object_mask_t = np.zeros((cfg.max_objects,), dtype=np.float32)
        for i in range(num_objects):
            state_t[i] = self._sample_object_state(rng, cfg, occluders, object_idx=i)
            object_mask_t[i] = 1.0

        states = np.zeros((cfg.seq_len, cfg.max_objects, STATE_DIM), dtype=np.float32)
        object_mask = np.zeros((cfg.seq_len, cfg.max_objects), dtype=np.float32)
        frames = np.zeros((cfg.seq_len, cfg.image_size, cfg.image_size, 3), dtype=np.uint8)

        for t in range(cfg.seq_len):
            if t > 0:
                for i in range(num_objects):
                    state_t[i] = self._advance_one_step(state_t[i], cfg)

            for i in range(num_objects):
                x = float(state_t[i, STATE_INDEX["x"]])
                y = float(state_t[i, STATE_INDEX["y"]])
                occluded = any(point_in_rect(x, y, occ) for occ in occluders if np.any(occ))
                state_t[i, STATE_INDEX["visible"]] = 0.0 if occluded else 1.0
                state_t[i, STATE_INDEX["occluded"]] = 1.0 if occluded else 0.0

            states[t] = state_t
            object_mask[t] = object_mask_t
            frames[t] = self.render_state(state_t, object_mask_t, occluders, cfg=cfg)

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

