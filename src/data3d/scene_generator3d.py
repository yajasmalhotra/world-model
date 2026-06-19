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

            frames, depth = self.render_sequence_with_depth(states, object_mask, occluders)
            return {
                "frames": frames,
                "depth": depth,
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
