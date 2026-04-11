from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import read_manifest
from .scene_generator import SceneConfig, SyntheticSceneGenerator


SCENE_CONFIG_KEYS = {
    "image_size",
    "seq_len",
    "obs_len",
    "min_objects",
    "max_objects",
    "min_occluders",
    "max_occluders",
    "velocity_scale",
    "object_size_min",
    "object_size_max",
    "occluder_layout",
}


def scene_config_from_data_cfg(data_cfg: Dict[str, Any]) -> SceneConfig:
    subset = {k: v for k, v in data_cfg.items() if k in SCENE_CONFIG_KEYS}
    return SceneConfig(**subset)


class SyntheticSceneDataset(Dataset):
    def __init__(self, manifest_path: str | Path, data_cfg: Dict[str, Any]):
        self.manifest_path = Path(manifest_path)
        self.rows = read_manifest(self.manifest_path)
        self.scene_cfg = scene_config_from_data_cfg(data_cfg)
        self.generator = SyntheticSceneGenerator(self.scene_cfg)
        self.obs_len = self.scene_cfg.obs_len
        self.seq_len = self.scene_cfg.seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        sample = self.generator.generate(
            seed=int(row["seed"]),
            overrides=row.get("overrides", {}),
            tags=row.get("tags", []),
        )
        frames = sample["frames"].astype(np.float32) / 255.0
        state = sample["state"].astype(np.float32)
        object_mask = sample["object_mask"].astype(np.float32)
        occluders = sample["occluders"].astype(np.float32)

        obs_frames = frames[: self.obs_len]
        future_frames = frames[self.obs_len :]
        obs_state = state[: self.obs_len]
        future_state = state[self.obs_len :]
        obs_mask = object_mask[: self.obs_len]
        future_mask = object_mask[self.obs_len :]

        return {
            "scene_id": row["scene_id"],
            "seed": int(row["seed"]),
            "tags": row.get("tags", []),
            "obs_frames": torch.from_numpy(obs_frames).permute(0, 3, 1, 2).contiguous(),
            "future_frames": torch.from_numpy(future_frames).permute(0, 3, 1, 2).contiguous(),
            "obs_state": torch.from_numpy(obs_state),
            "future_state": torch.from_numpy(future_state),
            "obs_mask": torch.from_numpy(obs_mask),
            "future_mask": torch.from_numpy(future_mask),
            "occluders": torch.from_numpy(occluders),
        }


def collate_scenes(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    tensor_keys = {
        "obs_frames",
        "future_frames",
        "obs_state",
        "future_state",
        "obs_mask",
        "future_mask",
        "occluders",
    }
    for key in batch[0].keys():
        if key in tensor_keys:
            output[key] = torch.stack([item[key] for item in batch], dim=0)
        else:
            output[key] = [item[key] for item in batch]
    return output


def split_paths(manifest_dir: str | Path) -> Tuple[Path, Path]:
    manifest_dir = Path(manifest_dir)
    return manifest_dir / "train.jsonl", manifest_dir / "val.jsonl"
