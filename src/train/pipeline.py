from __future__ import annotations

from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader

from src.data.dataset import SyntheticSceneDataset, collate_scenes


def make_loader(
    manifest_path: str | Path,
    data_cfg: Dict,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    dataset = SyntheticSceneDataset(manifest_path=manifest_path, data_cfg=data_cfg)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collate_scenes,
    )


def make_train_val_loaders(config: Dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    manifest_dir = Path(data_cfg["manifest_dir"])
    train_loader = make_loader(
        manifest_path=manifest_dir / "train.jsonl",
        data_cfg=data_cfg,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
    )
    val_loader = make_loader(
        manifest_path=manifest_dir / "val.jsonl",
        data_cfg=data_cfg,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=False,
    )
    return train_loader, val_loader

