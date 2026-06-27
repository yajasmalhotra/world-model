#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.manifest import ManifestSpec, build_manifest_rows, write_manifest
from src.data3d.dataset3d import scene3d_config_from_data_cfg
from src.data3d.scene_generator3d import SyntheticScene3DGenerator


TARGET_METADATA_FIELDS = [
    "target_object_index",
    "object_index",
    "scenario",
    "path_mode",
    "planned_occlusion_start",
    "planned_occlusion_end",
    "planned_reappearance_frame",
    "occlusion_start",
    "occlusion_end",
    "reappearance_frame",
    "hidden_frames",
    "visible_before",
    "visible_after",
    "occluder_ids",
    "obstacle_ids",
    "collision_or_turn_frames",
    "valid_route_id",
    "is_impossible_event",
    "num_target_occluders",
]


def load_config(path: str) -> dict:
    config_path = Path(path)
    if config_path.suffix.lower() == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    import yaml  # type: ignore

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic 3D hidden-trajectory manifests.")
    parser.add_argument("--config", type=str, default="configs/belief3d.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--no-target-metadata",
        action="store_true",
        help="Do not attach regenerated target metadata to targeted split rows.",
    )
    return parser.parse_args()


def compact_target_metadata(target: Dict[str, Any]) -> Dict[str, Any]:
    return {key: target.get(key) for key in TARGET_METADATA_FIELDS if key in target}


def attach_target_metadata(
    rows: List[Dict[str, object]],
    generator: SyntheticScene3DGenerator,
) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    for row in rows:
        sample = generator.generate(
            seed=int(row["seed"]),
            overrides=row.get("overrides", {}),
            tags=row.get("tags", []),
        )
        target = sample["metadata"].get("target", {})
        enriched_row = dict(row)
        enriched_row["target"] = compact_target_metadata(target)
        enriched_row["geometry"] = sample["metadata"].get("geometry", {})
        enriched.append(enriched_row)
    return enriched


def is_targeted_spec(spec: ManifestSpec) -> bool:
    overrides = spec.overrides or {}
    return str(overrides.get("scenario", "")) in {
        "targeted_occlusion",
        "structured_occlusion",
        "impossible_reappearance",
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data3d"]
    output_dir = Path(args.output_dir or data_cfg["manifest_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_count = int(data_cfg["train_count"])
    val_count = int(data_cfg["val_count"])
    test_count = int(data_cfg["test_count"])
    seq_len = int(data_cfg["seq_len"])
    obs_len = int(data_cfg["obs_len"])
    velocity_scale = float(data_cfg["velocity_scale"])
    targeted_seq_len = max(seq_len, obs_len + 14, 24)

    specs = [
        ManifestSpec("train", 10_000_000, train_count, tags=["train"], overrides={}),
        ManifestSpec(
            "train_structured_occlusion",
            19_000_000,
            train_count,
            tags=["train", "targeted_occlusion", "structured_occlusion"],
            overrides={
                "scenario": "structured_occlusion",
                "path_mode": "linear",
                "seq_len": targeted_seq_len,
                "obs_len": obs_len,
            },
        ),
        ManifestSpec("val", 11_000_000, val_count, tags=["val"], overrides={}),
        ManifestSpec("test", 12_000_000, test_count, tags=["test"], overrides={}),
        ManifestSpec(
            "test_long_occlusion",
            13_000_000,
            test_count,
            tags=["test", "long_occlusion"],
            overrides={"seq_len": max(seq_len + 8, 32), "obs_len": obs_len},
        ),
        ManifestSpec(
            "test_unseen_speed",
            14_000_000,
            test_count,
            tags=["test", "unseen_speed"],
            overrides={"velocity_scale": velocity_scale * 1.6},
        ),
        ManifestSpec(
            "test_unseen_occluders",
            15_000_000,
            test_count,
            tags=["test", "unseen_occluders"],
            overrides={"occluder_layout": "edge_bias"},
        ),
        ManifestSpec(
            "test_targeted_occlusion",
            16_000_000,
            test_count,
            tags=["test", "targeted_occlusion"],
            overrides={
                "scenario": "targeted_occlusion",
                "seq_len": targeted_seq_len,
                "obs_len": obs_len,
            },
        ),
        ManifestSpec(
            "test_structured_occlusion",
            17_000_000,
            test_count,
            tags=["test", "targeted_occlusion", "structured_occlusion"],
            overrides={
                "scenario": "structured_occlusion",
                "path_mode": "linear",
                "seq_len": targeted_seq_len,
                "obs_len": obs_len,
            },
        ),
        ManifestSpec(
            "test_impossible_reappearance",
            18_000_000,
            test_count,
            tags=["test", "targeted_occlusion", "impossible_reappearance"],
            overrides={
                "scenario": "impossible_reappearance",
                "path_mode": "impossible_jump",
                "seq_len": targeted_seq_len,
                "obs_len": obs_len,
            },
        ),
    ]

    generator = SyntheticScene3DGenerator(scene3d_config_from_data_cfg(data_cfg))
    index = {}
    for spec in specs:
        rows = build_manifest_rows(spec)
        metadata_attached = bool(is_targeted_spec(spec) and not args.no_target_metadata)
        if metadata_attached:
            rows = attach_target_metadata(rows, generator)
        path = output_dir / f"{spec.name}.jsonl"
        write_manifest(path, rows)
        index[spec.name] = {
            "path": str(path),
            "count": len(rows),
            "tags": spec.tags,
            "overrides": spec.overrides or {},
            "target_metadata": metadata_attached,
        }
        print(f"Wrote {len(rows):4d} rows -> {path}")

    index_path = output_dir / "manifest_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote index -> {index_path}")


if __name__ == "__main__":
    main()
