#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.manifest import ManifestSpec, build_manifest_rows, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic manifest files.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def load_project_config(path: str) -> dict:
    config_path = Path(path)
    if config_path.suffix.lower() == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    try:
        import yaml  # type: ignore

        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to read YAML configs. Install dependencies from environment.yml "
            "or pass a JSON config to --config."
        ) from exc


def main() -> None:
    args = parse_args()
    cfg = load_project_config(args.config)
    data_cfg = cfg["data"]
    output_dir = Path(args.output_dir or data_cfg["manifest_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_count = int(data_cfg["train_count"])
    val_count = int(data_cfg["val_count"])
    test_count = int(data_cfg["test_count"])

    specs = [
        ManifestSpec("train", 0, train_count, tags=["train"], overrides={}),
        ManifestSpec("val", 1_000_000, val_count, tags=["val"], overrides={}),
        ManifestSpec("test", 2_000_000, test_count, tags=["test"], overrides={}),
        ManifestSpec(
            "test_long_occlusion",
            3_000_000,
            test_count,
            tags=["test", "long_occlusion"],
            overrides={"seq_len": 28, "obs_len": 8},
        ),
        ManifestSpec(
            "test_unseen_speed",
            4_000_000,
            test_count,
            tags=["test", "unseen_speed"],
            overrides={"velocity_scale": float(data_cfg["velocity_scale"]) * 1.6},
        ),
        ManifestSpec(
            "test_unseen_occluders",
            5_000_000,
            test_count,
            tags=["test", "unseen_occluders"],
            overrides={"occluder_layout": "edge_bias"},
        ),
    ]

    index = {}
    for spec in specs:
        rows = build_manifest_rows(spec)
        path = output_dir / f"{spec.name}.jsonl"
        write_manifest(path, rows)
        index[spec.name] = {"path": str(path), "count": len(rows), "tags": spec.tags, "overrides": spec.overrides or {}}
        print(f"Wrote {len(rows):4d} rows -> {path}")

    index_path = output_dir / "manifest_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote index -> {index_path}")


if __name__ == "__main__":
    main()
