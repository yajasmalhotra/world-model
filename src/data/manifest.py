from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class ManifestSpec:
    name: str
    start_seed: int
    count: int
    tags: List[str]
    overrides: Optional[Dict[str, object]] = None


def write_manifest(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_manifest_rows(spec: ManifestSpec) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for idx in range(spec.count):
        seed = spec.start_seed + idx
        rows.append(
            {
                "scene_id": f"{spec.name}_{idx:06d}",
                "seed": seed,
                "tags": spec.tags,
                "overrides": spec.overrides or {},
            }
        )
    return rows


def read_manifest(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows

