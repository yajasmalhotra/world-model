#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.manifest import read_manifest
from src.data3d.dataset3d import scene3d_config_from_data_cfg
from src.data3d.scene_generator3d import SyntheticScene3DGenerator
from src.train.utils import load_config


REQUIRED_SPLITS = {
    "train",
    "val",
    "test",
    "test_long_occlusion",
    "test_unseen_speed",
    "test_unseen_occluders",
    "test_targeted_occlusion",
    "test_structured_occlusion",
    "test_impossible_reappearance",
}
TARGET_METADATA_KEYS = {
    "target_object_index",
    "scenario",
    "path_mode",
    "occlusion_start",
    "occlusion_end",
    "reappearance_frame",
    "hidden_frames",
    "obstacle_ids",
    "occluder_ids",
    "collision_or_turn_frames",
    "valid_route_id",
    "is_impossible_event",
}
DEMO_METHODS = {"constant", "geometry", "image", "jepa"}
DEMO_METRICS = {
    "expected_distance",
    "mass_radius",
    "density_nll",
    "surprise",
    "entropy",
    "coverage_50",
    "coverage_70",
    "coverage_90",
    "calibration_error_50",
    "calibration_error_70",
    "calibration_error_90",
}


@dataclass
class Check:
    name: str
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Belief-JEPA 3D evidence artifacts against the goal spec.")
    parser.add_argument("--config", type=str, default="configs/belief3d_smoke.yaml")
    parser.add_argument("--demo-json", type=str, default="results/belief3d_demo_compare_all/seed_2026_belief3d_metrics.json")
    parser.add_argument("--report-json", type=str, default="results/belief3d_report/belief3d_report.json")
    parser.add_argument("--output-dir", type=str, default="results/belief3d_audit")
    parser.add_argument("--sample-count", type=int, default=3, help="Targeted manifest rows to regenerate per split.")
    return parser.parse_args()


def pass_check(name: str, detail: str) -> Check:
    return Check(name=name, status="pass", detail=detail)


def fail_check(name: str, detail: str) -> Check:
    return Check(name=name, status="fail", detail=detail)


def warn_check(name: str, detail: str) -> Check:
    return Check(name=name, status="warn", detail=detail)


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def finite_values(values: Iterable[Any]) -> List[float]:
    return [float(value) for value in values if is_finite_number(value)]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_artifact_path(path_value: Any, base_dir: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def nonzero_box_count(boxes: Any) -> int:
    count = 0
    for box in boxes:
        try:
            if any(float(value) != 0.0 for value in box):
                count += 1
        except TypeError:
            continue
    return count


def audit_manifests(config: Dict[str, Any], sample_count: int) -> List[Check]:
    checks: List[Check] = []
    manifest_dir = Path(config["data3d"]["manifest_dir"])
    index_path = manifest_dir / "manifest_index.json"
    if not index_path.exists():
        return [fail_check("manifests.index", f"Missing {index_path}")]
    index = load_json(index_path)
    missing = sorted(REQUIRED_SPLITS - set(index.keys()))
    if missing:
        checks.append(fail_check("manifests.required_splits", f"Missing splits: {missing}"))
    else:
        checks.append(pass_check("manifests.required_splits", f"Found {len(REQUIRED_SPLITS)} required splits."))

    for split in sorted(REQUIRED_SPLITS):
        path = manifest_dir / f"{split}.jsonl"
        if not path.exists():
            checks.append(fail_check(f"manifests.{split}.file", f"Missing {path}"))
            continue
        rows = read_manifest(path)
        if rows:
            checks.append(pass_check(f"manifests.{split}.rows", f"{len(rows)} deterministic seed rows."))
        else:
            checks.append(fail_check(f"manifests.{split}.rows", "Manifest is empty."))

    generator = SyntheticScene3DGenerator(scene3d_config_from_data_cfg(config["data3d"]))
    targeted_expectations = {
        "test_targeted_occlusion": ("targeted_occlusion", False),
        "test_structured_occlusion": ("structured_occlusion", False),
        "test_impossible_reappearance": ("impossible_reappearance", True),
    }
    for split, (expected_scenario, expected_impossible) in targeted_expectations.items():
        rows = read_manifest(manifest_dir / f"{split}.jsonl")[: max(1, int(sample_count))]
        regenerated = []
        split_failures = []
        for row in rows:
            sample = generator.generate(seed=int(row["seed"]), overrides=row.get("overrides", {}), tags=row.get("tags", []))
            target = sample["metadata"].get("target", {})
            regenerated.append(target)
            missing_keys = sorted(TARGET_METADATA_KEYS - set(target.keys()))
            if missing_keys:
                split_failures.append(f"{row['scene_id']} missing target keys {missing_keys}")
            if target.get("scenario") != expected_scenario:
                split_failures.append(f"{row['scene_id']} scenario={target.get('scenario')!r}")
            if bool(target.get("is_impossible_event")) != expected_impossible:
                split_failures.append(f"{row['scene_id']} is_impossible_event={target.get('is_impossible_event')!r}")
            if not target.get("hidden_frames"):
                split_failures.append(f"{row['scene_id']} has no hidden frames")
            if not isinstance(target.get("reappearance_frame"), int):
                split_failures.append(f"{row['scene_id']} has no integer reappearance_frame")
            if split == "test_structured_occlusion":
                if not target.get("collision_or_turn_frames"):
                    split_failures.append(f"{row['scene_id']} has no collision/turn frames")
                if target.get("path_mode") not in {"bounce", "curved"}:
                    split_failures.append(f"{row['scene_id']} has unsupported structured path_mode={target.get('path_mode')!r}")
                if not target.get("valid_route_id"):
                    split_failures.append(f"{row['scene_id']} has no valid_route_id")
                if target.get("path_mode") != "curved" and not target.get("obstacle_ids"):
                    split_failures.append(f"{row['scene_id']} has no physical obstacle ids for non-curved path")
            if split == "test_impossible_reappearance" and target.get("path_mode") != "impossible_jump":
                split_failures.append(f"{row['scene_id']} path_mode={target.get('path_mode')!r}")
            if nonzero_box_count(sample.get("visual_occluders", [])) <= 0:
                split_failures.append(f"{row['scene_id']} has no visual occluders")
            if split in {"test_structured_occlusion", "test_impossible_reappearance"} and target.get("path_mode") != "curved":
                physical_count = nonzero_box_count(sample.get("physical_obstacles", [])) + nonzero_box_count(
                    sample.get("solid_screens", [])
                )
                if physical_count <= 0:
                    split_failures.append(f"{row['scene_id']} has no physical constraints")
        if split_failures:
            checks.append(fail_check(f"targeted_metadata.{split}", "; ".join(split_failures[:5])))
        else:
            checks.append(
                pass_check(
                    f"targeted_metadata.{split}",
                    f"Regenerated {len(regenerated)} samples with required target metadata and scenario constraints.",
                )
            )
    return checks


def audit_demo(demo_json_path: Path, base_dir: Path) -> List[Check]:
    checks: List[Check] = []
    if not demo_json_path.is_absolute():
        demo_json_path = base_dir / demo_json_path
    if not demo_json_path.exists():
        return [fail_check("demo.json", f"Missing {demo_json_path}")]
    demo = load_json(demo_json_path)
    checks.append(pass_check("demo.json", f"Loaded {demo_json_path}."))

    artifacts = demo.get("artifacts", {})
    for key in ("gif", "mp4", "preview"):
        path = resolve_artifact_path(artifacts.get(key), base_dir)
        if path is not None and path.exists() and path.stat().st_size > 0:
            checks.append(pass_check(f"demo.artifact.{key}", f"{path} ({path.stat().st_size} bytes)."))
        else:
            checks.append(fail_check(f"demo.artifact.{key}", f"Missing or empty artifact for {key}: {artifacts.get(key)!r}"))

    methods = set((demo.get("comparison_metrics") or {}).keys())
    missing_methods = sorted(DEMO_METHODS - methods)
    if missing_methods:
        checks.append(fail_check("demo.comparison_methods", f"Missing comparison methods: {missing_methods}"))
    else:
        checks.append(pass_check("demo.comparison_methods", "Constant, geometry, image, and JEPA traces are present."))

    metrics = demo.get("metrics", {})
    missing_metrics = sorted(DEMO_METRICS - set(metrics.keys()))
    empty_metrics = sorted(key for key in DEMO_METRICS & set(metrics.keys()) if not finite_values(metrics.get(key, [])))
    if missing_metrics or empty_metrics:
        checks.append(fail_check("demo.metrics", f"Missing={missing_metrics}; empty={empty_metrics}"))
    else:
        checks.append(pass_check("demo.metrics", "Distance, mass, surprise, entropy, and calibration series are finite."))

    phases = [str(value) for value in metrics.get("phase", [])]
    required_phase_substrings = ["observed target", "belief initialized", "hidden rollout", "reappearance"]
    missing_phases = [phase for phase in required_phase_substrings if not any(phase in value for value in phases)]
    if missing_phases:
        checks.append(fail_check("demo.phases", f"Missing phase labels containing: {missing_phases}"))
    else:
        checks.append(pass_check("demo.phases", "Observed, initialized, hidden rollout, and reappearance phases are labeled."))
    if demo.get("scenario") == "impossible_reappearance":
        if any("impossible" in value for value in phases):
            checks.append(pass_check("demo.impossible_phase", "Impossible-event phase is labeled."))
        else:
            checks.append(fail_check("demo.impossible_phase", "Impossible scenario lacks an impossible-event phase label."))

    metadata = demo.get("method_metadata", {})
    constant = metadata.get("constant", {})
    constant_distance = constant.get("mean_expected_distance")
    stronger = [
        method
        for method, row in metadata.items()
        if method != "constant"
        and is_finite_number(row.get("mean_expected_distance"))
        and is_finite_number(constant_distance)
        and float(row["mean_expected_distance"]) < float(constant_distance)
    ]
    if stronger:
        checks.append(pass_check("demo.visible_baseline_failure", f"Methods beating constant baseline: {stronger}."))
    else:
        checks.append(fail_check("demo.visible_baseline_failure", "No available method beats constant expected distance."))
    return checks


def audit_report(report_json_path: Path) -> List[Check]:
    checks: List[Check] = []
    if not report_json_path.is_absolute():
        report_json_path = ROOT / report_json_path
    if not report_json_path.exists():
        return [fail_check("report.json", f"Missing {report_json_path}")]
    report = load_json(report_json_path)
    claims = report.get("claims", {})
    checks.append(pass_check("report.json", f"Loaded {report_json_path}."))

    gain = claims.get("structured_geometry_gain", {})
    if is_finite_number(gain.get("surprise_reduction")) and float(gain["surprise_reduction"]) > 0.0:
        checks.append(pass_check("report.geometry_gain", f"Surprise reduction={float(gain['surprise_reduction']):.4f}."))
    else:
        checks.append(fail_check("report.geometry_gain", f"Invalid geometry gain claim: {gain}"))

    cf = claims.get("structured_counterfactual", {})
    if (
        is_finite_number(cf.get("physical_delta"))
        and is_finite_number(cf.get("visual_delta"))
        and is_finite_number(cf.get("selectivity"))
        and float(cf["physical_delta"]) > float(cf["visual_delta"])
        and float(cf["selectivity"]) > 0.0
    ):
        checks.append(
            pass_check(
                "report.counterfactual_selectivity",
                f"physical={float(cf['physical_delta']):.4f}, visual={float(cf['visual_delta']):.4f}.",
            )
        )
    else:
        checks.append(fail_check("report.counterfactual_selectivity", f"Invalid counterfactual claim: {cf}"))

    jepa = claims.get("jepa", {})
    if bool(jepa.get("ema_enabled")) and is_finite_number(jepa.get("mean_latent_mse")):
        checks.append(pass_check("report.jepa_diagnostics", f"EMA JEPA latent MSE={float(jepa['mean_latent_mse']):.4f}."))
    else:
        checks.append(fail_check("report.jepa_diagnostics", f"Invalid JEPA diagnostics: {jepa}"))

    rows = report.get("rows", [])
    required_row_metrics = {"target_hidden_expected_distance", "target_reappearance_surprise", "target_hidden_entropy"}
    if rows and all(any(metric in row for row in rows) for metric in required_row_metrics):
        checks.append(pass_check("report.target_metrics", "Target-only distance, entropy, and reappearance metrics are present."))
    else:
        checks.append(fail_check("report.target_metrics", f"Missing one of {sorted(required_row_metrics)}"))
    return checks


def summarize_checks(checks: List[Check]) -> Dict[str, Any]:
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return {
        "status": "pass" if counts.get("fail", 0) == 0 else "fail",
        "counts": counts,
        "checks": [asdict(check) for check in checks],
    }


def markdown_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# Belief-JEPA 3D Evidence Audit",
        "",
        f"Overall status: `{summary['status']}`",
        "",
        "| status | count |",
        "| --- | ---: |",
    ]
    for status in ("pass", "warn", "fail"):
        lines.append(f"| {status} | {summary['counts'].get(status, 0)} |")
    lines.extend(["", "## Checks", "", "| status | check | detail |", "| --- | --- | --- |"])
    for check in summary["checks"]:
        detail = str(check["detail"]).replace("\n", " ")
        lines.append(f"| {check['status']} | `{check['name']}` | {detail} |")
    lines.append("")
    return "\n".join(lines)


def write_outputs(summary: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "belief3d_evidence_audit.json"
    md_path = output_dir / "belief3d_evidence_audit.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(markdown_report(summary), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checks: List[Check] = []
    checks.extend(audit_manifests(config, sample_count=int(args.sample_count)))
    checks.extend(audit_demo(Path(args.demo_json), base_dir=ROOT))
    checks.extend(audit_report(Path(args.report_json)))
    summary = summarize_checks(checks)
    artifacts = write_outputs(summary, Path(args.output_dir))
    print(json.dumps({"status": summary["status"], "counts": summary["counts"], "artifacts": artifacts}, indent=2))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
