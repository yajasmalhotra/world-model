#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_belief3d import evaluate_split, load_belief_jepa, make_loader
from src.train.utils import get_device, load_checkpoint, load_config, set_seed


DEFAULT_SPLITS = ["test_structured_occlusion", "test_impossible_reappearance"]
TABLE_METRICS = [
    "target_hidden_expected_distance",
    "target_reappearance_surprise",
    "target_hidden_nll",
    "jepa_latent_mse",
    "jepa_mixture_nll",
    "jepa_mixture_entropy",
    "jepa_pred_target_cosine",
    "jepa_target_latent_std",
    "jepa_pred_latent_std",
    "jepa_target_counterfactual_selectivity",
]
LOWER_IS_BETTER = [
    "target_hidden_expected_distance",
    "target_reappearance_surprise",
    "target_hidden_nll",
    "jepa_latent_mse",
    "jepa_mixture_nll",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate named Belief-JEPA EMA/SIGReg checkpoint ablations.")
    parser.add_argument("--config", type=str, default="configs/belief3d_smoke.yaml")
    parser.add_argument("--output-dir", type=str, default="results/belief_jepa3d_ablation")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Named checkpoint as NAME=PATH. Repeat for EMA/no-EMA/no-SIGReg variants.",
    )
    parser.add_argument("--split", action="append", default=None, help="Target split to evaluate. Repeatable.")
    return parser.parse_args()


def numeric(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_float if math.isfinite(value_float) else None


def parse_checkpoint_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Checkpoint spec has an empty name: {spec!r}")
        return name, Path(path).expanduser()
    path = Path(spec).expanduser()
    run_name = path.parent.parent.name if path.name else path.name
    return run_name or "jepa", path


def latest_checkpoint(pattern: str, reject_tokens: Iterable[str] = ()) -> Optional[Path]:
    candidates = sorted((ROOT / "runs").glob(pattern))
    rejects = tuple(reject_tokens)
    for candidate in reversed(candidates):
        text = str(candidate)
        if all(token not in text for token in rejects):
            return candidate
    return None


def discover_checkpoints() -> List[tuple[str, Path]]:
    discovered: List[tuple[str, Path]] = []
    ema = latest_checkpoint("*_train_belief_jepa3d/checkpoints/best.pt", reject_tokens=("noema", "nosigreg"))
    no_ema = latest_checkpoint("*_train_belief_jepa3d_noema/checkpoints/best.pt")
    no_sigreg = latest_checkpoint("*_train_belief_jepa3d_nosigreg/checkpoints/best.pt")
    if ema is not None:
        discovered.append(("ema_sigreg", ema))
    if no_ema is not None:
        discovered.append(("no_ema", no_ema))
    if no_sigreg is not None:
        discovered.append(("no_sigreg", no_sigreg))
    return discovered


def checkpoint_metadata(name: str, path: Path, device: torch.device) -> Dict[str, object]:
    ckpt = load_checkpoint(path, device)
    config = ckpt.get("config", {})
    train_cfg = config.get("train_belief3d", {}) if isinstance(config, dict) else {}
    sigreg_weight = ckpt.get("sigreg_weight", train_cfg.get("sigreg_weight"))
    sigreg_sketches = ckpt.get("sigreg_sketches", train_cfg.get("sigreg_sketches"))
    sigreg_scale = ckpt.get("sigreg_scale", train_cfg.get("sigreg_scale"))
    visual_invariance_weight = ckpt.get("visual_invariance_weight", train_cfg.get("visual_invariance_weight"))
    return {
        "variant": name,
        "checkpoint": str(path),
        "ema_enabled": bool(ckpt.get("ema_enabled", False)),
        "ema_decay": numeric(ckpt.get("ema_decay")),
        "rgbd": bool(ckpt.get("rgbd", False)),
        "sigreg_weight": numeric(sigreg_weight),
        "sigreg_sketches": numeric(sigreg_sketches),
        "sigreg_scale": numeric(sigreg_scale),
        "visual_invariance_weight": numeric(visual_invariance_weight),
        "target_encoder": ckpt.get("target_encoder", "legacy_per_state"),
        "belief_head": ckpt.get("belief_head", "single_gaussian"),
        "mixture_components": numeric(ckpt.get("mixture_components")),
        "structured_context": bool(ckpt.get("structured_context", False)),
        "visual_geometry_weight": numeric(ckpt.get("visual_geometry_weight", 1.0)),
        "context_encoder": ckpt.get("context_encoder", "rgb"),
        "epoch": ckpt.get("epoch"),
        "best_metric": numeric(ckpt.get("best_metric")),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = ["variant", "split", "ema_enabled", "sigreg_weight", "checkpoint"]
    for metric in TABLE_METRICS:
        if any(metric in row for row in rows):
            fieldnames.append(metric)
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_anchor(rows: List[Dict[str, object]], split: str) -> Optional[Dict[str, object]]:
    candidates = [
        row
        for row in rows
        if row.get("split") == split and bool(row.get("ema_enabled")) and (numeric(row.get("sigreg_weight")) or 0.0) > 0.0
    ]
    if not candidates:
        return None
    named = [row for row in candidates if str(row.get("variant")) == "ema_sigreg"]
    return (named or candidates)[0]


def best_variant(rows: List[Dict[str, object]], split: str, metric: str, higher_is_better: bool = False) -> Optional[Dict[str, object]]:
    scored = []
    for row in rows:
        if row.get("split") != split:
            continue
        value = numeric(row.get(metric))
        if value is not None:
            scored.append((value, row))
    if not scored:
        return None
    return sorted(scored, key=lambda item: item[0], reverse=higher_is_better)[0][1]


def summarize_claims(rows: List[Dict[str, object]]) -> Dict[str, object]:
    splits = sorted({str(row.get("split")) for row in rows})
    variants = sorted({str(row.get("variant")) for row in rows})
    claims: Dict[str, object] = {"splits": splits, "variants": variants, "variant_count": len(variants)}
    claims["best_by_split"] = {}
    for split in splits:
        split_best: Dict[str, object] = {}
        for metric in LOWER_IS_BETTER:
            row = best_variant(rows, split, metric)
            if row is not None:
                split_best[metric] = {"variant": row.get("variant"), "value": numeric(row.get(metric))}
        cosine_best = best_variant(rows, split, "jepa_pred_target_cosine", higher_is_better=True)
        if cosine_best is not None:
            split_best["jepa_pred_target_cosine"] = {
                "variant": cosine_best.get("variant"),
                "value": numeric(cosine_best.get("jepa_pred_target_cosine")),
            }
        selectivity_best = best_variant(rows, split, "jepa_target_counterfactual_selectivity", higher_is_better=True)
        if selectivity_best is not None:
            split_best["jepa_target_counterfactual_selectivity"] = {
                "variant": selectivity_best.get("variant"),
                "value": numeric(selectivity_best.get("jepa_target_counterfactual_selectivity")),
            }
        claims["best_by_split"][split] = split_best

    comparisons = []
    for split in splits:
        anchor = find_anchor(rows, split)
        if anchor is None:
            continue
        for row in rows:
            if row is anchor or row.get("split") != split:
                continue
            delta: Dict[str, object] = {"split": split, "baseline_variant": anchor.get("variant"), "variant": row.get("variant")}
            for metric in LOWER_IS_BETTER:
                anchor_value = numeric(anchor.get(metric))
                row_value = numeric(row.get(metric))
                if anchor_value is not None and row_value is not None:
                    delta[f"{metric}_anchor_improvement"] = row_value - anchor_value
            comparisons.append(delta)
    claims["comparisons"] = comparisons
    return claims


def format_float(value: object) -> str:
    value_float = numeric(value)
    if value_float is None:
        return "n/a"
    return f"{value_float:.4f}"


def markdown_report(rows: List[Dict[str, object]], claims: Dict[str, object]) -> str:
    lines = [
        "# Belief-JEPA EMA/SIGReg Ablation",
        "",
        "This table evaluates context-only Belief-JEPA predictions while using privileged future state only for latent diagnostics.",
        "",
        "## Variants",
        "",
        "| variant | EMA | SIGReg | visual inv | visual geom | target encoder | belief head | structured | RGB-D | checkpoint |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]
    seen = set()
    for row in rows:
        variant = str(row.get("variant"))
        if variant in seen:
            continue
        seen.add(variant)
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    str(bool(row.get("ema_enabled"))),
                    format_float(row.get("sigreg_weight")),
                    format_float(row.get("visual_invariance_weight")),
                    format_float(row.get("visual_geometry_weight")),
                    str(row.get("target_encoder", "")),
                    str(row.get("belief_head", "")),
                    str(bool(row.get("structured_context"))),
                    str(bool(row.get("rgbd"))),
                    f"`{row.get('checkpoint')}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Targeted Split Metrics",
            "",
            "| split | variant | target dist | reappear surprise | target NLL | latent MSE | mix NLL | mix entropy | cosine | JEPA cf selectivity | target std | pred std |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item.get("split")), str(item.get("variant")))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("split", "")),
                    str(row.get("variant", "")),
                    format_float(row.get("target_hidden_expected_distance")),
                    format_float(row.get("target_reappearance_surprise")),
                    format_float(row.get("target_hidden_nll")),
                    format_float(row.get("jepa_latent_mse")),
                    format_float(row.get("jepa_mixture_nll")),
                    format_float(row.get("jepa_mixture_entropy")),
                    format_float(row.get("jepa_pred_target_cosine")),
                    format_float(row.get("jepa_target_counterfactual_selectivity")),
                    format_float(row.get("jepa_target_latent_std")),
                    format_float(row.get("jepa_pred_latent_std")),
                ]
            )
            + " |"
        )

    comparisons = claims.get("comparisons", [])
    if isinstance(comparisons, list) and comparisons:
        lines.extend(
            [
                "",
                "## EMA+SIGReg Deltas",
                "",
                "Positive values mean the EMA+SIGReg anchor was lower on lower-is-better metrics.",
                "",
                "| split | variant | target dist | reappear surprise | target NLL | latent MSE |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in comparisons:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("split", "")),
                        str(item.get("variant", "")),
                        format_float(item.get("target_hidden_expected_distance_anchor_improvement")),
                        format_float(item.get("target_reappearance_surprise_anchor_improvement")),
                        format_float(item.get("target_hidden_nll_anchor_improvement")),
                        format_float(item.get("jepa_latent_mse_anchor_improvement")),
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def evaluate_ablation(
    config: Dict[str, Any],
    checkpoint_specs: List[tuple[str, Path]],
    splits: List[str],
    device: torch.device,
) -> List[Dict[str, object]]:
    manifest_dir = ROOT / config["data3d"]["manifest_dir"]
    batch_size = int(config["eval"].get("batch_size", 8))
    rows: List[Dict[str, object]] = []
    for name, checkpoint in checkpoint_specs:
        checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
        metadata = checkpoint_metadata(name, checkpoint, device)
        model, jepa_rgbd, jepa_ema_enabled = load_belief_jepa(config, device, str(checkpoint))
        for split in splits:
            manifest_path = manifest_dir / f"{split}.jsonl"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing split manifest: {manifest_path}")
            loader = make_loader(manifest_path, config["data3d"], batch_size=batch_size)
            metrics = evaluate_split(
                loader,
                config,
                device,
                mode="jepa",
                jepa=model,
                jepa_rgbd=jepa_rgbd,
                jepa_ema_enabled=jepa_ema_enabled,
            )
            row = {**metadata, "split": split, **metrics, "jepa_ema_enabled": float(jepa_ema_enabled)}
            rows.append(row)
            print({"variant": name, "split": split, **{metric: row.get(metric) for metric in TABLE_METRICS}})
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    device = get_device(config["project"].get("device", "auto"))
    splits = args.split or DEFAULT_SPLITS
    checkpoint_specs = [parse_checkpoint_spec(spec) for spec in args.checkpoint] if args.checkpoint else discover_checkpoints()
    if len(checkpoint_specs) < 2:
        raise RuntimeError("Need at least two JEPA checkpoints for an ablation. Pass --checkpoint NAME=PATH.")

    rows = evaluate_ablation(config, checkpoint_specs, splits, device)
    claims = summarize_claims(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "belief_jepa3d_ablation.csv"
    json_path = output_dir / "belief_jepa3d_ablation.json"
    md_path = output_dir / "belief_jepa3d_ablation.md"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps({"claims": claims, "rows": rows}, indent=2), encoding="utf-8")
    md_path.write_text(markdown_report(rows, claims), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
