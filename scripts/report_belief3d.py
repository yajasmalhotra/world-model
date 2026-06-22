#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


IMPORTANT_METRICS = [
    "target_hidden_expected_distance",
    "target_hidden_mass_radius",
    "target_hidden_nll",
    "target_reappearance_surprise",
    "target_counterfactual_physical_belief_delta",
    "target_counterfactual_visual_belief_delta",
    "target_counterfactual_selectivity",
    "jepa_latent_mse",
    "jepa_mixture_nll",
    "jepa_mixture_entropy",
    "jepa_pred_target_cosine",
    "jepa_ema_enabled",
    "jepa_mixture_enabled",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a concise Belief-JEPA 3D benchmark report.")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--run-dir", type=str, default=None, help="Specific evaluate_belief3d run directory.")
    parser.add_argument("--output-dir", type=str, default="results/belief3d_report")
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_belief3d_eval_run(runs_dir: Path) -> Path:
    candidates = [
        summary_path.parent
        for summary_path in runs_dir.glob("*/summary.json")
        if load_json(summary_path).get("run_type") == "evaluate_belief3d"
    ]
    if not candidates:
        raise FileNotFoundError(f"No evaluate_belief3d summary found under {runs_dir}")
    return sorted(candidates)[-1]


def flatten_summary(run_dir: Path, summary: Dict) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key, metrics in summary.get("splits", {}).items():
        if "::" not in key:
            continue
        mode, split = key.split("::", 1)
        row: Dict[str, object] = {"run_name": run_dir.name, "mode": mode, "split": split}
        row.update(metrics)
        rows.append(row)
    return rows


def numeric(row: Dict[str, object], key: str) -> Optional[float]:
    value = row.get(key)
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_float):
        return None
    return value_float


def row_for(rows: Iterable[Dict[str, object]], split: str, mode: str) -> Optional[Dict[str, object]]:
    for row in rows:
        if row.get("split") == split and row.get("mode") == mode:
            return row
    return None


def mean_metric(rows: Iterable[Dict[str, object]], metric: str) -> Optional[float]:
    values = [value for row in rows if (value := numeric(row, metric)) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def best_by_metric(rows: Iterable[Dict[str, object]], split: str, metric: str, higher_is_better: bool = False) -> Optional[Dict[str, object]]:
    scored = []
    for row in rows:
        if row.get("split") != split:
            continue
        value = numeric(row, metric)
        if value is not None:
            scored.append((value, row))
    if not scored:
        return None
    return sorted(scored, key=lambda item: item[0], reverse=higher_is_better)[0][1]


def summarize_claims(rows: List[Dict[str, object]]) -> Dict[str, object]:
    splits = sorted({str(row["split"]) for row in rows})
    claims: Dict[str, object] = {"splits": splits}
    structured = "test_structured_occlusion"
    geometry = row_for(rows, structured, "geometry")
    constant = row_for(rows, structured, "constant")
    if geometry:
        claims["structured_counterfactual"] = {
            "physical_delta": numeric(geometry, "target_counterfactual_physical_belief_delta"),
            "visual_delta": numeric(geometry, "target_counterfactual_visual_belief_delta"),
            "selectivity": numeric(geometry, "target_counterfactual_selectivity"),
        }
    if geometry and constant:
        constant_surprise = numeric(constant, "target_reappearance_surprise")
        geometry_surprise = numeric(geometry, "target_reappearance_surprise")
        if constant_surprise is not None and geometry_surprise is not None:
            claims["structured_geometry_gain"] = {
                "constant_target_reappearance_surprise": constant_surprise,
                "geometry_target_reappearance_surprise": geometry_surprise,
                "surprise_reduction": constant_surprise - geometry_surprise,
            }
    best_structured = best_by_metric(rows, structured, "target_reappearance_surprise")
    if best_structured:
        claims["best_structured_reappearance_mode"] = {
            "mode": best_structured["mode"],
            "target_reappearance_surprise": numeric(best_structured, "target_reappearance_surprise"),
        }
    impossible = "test_impossible_reappearance"
    if impossible in splits:
        best_impossible = best_by_metric(rows, impossible, "target_reappearance_surprise", higher_is_better=True)
        if best_impossible:
            claims["highest_impossible_surprise_mode"] = {
                "mode": best_impossible["mode"],
                "target_reappearance_surprise": numeric(best_impossible, "target_reappearance_surprise"),
            }
    jepa_rows = [row for row in rows if row.get("mode") == "jepa"]
    if jepa_rows:
        claims["jepa"] = {
            "ema_enabled": any(numeric(row, "jepa_ema_enabled") == 1.0 for row in jepa_rows),
            "mixture_enabled": any(numeric(row, "jepa_mixture_enabled") == 1.0 for row in jepa_rows),
            "mean_latent_mse": mean_metric(jepa_rows, "jepa_latent_mse"),
            "mean_mixture_nll": mean_metric(jepa_rows, "jepa_mixture_nll"),
            "mean_mixture_entropy": mean_metric(jepa_rows, "jepa_mixture_entropy"),
        }
    return claims


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = ["run_name", "split", "mode"]
    for metric in IMPORTANT_METRICS:
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


def format_float(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{value_float:.4f}"


def path_mode_tokens(row: Dict[str, object]) -> List[str]:
    prefix = "path_mode_"
    suffix = "_target_hidden_expected_distance"
    tokens = []
    for key in row.keys():
        if key.startswith(prefix) and key.endswith(suffix):
            tokens.append(key[len(prefix) : -len(suffix)])
    return sorted(tokens)


def path_mode_label(token: str) -> str:
    return token.replace("_", " ")


def markdown_report(run_dir: Path, rows: List[Dict[str, object]], claims: Dict[str, object]) -> str:
    lines = [
        "# Belief-JEPA 3D Benchmark Report",
        "",
        f"Source run: `{run_dir.name}`",
        "",
        "## Key Claims",
    ]
    geometry_gain = claims.get("structured_geometry_gain")
    if isinstance(geometry_gain, dict):
        lines.append(
            "- Geometry-aware belief reduces structured target reappearance surprise "
            f"from {format_float(geometry_gain.get('constant_target_reappearance_surprise'))} "
            f"to {format_float(geometry_gain.get('geometry_target_reappearance_surprise'))}."
        )
    counterfactual = claims.get("structured_counterfactual")
    if isinstance(counterfactual, dict):
        lines.append(
            "- Counterfactual selectivity is "
            f"{format_float(counterfactual.get('selectivity'))}: physical obstacle movement changes belief "
            f"({format_float(counterfactual.get('physical_delta'))}), visual-only control does not "
            f"({format_float(counterfactual.get('visual_delta'))})."
        )
    jepa = claims.get("jepa")
    if isinstance(jepa, dict):
        lines.append(
            "- Belief-JEPA evaluation includes latent diagnostics "
            f"(EMA enabled: `{bool(jepa.get('ema_enabled'))}`, mean latent MSE: {format_float(jepa.get('mean_latent_mse'))}, "
            f"mixture enabled: `{bool(jepa.get('mixture_enabled'))}`, mean mixture NLL: {format_float(jepa.get('mean_mixture_nll'))})."
        )
    if len(lines) == 5:
        lines.append("- No structured claims were available in this run.")

    lines.extend(
        [
            "",
            "## Target Metrics By Split",
            "",
            "| split | mode | target dist | target mass | target NLL | target reappear surprise | cf selectivity |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item.get("split")), str(item.get("mode")))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("split", "")),
                    str(row.get("mode", "")),
                    format_float(row.get("target_hidden_expected_distance")),
                    format_float(row.get("target_hidden_mass_radius")),
                    format_float(row.get("target_hidden_nll")),
                    format_float(row.get("target_reappearance_surprise")),
                    format_float(row.get("target_counterfactual_selectivity")),
                ]
            )
            + " |"
        )
    path_rows = []
    for row in sorted(rows, key=lambda item: (str(item.get("split")), str(item.get("mode")))):
        for token in path_mode_tokens(row):
            base = f"path_mode_{token}_target_"
            if numeric(row, f"{base}hidden_expected_distance") is None:
                continue
            path_rows.append((row, token, base))
    if path_rows:
        lines.extend(
            [
                "",
                "## Target Metrics By Path Mode",
                "",
                "| split | mode | path mode | target dist | target NLL | target reappear surprise | hidden count | cf selectivity |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row, token, base in path_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("split", "")),
                        str(row.get("mode", "")),
                        path_mode_label(token),
                        format_float(row.get(f"{base}hidden_expected_distance")),
                        format_float(row.get(f"{base}hidden_nll")),
                        format_float(row.get(f"{base}reappearance_surprise")),
                        format_float(row.get(f"{base}hidden_count")),
                        format_float(row.get(f"path_mode_{token}_target_counterfactual_selectivity")),
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_belief3d_eval_run(Path(args.runs_dir))
    summary = load_json(run_dir / "summary.json")
    rows = flatten_summary(run_dir, summary)
    if not rows:
        raise RuntimeError(f"No Belief3D split metrics found in {run_dir / 'summary.json'}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    claims = summarize_claims(rows)

    csv_path = output_dir / "belief3d_metrics_table.csv"
    json_path = output_dir / "belief3d_report.json"
    md_path = output_dir / "belief3d_report.md"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps({"run_dir": str(run_dir), "claims": claims, "rows": rows}, indent=2), encoding="utf-8")
    md_path.write_text(markdown_report(run_dir, rows, claims), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
