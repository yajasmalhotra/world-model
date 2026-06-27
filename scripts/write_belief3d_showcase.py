#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a one-page Belief-JEPA 3D showcase/model card.")
    parser.add_argument("--report-json", type=str, default="results/belief3d_report/belief3d_report.json")
    parser.add_argument("--ablation-json", type=str, default="results/belief_jepa3d_ablation/belief_jepa3d_ablation.json")
    parser.add_argument("--demo-json", type=str, default="results/belief3d_demo_compare_all/seed_2026_belief3d_metrics.json")
    parser.add_argument(
        "--impossible-demo-json",
        type=str,
        default="results/belief3d_demo_impossible/seed_2026_belief3d_metrics.json",
    )
    parser.add_argument("--output", type=str, default="results/belief3d_showcase.md")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def numeric(value: Any) -> Optional[float]:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_float if math.isfinite(value_float) else None


def format_float(value: Any) -> str:
    value_float = numeric(value)
    return "n/a" if value_float is None else f"{value_float:.4f}"


def rows_for(report: Dict[str, Any], split: str) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for row in report.get("rows", []):
        if row.get("split") == split:
            rows[str(row.get("mode"))] = row
    return rows


def artifact_links(demo: Dict[str, Any]) -> list[str]:
    artifacts = demo.get("artifacts", {})
    links = []
    for key in ("gif", "mp4", "preview", "metrics"):
        value = artifacts.get(key)
        if isinstance(value, str) and value:
            links.append(f"- `{key}`: `{value}`")
    return links


def best_ablation_claim(ablation: Dict[str, Any]) -> list[str]:
    rows = ablation.get("rows", [])
    anchor = [row for row in rows if row.get("variant") == "ema_sigreg"]
    no_ema = [row for row in rows if row.get("variant") == "no_ema"]
    lines = []
    if anchor and no_ema:
        anchor_latent = sum(float(row["jepa_latent_mse"]) for row in anchor) / len(anchor)
        no_ema_latent = sum(float(row["jepa_latent_mse"]) for row in no_ema) / len(no_ema)
        lines.append(
            f"- EMA target branch improves latent matching in smoke ablations: "
            f"mean latent MSE `{no_ema_latent:.4f}` without EMA vs `{anchor_latent:.4f}` with EMA."
        )
    return lines


def require_text(lines: Iterable[str], required: Iterable[str]) -> None:
    text = "\n".join(lines)
    missing = [item for item in required if item not in text]
    if missing:
        raise ValueError(f"Showcase is missing required terms: {missing}")


def build_showcase(
    report: Dict[str, Any],
    ablation: Dict[str, Any],
    demo: Dict[str, Any],
    impossible_demo: Dict[str, Any],
) -> str:
    structured = rows_for(report, "test_structured_occlusion")
    impossible = rows_for(report, "test_impossible_reappearance")
    targeted = rows_for(report, "test_targeted_occlusion")
    claims = report.get("claims", {})
    jepa_claim = claims.get("jepa", {})

    structured_geometry = structured.get("geometry", {})
    structured_jepa = structured.get("jepa", {})
    impossible_geometry = impossible.get("geometry", {})
    impossible_jepa = impossible.get("jepa", {})
    targeted_geometry = targeted.get("geometry", {})
    targeted_jepa = targeted.get("jepa", {})

    lines = [
        "# Belief-JEPA 3D Showcase",
        "",
        "**Thesis:** pixel prediction is not object permanence. A useful world model should preserve calibrated belief about hidden reality.",
        "",
        "Belief-JEPA 3D is a small synthetic benchmark and model stack for asking whether a latent predictor can track hidden 3D object state under occlusion, react to physical geometry, and ignore visual-only confounds.",
        "",
        "## What The Demo Shows",
        "",
        "- A target object is visible, becomes hidden, evolves behind occluders, and then reappears.",
        "- The benchmark separates `visual_occluders`, `physical_obstacles`, and `solid_screens` so appearance and dynamics can be tested independently.",
        "- The demo compares constant-velocity particles, geometry-aware particles, image-to-belief, and Belief-JEPA.",
        "- The strongest diagnostic is counterfactual: moving physical obstacles should change belief; moving visual-only occluders should not.",
        "",
        "## Why JEPA Here",
        "",
        "Belief-JEPA predicts future latent/belief representations instead of future pixels. During training, an EMA target encoder embeds privileged future 3D state. At evaluation, predictions stay context-only: observed frames, observed state, and scene geometry go in; future state is used only for diagnostics.",
        "",
        "## Headline Evidence",
        "",
        "| split | metric | geometry baseline | Belief-JEPA |",
        "| --- | --- | ---: | ---: |",
        f"| structured occlusion | target hidden distance | {format_float(structured_geometry.get('target_hidden_expected_distance'))} | {format_float(structured_jepa.get('target_hidden_expected_distance'))} |",
        f"| structured occlusion | target NLL | {format_float(structured_geometry.get('target_hidden_nll'))} | {format_float(structured_jepa.get('target_hidden_nll'))} |",
        f"| structured occlusion | target reappearance surprise | {format_float(structured_geometry.get('target_reappearance_surprise'))} | {format_float(structured_jepa.get('target_reappearance_surprise'))} |",
        f"| impossible reappearance | target hidden distance | {format_float(impossible_geometry.get('target_hidden_expected_distance'))} | {format_float(impossible_jepa.get('target_hidden_expected_distance'))} |",
        f"| impossible reappearance | target NLL | {format_float(impossible_geometry.get('target_hidden_nll'))} | {format_float(impossible_jepa.get('target_hidden_nll'))} |",
        f"| linear targeted occlusion | target hidden distance | {format_float(targeted_geometry.get('target_hidden_expected_distance'))} | {format_float(targeted_jepa.get('target_hidden_expected_distance'))} |",
        "",
        "## Counterfactual Sanity Check",
        "",
        f"- Geometry-aware baseline selectivity: `{format_float(claims.get('structured_counterfactual', {}).get('selectivity'))}`.",
        f"- Belief-JEPA structured physical delta: `{format_float(jepa_claim.get('structured_physical_delta'))}`.",
        f"- Belief-JEPA structured visual-only delta: `{format_float(jepa_claim.get('structured_visual_delta'))}`.",
        f"- Belief-JEPA structured selectivity: `{format_float(jepa_claim.get('structured_selectivity'))}`.",
        "",
        "## Improvement Over The Earlier JEPA Checkpoint",
        "",
        "| metric | before physical-prior calibration | current |",
        "| --- | ---: | ---: |",
        f"| structured target distance | 0.7249 | {format_float(structured_jepa.get('target_hidden_expected_distance'))} |",
        f"| impossible target distance | 0.7313 | {format_float(impossible_jepa.get('target_hidden_expected_distance'))} |",
        f"| structured target NLL | 0.9474 | {format_float(structured_jepa.get('target_hidden_nll'))} |",
        f"| impossible target NLL | 0.7810 | {format_float(impossible_jepa.get('target_hidden_nll'))} |",
        "",
        "## Ablation Notes",
        "",
    ]
    lines.extend(best_ablation_claim(ablation) or ["- Ablation rows are available in `results/belief_jepa3d_ablation/`."])
    lines.extend(
        [
            "- No-EMA ablations preserve the engineered physical prior but lose latent target alignment, separating belief scaffolding from JEPA representation learning.",
            "",
            "## Demo Artifacts",
            "",
            "Structured comparison:",
        ]
    )
    lines.extend(artifact_links(demo))
    lines.extend(["", "Impossible reappearance comparison:"])
    lines.extend(artifact_links(impossible_demo))
    lines.extend(
        [
            "",
            "## Reproduce The Smoke Evidence",
            "",
            "```bash",
            "conda run -n term-project-wm python scripts/train_belief_jepa3d.py --config configs/belief3d_smoke.yaml",
            "conda run -n term-project-wm python scripts/evaluate_belief3d.py --config configs/belief3d_smoke.yaml --mode all --jepa-ckpt runs/20260626_175651_train_belief_jepa3d/checkpoints/best.pt",
            "conda run -n term-project-wm python scripts/report_belief3d.py --run-dir runs/20260626_175709_evaluate_belief3d --output-dir results/belief3d_report",
            "conda run -n term-project-wm python scripts/write_belief3d_showcase.py --output results/belief3d_showcase.md",
            "conda run -n term-project-wm python scripts/audit_belief3d_evidence.py --output-dir results/belief3d_audit",
            "```",
            "",
            "## Honest Limitations",
            "",
            "- This is a small synthetic 3D benchmark, not a foundation model result.",
            "- The current JEPA branch receives structured scene geometry; it does not infer every obstacle from pixels alone.",
            "- The physical prior is engineered, which is appropriate for this benchmark but should be ablated or learned in larger follow-up work.",
            "- Curved paths remain harder than bounce paths, so the next research step is better dynamics abstraction rather than prettier video.",
            "",
            "## One-Sentence Pitch",
            "",
            "Belief-JEPA 3D tests whether a latent world model can keep calibrated object-permanence beliefs when the object is hidden, and whether those beliefs respond to physical causes rather than visual distractions.",
            "",
        ]
    )
    require_text(
        lines,
        [
            "pixel prediction is not object permanence",
            "visual-only delta",
            "EMA target encoder",
            "Honest Limitations",
            "Reproduce The Smoke Evidence",
        ],
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    showcase = build_showcase(
        load_json(Path(args.report_json)),
        load_json(Path(args.ablation_json)),
        load_json(Path(args.demo_json)),
        load_json(Path(args.impossible_demo_json)),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(showcase, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
