#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate run summaries into tables/plots.")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def flatten_eval_results(run_name: str, results: Dict[str, Dict[str, float]]) -> List[Dict[str, object]]:
    rows = []
    for key, metrics in results.items():
        mode, split = key.split("::", 1)
        row = {"run_name": run_name, "mode": mode, "split": split}
        row.update(metrics)
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, object]] = []
    eval_rows: List[Dict[str, object]] = []

    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        run_name = summary_path.parent.name
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        run_type = summary.get("run_type", "unknown")
        row = {"run_name": run_name, "run_type": run_type}
        for key in (
            "best_val_rollout_rmse",
            "best_val_pos_rmse",
            "best_val_mse",
            "best_metric",
            "best_checkpoint",
        ):
            if key in summary:
                row[key] = summary[key]
        run_rows.append(row)

        if run_type == "evaluate" and "results" in summary:
            eval_rows.extend(flatten_eval_results(run_name, summary["results"]))

    run_df = pd.DataFrame(run_rows)
    run_csv = out_dir / "run_summaries.csv"
    run_df.to_csv(run_csv, index=False)
    print(f"Wrote {run_csv}")

    if not eval_rows:
        print("No evaluation summaries found.")
        return

    eval_df = pd.DataFrame(eval_rows)
    eval_csv = out_dir / "eval_metrics.csv"
    eval_df.to_csv(eval_csv, index=False)
    print(f"Wrote {eval_csv}")

    if "rollout_rmse" in eval_df.columns:
        pivot = (
            eval_df.groupby(["mode", "split"], as_index=False)["rollout_rmse"]
            .mean()
            .sort_values(["split", "rollout_rmse"])
        )
        pivot_csv = out_dir / "eval_rollout_rmse_table.csv"
        pivot.to_csv(pivot_csv, index=False)
        print(f"Wrote {pivot_csv}")

        plt.figure(figsize=(9, 5))
        for mode in sorted(pivot["mode"].unique()):
            subset = pivot[pivot["mode"] == mode]
            plt.plot(subset["split"], subset["rollout_rmse"], marker="o", label=mode)
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("Rollout RMSE")
        plt.title("World Model Rollout Error by Split")
        plt.tight_layout()
        plt.legend()
        fig_path = out_dir / "eval_rollout_rmse.png"
        plt.savefig(fig_path, dpi=140)
        plt.close()
        print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()

