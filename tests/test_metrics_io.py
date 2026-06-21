from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.train.utils import append_metrics


class MetricsIOTest(unittest.TestCase):
    def test_append_metrics_preserves_union_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            append_metrics(run_dir, {"split": "structured", "mode": "constant", "hidden_nll": 1.0})
            append_metrics(
                run_dir,
                {
                    "split": "structured",
                    "mode": "geometry",
                    "hidden_nll": 0.5,
                    "counterfactual_selectivity": 0.12,
                },
            )
            append_metrics(run_dir, {"split": "structured", "mode": "jepa", "jepa_latent_mse": 1.3})

            metrics_path = run_dir / "metrics.csv"
            with metrics_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 3)
            self.assertIn("counterfactual_selectivity", rows[0])
            self.assertIn("jepa_latent_mse", rows[0])
            self.assertEqual(rows[0]["hidden_nll"], "1.0")
            self.assertEqual(rows[1]["counterfactual_selectivity"], "0.12")
            self.assertEqual(rows[2]["jepa_latent_mse"], "1.3")


if __name__ == "__main__":
    unittest.main()
