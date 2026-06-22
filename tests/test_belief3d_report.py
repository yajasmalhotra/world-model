from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.report_belief3d import flatten_summary, markdown_report, summarize_claims, write_csv


class Belief3DReportTest(unittest.TestCase):
    def _summary(self) -> dict:
        return {
            "run_type": "evaluate_belief3d",
            "splits": {
                "constant::test_structured_occlusion": {
                    "target_reappearance_surprise": 12.0,
                    "target_hidden_expected_distance": 0.3,
                },
                "geometry::test_structured_occlusion": {
                    "target_reappearance_surprise": 4.0,
                    "target_hidden_expected_distance": 0.1,
                    "target_counterfactual_physical_belief_delta": 0.2,
                    "target_counterfactual_visual_belief_delta": 0.0,
                    "target_counterfactual_selectivity": 0.2,
                },
                "jepa::test_structured_occlusion": {
                    "target_reappearance_surprise": 8.0,
                    "jepa_latent_mse": 1.5,
                    "jepa_mixture_nll": 2.5,
                    "jepa_mixture_entropy": 0.9,
                    "jepa_ema_enabled": 1.0,
                    "jepa_mixture_enabled": 1.0,
                },
            },
        }

    def test_report_claims_capture_geometry_and_jepa_evidence(self) -> None:
        run_dir = Path("runs/example_evaluate_belief3d")
        rows = flatten_summary(run_dir, self._summary())
        claims = summarize_claims(rows)

        self.assertEqual(claims["structured_geometry_gain"]["surprise_reduction"], 8.0)
        self.assertEqual(claims["structured_counterfactual"]["selectivity"], 0.2)
        self.assertTrue(claims["jepa"]["ema_enabled"])
        self.assertTrue(claims["jepa"]["mixture_enabled"])
        self.assertEqual(claims["jepa"]["mean_mixture_nll"], 2.5)

        report = markdown_report(run_dir, rows, claims)
        self.assertIn("Geometry-aware belief reduces", report)
        self.assertIn("Counterfactual selectivity", report)
        self.assertIn("Belief-JEPA evaluation includes latent diagnostics", report)
        self.assertIn("mean mixture NLL", report)

    def test_report_csv_uses_union_schema(self) -> None:
        rows = flatten_summary(Path("runs/example"), self._summary())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "table.csv"
            write_csv(out, rows)
            text = out.read_text(encoding="utf-8")
        self.assertIn("target_counterfactual_selectivity", text.splitlines()[0])
        self.assertIn("jepa_latent_mse", text.splitlines()[0])
        self.assertIn("jepa_mixture_nll", text.splitlines()[0])
        self.assertIn("jepa_mixture_enabled", text.splitlines()[0])


if __name__ == "__main__":
    unittest.main()
