from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_belief3d_evidence import audit_jepa_ablation
from scripts.report_belief_jepa3d_ablation import markdown_report, summarize_claims, write_csv


class BeliefJEPA3DAblationReportTest(unittest.TestCase):
    def _rows(self) -> list[dict[str, object]]:
        return [
            {
                "variant": "ema_sigreg",
                "split": "test_structured_occlusion",
                "checkpoint": "runs/ema/checkpoints/best.pt",
                "ema_enabled": True,
                "rgbd": False,
                "sigreg_weight": 0.05,
                "target_hidden_expected_distance": 0.5,
                "target_reappearance_surprise": 7.0,
                "target_hidden_nll": 1.1,
                "jepa_latent_mse": 1.2,
                "jepa_pred_target_cosine": 0.3,
                "jepa_target_latent_std": 0.14,
                "jepa_pred_latent_std": 0.06,
            },
            {
                "variant": "no_ema",
                "split": "test_structured_occlusion",
                "checkpoint": "runs/noema/checkpoints/best.pt",
                "ema_enabled": False,
                "rgbd": False,
                "sigreg_weight": 0.05,
                "target_hidden_expected_distance": 0.7,
                "target_reappearance_surprise": 9.0,
                "target_hidden_nll": 1.5,
                "jepa_latent_mse": 1.4,
                "jepa_pred_target_cosine": 0.2,
                "jepa_target_latent_std": 0.15,
                "jepa_pred_latent_std": 0.05,
            },
            {
                "variant": "no_sigreg",
                "split": "test_structured_occlusion",
                "checkpoint": "runs/nosigreg/checkpoints/best.pt",
                "ema_enabled": True,
                "rgbd": False,
                "sigreg_weight": 0.0,
                "target_hidden_expected_distance": 0.6,
                "target_reappearance_surprise": 8.0,
                "target_hidden_nll": 1.3,
                "jepa_latent_mse": 1.3,
                "jepa_pred_target_cosine": 0.25,
                "jepa_target_latent_std": 0.1,
                "jepa_pred_latent_std": 0.04,
            },
        ]

    def test_claims_compare_ema_sigreg_anchor_to_ablations(self) -> None:
        claims = summarize_claims(self._rows())
        self.assertEqual(claims["variant_count"], 3)
        self.assertEqual(
            claims["best_by_split"]["test_structured_occlusion"]["target_reappearance_surprise"]["variant"],
            "ema_sigreg",
        )
        comparisons = claims["comparisons"]
        self.assertEqual(len(comparisons), 2)
        no_ema = next(item for item in comparisons if item["variant"] == "no_ema")
        self.assertAlmostEqual(no_ema["target_reappearance_surprise_anchor_improvement"], 2.0)

    def test_markdown_and_csv_include_variant_metadata(self) -> None:
        rows = self._rows()
        claims = summarize_claims(rows)
        report = markdown_report(rows, claims)
        self.assertIn("EMA/SIGReg Ablation", report)
        self.assertIn("no_sigreg", report)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ablation.csv"
            write_csv(path, rows)
            header = path.read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("sigreg_weight", header)
        self.assertIn("jepa_pred_target_cosine", header)

    def test_evidence_audit_accepts_ablation_report(self) -> None:
        rows = self._rows()
        payload = {"claims": summarize_claims(rows), "rows": rows}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ablation.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            checks = audit_jepa_ablation(path)
        failures = [check for check in checks if check.status == "fail"]
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
