from __future__ import annotations

import unittest

from scripts.write_belief3d_showcase import build_showcase


class Belief3DShowcaseTest(unittest.TestCase):
    def test_showcase_summarizes_claims_artifacts_and_limits(self) -> None:
        report = {
            "claims": {
                "structured_counterfactual": {"selectivity": 0.12},
                "jepa": {
                    "structured_physical_delta": 0.10,
                    "structured_visual_delta": 0.0,
                    "structured_selectivity": 0.10,
                },
            },
            "rows": [
                {
                    "split": "test_structured_occlusion",
                    "mode": "geometry",
                    "target_hidden_expected_distance": 0.16,
                    "target_hidden_nll": -3.2,
                    "target_reappearance_surprise": 5.8,
                },
                {
                    "split": "test_structured_occlusion",
                    "mode": "jepa",
                    "target_hidden_expected_distance": 0.18,
                    "target_hidden_nll": -3.0,
                    "target_reappearance_surprise": 8.1,
                },
                {
                    "split": "test_impossible_reappearance",
                    "mode": "geometry",
                    "target_hidden_expected_distance": 0.14,
                    "target_hidden_nll": -3.5,
                },
                {
                    "split": "test_impossible_reappearance",
                    "mode": "jepa",
                    "target_hidden_expected_distance": 0.17,
                    "target_hidden_nll": -3.2,
                },
                {
                    "split": "test_targeted_occlusion",
                    "mode": "geometry",
                    "target_hidden_expected_distance": 0.12,
                },
                {
                    "split": "test_targeted_occlusion",
                    "mode": "jepa",
                    "target_hidden_expected_distance": 0.16,
                },
            ],
        }
        ablation = {
            "rows": [
                {"variant": "ema_sigreg", "jepa_latent_mse": 0.26},
                {"variant": "no_ema", "jepa_latent_mse": 0.63},
            ]
        }
        demo = {"artifacts": {"gif": "demo.gif", "mp4": "demo.mp4", "metrics": "demo.json"}}
        impossible = {"artifacts": {"gif": "impossible.gif", "preview": "impossible.png"}}

        text = build_showcase(report, ablation, demo, impossible)

        self.assertIn("pixel prediction is not object permanence", text)
        self.assertIn("visual-only delta", text)
        self.assertIn("EMA target encoder", text)
        self.assertIn("Honest Limitations", text)
        self.assertIn("demo.gif", text)
        self.assertIn("impossible.png", text)
        self.assertIn("0.6300", text)
        self.assertIn("0.2600", text)


if __name__ == "__main__":
    unittest.main()
