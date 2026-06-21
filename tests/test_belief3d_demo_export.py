from __future__ import annotations

import json
import math
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.export_belief3d_demo_assets import build_demo_for_seed, choose_primary_trace, finite_mean
from src.data3d.dataset3d import scene3d_config_from_data_cfg
from src.train.utils import get_device, load_config


class Belief3DDemoExportTest(unittest.TestCase):
    def test_choose_primary_trace_uses_lowest_finite_expected_distance(self) -> None:
        comparison = {
            "constant": {"expected_distance": [math.nan, 0.8, 0.7]},
            "geometry": {"expected_distance": [math.nan, 0.2, 0.3]},
            "jepa": {"expected_distance": [math.nan, 0.5, 0.4]},
        }
        self.assertAlmostEqual(finite_mean(comparison["constant"]["expected_distance"]), 0.75)
        self.assertEqual(choose_primary_trace(comparison), "geometry")

    def test_compare_all_export_records_missing_learned_methods(self) -> None:
        config = load_config("configs/belief3d_smoke.yaml")
        config["belief"]["num_particles"] = 8
        scene_cfg = scene3d_config_from_data_cfg(config["data3d"])
        with tempfile.TemporaryDirectory() as tmpdir:
            with warnings.catch_warnings(), redirect_stdout(StringIO()):
                warnings.simplefilter("ignore")
                build_demo_for_seed(
                    seed=2026,
                    config=config,
                    scene_cfg=scene_cfg,
                    output_dir=Path(tmpdir),
                    device=get_device("cpu"),
                    fps=4,
                    max_particles=8,
                    panel_scale=1,
                    scenario="structured_occlusion",
                    mode="compare_all",
                )
            metrics_path = Path(tmpdir) / "seed_2026_belief3d_metrics.json"
            payload = json.loads(metrics_path.read_text())
            self.assertIn(payload["primary_method"], {"constant", "geometry"})
            self.assertIn("constant", payload["comparison_metrics"])
            self.assertIn("geometry", payload["comparison_metrics"])
            self.assertIn("entropy", payload["metrics"])
            self.assertIn("coverage_90", payload["metrics"])
            self.assertIn("calibration_error_90", payload["metrics"])
            self.assertIn("mean_entropy", payload["method_metadata"]["constant"])
            self.assertIn("mean_coverage_90", payload["method_metadata"]["constant"])
            self.assertFalse(payload["method_metadata"]["image"]["available"])
            self.assertFalse(payload["method_metadata"]["jepa"]["available"])
            self.assertTrue(Path(payload["artifacts"]["gif"]).exists())
            if payload["artifacts"]["mp4"] is None:
                self.assertIsInstance(payload["artifacts"]["mp4_error"], str)
            else:
                self.assertTrue(Path(payload["artifacts"]["mp4"]).exists())
                self.assertIsNone(payload["artifacts"]["mp4_error"])
            self.assertTrue(Path(payload["artifacts"]["preview"]).exists())


if __name__ == "__main__":
    unittest.main()
