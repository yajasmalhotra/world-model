from __future__ import annotations

import json
import math
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.export_belief3d_demo_assets import (
    build_demo_for_seed,
    choose_demo_primary_method,
    choose_primary_trace,
    combine_metrics,
    finite_mean,
    select_preferred_jepa_checkpoint,
    visible_prefix_metrics,
)
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

    def test_choose_demo_primary_method_honors_explicit_available_method(self) -> None:
        traces = {"constant": {}, "geometry": {}, "jepa": {}}
        comparison = {
            "constant": {"expected_distance": [0.7]},
            "geometry": {"expected_distance": [0.2]},
            "jepa": {"expected_distance": [0.5]},
        }
        self.assertEqual(choose_demo_primary_method("compare_all", traces, comparison, requested="jepa"), "jepa")
        self.assertEqual(choose_demo_primary_method("compare_all", traces, comparison, requested="auto"), "geometry")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            choose_demo_primary_method("compare_all", traces, comparison, requested="image")

    def test_jepa_checkpoint_selection_prefers_ema_sigreg(self) -> None:
        candidates = [
            Path("runs/20260101_train_belief_jepa3d_noema/checkpoints/best.pt"),
            Path("runs/20260102_train_belief_jepa3d/checkpoints/best.pt"),
            Path("runs/20260103_train_belief_jepa3d_nosigreg/checkpoints/best.pt"),
        ]
        self.assertEqual(select_preferred_jepa_checkpoint(candidates), candidates[1])

    def test_impossible_reappearance_phase_is_labeled(self) -> None:
        rollout = {
            "expected_distance": [0.2, 0.3, 0.4],
            "mean_error": [0.2, 0.3, 0.4],
            "mass_radius": [0.8, 0.6, 0.2],
            "density_nll": [1.0, 2.0, 3.0],
            "surprise": [0.1, 0.5, 1.5],
            "entropy": [2.0, 2.0, 2.0],
            "coverage_50": [1.0, 1.0, 0.0],
            "coverage_70": [1.0, 1.0, 0.0],
            "coverage_90": [1.0, 1.0, 0.0],
            "calibration_error_50": [0.5, 0.5, 0.5],
            "calibration_error_70": [0.3, 0.3, 0.7],
            "calibration_error_90": [0.1, 0.1, 0.9],
            "hidden": [True, False, False],
        }
        metrics = combine_metrics(
            visible_prefix_metrics(2),
            rollout,
            obs_len=2,
            target_metadata={"is_impossible_event": True, "reappearance_frame": 3},
        )
        self.assertEqual(metrics["phase"][3], "impossible event")

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
            self.assertEqual(payload["primary_method_requested"], "auto")
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
