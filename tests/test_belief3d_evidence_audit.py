from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_belief3d_evidence import (
    Check,
    audit_demo,
    audit_report,
    summarize_checks,
)


class Belief3DEvidenceAuditTest(unittest.TestCase):
    def test_summarize_checks_marks_any_failure_as_failed(self) -> None:
        summary = summarize_checks(
            [
                Check(name="one", status="pass", detail="ok"),
                Check(name="two", status="warn", detail="soft"),
                Check(name="three", status="fail", detail="bad"),
            ]
        )
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["counts"]["pass"], 1)
        self.assertEqual(summary["counts"]["warn"], 1)
        self.assertEqual(summary["counts"]["fail"], 1)

    def test_audit_demo_accepts_complete_compare_all_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ("demo.gif", "demo.mp4", "preview.png"):
                (root / name).write_bytes(b"not-empty")
            series = [float("nan"), 0.5, 0.4, 0.3]
            payload = {
                "scenario": "structured_occlusion",
                "artifacts": {"gif": "demo.gif", "mp4": "demo.mp4", "preview": "preview.png"},
                "metrics": {
                    "phase": ["observed target", "belief initialized", "hidden rollout", "reappearance / visible"],
                    "expected_distance": series,
                    "mass_radius": series,
                    "density_nll": series,
                    "surprise": series,
                    "entropy": series,
                    "coverage_50": series,
                    "coverage_70": series,
                    "coverage_90": series,
                    "calibration_error_50": series,
                    "calibration_error_70": series,
                    "calibration_error_90": series,
                },
                "comparison_metrics": {"constant": {}, "geometry": {}, "image": {}, "jepa": {}},
                "method_metadata": {
                    "constant": {"mean_expected_distance": 0.8},
                    "geometry": {"mean_expected_distance": 0.4},
                    "image": {"mean_expected_distance": 0.9},
                    "jepa": {"mean_expected_distance": 0.7},
                },
            }
            path = root / "demo.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            checks = audit_demo(path, base_dir=root)
            failures = [check for check in checks if check.status == "fail"]
            self.assertEqual(failures, [])

    def test_audit_demo_accepts_impossible_event_phase_without_baseline_win(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ("demo.gif", "demo.mp4", "preview.png"):
                (root / name).write_bytes(b"not-empty")
            series = [float("nan"), 0.5, 0.4, 0.3]
            payload = {
                "scenario": "impossible_reappearance",
                "artifacts": {"gif": "demo.gif", "mp4": "demo.mp4", "preview": "preview.png"},
                "metrics": {
                    "phase": [
                        "observed target",
                        "belief initialized",
                        "hidden rollout",
                        "impossible event",
                        "reappearance / visible",
                    ],
                    "expected_distance": series,
                    "mass_radius": series,
                    "density_nll": series,
                    "surprise": series,
                    "entropy": series,
                    "coverage_50": series,
                    "coverage_70": series,
                    "coverage_90": series,
                    "calibration_error_50": series,
                    "calibration_error_70": series,
                    "calibration_error_90": series,
                },
                "comparison_metrics": {"constant": {}, "geometry": {}, "image": {}, "jepa": {}},
                "method_metadata": {
                    "constant": {"mean_expected_distance": 0.4},
                    "geometry": {"mean_expected_distance": 0.8},
                    "image": {"mean_expected_distance": 0.9},
                    "jepa": {"mean_expected_distance": 0.7},
                },
            }
            path = root / "impossible.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            checks = audit_demo(path, base_dir=root, prefix="impossible_demo", require_baseline_failure=False)
            failures = [check for check in checks if check.status == "fail"]
            self.assertEqual(failures, [])
            self.assertIn("impossible_demo.impossible_phase", {check.name for check in checks})

    def test_audit_report_requires_geometry_counterfactual_and_jepa_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            path.write_text(
                json.dumps(
                    {
                        "claims": {
                            "structured_geometry_gain": {"surprise_reduction": 3.0},
                            "structured_counterfactual": {
                                "physical_delta": 0.2,
                                "visual_delta": 0.0,
                                "selectivity": 0.2,
                            },
                            "jepa": {"ema_enabled": True, "mean_latent_mse": 1.2},
                        },
                        "rows": [
                            {
                                "target_hidden_expected_distance": 0.1,
                                "target_reappearance_surprise": 2.0,
                                "target_hidden_entropy": 4.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            checks = audit_report(path)
            failures = [check for check in checks if check.status == "fail"]
            self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
