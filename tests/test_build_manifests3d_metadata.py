from __future__ import annotations

import unittest

from scripts.build_manifests3d import compact_target_metadata, is_targeted_spec
from src.data.manifest import ManifestSpec


class BuildManifests3DMetadataTest(unittest.TestCase):
    def test_targeted_specs_are_detected_from_scenario_override(self) -> None:
        targeted = ManifestSpec(
            "test_structured_occlusion",
            17_000_000,
            1,
            tags=["test"],
            overrides={"scenario": "structured_occlusion"},
        )
        random = ManifestSpec("test", 12_000_000, 1, tags=["test"], overrides={})
        self.assertTrue(is_targeted_spec(targeted))
        self.assertFalse(is_targeted_spec(random))

    def test_compact_target_metadata_keeps_required_traceability_fields(self) -> None:
        compact = compact_target_metadata(
            {
                "target_object_index": 0,
                "scenario": "impossible_reappearance",
                "path_mode": "impossible_jump",
                "occlusion_start": 8,
                "occlusion_end": 15,
                "reappearance_frame": 16,
                "hidden_frames": [8, 9, 10],
                "obstacle_ids": [0],
                "occluder_ids": [0],
                "collision_or_turn_frames": [16],
                "valid_route_id": "teleport_reappearance",
                "is_impossible_event": True,
                "trajectory_velocities": [[0.0, 0.0, 0.0]],
            }
        )
        self.assertEqual(compact["scenario"], "impossible_reappearance")
        self.assertEqual(compact["path_mode"], "impossible_jump")
        self.assertTrue(compact["is_impossible_event"])
        self.assertNotIn("trajectory_velocities", compact)


if __name__ == "__main__":
    unittest.main()
