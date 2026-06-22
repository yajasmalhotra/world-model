from __future__ import annotations

import unittest

import torch

from scripts.evaluate_belief3d import metric_token, target_path_mode_masks


class Belief3DEvaluateSlicesTest(unittest.TestCase):
    def test_target_path_mode_masks_select_target_object_by_path_family(self) -> None:
        future_mask = torch.ones(3, 4, 2)
        batch = {
            "metadata": [
                {"target": {"object_index": 0, "path_mode": "bounce"}},
                {"target": {"object_index": 1, "path_mode": "curved hidden"}},
                {"target": {"target_object_index": 0, "path_mode": "impossible_jump"}},
            ]
        }

        masks = target_path_mode_masks(batch, future_mask)

        self.assertEqual(set(masks), {"bounce", "curved_hidden", "impossible_jump"})
        self.assertEqual(float(masks["bounce"][0, :, 0].sum()), 4.0)
        self.assertEqual(float(masks["bounce"][1:].sum()), 0.0)
        self.assertEqual(float(masks["curved_hidden"][1, :, 1].sum()), 4.0)
        self.assertEqual(float(masks["impossible_jump"][2, :, 0].sum()), 4.0)

    def test_metric_token_is_stable_for_report_columns(self) -> None:
        self.assertEqual(metric_token("Impossible Jump"), "impossible_jump")
        self.assertEqual(metric_token(" curved/hidden "), "curved_hidden")
        self.assertEqual(metric_token(""), "unknown")


if __name__ == "__main__":
    unittest.main()
