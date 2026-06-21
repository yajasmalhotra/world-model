from __future__ import annotations

import unittest

import torch

from src.eval.counterfactual import counterfactual_delta_metrics, move_boxes_to_far_corner
from src.models.belief_state import ParticleBeliefConfig, rollout_geometry_aware_particle_belief


class CounterfactualSensitivityTest(unittest.TestCase):
    def test_physical_obstacle_moves_belief_but_visual_control_does_not(self) -> None:
        cfg = ParticleBeliefConfig(
            num_particles=8,
            init_pos_noise=0.0,
            init_vel_noise=0.0,
            process_pos_noise=0.0,
            process_vel_noise=0.0,
            world_min=-1.0,
            world_max=1.0,
        )
        init_state = torch.zeros(1, 1, 12)
        init_state[..., 0:3] = torch.tensor([-0.10, 0.0, 0.0])
        init_state[..., 3:6] = torch.tensor([0.08, 0.0, 0.0])
        init_state[..., 8] = 0.02
        object_mask = torch.ones(1, 1)
        obstacles = torch.tensor([[[0.02, -0.18, -0.18, 0.10, 0.18, 0.18]]], dtype=torch.float32)
        moved_obstacles = move_boxes_to_far_corner(obstacles, world_min=-1.0, world_max=1.0)

        base_particles, base_weights = rollout_geometry_aware_particle_belief(
            init_state,
            object_mask,
            obstacles,
            horizon=5,
            cfg=cfg,
        )
        moved_particles, moved_weights = rollout_geometry_aware_particle_belief(
            init_state,
            object_mask,
            moved_obstacles,
            horizon=5,
            cfg=cfg,
        )
        visual_particles, visual_weights = rollout_geometry_aware_particle_belief(
            init_state,
            object_mask,
            obstacles,
            horizon=5,
            cfg=cfg,
        )
        target_state = torch.zeros(1, 5, 1, 12)
        target_state[..., 7] = 1.0
        future_mask = torch.ones(1, 5, 1)

        physical = counterfactual_delta_metrics(
            base_particles,
            base_weights,
            moved_particles,
            moved_weights,
            target_state,
            future_mask,
            prefix="physical",
        )
        visual = counterfactual_delta_metrics(
            base_particles,
            base_weights,
            visual_particles,
            visual_weights,
            target_state,
            future_mask,
            prefix="visual",
        )

        self.assertGreater(physical["physical_belief_delta"], 0.05)
        self.assertAlmostEqual(visual["visual_belief_delta"], 0.0, places=6)
        self.assertGreater(physical["physical_belief_delta"] - visual["visual_belief_delta"], 0.05)


if __name__ == "__main__":
    unittest.main()
