from __future__ import annotations

import unittest

import torch

from src.models.belief_jepa3d import (
    BeliefJEPA3D,
    belief_jepa_diagnostics,
    belief_jepa_loss,
    sketched_isotropic_gaussian_regularizer,
)
from src.models.belief_state import ParticleBeliefConfig, particles_from_gaussian_mixture_sequence
from scripts.evaluate_belief3d import jepa_diagnostic_outputs


class BeliefJEPAEMATest(unittest.TestCase):
    def _make_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(7)
        frames = torch.randn(2, 3, 3, 32, 32)
        future_state = torch.randn(2, 4, 2, 12)
        future_state[..., 7] = torch.tensor([0.0, 1.0, 1.0, 0.0])[None, :, None]
        future_mask = torch.ones(2, 4, 2)
        return frames, future_state, future_mask

    def _make_model(self) -> BeliefJEPA3D:
        torch.manual_seed(11)
        return BeliefJEPA3D(
            max_objects=2,
            horizon=4,
            input_channels=3,
            cnn_dim=8,
            rnn_dim=16,
            latent_dim=8,
        )

    def test_ema_target_encoder_is_frozen_and_updates(self) -> None:
        model = self._make_model()
        model.sync_ema_target_encoder()
        frames, future_state, future_mask = self._make_batch()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

        outputs = model(frames, future_state=future_state, use_ema_target=True)
        losses = belief_jepa_loss(outputs, future_state, future_mask)
        losses["total"].backward()

        ema_params = [param for name, param in model.named_parameters() if name.startswith("ema_target")]
        self.assertFalse(any(param.grad is not None for param in ema_params))
        ema_before = [param.detach().clone() for param in ema_params]

        optimizer.step()
        drift_before_update = float(model.ema_online_drift())
        self.assertGreater(drift_before_update, 0.0)

        model.update_ema_target_encoder(decay=0.5)
        ema_after = [param for name, param in model.named_parameters() if name.startswith("ema_target")]
        changed = [not torch.allclose(before, after) for before, after in zip(ema_before, ema_after)]
        self.assertTrue(any(changed))
        self.assertLess(float(model.ema_online_drift()), drift_before_update)

    def test_future_state_diagnostics_do_not_change_predictions(self) -> None:
        model = self._make_model()
        model.sync_ema_target_encoder()
        frames, future_state, future_mask = self._make_batch()

        base = model(frames)
        diagnostics_only = model(
            frames,
            future_state=future_state,
            use_ema_target=True,
            include_target_reconstruction=False,
        )

        self.assertNotIn("target_reconstruction", diagnostics_only)
        self.assertTrue(torch.allclose(base["mean"], diagnostics_only["mean"]))
        self.assertTrue(torch.allclose(base["log_std"], diagnostics_only["log_std"]))

        diagnostics = belief_jepa_diagnostics(diagnostics_only, future_state, future_mask)
        self.assertIn("latent_mse", diagnostics)
        self.assertIn("pred_target_cosine", diagnostics)
        self.assertTrue(torch.isfinite(diagnostics["latent_mse"]))

    def test_target_encoder_uses_future_trajectory_context(self) -> None:
        model = self._make_model()
        model.sync_ema_target_encoder()
        frames, future_state, _future_mask = self._make_batch()
        alternate_future = future_state.clone()
        alternate_future[:, 1:, :, 0:3] += 3.0
        alternate_future[:, 0] = future_state[:, 0]

        base = model(frames, future_state=future_state, use_ema_target=True)
        changed = model(frames, future_state=alternate_future, use_ema_target=True)

        self.assertTrue(torch.allclose(future_state[:, 0], alternate_future[:, 0]))
        self.assertFalse(torch.allclose(base["target_latent"][:, 0], changed["target_latent"][:, 0]))

    def test_mixture_belief_outputs_loss_and_particles(self) -> None:
        model = self._make_model()
        model.sync_ema_target_encoder()
        frames, future_state, future_mask = self._make_batch()
        outputs = model(frames, future_state=future_state, use_ema_target=True)

        self.assertEqual(outputs["mixture_logits"].shape, (2, 4, 2, 3))
        self.assertEqual(outputs["mixture_mean"].shape, (2, 4, 2, 3, 6))
        self.assertEqual(outputs["mixture_log_std"].shape, (2, 4, 2, 3, 6))

        losses = belief_jepa_loss(outputs, future_state, future_mask, mixture_belief_weight=0.25)
        self.assertIn("mixture_nll", losses)
        self.assertIn("mixture_entropy", losses)
        self.assertTrue(torch.isfinite(losses["mixture_nll"]))
        self.assertTrue(torch.isfinite(losses["mixture_entropy"]))

        particles, weights = particles_from_gaussian_mixture_sequence(
            outputs["mixture_logits"],
            outputs["mixture_mean"],
            outputs["mixture_log_std"],
            torch.ones(2, 2),
            ParticleBeliefConfig(num_particles=9),
        )
        self.assertEqual(particles.shape, (2, 4, 2, 9, 6))
        self.assertEqual(weights.shape, (2, 4, 2, 9))

    def test_structured_context_branch_changes_context_predictions(self) -> None:
        model = BeliefJEPA3D(
            max_objects=2,
            horizon=4,
            input_channels=3,
            cnn_dim=8,
            rnn_dim=16,
            latent_dim=8,
            structured_context=True,
            structured_dim=8,
        )
        frames, future_state, _future_mask = self._make_batch()
        obs_state = torch.zeros(2, 3, 2, 12)
        obs_mask = torch.ones(2, 3, 2)
        boxes = torch.zeros(2, 1, 6)
        structured = {
            "obs_state": obs_state,
            "obs_mask": obs_mask,
            "visual_occluders": boxes,
            "physical_obstacles": boxes,
            "solid_screens": boxes,
        }
        changed_structured = {key: value.clone() for key, value in structured.items()}
        changed_structured["obs_state"][..., 0:6] += 0.5
        changed_structured["physical_obstacles"][:, 0] = torch.tensor([-0.2, -0.2, -0.1, 0.2, 0.2, 0.1])

        base = model(frames, future_state=future_state, structured_context=structured, use_ema_target=True)
        changed = model(frames, future_state=future_state, structured_context=changed_structured, use_ema_target=True)

        self.assertTrue(model.use_structured_context)
        self.assertFalse(torch.allclose(base["pred_latent"], changed["pred_latent"]))

    def test_visual_geometry_weight_ignores_visual_occluder_motion(self) -> None:
        model = BeliefJEPA3D(
            max_objects=2,
            horizon=4,
            input_channels=3,
            cnn_dim=8,
            rnn_dim=16,
            latent_dim=8,
            structured_context=True,
            structured_dim=8,
            visual_geometry_weight=0.0,
        )
        frames, future_state, _future_mask = self._make_batch()
        obs_state = torch.zeros(2, 3, 2, 12)
        obs_mask = torch.ones(2, 3, 2)
        boxes = torch.zeros(2, 1, 6)
        structured = {
            "obs_state": obs_state,
            "obs_mask": obs_mask,
            "visual_occluders": boxes.clone(),
            "physical_obstacles": boxes.clone(),
            "solid_screens": boxes.clone(),
        }
        visual_changed = {key: value.clone() for key, value in structured.items()}
        visual_changed["visual_occluders"][:, 0] = torch.tensor([-0.8, -0.8, -0.8, -0.4, -0.4, -0.4])
        physical_changed = {key: value.clone() for key, value in structured.items()}
        physical_changed["physical_obstacles"][:, 0] = torch.tensor([-0.2, -0.2, -0.1, 0.2, 0.2, 0.1])

        base = model(frames, future_state=future_state, structured_context=structured, use_ema_target=True)
        visual = model(frames, future_state=future_state, structured_context=visual_changed, use_ema_target=True)
        physical = model(frames, future_state=future_state, structured_context=physical_changed, use_ema_target=True)

        self.assertTrue(torch.allclose(base["pred_latent"], visual["pred_latent"]))
        self.assertFalse(torch.allclose(base["pred_latent"], physical["pred_latent"]))

    def test_legacy_diagnostics_strip_untrained_mixture_outputs(self) -> None:
        outputs = {
            "mean": torch.zeros(1),
            "pred_latent": torch.zeros(1),
            "target_latent": torch.zeros(1),
            "mixture_logits": torch.zeros(1),
            "mixture_mean": torch.zeros(1),
            "mixture_log_std": torch.zeros(1),
        }
        legacy = jepa_diagnostic_outputs(outputs, mixture_enabled=False)
        current = jepa_diagnostic_outputs(outputs, mixture_enabled=True)

        self.assertNotIn("mixture_logits", legacy)
        self.assertNotIn("mixture_mean", legacy)
        self.assertNotIn("mixture_log_std", legacy)
        self.assertIn("mixture_logits", current)

    def test_sigreg_is_finite_and_contributes_to_loss(self) -> None:
        model = self._make_model()
        model.sync_ema_target_encoder()
        frames, future_state, future_mask = self._make_batch()
        outputs = model(frames, future_state=future_state, use_ema_target=True)

        no_sigreg = belief_jepa_loss(outputs, future_state, future_mask, sigreg_weight=0.0)
        with_sigreg = belief_jepa_loss(outputs, future_state, future_mask, sigreg_weight=0.2, sigreg_sketches=8)

        self.assertIn("sigreg", with_sigreg)
        self.assertIn("pred_sigreg", with_sigreg)
        self.assertIn("target_sigreg", with_sigreg)
        self.assertTrue(torch.isfinite(with_sigreg["sigreg"]))
        self.assertGreater(float(with_sigreg["total"] - no_sigreg["total"]), 0.0)

    def test_sigreg_handles_empty_masks(self) -> None:
        latents = torch.randn(2, 3, 4, 5)
        mask = torch.zeros(2, 3, 4)
        loss = sketched_isotropic_gaussian_regularizer(latents, mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
