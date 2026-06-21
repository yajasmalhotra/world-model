from __future__ import annotations

import unittest

import torch

from src.models.belief_jepa3d import BeliefJEPA3D, belief_jepa_diagnostics, belief_jepa_loss


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

        self.assertFalse(any(param.grad is not None for param in model.ema_target_encoder.parameters()))
        ema_before = [param.detach().clone() for param in model.ema_target_encoder.parameters()]

        optimizer.step()
        drift_before_update = float(model.ema_online_drift())
        self.assertGreater(drift_before_update, 0.0)

        model.update_ema_target_encoder(decay=0.5)
        changed = [not torch.allclose(before, after) for before, after in zip(ema_before, model.ema_target_encoder.parameters())]
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


if __name__ == "__main__":
    unittest.main()
