from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.evaluate_belief3d import load_image_encoder as load_eval_image_encoder
from scripts.export_belief3d_demo_assets import load_image_encoder as load_demo_image_encoder
from scripts.train_belief3d import build_model, context_frames
from src.models.belief_encoder3d import ImageToBeliefEncoder3D
from src.train.utils import load_config, save_checkpoint


class BeliefEncoderRGBDTest(unittest.TestCase):
    def test_encoder_accepts_rgb_and_rgbd_inputs(self) -> None:
        rgb = ImageToBeliefEncoder3D(max_objects=2, input_channels=3, cnn_dim=8, rnn_dim=8)
        rgbd = ImageToBeliefEncoder3D(max_objects=2, input_channels=4, cnn_dim=8, rnn_dim=8)
        self.assertEqual(rgb(torch.zeros(1, 2, 3, 16, 16))["mean"].shape, (1, 2, 6))
        self.assertEqual(rgbd(torch.zeros(1, 2, 4, 16, 16))["mean"].shape, (1, 2, 6))

    def test_context_frames_concatenates_depth_for_rgbd(self) -> None:
        batch = {
            "obs_frames": torch.zeros(2, 3, 3, 8, 8),
            "obs_depth": torch.ones(2, 3, 1, 8, 8),
        }
        rgb = context_frames(batch, torch.device("cpu"), rgbd=False)
        rgbd = context_frames(batch, torch.device("cpu"), rgbd=True)
        self.assertEqual(rgb.shape[2], 3)
        self.assertEqual(rgbd.shape[2], 4)
        self.assertTrue(torch.all(rgbd[:, :, 3:] == 1.0))

    def test_rgbd_checkpoint_loaders_restore_four_channel_model(self) -> None:
        config = load_config("configs/belief3d_smoke.yaml")
        device = torch.device("cpu")
        model = build_model(config, device, rgbd=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best.pt"
            save_checkpoint(
                ckpt_path,
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": 1,
                    "best_metric": 0.0,
                    "rgbd": True,
                },
            )
            eval_model, eval_rgbd = load_eval_image_encoder(config, device, str(ckpt_path))
            demo_model, demo_rgbd = load_demo_image_encoder(config, device, str(ckpt_path))
        self.assertTrue(eval_rgbd)
        self.assertTrue(demo_rgbd)
        self.assertEqual(eval_model.input_channels, 4)
        self.assertEqual(demo_model.input_channels, 4)


if __name__ == "__main__":
    unittest.main()
