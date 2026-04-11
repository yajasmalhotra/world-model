from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.scene_generator import SceneConfig, SyntheticSceneGenerator
from src.models.encoder import PixelToStateEncoder
from src.models.state_dynamics import ObjectCentricDynamics, apply_counterfactual
from src.train.utils import get_device, load_checkpoint, load_config


def latest_checkpoint(pattern: str) -> str | None:
    candidates = sorted((ROOT / "runs").glob(pattern))
    if not candidates:
        return None
    return str(candidates[-1])


def to_scene_config(data_cfg: Dict) -> SceneConfig:
    keys = {
        "image_size",
        "seq_len",
        "obs_len",
        "min_objects",
        "max_objects",
        "min_occluders",
        "max_occluders",
        "velocity_scale",
        "object_size_min",
        "object_size_max",
        "occluder_layout",
    }
    subset = {k: v for k, v in data_cfg.items() if k in keys}
    return SceneConfig(**subset)


@st.cache_resource
def load_models(config_path: str, joint_ckpt: str | None, dynamics_ckpt: str | None, encoder_ckpt: str | None):
    config = load_config(config_path)
    device = get_device(config["project"].get("device", "auto"))
    model_cfg = config["model"]

    encoder = PixelToStateEncoder(
        max_objects=int(model_cfg["max_objects"]),
        state_dim=int(model_cfg["state_dim"]),
        cnn_dim=int(model_cfg["encoder_cnn_dim"]),
        rnn_dim=int(model_cfg["encoder_rnn_dim"]),
        image_size=int(config["data"]["image_size"]),
    ).to(device)
    dynamics = ObjectCentricDynamics(
        state_dim=int(model_cfg["state_dim"]),
        dynamic_dim=int(model_cfg["dynamic_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        interaction_dim=int(model_cfg["interaction_dim"]),
        max_occluders=int(model_cfg["max_occluders"]),
    ).to(device)

    if joint_ckpt:
        ckpt = load_checkpoint(joint_ckpt, device)
        encoder.load_state_dict(ckpt["encoder_state"], strict=False)
        dynamics.load_state_dict(ckpt["dynamics_state"], strict=False)
    else:
        if encoder_ckpt:
            ckpt = load_checkpoint(encoder_ckpt, device)
            encoder.load_state_dict(ckpt.get("model_state", ckpt.get("encoder_state")), strict=False)
        if dynamics_ckpt:
            ckpt = load_checkpoint(dynamics_ckpt, device)
            dynamics.load_state_dict(ckpt.get("model_state", ckpt.get("dynamics_state")), strict=False)

    encoder.eval()
    dynamics.eval()
    return config, device, encoder, dynamics


def plot_trajectories(gt_state: np.ndarray, pred_state: np.ndarray, cf_state: np.ndarray, object_mask: np.ndarray):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    num_obj = gt_state.shape[1]
    for o in range(num_obj):
        valid = object_mask[:, o] > 0.5
        if not np.any(valid):
            continue
        ax.plot(gt_state[valid, o, 0], gt_state[valid, o, 1], "-", linewidth=2, label=f"GT obj{o}")
        ax.plot(pred_state[valid, o, 0], pred_state[valid, o, 1], "--", linewidth=2, label=f"Pred obj{o}")
        ax.plot(cf_state[valid, o, 0], cf_state[valid, o, 1], ":", linewidth=2, label=f"CF obj{o}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Future Trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="World Model Demo", layout="wide")
    st.title("Object-Centric World Model Demo")
    st.caption("Observed frames -> world state -> future rollout with counterfactual intervention")

    config_path = str(ROOT / "configs" / "default.yaml")
    default_joint = latest_checkpoint("*_train_joint/checkpoints/best.pt")
    default_dynamics = latest_checkpoint("*_train_dynamics/checkpoints/best.pt")
    default_encoder = latest_checkpoint("*_train_encoder/checkpoints/best.pt")

    with st.sidebar:
        st.header("Model and Scene")
        joint_ckpt = st.text_input("Joint checkpoint", value=default_joint or "")
        dynamics_ckpt = st.text_input("Dynamics checkpoint", value=default_dynamics or "")
        encoder_ckpt = st.text_input("Encoder checkpoint", value=default_encoder or "")
        seed = st.number_input("Scene seed", min_value=0, max_value=10_000_000, value=2026, step=1)
        counterfactual_object = st.slider("Counterfactual object idx", min_value=0, max_value=3, value=0, step=1)
        dvx = st.slider("Delta vx", min_value=-0.1, max_value=0.1, value=0.04, step=0.005)
        dvy = st.slider("Delta vy", min_value=-0.1, max_value=0.1, value=-0.02, step=0.005)
        run_button = st.button("Run Demo")

    try:
        config, device, encoder, dynamics = load_models(
            config_path,
            joint_ckpt.strip() or None,
            dynamics_ckpt.strip() or None,
            encoder_ckpt.strip() or None,
        )
    except Exception as exc:
        st.error(f"Model load failed: {exc}")
        return

    if not run_button:
        st.info("Set options in the sidebar and click 'Run Demo'.")
        return

    scene_cfg = to_scene_config(config["data"])
    generator = SyntheticSceneGenerator(scene_cfg)
    sample = generator.generate(seed=int(seed))

    obs_len = scene_cfg.obs_len
    horizon = sample["state"].shape[0] - obs_len

    obs_frames = torch.from_numpy(sample["frames"][:obs_len].astype(np.float32) / 255.0).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    obs_state = torch.from_numpy(sample["state"][:obs_len]).unsqueeze(0).to(device)
    object_mask = torch.from_numpy(sample["object_mask"][obs_len - 1]).unsqueeze(0).to(device)
    occluders = torch.from_numpy(sample["occluders"]).unsqueeze(0).to(device)

    with torch.no_grad():
        init_state = encoder(obs_frames)
        init_state[..., 6:] = obs_state[:, -1, :, 6:]
        pred_state = dynamics(init_state, object_mask, occluders, horizon=horizon)
        cf_init = apply_counterfactual(
            init_state,
            object_idx=int(counterfactual_object),
            intervention={"vx": float(dvx), "vy": float(dvy)},
        )
        cf_state = dynamics(cf_init, object_mask, occluders, horizon=horizon)

    gt_future = sample["state"][obs_len:]
    future_mask = sample["object_mask"][obs_len:]
    pred_np = pred_state.squeeze(0).cpu().numpy()
    cf_np = cf_state.squeeze(0).cpu().numpy()

    pred_frames = generator.render_sequence(pred_np, future_mask, sample["occluders"])
    cf_frames = generator.render_sequence(cf_np, future_mask, sample["occluders"])
    gt_frames = sample["frames"][obs_len:]

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.subheader("Observed Frames")
        st.image(list(sample["frames"][:obs_len]), width=120)
    with col_b:
        st.subheader("Predicted Future")
        st.image(list(pred_frames), width=120)
    with col_c:
        st.subheader("Counterfactual Future")
        st.image(list(cf_frames), width=120)

    st.subheader("Trajectory Comparison")
    fig = plot_trajectories(gt_future, pred_np, cf_np, future_mask)
    st.pyplot(fig)

    error = np.sqrt(((pred_np[..., 0:2] - gt_future[..., 0:2]) ** 2).sum(axis=-1))
    masked_error = (error * future_mask).sum() / max(future_mask.sum(), 1.0)
    st.metric("Future Position RMSE (approx)", f"{masked_error:.4f}")

    st.subheader("Ground Truth vs Predicted (First Future Frame)")
    compare = np.concatenate([gt_frames[0], pred_frames[0], cf_frames[0]], axis=1)
    st.image(compare, caption="left: GT, middle: predicted, right: counterfactual", width=420)


if __name__ == "__main__":
    main()

