# Project Skeleton: Hidden-Trajectory Calibration

## One-Line Concept

A controlled 3D benchmark and baseline system for measuring whether object-permanence world models maintain calibrated beliefs over true hidden object trajectories during occlusion.

## Why This Is Not Just Another Object-Permanence Demo

Close prior work already covers pieces of the space:

- Object permanence with physics and Gaussian splatting: PersistGS.
- Broad 3D embodied belief inference from partial observations: 3D-Belief.
- Particle-filter hidden-object tracking: Object Permanence Filter.
- Object-centric latent dynamics: SlotFormer and related models.
- Violation-of-expectation benchmarks: X-VoE, IntPhys, V-JEPA-style intuitive physics work.
- Uncertainty-aware Gaussian splatting: GAVIS, USplat4D, PhysGS-like directions.

The project should therefore focus on a narrower contribution: frame-by-frame hidden-trajectory calibration. The rendered belief field is the interface; the scientific object is whether probability mass stays correlated and calibrated with the true hidden 3D trajectory.

## Core Research Question

When an object is fully occluded, does a world model preserve a belief distribution that remains correlated with the object's true hidden 3D trajectory, and is that belief calibrated rather than merely visually plausible?

## Proposed Repository Additions

```text
configs/
  belief3d_smoke.yaml          # tiny CPU/Kaggle smoke config
  belief3d.yaml                # main Kaggle/Colab config

src/data3d/
  __init__.py
  scene_generator3d.py         # synthetic 3D states, cameras, occluders
  camera.py                    # camera intrinsics/extrinsics, projection helpers
  render3d.py                  # simple low-res projection renderer
  manifest3d.py                # seed manifests and split metadata
  dataset3d.py                 # PyTorch dataset, generated on the fly

src/models/
  belief_state.py              # particle/Gaussian belief container utilities
  belief_encoder3d.py          # supervised image-to-belief encoder
  belief_dynamics3d.py         # propagates belief through time
  encoder3d.py                 # optional multi-view/image-to-belief encoder
  belief_world_model3d.py      # encoder + belief dynamics wrapper

src/eval/
  belief_metrics.py            # NLL, coverage, mass around truth, calibration
  belief_runner3d.py           # evaluation loops and plots

scripts/
  build_manifests3d.py
  evaluate_belief3d.py
  train_belief3d.py
  export_belief3d_demo_assets.py

notebooks/
  kaggle_belief3d_train.ipynb  # optional reproducible Kaggle notebook

docs/
  FRESH_AGENT_PROMPT.md
  PROJECT_SKELETON_3D_BELIEF.md
  KAGGLE_WORKFLOW.md
```

This skeleton should be introduced incrementally. Do not rename or remove the existing 2D modules until the 3D path is stable.

## V1 Technical Scope

### Data

- World bounds: unit cube or `[-1, 1]^3`.
- Objects: 2-5 spheres/cubes with colors and radii.
- Motion: constant velocity with wall bounces.
- Occluders: axis-aligned boxes between camera and objects.
- Cameras: start with one fixed perspective camera; add 2-3 views later.
- Sequence length: 16-32.
- Resolution: 64x64 for smoke/main, with a local 128x128 stress-test config for nicer 3D renders.
- State: `[x, y, z, vx, vy, vz, visible, occluded, size, shape_id, color_id, object_id]`.
- Observations: RGB camera frames plus normalized depth maps. The raster resolution is 2D, but the scene state, projection, occlusion, and belief target are 3D.

### Belief Representation

Start with one of these, in this order:

1. Particle belief: `N` weighted particles per object in `(x, y, z, vx, vy, vz)`.
2. Gaussian mixture belief: `K` components with mean/covariance/weight.
3. Dense voxel belief only if needed later; avoid for v1.

Particle belief is easiest to debug and naturally supports physics constraints.

### Dynamics

- Deterministic visible update when object is observed.
- During occlusion, propagate particles with velocity noise and wall constraints.
- Optional learned correction network predicts velocity noise, diffusion scale, or mixture updates.
- Later: SDF constraints for occluder/wall geometry and collision constraints.

### Rendering

- Render observed RGB frames with a CGAI-inspired perspective camera: projection, depth buffering, shaded spheres/cubes, and projected occluder faces.
- Emit per-frame depth maps so later models can compare RGB-only, depth-only, and RGB-D belief inference.
- Render belief by projecting particles/Gaussians into the camera as translucent splats.
- For 3D demo views, use matplotlib/plotly/three.js-style point clouds first.
- Avoid training a full 3DGS/NeRF renderer in v1.

## Metrics

Let `B_t(p)` be the normalized belief density over 3D position and `x_t` the true hidden position.

- **NLL:** `-log B_t(x_t)`
- **Belief mass around truth:** `sum_{||p - x_t|| < r} B_t(p)`
- **Expected distance:** `E_{p ~ B_t}[||p - x_t||]`
- **Top-k particle distance:** distance from true position to nearest high-weight particles.
- **Credible-region calibration:** 50/70/90 percent regions should contain truth about 50/70/90 percent of the time.
- **Occlusion degradation:** plot each metric against occlusion duration.
- **Surprise:** low belief density at a reappearing location should produce high surprise.

## Baselines

- Constant-velocity point prediction.
- Physics-only particle filter with hand-tuned noise.
- Existing deterministic state dynamics lifted to 3D.
- Optional direct pixel predictor if time allows.

## Kaggle/Colab Plan

Design for free GPU notebooks:

- Generate scenes from seed manifests instead of storing rendered datasets.
- Checkpoint every epoch.
- Keep jobs under 8 hours.
- Use smoke configs before main configs.
- Save metrics/plots/checkpoints as notebook outputs.

Approximate first-pass compute target:

- Smoke run: CPU or small GPU, minutes.
- Main run: 2-6 hours on Kaggle T4/P100-class GPU.
- Ablations: several short runs rather than one giant run.

## Milestones

### Milestone 1: 3D Simulator and Renderer

- Generate deterministic 3D scenes from seeds.
- Render low-res camera observations.
- Record ground-truth hidden trajectories.
- Add smoke tests for shape, visibility, and reproducibility.

### Milestone 2: Physics Particle Belief Baseline

- Initialize particle belief from visible state.
- Propagate belief during occlusion.
- Render belief splats.
- Compute NLL/mass/distance/calibration.

Initial implementation status: `scripts/build_manifests3d.py`, `scripts/evaluate_belief3d.py`, `src/data3d/`, `src/models/belief_state.py`, and `src/eval/belief_metrics.py` provide the first smoke-testable version of this baseline.

Renderer status: `src/data3d/scene_generator3d.py` now generates perspective RGB frames and normalized depth maps from the same 3D state. `src/data3d/dataset3d.py` exposes those as `obs_frames`, `future_frames`, `obs_depth`, and `future_depth`.

### Milestone 2b: Supervised Image-to-Belief Encoder

- Encode observed RGB frame sequences with a small CNN+GRU.
- Predict per-slot Gaussian belief parameters for final observed 3D position and velocity.
- Train with supervised Gaussian NLL against synthetic ground-truth state.
- Initialize belief particles from the learned Gaussian and evaluate hidden-trajectory calibration after rollout.

Initial implementation status: `src/models/belief_encoder3d.py` and `scripts/train_belief3d.py` provide the first learned image-to-belief path.

### Milestone 3: Learned Belief Dynamics

- Add a small model to predict belief update parameters.
- Compare against physics-only baseline.
- Track occlusion-duration degradation.

### Milestone 4: Surprise and Impossible Events

- Add impossible reappearance splits: teleportation, identity swap, wall penetration, impossible bounce.
- Score surprise from belief density at reappearance.
- Compare to deterministic baselines.

### Milestone 5: Demo and Writeup

- Interactive demo: observed frames, belief splats, true trajectory toggle, surprise score.
- README figures.
- Short methodology/results report.

Initial implementation status: `scripts/export_belief3d_demo_assets.py` exports GIF/PNG/JSON demo assets showing RGB observations, depth maps, camera-projected belief particles, true 3D trajectory, expected distance, mass near truth, and surprise.

## Design Principle

Do not chase photorealism or scale. The project wins by making hidden beliefs measurable, visual, and physically interpretable.
