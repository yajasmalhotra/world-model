---
license: mit
datasets:
- yajasm/world-model
papers:
- https://doi.org/10.5281/zenodo.20725115
---

# World Model Project

This project implements a CPU-feasible object-centric world model for synthetic visual scenes with occlusion, plus:

- pixel-to-state encoder
- state dynamics model
- direct pixel baseline
- evaluation and result aggregation pipeline
- Streamlit demo with counterfactual rollouts

## Project Direction

The next research direction is documented in:

- [Fresh agent prompt](docs/FRESH_AGENT_PROMPT.md)
- [3D belief project skeleton](docs/PROJECT_SKELETON_3D_BELIEF.md)
- [Kaggle workflow](docs/KAGGLE_WORKFLOW.md)

The short version: evolve the existing 2D object-centric world model into a 3D hidden-trajectory calibration benchmark. The model maintains a belief distribution over hidden object locations during occlusion, and the benchmark scores whether that belief stays correlated and calibrated with the true hidden 3D trajectory.

The 3D path now renders low-resolution camera observations from true 3D scene state: `128x128` means the camera image size, while object positions, velocities, occluders, visibility, and belief metrics live in `(x, y, z)`. The renderer uses a fixed perspective camera, depth buffering, simple shaded spheres/cubes, projected occluder faces, and per-frame normalized depth maps inspired by the projection, lighting, and visibility concepts from CGAI.

## Datasets

This repository includes the lightweight manifest datasets used by the project under `data/manifests/`. The full image/state sequences are regenerated deterministically from manifest seeds, so the repo stays small while preserving reproducibility.

Included splits:

- `train.jsonl`: in-distribution training scenes
- `val.jsonl`: validation scenes
- `test.jsonl`: in-distribution test scenes
- `test_long_occlusion.jsonl`: longer rollouts with extended occlusion
- `test_unseen_speed.jsonl`: higher-speed generalization scenes
- `test_unseen_occluders.jsonl`: shifted occluder-layout generalization scenes
- `test_targeted_occlusion.jsonl`: controlled 3D object-permanence episodes where a target object starts visible, passes behind one or more varied occluders, and reappears
- `test_structured_occlusion.jsonl`: targeted hidden episodes with curved or bounced hidden dynamics around physical obstacles
- `test_impossible_reappearance.jsonl`: targeted episodes where the target reappears in a physically unlikely/impossible location for surprise scoring
- `manifest_index.json`: split metadata and counts

The corresponding Hugging Face dataset is `yajasm/world-model`, listed in the metadata block above. Generated caches remain local under `data/cache/` and are intentionally ignored.

## Setup

```bash
cd "Term Project/world_model_project"
conda env create -f environment.yml
conda activate term-project-wm
```

## Quick Start

1. Build manifests:
```bash
python scripts/build_manifests.py --config configs/default.yaml
```
2. Train state dynamics:
```bash
python scripts/train_dynamics.py --config configs/default.yaml
```
3. Train pixel encoder:
```bash
python scripts/train_encoder.py --config configs/default.yaml
```
4. Train joint model:
```bash
python scripts/train_joint.py --config configs/default.yaml
```
5. Evaluate:
```bash
python scripts/evaluate.py --config configs/default.yaml
```
6. Aggregate results:
```bash
python scripts/aggregate_results.py --runs-dir runs --output-dir results
```
7. Demo:
```bash
streamlit run app.py
```

## Smoke Test (fast)

Use the tiny config to verify the full stack in minutes:

```bash
python scripts/build_manifests.py --config configs/smoke.yaml
python scripts/train_dynamics.py --config configs/smoke.yaml
python scripts/train_encoder.py --config configs/smoke.yaml
python scripts/train_joint.py --config configs/smoke.yaml
python scripts/train_pixel_baseline.py --config configs/smoke.yaml
python scripts/evaluate.py --config configs/smoke.yaml --mode all
python scripts/aggregate_results.py --runs-dir runs --output-dir results
```

## 3D Belief Smoke Test

Build tiny 3D manifests and evaluate the physics-only particle belief baseline:

```bash
python scripts/build_manifests3d.py --config configs/belief3d_smoke.yaml
python scripts/evaluate_belief3d.py --config configs/belief3d_smoke.yaml
```

Train and evaluate the learned baselines:

```bash
python scripts/train_belief3d.py --config configs/belief3d_smoke.yaml
python scripts/train_belief_jepa3d.py --config configs/belief3d_smoke.yaml
python scripts/evaluate_belief3d.py --config configs/belief3d_smoke.yaml --mode all
```

Train the supervised image-to-belief baseline with RGB-D context:

```bash
python scripts/train_belief3d.py --config configs/belief3d_smoke.yaml --rgbd
```

Belief-JEPA training uses an EMA target encoder by default. For an ablation:

```bash
python scripts/train_belief_jepa3d.py --config configs/belief3d_smoke.yaml --no-ema
python scripts/train_belief_jepa3d.py --config configs/belief3d_smoke.yaml --sigreg-weight 0.0
```

The JEPA run logs latent diagnostics including `latent_mse`, `target_recon_mse`, `pred_target_cosine`, `target_latent_std`, `pred_latent_std`, and `ema_online_drift`; evaluation adds `jepa_*` diagnostics while keeping predictions context-only. The context branch fuses RGB/RGB-D with observed object state, object masks, physical obstacles, and solid screens when `jepa_structured_context: true`; visual-only occluder boxes are weighted by `jepa_visual_geometry_weight` and default to zero direct dynamics influence. Training can also add visual-only counterfactual invariance with `visual_invariance_weight` and deterministic geometry-teacher counterfactual deltas with `geometry_teacher_weight`. The prediction head can blend in a deterministic physical rollout prior through `jepa_geometry_prior_weight` and tighten prior uncertainty with `jepa_geometry_prior_log_std`, giving the latent predictor a stronger obstacle-sensitive belief scaffold while keeping future targets unavailable at inference. The target branch encodes privileged future state with a trajectory-aware temporal encoder before the stop-gradient/EMA target loss. The learned belief head predicts both a bounded Gaussian and a small Gaussian mixture for multimodal hidden futures. Belief-JEPA also includes a lightweight LeJEPA-inspired sketched Gaussian latent regularizer via `sigreg_weight`, `sigreg_sketches`, and `sigreg_scale`; set `sigreg_weight: 0.0` for ablation.

Write an EMA/SIGReg ablation table from the latest JEPA checkpoints:

```bash
python scripts/report_belief_jepa3d_ablation.py --config configs/belief3d_smoke.yaml
```

`evaluate_belief3d.py --mode all` compares:

- `constant_velocity_particle_belief`
- `geometry_aware_particle_belief`
- `image_to_belief`
- `belief_jepa_latent_predictor`

Geometry-aware evaluation also reports counterfactual sensitivity: `counterfactual_physical_belief_delta` should rise when physical obstacles are moved, while `counterfactual_visual_belief_delta` should stay near zero for visual-only changes.

Write a compact benchmark report from the latest Belief3D evaluation run:

```bash
python scripts/report_belief3d.py --output-dir results/belief3d_report
```

The report includes target-only aggregate metrics and path-mode slices for linear, bounce, curved, and impossible-jump hidden dynamics, so structured failures are visible instead of averaged away.

For a more meaningful local MVP run:

```bash
python scripts/build_manifests3d.py --config configs/belief3d_mvp.yaml
python scripts/train_belief3d.py --config configs/belief3d_mvp.yaml
python scripts/evaluate_belief3d.py --config configs/belief3d_mvp.yaml --mode all
```

For the larger local stress-test scene setting (`128x128`, 40 frames, up to 10 objects, 512 particles/object):

```bash
python scripts/build_manifests3d.py --config configs/belief3d_large.yaml
python scripts/train_belief3d.py --config configs/belief3d_large.yaml
python scripts/evaluate_belief3d.py --config configs/belief3d_large.yaml --mode all
```

Export a visual 3D belief demo with GIF/MP4/PNG assets plus JSON metrics for RGB frames, depth, projected particles, the true hidden trajectory, and per-frame distance, mass, surprise, entropy, and calibration:

```bash
python scripts/export_belief3d_demo_assets.py --config configs/belief3d_smoke.yaml --seeds 2026 --scenario structured_occlusion --mode compare
```

Use `--skip-mp4` on machines without an MP4 writer.

Include learned image-to-belief and Belief-JEPA branches when checkpoints are available:

```bash
python scripts/export_belief3d_demo_assets.py --config configs/belief3d_smoke.yaml --seeds 2026 --scenario structured_occlusion --mode compare_all
```

Render the comparison chart while putting the learned JEPA trace in the main belief panels:

```bash
python scripts/export_belief3d_demo_assets.py --config configs/belief3d_smoke.yaml --seeds 2026 --scenario structured_occlusion --mode compare_all --primary-method jepa --output-dir results/belief3d_demo_jepa
```

Export an impossible-reappearance companion demo to exercise the impossible-event phase label:

```bash
python scripts/export_belief3d_demo_assets.py --config configs/belief3d_smoke.yaml --seeds 2026 --scenario impossible_reappearance --mode compare_all --output-dir results/belief3d_demo_impossible
```

Audit the current Belief3D evidence package against the research/demo requirements:

```bash
python scripts/audit_belief3d_evidence.py --output-dir results/belief3d_audit
```

The 3D generator distinguishes visual occluders, physical obstacles, and solid screens. Targeted manifest rows store compact `target` metadata including `target_object_index`, `scenario`, `path_mode`, `occlusion_start`, `occlusion_end`, `reappearance_frame`, `hidden_frames`, `obstacle_ids`, `occluder_ids`, `collision_or_turn_frames`, `valid_route_id`, and `is_impossible_event`.

This path is additive and does not modify the original 2D training pipeline.

3D batches expose `obs_depth` and `future_depth` tensors. The supervised image-to-belief and Belief-JEPA training scripts can use RGB-D context via `--rgbd`, while the default RGB path remains unchanged.

## Storage Hygiene

- Keep only the best checkpoint per run.
- Remove large media files from older runs.
- Regenerate synthetic training scenes on the fly instead of saving full datasets.
