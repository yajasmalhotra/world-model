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

Train and evaluate the first supervised image-to-belief encoder:

```bash
python scripts/train_belief3d.py --config configs/belief3d_smoke.yaml
python scripts/evaluate_belief3d.py --config configs/belief3d_smoke.yaml --mode all
```

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

Export a visual 3D belief demo with RGB frames, depth, projected particles, the true hidden trajectory, and per-frame belief metrics:

```bash
python scripts/export_belief3d_demo_assets.py --config configs/belief3d_smoke.yaml --seeds 2026
```

By default the 3D demo uses the targeted occlusion scenario and writes target metadata, including `object_index`, `occlusion_start`, `occlusion_end`, `reappearance_frame`, `hidden_frames`, and target occluder indices.

This path is additive and does not modify the original 2D training pipeline.

3D batches also expose `obs_depth` and `future_depth` tensors for future RGB-D or depth-supervised experiments. Current training scripts continue to use RGB `obs_frames`, so the extra depth channel is available without changing the existing MVP training loop.

## Storage Hygiene

- Keep only the best checkpoint per run.
- Remove large media files from older runs.
- Regenerate synthetic training scenes on the fly instead of saving full datasets.
