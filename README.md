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

## Datasets

This repository includes the lightweight manifest datasets used by the project under `data/manifests/`. The full image/state sequences are regenerated deterministically from manifest seeds, so the repo stays small while preserving reproducibility.

Included splits:

- `train.jsonl`: in-distribution training scenes
- `val.jsonl`: validation scenes
- `test.jsonl`: in-distribution test scenes
- `test_long_occlusion.jsonl`: longer rollouts with extended occlusion
- `test_unseen_speed.jsonl`: higher-speed generalization scenes
- `test_unseen_occluders.jsonl`: shifted occluder-layout generalization scenes
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

## Storage Hygiene

- Keep only the best checkpoint per run.
- Remove large media files from older runs.
- Regenerate synthetic training scenes on the fly instead of saving full datasets.
