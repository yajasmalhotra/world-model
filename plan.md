# World Model Implementation Spec

## Summary
- Build a self-contained project at `Term Project/world_model_project` with a dedicated conda env `term-project-wm`.
- Use `Python 3.11`, `PyTorch 2.x`, `numpy`, `pandas`, `matplotlib`, `pillow`, `imageio`, `pyyaml`, `tqdm`, `scikit-learn`, `tensorboard`, and `streamlit`.
- Keep the system technically faithful to the pitch by supporting both required paths:
  - `ground-truth state -> dynamics model -> renderer`
  - `pixels -> temporal object encoder -> dynamics model -> renderer`
- Make the structured world model the core of the system. The pixel path feeds that same core rather than replacing it with a generic video predictor.
- Keep the environment CPU-feasible on an Apple Silicon MacBook Air. If `mps` is available it can be used opportunistically, but the spec must not depend on it.

## Project Structure
- Create `Term Project/world_model_project` as the only new code root.
- Inside it, use this layout:
```text
world_model_project/
  app.py
  configs/
  scripts/
  src/
  data/
  runs/
  results/
  assets/
  docs/
```
- `src` contains four subsystems only: `data`, `models`, `train`, and `eval`.
- `data` stores small manifests and cached demo/eval scenes only. Training data is generated on demand from seeds to avoid wasting disk.
- `runs` stores per-run checkpoints, logs, and media.
- `results` stores aggregated tables, plots, and paper-ready figures.

## Technical Implementation
- Scene generator:
  - Generate `64x64` RGB sequences with `2-4` uniquely identifiable objects, `1-2` static rectangular occluders, and fixed object IDs.
  - Use simple motion with wall bounces and no object-object collisions.
  - Store object state per timestep: `x`, `y`, `vx`, `vy`, `visible`, `occluded`, `shape_id`, `color_id`, `size`, and `object_id`.
  - Generate fixed train/val/test manifests from seeds so experiments are reproducible.
- Core world model:
  - Implement a state-space dynamics model that consumes the observed scene state at the final observation step and rolls it forward autoregressively.
  - Use padded fixed slots for up to `4` objects plus a mask.
  - Use a lightweight recurrent transition model with per-object updates and scene-context mixing.
  - Render predicted future frames from predicted states with the deterministic renderer from data generation.
- Pixel-input path:
  - Implement a temporal encoder that ingests observed frame sequences and predicts structured object state at the final observed timestep.
  - Train this encoder with synthetic labels.
  - Compose `encoder + dynamics + renderer` into the final demo pipeline.
  - Keep a direct `ground-truth state` entrypoint for ablation/debugging.
- Baselines:
  - Kinematic extrapolation baseline from visible states.
  - Small direct pixel predictor baseline for comparison.
- Counterfactuals:
  - Support interventions on `x/y/vx/vy` for one object at one timestep and re-roll future predictions.
  - Expose this in scripts and Streamlit.

## Interfaces and Result Tracking
- Canonical scene format:
  - `frames`: `[T, H, W, 3]`
  - `state`: `[T, O, F]` with `O=4` padded slots and a slot mask
  - `metadata`: sequence seed, split, occluder layout, and generalization tags
- Canonical run format:
  - `runs/<timestamp>_<name>/config.yaml`
  - `runs/<timestamp>_<name>/metrics.csv`
  - `runs/<timestamp>_<name>/summary.json`
  - `runs/<timestamp>_<name>/checkpoints/best.pt`
  - `runs/<timestamp>_<name>/media/`
- Required scripts:
  - `scripts/build_manifests.py`
  - `scripts/train_dynamics.py`
  - `scripts/train_encoder.py`
  - `scripts/train_joint.py`
  - `scripts/train_pixel_baseline.py`
  - `scripts/evaluate.py`
  - `scripts/aggregate_results.py`
  - `scripts/export_demo_assets.py`
- `summary.json` includes model type, split, seed, metrics, checkpoint paths, and artifact paths.
- `aggregate_results.py` outputs paper-ready CSV/PNG artifacts in `results/`.

## Demo and Evaluation
- Streamlit app loads checkpoints, runs a seed scene, and displays observed frames + predicted future trajectories and frames.
- Counterfactual controls: object selection, timestep, velocity/position edits.
- Required evaluation slices:
  - in-distribution test scenes
  - longer occlusions
  - unseen speed ranges
  - unseen occluder layouts
- Required metrics:
  - position error during occlusion
  - position error at first reappearance
  - identity consistency after occlusion
  - rollout frame error
  - counterfactual locality

## Assumptions and Storage
- The existing `ml4t` conda env can be deleted (estimated `~568 MB` reclaimed).
- Recommended free space before experimentation: at least `12 GiB`.
- Estimated steady-state storage:
  - conda env: `2.5-3.5 GiB`
  - code/config/docs: `<100 MB`
  - manifests and cached eval/demo scenes: `0.3-1.0 GiB`
  - checkpoints/logs/plots/media: `1.0-2.5 GiB`
  - total: `4.0-7.0 GiB`
- Estimated temporary peak during install and heavier experimentation: `6.0-9.0 GiB`.
